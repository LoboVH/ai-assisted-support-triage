
import json
import os
import time
import hashlib
import re

from google import genai
from google.genai import errors

try:
    from .env_config import load_project_env
except ImportError:
    from env_config import load_project_env

try:
    from .ticket_reporting import customer_report_summary
except ImportError:
    from ticket_reporting import customer_report_summary

load_project_env()

try:
    from .llm_cache import LLMCache
except ImportError:
    from llm_cache import LLMCache

try:
    from .models import LLMCallResult
except ImportError:
    from models import LLMCallResult


MODEL_NAME = "gemini-3.8-flash"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2

VALID_CATEGORIES = {
    "withdrawal_delay",
    "account_security",
    "account_access",
    "trade_dispute",
    "general_query",
}
VALID_URGENCIES = {"critical", "high", "medium", "low"}
VALID_SENTIMENTS = {"negative", "neutral", "positive", "mixed"}
VALID_RESOLUTION_MODES = {
    "reply_only",
    "needs_human_review",
    "needs_specialist_escalation",
}
VALID_QUEUES = {
    "payments_ops",
    "trust_and_safety",
    "customer_support",
    "trading_ops",
    "account_operations",
}


def _redact_provider_secrets(message) -> str:
    """Remove configured API-key values before errors reach logs or callers."""
    safe_message = str(message)
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        secret = os.getenv(name)
        if secret:
            safe_message = safe_message.replace(secret, "[REDACTED]")
    return safe_message



class _ProviderResponse:
    def __init__(self, text):
        self.text = text


def _make_strict_json_schema(schema):
    """Convert the existing JSON schema to Groq strict-output form."""
    schema = json.loads(json.dumps(schema))

    def visit(node):
        if not isinstance(node, dict):
            return

        if node.get("type") == "object":
            properties = node.get("properties", {})
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
            for child in properties.values():
                visit(child)

        if node.get("type") == "array":
            visit(node.get("items", {}))

    visit(schema)
    return schema


def _matches_json_schema(value, schema):
    """Small local schema check before accepting a provider response."""
    kind = schema.get("type")

    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not all(key in value for key in required):
            return False
        if schema.get("additionalProperties") is False:
            if not set(value).issubset(properties):
                return False
        if not all(
            _matches_json_schema(value[key], child_schema)
            for key, child_schema in properties.items()
            if key in value
        ):
            return False

    elif kind == "array":
        if not isinstance(value, list):
            return False
        if not all(
            _matches_json_schema(item, schema.get("items", {}))
            for item in value
        ):
            return False

    elif kind == "string" and not isinstance(value, str):
        return False

    elif kind == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        return False

    elif kind == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return False

    elif kind == "boolean" and not isinstance(value, bool):
        return False

    if "enum" in schema and value not in schema["enum"]:
        return False

    return True


class _ProviderRouter:
    """Adapter that preserves the existing Gemini-style call interface."""

    def __init__(self, owner):
        self.owner = owner

    @property
    def models(self):
        return self

    def generate_content(self, model, contents, config=None):
        config = config or {}
        schema = config.get("response_schema")
        errors_seen = []
        attempts_total = 0
        provider_attempts = []

        for provider, provider_model, client in self.owner.provider_clients:
            provider_attempt_count = 0
            provider_error_type = None

            for attempt in range(MAX_RETRIES + 1):
                attempts_total += 1
                provider_attempt_count += 1
                self.owner._last_model = f"{provider}:{provider_model}"

                try:
                    if provider == "gemini":
                        response = client.models.generate_content(
                            model=provider_model,
                            contents=contents,
                            config=config,
                        )
                        text = response.text or ""

                    elif provider == "groq":
                        if not schema:
                            raise ValueError(
                                "A response schema is required for Groq."
                            )

                        strict_schema = _make_strict_json_schema(schema)
                        response = client.chat.completions.create(
                            model=provider_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Return only one JSON object matching "
                                        "the supplied response schema. "
                                        "Do not use Markdown fences."
                                    ),
                                },
                                {"role": "user", "content": contents},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "support_pipeline_output",
                                    "strict": True,
                                    "schema": strict_schema,
                                },
                            },
                            max_completion_tokens=8192,
                            reasoning_effort="low",
                            extra_body={"include_reasoning": False},
                        )
                        text = (
                            response.choices[0].message.content or ""
                        )

                    else:
                        raise ValueError(
                            f"Unsupported provider: {provider}"
                        )

                    if not text.strip():
                        raise ValueError(
                            f"{provider} returned empty response text."
                        )

                    parsed = json.loads(text)
                    if schema and not _matches_json_schema(parsed, schema):
                        raise ValueError(
                            f"{provider} response did not match the schema."
                        )

                    provider_attempts.append({
                        "provider": provider,
                        "model": provider_model,
                        "attempts": provider_attempt_count,
                        "outcome": "success",
                        "error_type": None,
                    })

                    self.owner._last_provider = provider
                    self.owner._last_model = f"{provider}:{provider_model}"
                    self.owner._last_attempts = attempts_total
                    self.owner._last_provider_attempts = provider_attempts

                    return _ProviderResponse(text)

                except Exception as exc:
                    provider_error_type = type(exc).__name__
                    message = str(exc)
                    errors_seen.append(_redact_provider_secrets(
                        f"{provider}/{provider_model}: "
                        f"{type(exc).__name__}: {message}"
                    ))

                    status = getattr(exc, "status_code", None)
                    if status is None:
                        status = getattr(exc, "code", None)
                    try:
                        status = int(status)
                    except (TypeError, ValueError):
                        status = None

                    lowered = message.lower()
                    quota_error = (
                        status == 429
                        or "resource_exhausted" in lowered
                        or "quota exceeded" in lowered
                        or "rate limit" in lowered
                    )
                    server_error = (
                        (status is not None and status >= 500)
                        or "503" in lowered
                        or "server error" in lowered
                        or "temporarily unavailable" in lowered
                    )

                    # Never waste retries on an exhausted quota.
                    if quota_error:
                        print(
                            f"{provider} quota/rate limit reached; "
                            "trying the next configured provider."
                        )
                        break

                    if server_error and attempt < MAX_RETRIES:
                        delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                        print(
                            f"{provider} server error; retrying in "
                            f"{delay}s..."
                        )
                        time.sleep(delay)
                        continue

                    # For other errors, move to the next provider.
                    break

            provider_attempts.append({
                "provider": provider,
                "model": provider_model,
                "attempts": provider_attempt_count,
                "outcome": "failed",
                "error_type": provider_error_type,
            })
        self.owner._last_attempts = attempts_total
        self.owner._last_provider_attempts = provider_attempts
        details = " | ".join(errors_seen)
        raise RuntimeError(
            "All configured LLM providers failed. " + details
        )


class LLMClient:
    def __init__(self):
        self.mode = os.getenv("LLM_MODE", "mock").lower()
        self.client = None
        self.initialization_error = None
        self.cache = LLMCache()

        self.preferred_provider = os.getenv(
            "LLM_PROVIDER", "gemini"
        ).strip().lower()
        if self.preferred_provider not in {"gemini", "groq"}:
            raise ValueError(
                "Invalid LLM_PROVIDER. Use gemini or groq."
            )
        if self.mode not in {"mock", "live", "off"}:
            raise ValueError(
                f"Invalid LLM_MODE: {self.mode}. "
                "Use mock, live, or off."
            )

        self.provider_clients = []
        self._last_provider = self.preferred_provider
        self._last_attempts = 0
        self._last_model = MODEL_NAME
        self.cache_identity_model = f"mode:{self.mode}"

        if self.mode == "live":
            available = {}

            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    available["gemini"] = (
                        MODEL_NAME,
                        genai.Client(api_key=gemini_key),
                    )
                except Exception as exc:
                    print(
                        "Gemini client initialization failed: "
                        f"{type(exc).__name__}"
                    )

            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    from openai import OpenAI

                    groq_model = os.getenv(
                        "GROQ_MODEL_NAME", "openai/gpt-oss-20b"
                    )
                    available["groq"] = (
                        groq_model,
                        OpenAI(
                            api_key=groq_key,
                            base_url="https://api.groq.com/openai/v1",
                        ),
                    )
                except Exception as exc:
                    print(
                        "Groq client initialization failed: "
                        f"{type(exc).__name__}"
                    )

            order = [self.preferred_provider]
            order += [
                name for name in ("groq", "gemini")
                if name != self.preferred_provider
            ]

            for name in order:
                if name in available:
                    provider_model, provider_client = available[name]
                    self.provider_clients.append(
                        (name, provider_model, provider_client)
                    )

            if not self.provider_clients:
                self.initialization_error = (
                    "No usable live provider is configured. Set a nonempty "
                    "GROQ_API_KEY or GEMINI_API_KEY and ensure its provider "
                    "SDK is installed."
                )
            else:
                self.client = _ProviderRouter(self)
                self._last_provider, first_model, _ = (
                    self.provider_clients[0]
                )
                self._last_model = (
                    f"{self._last_provider}:{first_model}"
                )
                self.cache_identity_model = "|".join(
                    f"{name}:{provider_model}"
                    for name, provider_model, _
                    in self.provider_clients
                )
                if self._last_provider != self.preferred_provider:
                    print(
                        f"Preferred provider {self.preferred_provider!r} "
                        "is unavailable; using the configured fallback."
                    )
    def _result(
        self,
        success,
        output,
        model,
        attempts,
        start_time,
        error_type=None,
        error_message=None,
        cache_hit=False,
        provider_attempts=None,
    ):
        return LLMCallResult(
            success=success,
            output=output,
            model=model,
            attempts=attempts,
            duration_ms=int(
                (time.perf_counter() - start_time) * 1000
            ),
            error_type=error_type,
            error_message=(
                _redact_provider_secrets(error_message)
                if error_message is not None
                else None
            ),
            cache_hit=cache_hit,
            provider_attempts=(
                provider_attempts
                if provider_attempts is not None
                else []
            ),
        )

    def _mock_triage(self, prompt):
        ticket_section = prompt.split("Ticket:", 1)[-1]
        ticket_text = ticket_section.split(
            "Retrieved knowledge:", 1
        )[0].lower()

        message_match = re.search(
            r"(?im)^Message:\s*(.*?)(?=^\S.*?:|\Z)",
            ticket_section,
            re.DOTALL,
        )

        subject_match = re.search(
            r"(?im)^Subject:\s*(.*?)(?=^\S.*?:|\Z)",
            ticket_section,
            re.DOTALL,
        )

        message = message_match.group(1).strip() if message_match else ""
        subject = subject_match.group(1).strip() if subject_match else ""

        knowledge_text = (
            prompt.split("Retrieved knowledge:", 1)[1]
            if "Retrieved knowledge:" in prompt
            else ""
        )

        source_chunk_ids = re.findall(
            r"(?m)^\[([A-Za-z0-9_-]+)\]\s*$",
            knowledge_text,
        )

        negative_signals = (
            "worried", "urgent", "locked", "pending", "losing",
            "loss", "mistake", "did not request",
            "someone may have accessed", "security review",
        )
        positive_signals = (
            "thank you", "great", "happy", "appreciate", "satisfied",
        )
        negative = any(x in ticket_text for x in negative_signals)
        positive = any(x in ticket_text for x in positive_signals)
        sentiment = (
            "mixed" if negative and positive
            else "negative" if negative
            else "positive" if positive
            else "neutral"
        )

        if (
            "did not request" in ticket_text
            or "unauthorized" in ticket_text
            or "unauthorised" in ticket_text
            or "someone may have accessed" in ticket_text
        ):
            category = "account_security"
            urgency = "critical"
            resolution_mode = "needs_specialist_escalation"
            recommended_queue = "trust_and_safety"
            reasoning_summary = customer_report_summary(message, subject)

        elif (
            "account locked" in ticket_text
            or "security review" in ticket_text
        ):
            is_security_review = "security review" in ticket_text
            category = (
                "account_security" if is_security_review else "account_access"
            )
            urgency = "high"
            resolution_mode = (
                "needs_specialist_escalation"
                if is_security_review
                else "needs_human_review"
            )
            recommended_queue = (
                "trust_and_safety"
                if is_security_review
                else "account_operations"
            )
            reasoning_summary = (
                "The ticket concerns an account restriction."
            )

        elif (
            "cannot log in" in ticket_text
            or "can't log in" in ticket_text
            or "unable to log in" in ticket_text
            or "cannot access my account" in ticket_text
            or "can't access my account" in ticket_text
        ):
            category = "account_access"
            urgency = "high"
            resolution_mode = "needs_human_review"
            recommended_queue = "account_operations"
            reasoning_summary = (
                "The ticket concerns difficulty accessing the account."
            )

        elif "withdrawal" in ticket_text or "payout" in ticket_text:
            category = "withdrawal_delay"
            urgency = "high"
            resolution_mode = "needs_human_review"
            recommended_queue = "payments_ops"
            reasoning_summary = (
                "The ticket concerns a pending withdrawal or payout."
            )

        elif (
            "reverse" in ticket_text
            or "restore my balance" in ticket_text
            or "losing trade" in ticket_text
        ):
            category = "trade_dispute"
            urgency = "high"
            resolution_mode = "needs_human_review"
            recommended_queue = "trading_ops"
            reasoning_summary = customer_report_summary(message, subject)

        else:
            category = "general_query"
            urgency = "medium"
            resolution_mode = "reply_only"
            recommended_queue = "customer_support"

            reasoning_summary = customer_report_summary(
                message,
                subject,
            )

        return {
            "category": category,
            "urgency": urgency,
            "sentiment": sentiment,
            "resolution_mode": resolution_mode,
            "recommended_queue": recommended_queue,
            "reasoning_summary": reasoning_summary,
            "source_chunk_ids": source_chunk_ids,
        }


    def _cache_key(self, prompt: str) -> str:
        cache_identity = {
            "model": getattr(self, "cache_identity_model", MODEL_NAME),
            "prompt": prompt,
            "schema_version": 2,
        }

        serialized = json.dumps(
            cache_identity,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()


    def _response_cache_key(self, prompt: str) -> str:
        identity = {
            "model": getattr(self, "cache_identity_model", MODEL_NAME),
            "purpose": "response_draft",
            "schema_version": 1,
            "prompt": prompt,
        }

        serialized = json.dumps(identity, sort_keys=True)

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()


    def _action_plan_cache_key(self, prompt: str) -> str:
        identity = {
            "model": getattr(self, "cache_identity_model", MODEL_NAME),
            "purpose": "action_plan",
            "schema_version": 2,
            "prompt": prompt,
        }

        serialized = json.dumps(identity, sort_keys=True)

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()



    def _valid_output(self, output: dict) -> bool:
        if not isinstance(output, dict):
            return False

        expected_fields = {
            "category",
            "urgency",
            "sentiment",
            "resolution_mode",
            "recommended_queue",
            "reasoning_summary",
            "source_chunk_ids",
        }

        if set(output) != expected_fields:
            return False

        string_fields = expected_fields - {"source_chunk_ids"}
        if not all(
            isinstance(output.get(field), str)
            and output[field].strip()
            for field in string_fields
        ):
            return False

        if not isinstance(output["source_chunk_ids"], list):
            return False

        if not all(
            isinstance(chunk_id, str)
            for chunk_id in output["source_chunk_ids"]
        ):
            return False

        return (
            output["category"] in VALID_CATEGORIES
            and output["urgency"] in VALID_URGENCIES
            and output["sentiment"] in VALID_SENTIMENTS
            and output["resolution_mode"] in VALID_RESOLUTION_MODES
            and output["recommended_queue"] in VALID_QUEUES
            and len(output["source_chunk_ids"])
                == len(set(output["source_chunk_ids"]))
        )


    def _valid_response_output(self, output: dict) -> bool:
        return (
            isinstance(output, dict)
            and isinstance(output.get("draft"), str)
            and bool(output["draft"].strip())
        )

    @staticmethod
    def _valid_grounding_review(output: dict, claim_count: int) -> bool:
        if not isinstance(output, dict):
            return False
        reviewed = output.get("reviewed_claim_indexes")
        unsupported = output.get("unsupported_claim_indexes")
        if (
            not isinstance(reviewed, list)
            or not isinstance(unsupported, list)
            or not isinstance(output.get("summary"), str)
            or not output["summary"].strip()
        ):
            return False
        valid_indexes = set(range(claim_count))
        return (
            all(type(index) is int for index in reviewed)
            and all(type(index) is int for index in unsupported)
            and len(reviewed) == len(set(reviewed))
            and len(unsupported) == len(set(unsupported))
            and set(reviewed) == valid_indexes
            and set(unsupported).issubset(valid_indexes)
        )

    def validate_grounding_claims(
        self,
        prompt: str,
        claim_records: list[dict],
    ) -> LLMCallResult:
        """Run Stage 4; mock/off modes explicitly mirror deterministic results."""
        start_time = time.perf_counter()
        self._last_provider_attempts = []
        claim_count = len(claim_records)

        if self.mode in {"mock", "off"}:
            unsupported = [
                index
                for index, item in enumerate(claim_records)
                if item.get("grounded") is not True
            ]
            return self._result(
                success=True,
                output={
                    "reviewed_claim_indexes": list(range(claim_count)),
                    "unsupported_claim_indexes": unsupported,
                    "summary": (
                        "Mock/offline validator explicitly reviewed the "
                        "deterministic claim records; it is not an independent "
                        "model judgment."
                    ),
                },
                model="mock-grounding-validator-v1",
                attempts=0,
                start_time=start_time,
            )

        if self.initialization_error:
            return self._result(
                success=False,
                output=None,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=0,
                start_time=start_time,
                error_type="LLMInitializationError",
                error_message=self.initialization_error,
            )

        schema = {
            "type": "object",
            "properties": {
                "reviewed_claim_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "unsupported_claim_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "summary": {"type": "string"},
            },
            "required": [
                "reviewed_claim_indexes",
                "unsupported_claim_indexes",
                "summary",
            ],
        }

        try:
            response = self.client.models.generate_content(
                model=getattr(self, "_last_model", MODEL_NAME),
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            output = json.loads(response.text)
            if not self._valid_grounding_review(output, claim_count):
                raise ValueError(
                    "Grounding validator did not review every claim index."
                )
            return self._result(
                success=True,
                output=output,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=getattr(self, "_last_attempts", 0) or 1,
                start_time=start_time,
                provider_attempts=getattr(
                    self, "_last_provider_attempts", []
                ),
            )
        except Exception as exc:
            return self._result(
                success=False,
                output=None,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=getattr(self, "_last_attempts", 0) or 1,
                start_time=start_time,
                provider_attempts=getattr(
                    self, "_last_provider_attempts", []
                ),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )



    def triage(self, prompt: str) -> LLMCallResult:
        start_time = time.perf_counter()
        self._last_provider_attempts = []

        if self.mode == "mock":
            return self._result(
                success=True,
                output=self._mock_triage(prompt),
                model="mock",
                attempts=0,
                start_time=start_time,
            )

        if self.mode == "off":
            return self._result(
                success=False,
                output=None,
                model="disabled",
                attempts=0,
                start_time=start_time,
                error_type="LLMDisabled",
                error_message="LLM calls are disabled.",
            )

        if self.initialization_error:
            return self._result(
                success=False,
                output=None,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=0,
                start_time=start_time,
                error_type="LLMInitializationError",
                error_message=self.initialization_error,
            )

        cache_key = self._cache_key(prompt)
        cached_output = self.cache.get(cache_key)

        if cached_output is not None:
            if self._valid_output(cached_output):
                self._last_model = f"cache:{getattr(self, 'cache_identity_model', MODEL_NAME)}"
                return self._result(
                    success=True,
                    output=cached_output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=0,
                    start_time=start_time,
                    cache_hit=True,
                )

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=getattr(self, "_last_model", MODEL_NAME),
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": sorted(VALID_CATEGORIES),
                                },
                                "urgency": {
                                    "type": "string",
                                    "enum": sorted(VALID_URGENCIES),
                                },
                                "sentiment": {
                                    "type": "string",
                                    "enum": sorted(VALID_SENTIMENTS),
                                },
                                "resolution_mode": {
                                    "type": "string",
                                    "enum": sorted(VALID_RESOLUTION_MODES),
                                },
                                "recommended_queue": {
                                    "type": "string",
                                    "enum": sorted(VALID_QUEUES),
                                },
                                "reasoning_summary": {
                                    "type": "string",
                                },
                                "source_chunk_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "category",
                                "urgency",
                                "sentiment",
                                "resolution_mode",
                                "recommended_queue",
                                "reasoning_summary",
                                "source_chunk_ids",
                            ],
                        },
                    },
                )

                output = json.loads(response.text)

                if not self._valid_output(output):
                    raise ValueError(
                        "Gemini returned an invalid triage structure."
                    )

                self.cache.set(cache_key, output)

                return self._result(
                    success=True,
                    output=output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                )

            except errors.ServerError as exc:
                if attempt == MAX_RETRIES:
                    return self._result(
                        False, None, MODEL_NAME, attempt + 1,
                        start_time, type(exc).__name__, str(exc),
                        provider_attempts=getattr(
                            self, "_last_provider_attempts", []
                        ),
                    )

                delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                print(
                    f"Gemini server error. Retrying in {delay}s..."
                )
                time.sleep(delay)

            except Exception as exc:
                return self._result(
                    success=False,
                    output=None,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                )




    def _mock_draft_response(self, prompt):
        text = prompt.lower()

        if "evidence is weak or missing" in text:
            return {
                "draft": (
                    "Thank you for contacting support. We cannot confirm "
                    "the specific status or outcome from the available "
                    "information. This case needs human review before a "
                    "policy-specific answer can be provided."
                )
            }

        if "retrieved knowledge:" in text:
            ticket_text = text.split("retrieved knowledge:", 1)[0]
            evidence_text = text.split("retrieved knowledge:", 1)[1]
        else:
            ticket_text = text
            evidence_text = ""

        if (
            ("pending withdrawal" in ticket_text or "payout" in ticket_text)
            and "within 24 hours" in evidence_text
            and "must not guarantee payout timing" in evidence_text
        ):
            draft = (
                "Thank you for contacting support about your pending "
                "withdrawal. Verified customers usually have withdrawals "
                "reviewed within 24 hours, although delays may occur "
                "when additional checks are required or account activity "
                "is under review. We cannot guarantee payout timing "
                "unless the payment provider has confirmed processing."
            )

        elif (
            ("unauthorised" in ticket_text or "unauthorized" in ticket_text
                or "did not request" in ticket_text
                or "someone may have accessed" in ticket_text)
            and "reset their password" in evidence_text
            and "enable two-factor authentication" in evidence_text
        ):
            draft = (
                "Thank you for contacting support about possible "
                "unauthorised account access. Please reset your password, "
                "enable two-factor authentication if available, and "
                "contact support immediately."
            )

        elif (
            ("account locked" in ticket_text or "security review" in ticket_text)
            and "temporarily restricted" in evidence_text
        ):
            draft = (
                "Thank you for contacting support about your locked "
                "account. Accounts may be temporarily restricted when "
                "suspicious login, device, password, or payment activity "
                "is detected."
            )

        elif (
            ("reverse" in ticket_text or "restore my balance" in ticket_text
             or "losing trade" in ticket_text)
            and "executed trades are final" in evidence_text
            and "confirmed platform error" in evidence_text
        ):
            draft = (
                "Thank you for contacting support about your trade. "
                "Executed trades are final unless there is a confirmed "
                "platform error or an approved operational adjustment."
            )

        else:
            draft = (
                "Thank you for contacting support. We cannot confirm "
                "the specific status or outcome of your case from the "
                "information currently available. Please contact support "
                "for assistance with your individual case."
            )

        return {"draft": draft}


    def draft_response(self, prompt: str) -> LLMCallResult:
        start_time = time.perf_counter()
        self._last_provider_attempts = []

        if self.mode == "mock":
            return self._result(
                success=True,
                output=self._mock_draft_response(prompt),
                model="mock",
                attempts=0,
                start_time=start_time,
            )

        if self.mode == "off":
            return self._result(
                success=False,
                output=None,
                model="disabled",
                attempts=0,
                start_time=start_time,
                error_type="LLMDisabled",
                error_message="LLM calls are disabled.",
            )

        if self.initialization_error:
            return self._result(
                success=False,
                output=None,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=0,
                start_time=start_time,
                error_type="LLMInitializationError",
                error_message=self.initialization_error,
            )

        cache_key = self._response_cache_key(prompt)
        cached_output = self.cache.get(cache_key)

        if cached_output is not None:
            if self._valid_response_output(cached_output):
                self._last_model = f"cache:{getattr(self, 'cache_identity_model', MODEL_NAME)}"
                return self._result(
                    success=True,
                    output=cached_output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=0,
                    start_time=start_time,
                    cache_hit=True,
                )

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=getattr(self, "_last_model", MODEL_NAME),
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "draft": {"type": "string"},
                            },
                            "required": ["draft"],
                        },
                    },
                )

                output = json.loads(response.text)

                if not self._valid_response_output(output):
                    raise ValueError(
                        "Gemini returned an invalid response draft."
                    )

                self.cache.set(cache_key, output)

                return self._result(
                    success=True,
                    output=output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                )

            except errors.ServerError as exc:
                if attempt == MAX_RETRIES:
                    return self._result(
                        success=False,
                        output=None,
                        model=getattr(self, "_last_model", MODEL_NAME),
                        attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                        start_time=start_time,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        provider_attempts=getattr(
                            self, "_last_provider_attempts", []
                        ),
                    )

                delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                print(
                    f"Gemini server error. Retrying in {delay}s..."
                )
                time.sleep(delay)

            except Exception as exc:
                return self._result(
                    success=False,
                    output=None,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                )




    def generate_action_plan(
        self,
        prompt: str,
        known_chunk_ids: set[str],
        selected_chunks=None,
    ) -> LLMCallResult:

        try:
            from .llm_action_plan import validate_action_plan
        except ImportError:
            from llm_action_plan import validate_action_plan


        start_time = time.perf_counter()
        self._last_provider_attempts = []

        if self.mode == "off":
            return self._result(
                success=False,
                output=None,
                model="disabled",
                attempts=0,
                start_time=start_time,
                error_type="LLMDisabled",
                error_message="LLM calls are disabled.",
            )

        if self.mode == "mock":
            ticket_context = ""
            if "Ticket:" in prompt and "Triage:" in prompt:
                ticket_context = prompt.split("Ticket:", 1)[1].split(
                    "Triage:", 1
                )[0]

            ticket_fields = {}
            for line in ticket_context.splitlines():
                if ":" in line:
                    field, value = line.split(":", 1)
                    ticket_fields[field.strip().lower()] = value.strip()

            ticket_id = ticket_fields.get("id", "UNKNOWN")
            subject = ticket_fields.get("subject", "support request")
            ticket_text = ticket_context.lower()

            if (
                "did not request" in ticket_text
                or "unauthorized" in ticket_text
                or "unauthorised" in ticket_text
                or "someone may have accessed" in ticket_text
            ):
                queue = "trust_and_safety"
                priority = "P0"
                description = (
                    "Review the reported possible unauthorized access "
                    "and follow the account-security procedure."
                )
            elif "withdrawal" in ticket_text or "payout" in ticket_text:
                queue = "payments_ops"
                priority = "P1"
                description = (
                    "Review the pending withdrawal and check for "
                    "available payment-provider status information."
                )
            elif (
                "reverse" in ticket_text
                or "restore my balance" in ticket_text
                or "losing trade" in ticket_text
            ):
                queue = "trading_ops"
                priority = "P1"
                description = (
                    "Review the customer's trade-reversal request "
                    "against the trading policy."
                )
            elif "account locked" in ticket_text or "security review" in ticket_text:
                queue = "trust_and_safety"
                priority = "P1"
                description = (
                    "Review the account restriction and provide the "
                    "customer with the permitted next steps."
                )
            elif any(
                signal in ticket_text
                for signal in (
                    "cannot log in",
                    "can't log in",
                    "unable to log in",
                    "cannot access my account",
                    "can't access my account",
                )
            ):
                queue = "account_operations"
                priority = "P1"
                description = (
                    "Review the account-access issue and establish "
                    "the permitted next step."
                )
            else:
                queue = "customer_support"
                priority = "P2"
                description = (
                    "Review the ticket and determine the next step "
                    "from the available evidence."
                )


            source_chunk_ids = []

            # Choose relevant evidence based on the ticket intent.
            # Do not use the entire prompt for this: it also contains
            # every retrieved chunk, including unrelated evidence.
            if queue == "payments_ops":
                evidence_keywords = [
                    "withdrawal",
                    "payment provider",
                    "payout timing",
                ]
            elif queue == "trading_ops":
                evidence_keywords = [
                    "trade finality",
                    "executed trades",
                    "platform error",
                ]
            elif queue == "trust_and_safety":
                evidence_keywords = [
                    "unauthorised access",
                    "unauthorized access",
                    "two-factor",
                    "account security review",
                    "support prioritisation",
                ]
            else:
                evidence_keywords = []

            for score, chunk in selected_chunks or []:
                evidence_text = (
                    chunk.section_title + " " + chunk.text
                ).lower()

                if any(
                    keyword in evidence_text
                    for keyword in evidence_keywords
                ):
                    source_chunk_ids.append(chunk.chunk_id)


            output = {
                "actions": [
                    {
                        "action_id": f"A-{ticket_id}-1",
                        "description": description,
                        "owner_queue": queue,
                        "priority": priority,
                        "depends_on": [],
                        "source_chunk_ids": source_chunk_ids,
                    }
                ],
                "handoff_note": {
                    "payments_ops": (
                        "Payments Ops: review ticket {ticket_id} about "
                        "'{subject}', check available provider status, "
                        "and avoid promising timing without confirmation."
                    ),
                    "trust_and_safety": (
                        "Trust and Safety: assess ticket {ticket_id} about "
                        "'{subject}' using the cited security guidance and "
                        "confirm safe next steps."
                    ),
                    "customer_support": (
                        "Customer Support: review ticket {ticket_id} about "
                        "'{subject}', answer from supported evidence, and "
                        "route unresolved account-specific questions."
                    ),
                    "trading_ops": (
                        "Trading Ops: review ticket {ticket_id} about "
                        "'{subject}' against trade-finality policy and "
                        "check whether a documented exception applies."
                    ),
                    "account_operations": (
                        "Account Operations: review ticket {ticket_id} "
                        "about '{subject}' and confirm permitted access "
                        "steps without assuming access has been restored."
                    ),
                }[queue].format(ticket_id=ticket_id, subject=subject),
            }

            try:
                validate_action_plan(output, known_chunk_ids)
            except (ValueError, TypeError) as exc:
                return self._result(
                    success=False,
                    output=None,
                    model="mock",
                    attempts=0,
                    start_time=start_time,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            return self._result(
                success=True,
                output=output,
                model="mock",
                attempts=0,
                start_time=start_time,
            )

        if self.initialization_error:
            return self._result(
                success=False,
                output=None,
                model=getattr(self, "_last_model", MODEL_NAME),
                attempts=0,
                start_time=start_time,
                error_type="LLMInitializationError",
                error_message=self.initialization_error,
            )

        cache_key = self._action_plan_cache_key(prompt)
        cached_output = self.cache.get(cache_key)

        if cached_output is not None:
            try:
                validate_action_plan(cached_output, known_chunk_ids)
                self._last_model = f"cache:{getattr(self, 'cache_identity_model', MODEL_NAME)}"
                return self._result(
                    success=True,
                    output=cached_output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=0,
                    start_time=start_time,
                    cache_hit=True,
                )
            except (ValueError, TypeError):
                # Ignore invalid cache entries and regenerate.
                pass

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=getattr(self, "_last_model", MODEL_NAME),
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action_id": {"type": "string"},
                                            "description": {"type": "string"},
                                            "owner_queue": {
                                                "type": "string",
                                                "enum": [
                                                    "payments_ops",
                                                    "trust_and_safety",
                                                    "customer_support",
                                                    "trading_ops",
                                                    "account_operations",
                                                ],
                                            },
                                            "priority": {
                                                "type": "string",
                                                "enum": [
                                                    "P0", "P1", "P2", "P3"
                                                ],
                                            },
                                            "depends_on": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "source_chunk_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "required": [
                                            "action_id",
                                            "description",
                                            "owner_queue",
                                            "priority",
                                            "depends_on",
                                            "source_chunk_ids",
                                        ],
                                    },
                                },
                                "handoff_note": {"type": "string"},
                            },
                            "required": ["actions", "handoff_note"],
                        },
                    },
                )

                output = json.loads(response.text)
                validate_action_plan(output, known_chunk_ids)

                self.cache.set(cache_key, output)

                return self._result(
                    success=True,
                    output=output,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                )

            except errors.ServerError as exc:
                if attempt == MAX_RETRIES:
                    return self._result(
                        success=False,
                        output=None,
                        model=getattr(self, "_last_model", MODEL_NAME),
                        attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                        start_time=start_time,
                        provider_attempts=getattr(
                            self, "_last_provider_attempts", []
                        ),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

                delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                print(
                    f"Gemini server error. Retrying in {delay}s..."
                )
                time.sleep(delay)

            except Exception as exc:
                return self._result(
                    success=False,
                    output=None,
                    model=getattr(self, "_last_model", MODEL_NAME),
                    attempts=getattr(self, "_last_attempts", 0) or attempt + 1,
                    start_time=start_time,
                    provider_attempts=getattr(
                        self, "_last_provider_attempts", []
                    ),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
