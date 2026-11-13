
from dataclasses import dataclass, field


@dataclass
class KnowledgeChunk:
    document_id: str
    source_file: str
    chunk_id: str
    section_title: str
    text: str
    character_count: int
    content_hash: str


@dataclass
class RetrievalResult:
    ticket_id: str
    query_text: str
    selected_chunk_ids: list[str]
    selection_reason: str
    omitted_relevant_risk: str | None


@dataclass
class Ticket:
    ticket_id: str
    submitted_at: str
    customer_tier: str
    language: str
    subject: str
    message: str


@dataclass(init=False)
class TriageResult:
    ticket_id: str
    category: str
    urgency: str
    sentiment: str
    resolution_mode: str
    recommended_queue: str
    reasoning_summary: str
    source_chunk_ids: list[str]

    def __init__(
        self,
        ticket_id: str,
        category: str,
        urgency: str | None = None,
        sentiment: str = "neutral",
        resolution_mode: str | None = None,
        recommended_queue: str | None = None,
        reasoning_summary: str | None = None,
        source_chunk_ids: list[str] | None = None,
        *,
        priority: str | None = None,
        owner: str | None = None,
        reasoning: str | None = None,
    ):
        # Temporary aliases allow existing tests and downstream code
        # to continue working while they are migrated.
        old_categories = {
            "payments": "withdrawal_delay",
            "security": "account_security",
            "trading": "trade_dispute",
            "general": "general_query",
        }
        old_queues = {
            "payments": "payments_ops",
            "security": "trust_and_safety",
            "support": "customer_support",
            "trading": "trading_ops",
        }

        self.ticket_id = ticket_id
        self.category = old_categories.get(category, category)
        self.urgency = urgency or priority or "medium"
        self.sentiment = sentiment
        self.recommended_queue = (
            recommended_queue
            or old_queues.get(owner, owner)
            or "customer_support"
        )
        self.reasoning_summary = reasoning_summary or reasoning or ""
        self.source_chunk_ids = list(source_chunk_ids or [])

        self.resolution_mode = resolution_mode or (
            "reply_only"
            if self.category == "general_query"
            else "needs_human_review"
        )

    # Temporary read-only compatibility aliases.
    @property
    def priority(self) -> str:
        return self.urgency

    @property
    def owner(self) -> str:
        old_queues = {
            "payments_ops": "payments",
            "trust_and_safety": "security",
            "customer_support": "support",
            "trading_ops": "trading",
            "account_operations": "support",
        }
        return old_queues.get(
            self.recommended_queue,
            self.recommended_queue,
        )

    @property
    def reasoning(self) -> str:
        return self.reasoning_summary


@dataclass
class LLMTriageSuggestion:
    category: str
    urgency: str
    sentiment: str
    resolution_mode: str
    recommended_queue: str
    reasoning_summary: str
    source_chunk_ids: list[str]

@dataclass
class LLMCallResult:
    success: bool
    output: dict | None
    model: str
    attempts: int
    duration_ms: int
    error_type: str | None
    error_message: str | None
    cache_hit: bool = False
    provider_attempts: list[dict] = field(default_factory=list)


@dataclass
class ResponseDraft:
    ticket_id: str
    draft: str
    evidence_chunk_ids: list[str]


@dataclass
class ActionItem:
    action_id: str
    description: str
    owner_queue: str
    priority: str
    depends_on: list[str]
    source_chunk_ids: list[str]


@dataclass
class ActionPlan:
    ticket_id: str
    actions: list[ActionItem]
    handoff_note: str
    status: str
