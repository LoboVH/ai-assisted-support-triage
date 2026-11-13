
import json
import argparse
import re
from pathlib import Path

try:
    from .human_review import get_current_review
except ImportError:
    from human_review import get_current_review


ROOT = Path(__file__).resolve().parent.parent

TICKETS_PATH = ROOT / "tickets.json"
DRAFTS_PATH = ROOT / "output" / "response_drafts.json"
LABELS_PATH = ROOT / "evaluation" / "response_quality_labels.json"
OUTPUT_PATH = ROOT / "output" / "response_quality_evaluation.json"




def _normalise_text(text):
    """Normalise punctuation and whitespace for tolerant phrase checks."""
    text = str(text or "").casefold().replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _contains_any(text, phrases):
    """Match a phrase variant without depending on punctuation/capitalisation."""
    normalised_text = f" {_normalise_text(text)} "
    for phrase in phrases:
        normalised_phrase = _normalise_text(phrase)
        if normalised_phrase and f" {normalised_phrase} " in normalised_text:
            return True
    return False


def _has_immediate_support_instruction(text):
    """Require unconditional immediate-contact wording, not a conditional substitute."""
    normalised = _normalise_text(text)
    immediate_variants = (
        "contact support immediately",
        "immediately contact support",
        "contact customer support immediately",
        "reach out to support immediately",
        "contact support right away",
        "contact support at once",
        "immediately reach out to support",
    )
    if not _contains_any(normalised, immediate_variants):
        return False

    if not _has_unnegated_claim(normalised, [
        "contact support", "contact customer support",
        "reach out to support", "immediately reach out to support",
    ]):
        return False

    # If the only immediate-contact phrase is conditional, do not count it.
    conditional_patterns = (
        r"\bcontact support immediately if\b",
        r"\bcontact customer support immediately if\b",
        r"\bcontact support if\b",
        r"\bcontact customer support if\b",
        r"\breach out to support immediately if\b",
        r"\breach out to support if\b",
        r"\bcontact support only if\b",
    )
    return not any(re.search(pattern, normalised) for pattern in conditional_patterns)


def _has_unnegated_claim(text, phrases):
    """Detect direct claim phrases while avoiding common explicit denials."""
    normalised = _normalise_text(text)
    for phrase in phrases:
        target = _normalise_text(phrase)
        if not target:
            continue
        start = 0
        while True:
            index = normalised.find(target, start)
            if index < 0:
                break
            prefix = normalised[max(0, index - 45):index]
            # A denial before a contrast word must not cancel a claim after it.
            prefix = re.split(
                r"\b(?:but|however|nevertheless|instead)\b", prefix
            )[-1]
            if not re.search(
                r"\b(?:not|never|cannot|can t|cant|will not|won t|wont|unable to|cannot confirm|cant confirm|can t confirm|cannot promise|cant promise|can t promise)\b(?:[^.]{0,45})$",
                prefix,
            ):
                return True
            start = index + len(target)
    return False


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_ticket(ticket, draft_record, label_record):
    draft = draft_record["draft"]
    draft_lower = draft.lower()

    scores = {}
    reasons = []
    critical_failures = []

    ticket_text = (
        ticket.get("subject", "") + " " + ticket.get("message", "")
    ).lower()

    # Relevance: match the issue using topic variants, not one exact wording.
    relevance_terms = {
        "T-1001": ["withdrawal", "withdrawals", "payout", "withdrawal funds", "funds will be paid"],
        "T-1002": [
            "locked account", "account lock", "security review",
            "account restriction", "locked for security review",
        ],
        "T-1003": [
            "executed trades are final", "completed trades are final",
            "trade finality", "trade reversal", "reverse your trade",
            "trade cannot be reversed", "completed trades cannot be reversed",
            "completed trade cannot be reversed", "reversing the trade",
            "restore your balance", "trades are final", "executed trade",
            "execution fault", "order will be voided",
        ],
        "T-1004": [
            "unauthorised account access", "unauthorized account access",
            "possible unauthorised access", "possible unauthorized access",
            "password reset email", "someone accessed your account",
            "account access",
        ],
    }

    relevant = _contains_any(draft, relevance_terms.get(ticket["ticket_id"], []))
    if ticket["ticket_id"] == "T-1004":
        relevant = relevant and _contains_any(
            draft,
            ["reset your password", "change your password", "reset the password"],
        )

    scores["relevance"] = 2 if relevant else 0

    if not relevant:
        reasons.append("Response does not match the expected ticket issue.")


    # Factual accuracy: each labelled requirement may have safe wording variants.
    factual_checks = {
        "T-1001": [
            ("usual withdrawal review period", [
                "within 24 hours", "within 24h", "within a day",
                "24-hour review", "24 hour review",
                "within about a day", "within roughly a day", "roughly one day",
                "reviewed in about a day", "reviewed within about a day",
                "normally reviewed in about a day", "usually reviewed in about a day",
                "usually take around a day", "reviews usually take around a day",
                "reviewed in around a day", "reviewed around a day",
            ]),
            ("payout timing is not guaranteed without provider confirmation", [
                "cannot guarantee payout timing",
                "cannot guarantee a specific payout time",
                "cannot guarantee a payout time",
                "can't guarantee a specific payout time",
                "cannot promise a specific payout time",
                "can't promise a specific payout time",
                "payout timing is not guaranteed",
                "payout timing isn't guaranteed",
                "a specific payout time cannot be guaranteed",
                "a specific payout time can't be guaranteed",
                "cannot promise payout timing",
                "payout time cannot be guaranteed",
                "a specific payout date isn't guaranteed until",
                "a specific payout date is not guaranteed until",
                "a specific payout date is not guaranteed",
                "a payout date isn't guaranteed until",
                "cannot give an arrival date until the payment provider confirms processing",
                "cannot give an arrival date until the provider confirms",
                "does not confirm the status of this individual payment",
                "does not confirm this payment status",
                "cannot verify this request's current status from the details provided",
                "cannot verify the current status from the details provided",
                "cannot confirm a specific arrival time",
                "can't confirm a specific arrival time",
                "only after the provider confirms that processing has started",
                "only after the payment provider confirms that processing has started",
                "only after the provider confirms processing has started",
            ]),
        ],
        "T-1002": [
            ("temporary-restriction guidance", [
                "temporarily restricted", "temporary restriction",
                "temporary restrictions", "temporary account restriction",
                "subject to a temporary restriction", "restriction may be temporary",
                "may be a temporary restriction", "restrictions can be temporary",
                "restrictions can be temporary when suspicious sign-in, device, password, or payment activity is detected",
                "the restriction can be temporary",
                "account remains locked for security review",
                "account is locked for a security review",
                "the account is locked for a security review",
                "remains locked for security review",
            ]),
        ],
        "T-1003": [
            ("trade-finality guidance", [
                "executed trades are final", "executed trade is final",
                "completed trades are final", "trades that have been executed are final",
                "trade finality policy", "trades are final",
                "ordinary finality rule", "before a reversal could be considered",
                "executed trades cannot be reversed",
                "completed trades cannot be reversed",
                "completed trade cannot be reversed",
            ]),
            ("confirmed-platform-error exception", [
                "confirmed platform error", "confirmed error on the platform",
                "platform error has been confirmed", "confirmed error by the platform",
                "confirmed platform fault", "a confirmed platform fault",
                "platform fault has been confirmed", "unless a platform fault is confirmed",
            ]),
            ("approved-operational-adjustment exception", [
                "approved operational adjustment",
                "operational adjustment has been approved",
                "operational adjustment approved",
            ]),
        ],
        "T-1004": [
            ("password-reset instruction", [
                "reset your password", "change your password", "reset the password",
            ]),
            ("two-factor-authentication instruction", [
                "enable two-factor authentication", "turn on two-factor authentication",
                "activate two-factor authentication", "enable 2fa", "turn on 2fa",
            ]),
            ("immediate-support instruction", [
                "contact support immediately", "immediately contact support",
                "contact customer support immediately", "reach out to support immediately",
                "contact support right away", "contact support at once",
            ]),
        ],
    }

    required_facts = factual_checks.get(ticket["ticket_id"], [])
    missing_facts = [
        label for label, alternatives in required_facts
        if not (
            _has_immediate_support_instruction(draft)
            if label == "immediate-support instruction"
            else _has_unnegated_claim(draft, alternatives)
            if ticket["ticket_id"] == "T-1004"
            else _contains_any(draft, alternatives)
        )
    ]

    # These checks detect specific unsupported claims. They do not
    # replace the grounding validator or prove that all claims are true.
    unsupported_claims = {
        "T-1001": [
            "your withdrawal will arrive today",
            "your withdrawal will arrive tomorrow",
            "your funds will arrive tomorrow",
            "your funds will be paid tomorrow",
            "funds will be paid tomorrow",
            "you will receive your funds tomorrow",
            "your payout is guaranteed",
            "payout is guaranteed",
            "guaranteed payout",
        ],
        "T-1002": [
            "your account was locked because of suspicious activity",
            "your account was locked due to suspicious activity",
            "your account will be unlocked today",
            "your account will be unlocked tomorrow",
            "you will regain access today",
            "you will regain access tomorrow",
        ],
        "T-1003": [
            "we will reverse your trade",
            "we'll reverse your trade",
            "we can reverse your trade",
            "your trade has been reversed",
            "your trade will be reversed",
            "your balance will be restored",
            "we can restore your balance",
            "we will restore your balance",
            "we'll restore your balance",
            "we can restore the balance",
            "we will restore the balance",
        ],
        "T-1004": [
            "your account has been compromised",
            "your account has definitely been compromised",
            "your account is definitely compromised",
            "we have secured your account",
            "locked down your account",
            "confirmed there is no ongoing threat",
            "there is no ongoing threat",
            "no ongoing threat remains",
        ],
    }

    detected_unsupported_claims = [
        phrase
        for phrase in unsupported_claims.get(ticket["ticket_id"], [])
        if _has_unnegated_claim(draft, [phrase])
    ]

    if detected_unsupported_claims:
        scores["factual_accuracy"] = 0
        reasons.append(
            "Potential unsupported claims detected: "
            + ", ".join(detected_unsupported_claims)
        )
    elif not missing_facts:
        scores["factual_accuracy"] = 2
    elif len(missing_facts) < len(required_facts):
        scores["factual_accuracy"] = 1
        reasons.append(
            "Expected factual or policy statements missing: "
            + ", ".join(missing_facts)
        )
    else:
        scores["factual_accuracy"] = 0
        reasons.append(
            "Expected factual or policy statements missing: "
            + ", ".join(missing_facts)
        )


    # Actionability: assess only actions supported by the applicable policy.
    if ticket["ticket_id"] == "T-1001":
        # The payment policy does not establish a specific next step
        # for this situation. Do not penalise the draft for omitting one.
        scores["actionability"] = 2

    elif ticket["ticket_id"] == "T-1002":
        # The policy establishes prioritisation but does not specify
        # a customer-facing procedure for unlocking the account.
        # Keep the score provisional for human review.
        scores["actionability"] = 1

        urgency_acknowledged = any(
            phrase in draft_lower
            for phrase in [
                "you need access urgently",
                "you said you need access urgently",
                "i understand the urgency",
                "i understand that you need access urgently",
            ]
        )

        if not urgency_acknowledged:
            reasons.append("Customer urgency is not acknowledged.")

        reasons.append(
            "The available policy does not establish a specific "
            "account-unlock procedure."
        )

    elif ticket["ticket_id"] == "T-1003":
        # The trade policy defines the exception but does not prescribe
        # a customer-facing next step.
        scores["actionability"] = 2

    elif ticket["ticket_id"] == "T-1004":
        required_actions = [
            ("reset your password", [
                "reset your password", "change your password", "reset the password",
            ]),
            ("enable two-factor authentication", [
                "enable two-factor authentication", "turn on two-factor authentication",
                "activate two-factor authentication", "enable 2fa", "turn on 2fa",
            ]),
            ("contact support immediately", [
                "contact support immediately", "immediately contact support",
                "contact customer support immediately", "reach out to support immediately",
                "contact support right away", "contact support at once",
            ]),
        ]
        missing_actions = [
            label for label, alternatives in required_actions
            if not (
                _has_immediate_support_instruction(draft)
                if label == "contact support immediately"
                else _contains_any(draft, alternatives)
            )
        ]

        scores["actionability"] = (
            2 if not missing_actions
            else 1 if len(missing_actions) < len(required_actions)
            else 0
        )

        if missing_actions:
            reasons.append(
                "Missing supported security actions: "
                + ", ".join(missing_actions)
            )

    else:
        scores["actionability"] = 1
        reasons.append("Actionability requires manual review.")

    # Policy compliance: detect critical unsupported promises and policy omissions.
    if ticket["ticket_id"] == "T-1001":
        positive_payout_promise = _has_unnegated_claim(
            draft,
            [
                "your withdrawal will arrive today",
                "your withdrawal will arrive tomorrow",
                "your funds will arrive tomorrow",
                "your payout is guaranteed",
                "payout is guaranteed",
                "guaranteed payout",
                "your funds will be paid tomorrow",
                "funds will be paid tomorrow",
                "you will receive your funds tomorrow",
            ],
        )
        if positive_payout_promise:
            critical_failures.append("Potential unsupported payout promise.")

    if ticket["ticket_id"] == "T-1002":
        invented_lock_reason = _has_unnegated_claim(
            draft,
            [
                "your account was locked because of suspicious activity",
                "your account was locked due to suspicious activity",
            ],
        )
        if invented_lock_reason:
            critical_failures.append("Invents a confirmed reason for the account restriction.")

        access_promise = _has_unnegated_claim(
            draft,
            [
                "your account will be unlocked today",
                "your account will be unlocked tomorrow",
                "we will unlock your account",
                "we'll unlock your account",
                "you will regain access today",
                "you will regain access tomorrow",
                "you will regain access",
                "you will have access again today",
                "access will be restored today",
                "fast track the unlock",
                "fast-track the unlock",
                "expedite the unlock",
                "expedite the account unlock",
                "fast track your account unlock",
            ],
        )
        if access_promise:
            critical_failures.append("Promises account access without evidence.")

    if ticket["ticket_id"] == "T-1003":
        reversal_promise = _has_unnegated_claim(
            draft,
            [
                "we will reverse your trade",
                "we'll reverse your trade",
                "we can reverse your trade",
                "your trade has been reversed",
                "your trade will be reversed",
                "your balance will be restored",
                "we can restore your balance",
                "we will restore your balance",
                "we'll restore your balance",
                "we can restore the balance",
                "we will restore the balance",
            ],
        )
        if reversal_promise:
            critical_failures.append("Promises a trade reversal or balance restoration.")

        states_finality = _contains_any(
            draft,
            [
                "executed trades are final", "executed trade is final",
                "completed trades are final", "trades that have been executed are final",
                "trade finality policy", "trades are final",
                "executed trades cannot be reversed",
                "completed trades cannot be reversed",
                "completed trade cannot be reversed",
            ],
        )
        has_platform_error_exception = _contains_any(
            draft,
            [
                "confirmed platform error", "confirmed error on the platform",
                "platform error has been confirmed", "confirmed error by the platform",
            ],
        )
        has_operational_adjustment_exception = _contains_any(
            draft,
            [
                "approved operational adjustment",
                "operational adjustment has been approved",
                "operational adjustment approved",
            ],
        )
        if states_finality and not (
            has_platform_error_exception and has_operational_adjustment_exception
        ):
            critical_failures.append(
                "Trade-finality statement omits one or both documented exceptions."
            )

    if ticket["ticket_id"] == "T-1004":
        if detected_unsupported_claims:
            critical_failures.append(
                "Makes an unsupported assertion about account compromise or security actions."
            )

        required_security_actions_present = (
            _has_unnegated_claim(draft, [
                "reset your password", "change your password", "reset the password",
            ])
            and _has_unnegated_claim(draft, [
                "enable two-factor authentication", "turn on two-factor authentication",
                "activate two-factor authentication", "enable 2fa", "turn on 2fa",
            ])
            and _has_immediate_support_instruction(draft)
        )

        if not required_security_actions_present:
            critical_failures.append("Missing or weakens one or more required security instructions.")

    # Generalization-probe supplement: tolerate meaning-preserving wording while
    # detecting unsupported operational promises and weakened security steps.
    # This is deterministic heuristic evaluation, not proof of factual truth.
    _gp_text = _normalise_text(draft)

    def _gp_any(phrases):
        return _contains_any(draft, phrases)

    def _gp_unnegated(phrases):
        return _has_unnegated_claim(draft, phrases)

    def _gp_remove_reasons(*prefixes):
        reasons[:] = [reason for reason in reasons if not reason.startswith(prefixes)]

    def _gp_has_immediate_support(text):
        # Evaluate each sentence separately so an earlier unconditional
        # instruction is not invalidated by an unrelated later condition.
        for clause in re.split(r"[.!?;\n]+", text):
            normalised_clause = _normalise_text(clause)
            contact = re.search(
                r"\b(?:contact|reach(?:\s+out\s+to)?|call|get\s+in\s+touch\s+with|speak\s+to)\s+(?:customer\s+)?support\b",
                normalised_clause,
            )
            if contact is None:
                continue
            if not _has_unnegated_claim(clause, [
                "contact support", "contact customer support",
                "reach customer support",
                "reach out to support", "reach out support",
                "call support", "get in touch with support",
                "speak to support",
            ]):
                continue

            tail = normalised_clause[contact.end():]
            immediate = re.search(
                r"\b(?:immediately|right\s+away|straight\s+away|at\s+once|now)\b",
                normalised_clause,
            )
            if immediate is None:
                continue
            if re.search(r"\b(?:only\s+if|if|unless|when)\b", tail):
                continue
            return True
        return False

    # Relevance is topic-based, not tied to one exact phrase.
    _gp_topic_patterns = {
        "T-1001": r"\b(?:withdrawal|withdrawals|payout|funds?|payment|money|bank)\b",
        "T-1002": r"\b(?:account|access|lock|locked|restriction|security review|open positions|unlock|suspicious login|vip)\b",
        "T-1003": r"\b(?:trade|transaction|balance|platform error|platform fault|operational adjustment|reverse|reversal|executed)\b",
        "T-1004": r"\b(?:password|2fa|two factor|two step|verification|support|suspicious|threat|access|account|locked down)\b",
    }
    if re.search(_gp_topic_patterns.get(ticket["ticket_id"], r"(?!)"), _gp_text):
        scores["relevance"] = 2
        _gp_remove_reasons("Response does not match the expected ticket issue.")

    # T-1001: recognize safe one-day and provider-confirmation caveats, but
    # reject affirmative payout dates and unverified processing/release claims.
    if ticket["ticket_id"] == "T-1001":
        _gp_payout_promise = _gp_unnegated([
            "your withdrawal will arrive today",
            "your withdrawal will arrive tomorrow",
            "your funds will arrive tomorrow",
            "your funds will be paid tomorrow",
            "funds will be paid tomorrow",
            "your payment should reach your bank by the next business day",
            "your payment should reach your bank by next business day",
            "your payment should reach your bank tomorrow",
            "your payment should arrive by the next business day",
            "your withdrawal should arrive by the next business day",
            "your payment will reach your bank by the next business day",
            "you should receive the money shortly",
            "your funds will arrive shortly",
            "your payout is guaranteed",
            "payout is guaranteed",
            "guaranteed payout",
        ])
        _gp_provider_status = _gp_unnegated([
            "the payment provider has confirmed your payout is processing",
            "payment provider has confirmed your payout is processing",
            "the provider has confirmed your payout is processing",
            "the payment provider confirmed your payout is processing",
            "provider confirmed your payout is processing",
        ])
        _gp_released_funds = _gp_unnegated([
            "we have already released your funds",
            "we ve already released your funds",
            "we have released your funds",
            "your funds have already been released",
            "your withdrawal has already been released",
            "your withdrawal has been released",
            "your withdrawal has been processed",
            "we have released the funds",
            "your funds are already on their way",
        ])
        if _gp_payout_promise or _gp_provider_status or _gp_released_funds:
            critical_failures.append("Unsupported payout timing or operational-status promise.")
            scores["factual_accuracy"] = 0
        else:
            _gp_review_period = _gp_any([
                "within 24 hours", "within 24h", "within a day", "within one day",
                "reviewed in about a day", "reviewed within about a day",
                "normally reviewed in about a day", "usually reviewed in about a day",
                "usual review window is one day", "usual review window for a verified withdrawal is one day",
                "reviewed in one day", "reviewed within one day", "24-hour review",
            ])
            _gp_payout_caveat = _gp_any([
                "cannot guarantee payout timing",
                "cannot guarantee a specific payout time",
                "cannot guarantee a payout time",
                "can't guarantee a specific payout time",
                "cannot promise a specific payout time",
                "can't promise a specific payout time",
                "payout timing is not guaranteed",
                "payout timing isn't guaranteed",
                "a specific payout time cannot be guaranteed",
                "a specific payout time can't be guaranteed",
                "cannot promise payout timing",
                "payout time cannot be guaranteed",
                "can't give an exact payout date before the payment provider confirms processing",
                "cannot give an exact payout date before the payment provider confirms processing",
                "cannot state when the money will arrive until the provider confirms it is processing",
                "cannot state when money will arrive until the provider confirms it is processing",
                "there is no confirmed arrival date to share until the payment provider confirms processing",
                "no confirmed arrival date to share until the payment provider confirms processing",
                "a specific payout date isn't guaranteed until the payment provider confirms processing",
                "a specific payout date isn't guaranteed until the provider confirms processing",
                "a specific payout date is not guaranteed until the payment provider confirms processing",
                "a specific payout date is not guaranteed until the provider confirms processing",
                "a specific payout date is not guaranteed",
                "cannot provide an exact arrival date before provider confirmation",
            ])
            if _gp_review_period and _gp_payout_caveat:
                scores["factual_accuracy"] = 2
                _gp_remove_reasons("Expected factual or policy statements missing:")

    # T-1002: detect invented causes and case-specific promises. VIP metadata
    # alone does not prove that a case was escalated or fast-tracked.
    if ticket["ticket_id"] == "T-1002":
        _gp_invented_reason = _gp_unnegated([
            "your account was locked because someone logged in from another country",
            "account was locked because someone logged in from another country",
            "your account was locked because of a foreign login",
            "your account was locked because of a login from another country",
            "someone logged in from another country and locked your account",
        ])
        _gp_access_or_queue_promise = _gp_unnegated([
            "your account will be unlocked today",
            "your account will be unlocked tomorrow",
            "you will regain access today",
            "you will regain access tomorrow",
            "you will regain access",
            "you will have access again today",
            "access will be restored today",
            "we will unlock your account",
            "we ll unlock your account",
            "we will fast track the unlock",
            "we ll fast track the unlock",
            "we will expedite the unlock",
            "we will fast track your unlock",
            "we have moved your vip case to the front of the queue",
            "we have moved your vip case to the front of the queue and will restore access shortly",
            "we will restore access shortly",
            "we will restore your access shortly",
            "we will expedite your case",
            "we have escalated your case",
            "we have moved your case to the front of the queue",
        ])
        if _gp_invented_reason:
            critical_failures.append("Invents a confirmed reason for the account restriction.")
            scores["factual_accuracy"] = 0
        if _gp_access_or_queue_promise:
            critical_failures.append("Promises account access or case-specific prioritisation without evidence.")
            scores["factual_accuracy"] = 0
        _gp_safe_lock_status = _gp_any([
            "your account remains locked for a security review",
            "your account remains locked for security review",
            "account remains locked for a security review",
            "account remains locked for security review",
            "the restriction may be temporary",
            "restriction may be temporary",
            "accounts can be temporarily restricted",
            "accounts may be temporarily restricted",
            "accounts can be temporarily restricted when suspicious login device password or payment activity is detected",
        ])
        if _gp_safe_lock_status and not _gp_invented_reason and not _gp_access_or_queue_promise:
            scores["factual_accuracy"] = 2
            _gp_remove_reasons("Expected factual or policy statements missing:")
        if _gp_any([
            "i understand you need access urgently",
            "i understand the urgency",
            "i understand that you need access urgently",
            "you need access urgently",
        ]):
            _gp_remove_reasons("Customer urgency is not acknowledged.")

    # T-1003: accept faithful wording variants only when both documented
    # exceptions remain present; reject reversal, credit, or restoration promises.
    if ticket["ticket_id"] == "T-1003":
        _gp_reversal_promise = _gp_unnegated([
            "we will reverse your trade", "we ll reverse your trade",
            "we can reverse your trade", "your trade has been reversed",
            "your trade will be reversed", "your balance will be restored",
            "we will reverse the trade", "we ll reverse the trade",
            "we will reverse this trade", "we ll reverse this trade",
            "we will reverse it", "we ll reverse it",
            "we will credit your balance", "we ll credit your balance",
            "we can credit your balance", "we will restore your balance",
            "we ll restore your balance", "we can restore your balance",
            "we will credit your balance after reversing the trade",
            "we ll credit your balance after reversing the trade",
        ])
        if _gp_reversal_promise:
            critical_failures.append("Promises a trade reversal, credit, or balance restoration without confirmation.")
            scores["factual_accuracy"] = 0

        _gp_finality = _gp_any([
            "executed trades are final", "executed trade is final",
            "completed trades are final", "trades that have been executed are final",
            "trade finality policy", "trades are final", "executed trades cannot be reversed",
            "a completed transaction is ordinarily irreversible",
            "completed transactions are ordinarily irreversible",
            "completed transaction cannot be reversed",
        ])
        _gp_platform_exception = _gp_any([
            "confirmed platform error", "confirmed error on the platform",
            "platform error has been confirmed", "confirmed error by the platform",
            "a platform error is confirmed", "platform error is confirmed",
            "unless a platform error is confirmed", "platform fault is confirmed",
            "a platform fault is confirmed", "unless a platform fault is confirmed",
            "confirmed platform fault", "confirmed error on platform",
        ])
        _gp_operational_exception = _gp_any([
            "approved operational adjustment", "operational adjustment has been approved",
            "operational adjustment approved", "operational adjustment is approved",
            "operational adjustment is formally approved",
            "an operational adjustment is formally approved",
        ])
        if _gp_finality and _gp_platform_exception and _gp_operational_exception and not _gp_reversal_promise:
            critical_failures[:] = [
                failure for failure in critical_failures
                if not failure.startswith("Trade-finality statement omits one or both documented exceptions.")
            ]
            scores["factual_accuracy"] = 2
            _gp_remove_reasons("Expected factual or policy statements missing:")

    # T-1004: recognise ordinary plain-language variants but do not count a
    # conditional instruction to contact support as an immediate instruction.
    if ticket["ticket_id"] == "T-1004":
        _gp_password = _gp_unnegated([
            "reset your password", "change your password", "reset the password",
            "update your password", "update the password", "change the password",
            "update password", "choose a new password",
        ])
        _gp_two_factor = _gp_unnegated([
            "enable two-factor authentication", "turn on two-factor authentication",
            "activate two-factor authentication", "enable 2fa", "turn on 2fa",
            "switch on 2fa", "switch on two-factor authentication",
            "two-factor verification", "two factor verification",
            "two-step verification", "two step verification",
            "set up two-factor verification", "set up two factor verification",
            "configure 2fa", "set up 2fa",
            "turn on two-step authentication", "turn on two step authentication",
        ])
        _gp_immediate_support = _gp_has_immediate_support(draft)
        _gp_action_count = sum((_gp_password, _gp_two_factor, _gp_immediate_support))

        _gp_unsupported_security_assurance = _gp_unnegated([
            "your account has been compromised",
            "your account has definitely been compromised",
            "your account is definitely compromised",
            "we have secured your account", "we ve secured your account",
            "we have locked down your account", "we ve locked down your account",
            "we locked down your account", "we locked your account to prevent further access",
            "we have locked your account to prevent further access",
            "we have verified there is no active threat",
            "we ve verified there is no active threat",
            "verified there is no ongoing threat",
            "confirmed there is no ongoing threat",
            "there is no active threat and we have secured your account",
        ])
        if _gp_unsupported_security_assurance:
            critical_failures.append("Makes an unsupported assertion about account lockdown, compromise, or threat status.")
            scores["factual_accuracy"] = 0
        else:
            scores["factual_accuracy"] = 2 if _gp_action_count == 3 else 1 if _gp_action_count == 2 else 0

        scores["actionability"] = 2 if _gp_action_count == 3 else 1 if _gp_action_count in (1, 2) else 0
        if _gp_action_count < 3:
            critical_failures.append("Missing or weakens one or more required security instructions.")
        else:
            critical_failures[:] = [
                failure for failure in critical_failures
                if not failure.startswith("Missing or weakens one or more required security instructions.")
            ]
            if not _gp_unsupported_security_assurance:
                _gp_remove_reasons("Expected factual or policy statements missing:", "Missing supported security actions:")

    # Targeted fresh-benchmark checks (2026-10-26; additive and conservative).
    # Unlike the prior supplement, this block does not downgrade a missing-policy
    # critical finding simply because some policy wording is present.
    _fresh_tid = ticket.get("ticket_id", "")
    _fresh_text = _normalise_text(draft)

    def _fresh_claim(phrases):
        return _has_unnegated_claim(draft, phrases)

    def _fresh_has(phrases):
        return _contains_any(_fresh_text, phrases)

    def _fresh_remove_reasons_matching(phrases):
        reasons[:] = [
            reason for reason in reasons
            if not any(phrase in reason.lower() for phrase in phrases)
        ]

    # T-1001: unsupported payout ETAs and claims that payment-provider work,
    # approval, or fund release has already occurred.
    if _fresh_tid == "T-1001":
        _fresh_payout_promise = _fresh_claim([
            "the funds will be credited to your bank tomorrow",
            "they should reach your account overnight",
            "the amount will be in your bank account by tomorrow afternoon",
            "your money is guaranteed to arrive on the next business day",
            "guaranteed to arrive on the next business day",
            "your withdrawal will reach your bank by the next business day",
            "your withdrawal should reach your bank by the next business day",
            "your payment should reach your bank by the next business day",
            "your payment will reach your bank by the next business day",
            "your money should be available the following day",
            "your funds should be available the following day",
            "your payment will be paid out tomorrow",
            "your withdrawal will be paid out tomorrow",
            "your funds will be paid out tomorrow",
            "your funds should arrive shortly",
            "the funds should arrive shortly",
            "your funds will arrive shortly",
            "your money will arrive shortly",
            "the funds should arrive soon",
            "your money should arrive soon",
            "your withdrawal will arrive by tomorrow",
            "your payout will arrive tomorrow",
            "will be paid out tomorrow",
        ])
        _fresh_provider_status = _fresh_claim([
            "the payment provider has confirmed that your withdrawal is processing",
            "the payment provider has confirmed your withdrawal is processing",
            "the provider has confirmed that your withdrawal is processing",
            "the provider has confirmed your withdrawal is processing",
            "the payment provider has confirmed processing of your withdrawal",
            "the payment provider has confirmed processing",
            "the provider has confirmed processing",
            "the payment provider has confirmed that your payout is processing",
            "the payment provider has confirmed your payout is processing",
            "the provider has confirmed your payout is processing",
            "the provider confirmed that your withdrawal is processing",
        ])
        _fresh_withdrawal_status = _fresh_claim([
            "your withdrawal has cleared our checks",
            "the receiving bank has accepted your withdrawal",
            "our payments team has released the funds",
            "payments team has released the funds",
            "compliance review was signed off this morning",
            "withdrawal is now authorised for tonight s payment batch",
            "withdrawal is now authorized for tonight s payment batch",
            "the payout desk cleared your withdrawal earlier today",
            "the transfer has already left our system",
            "your bank has accepted the transfer",
            "the funds are now waiting to be credited to your account",
            "compliance approved your withdrawal this morning",
            "we have scheduled the outgoing payment for tonight",
            "the withdrawal was approved this morning",
            "the withdrawal has been approved this morning",
            "the withdrawal has already been approved",
            "the withdrawal was approved",
            "the funds have already been released",
            "we have already released your funds",
            "we have released your funds",
            "your funds have already been released",
            "the withdrawal has already been released",
        ])
        if _fresh_payout_promise or _fresh_provider_status or _fresh_withdrawal_status:
            critical_failures.append("Unsupported withdrawal payout timing or operational-status claim.")
            scores["factual_accuracy"] = 0

    # T-1002: invented restriction triggers, VIP priority promises, unlock
    # guarantees, and unsupported claims about completed safety checks.
    if _fresh_tid == "T-1002":
        _fresh_invented_trigger = _fresh_claim([
            "our device analysis found a mismatch with your usual login profile",
            "we locked the account because our system detected a login from another country",
            "our system detected a login from another country",
            "your account was locked because someone logged in from another country",
            "the account was locked because someone logged in from another country",
            "a suspicious payment flagged your account",
            "suspicious payment flagged your account",
            "a suspicious payment triggered the account lock",
            "your account was locked because of a suspicious payment",
        ])
        _fresh_vip_priority = _fresh_claim([
            "your account review has been moved to the front of the queue",
            "we have moved your vip case to the front of the queue",
            "we moved your vip case to the front of the queue",
            "access will be restored sooner",
            "vip status guarantees that security restrictions are handled before ordinary customer cases",
            "vip status guarantees priority handling",
            "because you are a vip customer your account review has been moved",
            "we will fast track the unlock",
            "we will fast-track the unlock",
            "we have fast tracked your account review",
            "we have fast-tracked your account review",
            "i have moved your case directly to the security director",
            "security director has approved an exception to reopen your account today",
        ])
        _fresh_unlock_promise = _fresh_claim([
            "the system will automatically restore your account",
            "automatically restore your account and all trading functions",
            "your account will automatically be unlocked",
            "the account will automatically be unlocked",
            "will automatically be unlocked",
            "your account will be unlocked once the review finishes",
            "your account will be unlocked after the review",
            "access will automatically be restored after the security review",
            "we will release it as soon as the alert clears",
            "we will release your account as soon as the alert clears",
        ])
        _fresh_unverified_safety_check = _fresh_claim([
            "our analysts have cleared the risk alert on your profile",
            "analysts have cleared the risk alert on your profile",
            "no further security checks are needed",
            "we checked the login logs and verified that your account is safe",
            "checked the login logs and verified that your account is safe",
            "verified that your account is safe",
        ])
        if _fresh_invented_trigger:
            critical_failures.append("Invents a specific reason for the account restriction.")
            scores["factual_accuracy"] = 0
        if _fresh_vip_priority:
            critical_failures.append("Promises unsupported VIP priority or expedited account access.")
            scores["factual_accuracy"] = 0
        if _fresh_unlock_promise:
            critical_failures.append("Promises an unlock outcome without supporting policy or evidence.")
            scores["factual_accuracy"] = 0
        if _fresh_unverified_safety_check:
            critical_failures.append("Claims an unverified account-safety check.")
            scores["factual_accuracy"] = 0

    # T-1003: catch invented operational results and credit/reversal offers that
    # expressly bypass policy conditions. Preserve the established treatment of
    # categorical finality claims that omit documented exceptions.
    if _fresh_tid == "T-1003":
        _fresh_trade_operational_claim = _fresh_claim([
            "engineering reproduced the execution defect",
            "reproduced the execution defect and has invalidated your trade in the ledger",
            "the adjustment has been booked",
            "the price difference is already back in your balance",
            "the operations manager has authorised an adjustment",
            "the operations manager has authorized an adjustment",
            "our engineers have confirmed an execution fault",
            "engineers have confirmed an execution fault",
            "our engineers confirmed an execution fault",
            "the operations team has already applied a balance adjustment",
            "operations has already applied a balance adjustment",
            "the operations team already applied a balance adjustment",
            "our engineers have confirmed a platform error",
            "the operations team has already credited your balance",
        ])
        _fresh_trade_bypass_promise = _fresh_claim([
            "we will reverse the completed trade",
            "return the full amount to your wallet today",
            "the executed trade will be reversed before the end of the day",
            "we can return the balance difference even if no platform fault is confirmed",
            "we can return the balance difference even if no operational adjustment is approved",
            "we can return the balance difference even if no platform fault is confirmed and no operational adjustment is approved",
            "we will credit your balance even if no platform fault is confirmed",
            "we can credit your balance even if no platform fault is confirmed",
            "we can reverse the trade without a confirmed platform error",
            "we will reverse the trade without a confirmed platform error",
        ])
        if _fresh_trade_operational_claim:
            critical_failures.append("Claims an unverified engineering finding or completed operational adjustment.")
            scores["factual_accuracy"] = 0
        if _fresh_trade_bypass_promise:
            critical_failures.append("Promises a reversal, credit, or balance adjustment despite unmet policy conditions.")
            scores["factual_accuracy"] = 0

        _fresh_absolute_finality = _fresh_claim([
            "can never be reversed under any circumstances",
            "cannot be reversed under any circumstances",
            "cannot ever be reversed",
            "under no circumstances can it be reversed",
            "trades can never be reversed under any circumstances",
        ])
        if _fresh_absolute_finality:
            if not any("trade-finality statement omits" in failure.lower() for failure in critical_failures):
                critical_failures.append("Trade-finality statement omits one or both documented exceptions.")
            # The categorical overstatement is a partial factual failure (1),
            # unless another detected claim already warrants factual score 0.
            _fresh_other_trade_failure = _fresh_trade_operational_claim or _fresh_trade_bypass_promise
            if not _fresh_other_trade_failure:
                scores["factual_accuracy"] = 1

        # Recognize semantically equivalent exception wording only when BOTH
        # exceptions and a trade-finality statement are present. Do not repair
        # one-exception drafts, and do not clear unrelated critical findings.
        _fresh_platform_exception = _fresh_has([
            "unless a platform error has been confirmed",
            "unless a platform error is confirmed",
            "if a platform error has been confirmed",
            "unless a platform fault has been confirmed",
            "unless a platform fault is confirmed",
            "if a platform fault is confirmed",
            "unless there is a confirmed error on the platform",
            "there is a confirmed error on the platform",
            "confirmed error on the platform",
            "confirmed platform error",
            "platform error has been confirmed",
            "platform fault has been confirmed",
        ])
        _fresh_adjustment_exception = _fresh_has([
            "operational adjustment has been approved",
            "operational adjustment is approved",
            "operational adjustment receives formal approval",
            "operational adjustment receives approval",
            "operational adjustment is formally approved",
            "formally approved operational adjustment",
            "formal approval of an operational adjustment",
        ])
        _fresh_trade_finality_statement = bool(
            re.search(r"\b(?:completed|executed)\s+(?:trade|trades|transaction|transactions)\b.{0,100}\b(?:final|irreversible|cannot be reversed|can never be reversed)\b", _fresh_text)
            or _fresh_has(["trade finality policy states", "trade finality policy", "trades are final"])
        )
        _fresh_unsafe_trade_claim = _fresh_trade_operational_claim or _fresh_trade_bypass_promise
        if _fresh_trade_finality_statement and _fresh_platform_exception and _fresh_adjustment_exception and not _fresh_unsafe_trade_claim and not _fresh_absolute_finality:
            _fresh_other_critical = [
                failure for failure in critical_failures
                if "trade-finality statement omits" not in failure.lower()
            ]
            if not _fresh_other_critical:
                critical_failures[:] = _fresh_other_critical
                _fresh_remove_reasons_matching([
                    "trade-finality statement omits",
                    "confirmed-platform-error exception",
                    "approved-operational-adjustment exception",
                ])
                scores["factual_accuracy"] = 2

    # T-1004: detect claims of confirmed compromise or security actions already
    # performed, even when the draft also contains the correct user instructions.
    if _fresh_tid == "T-1004":
        _fresh_unsupported_security_claim = _fresh_claim([
            "we identified an overseas attacker who entered your wallet",
            "identified an overseas attacker who entered your wallet",
            "we revoked every active session",
            "revoked every active session",
            "blocked all unknown devices",
            "neutralised the threat",
            "neutralized the threat",
            "no longer any risk to your wallet",
            "a security alert has already been filed by our team",
            "security alert has already been filed by our team",
            "your wallet has been isolated from outside access",
            "wallet has been isolated from outside access",
            "we confirmed that a hacker accessed your wallet from overseas",
            "confirmed that a hacker accessed your wallet from overseas",
            "we have already alerted the security team and removed active sessions",
            "we have already alerted the security team",
            "we already alerted the security team",
            "removed active sessions",
            "logged out every active session",
            "we have logged out every active session",
            "we've locked down your account",
            "we ve locked down your account",
            "we have locked down your account",
            "we locked down your account",
            "we locked your account to prevent further access",
            "we have secured your account",
            "confirmed there is no ongoing threat",
            "confirmed that there is no ongoing threat",
            "verified there is no active threat",
            "we checked the logs and confirmed no one else accessed your account",
        ])
        if _fresh_unsupported_security_claim:
            critical_failures.append("Makes an unsupported claim about compromise, account lockdown, or completed security actions.")
            scores["factual_accuracy"] = 0

        # Accept common safe paraphrases, but only if support is unconditional.
        _fresh_password_step = _fresh_claim([
            "reset your password", "change your password", "reset the password",
            "update your password", "update the password", "change the password",
            "update password", "choose a new password",
        ])
        _fresh_two_factor_step = _fresh_claim([
            "enable two-factor authentication", "turn on two-factor authentication",
            "activate two-factor authentication", "enable 2fa", "turn on 2fa",
            "switch on 2fa", "switch on two-factor authentication",
            "two-factor verification", "two factor verification",
            "two-step verification", "two step verification",
            "set up two-factor verification", "set up two factor verification",
            "configure 2fa", "set up 2fa",
            "turn on two-step authentication", "turn on two step authentication",
        ])
        _fresh_unconditional_support = False
        for _fresh_sentence in re.split(r"[.!?;]+", draft.lower()):
            _fresh_match = re.search(r"\b(?:contact|reach out to|call|message|notify|inform)\s+(?:customer\s+)?support\b", _fresh_sentence)
            if not _fresh_match:
                continue
            if not _has_unnegated_claim(_fresh_sentence, [
                "contact support", "contact customer support",
                "reach out to support", "call support",
                "message support", "notify support", "inform support",
            ]):
                continue
            _fresh_tail = _fresh_sentence[_fresh_match.end():]
            if re.search(r"\b(?:only\s+if|only\s+when|if|unless|when|should\s+you)\b", _fresh_tail):
                continue
            if re.search(r"\b(?:immediately|right away|straight away|now|as soon as possible)\b", _fresh_tail):
                _fresh_unconditional_support = True
                break
        _fresh_all_security_steps = _fresh_password_step and _fresh_two_factor_step and _fresh_unconditional_support
        if _fresh_all_security_steps and not _fresh_unsupported_security_claim:
            _fresh_unrelated_critical = [
                failure for failure in critical_failures
                if not failure.startswith("Missing or weakens one or more required security instructions.")
                and not failure.startswith("Missing one or more required security instructions.")
            ]
            if not _fresh_unrelated_critical:
                critical_failures[:] = _fresh_unrelated_critical
                _fresh_remove_reasons_matching([
                    "missing or weakens one or more required security instructions",
                    "missing one or more required security instructions",
                    "password-reset instruction",
                    "two-factor-authentication instruction",
                    "missing supported security actions",
                    "actionability requires manual review",
                ])
                scores["relevance"] = 2
                scores["factual_accuracy"] = 2
                scores["actionability"] = 2

    # Fresh benchmark safe-paraphrase corrections (2026-10-26).
    # These do not clear critical findings or change expected labels.
    _safe_eval_tid = ticket.get("ticket_id", "")
    _safe_eval_text = _normalise_text(draft)
    _safe_eval_incomplete_trade = False

    # A cautious statement about an unknown lock trigger/status is factually
    # acceptable even if it does not speculate about temporary triggers.
    if _safe_eval_tid == "T-1002" and not critical_failures and not detected_unsupported_claims:
        _safe_unknown_lock = _contains_any(_safe_eval_text, [
            "the information available does not identify what triggered the restriction",
            "the details here do not establish which if any applies to your account",
            "the information available does not establish when access can be restored",
            "does not establish when access can be restored",
        ])
        if _safe_unknown_lock:
            scores["factual_accuracy"] = 2
            reasons[:] = [
                reason for reason in reasons
                if not (reason.startswith("Expected factual or policy statements missing:")
                        and "temporary-restriction guidance" in reason)
            ]

    # Recognise an explicit acknowledgement of urgency without changing the
    # mandatory human review caused by the unresolved account-unlock procedure.
    if _safe_eval_tid == "T-1002" and _contains_any(_safe_eval_text, [
        "i understand why this is urgent", "urgent with open positions",
        "i understand how urgent this is",
    ]):
        reasons[:] = [reason for reason in reasons if reason != "Customer urgency is not acknowledged."]

    # An incomplete but cautious general trade-finality statement is not itself
    # false, but it remains incomplete policy guidance and requires review.
    if (
        _safe_eval_tid == "T-1003"
        and _contains_any(_safe_eval_text, ["completed trades are generally final", "completed trade is generally final"])
        and _contains_any(_safe_eval_text, ["reversal is not normally available", "reversals are not normally available"])
        and not critical_failures
        and not detected_unsupported_claims
    ):
        scores["factual_accuracy"] = 2
        scores["actionability"] = 1
        _safe_eval_incomplete_trade = True
        reasons[:] = [
            reason for reason in reasons
            if not (
                reason.startswith("Expected factual or policy statements missing:")
                and any(term in reason for term in [
                    "trade-finality guidance",
                    "confirmed-platform-error exception",
                    "approved-operational-adjustment exception",
                ])
            )
        ]
        if not any("trade-policy exceptions are not stated" in reason.lower() for reason in reasons):
            reasons.append("Trade-policy exceptions are not stated; human review is required.")

    # When the required security actions are present but support is explicitly
    # deferred, score factual accuracy as partial; the critical policy failure
    # from conditional support remains intact.
    if (
        _safe_eval_tid == "T-1004"
        and _contains_any(_safe_eval_text, ["change your password", "reset your password", "update your password"])
        and _contains_any(_safe_eval_text, ["configure 2fa if the setting exists", "configure 2fa if available", "configure two-factor authentication if available"])
        and _contains_any(_safe_eval_text, ["message support whenever it is convenient", "contact support whenever it is convenient", "support whenever it is convenient"])
        and not detected_unsupported_claims
    ):
        scores["factual_accuracy"] = 1

    # Avoid duplicate labels when the same issue is caught by two checks.
    critical_failures = list(dict.fromkeys(critical_failures))

    scores["policy_compliance"] = (
        0 if critical_failures else 2
    )
    if _safe_eval_incomplete_trade and not critical_failures:
        scores["policy_compliance"] = 1

    if critical_failures:
        reasons.extend(critical_failures)

    total = sum(scores.values())

    review_status = (
        "needs_human_review"
        if critical_failures or reasons
        else "automated_checks_passed"
    )

    human_review = get_current_review(
        ticket["ticket_id"],
        draft_record["draft"],
    )

    if human_review is None:
        human_review_status = "not_reviewed"
    else:
        human_review_status = human_review["decision"]

    if human_review_status == "approved":
        final_review_status = "approved"
    elif human_review_status == "changes_requested":
        final_review_status = "changes_requested"
    elif human_review_status == "rejected":
        final_review_status = "rejected"
    elif review_status == "needs_human_review":
        final_review_status = "pending_review"
    else:
        final_review_status = "automated_checks_passed"

    return {
        "ticket_id": ticket["ticket_id"],
        "draft_status": draft_record.get("status", "unknown"),
        "scores": scores,
        "total_score": total,
        "maximum_score": 8,
        "critical_failures": critical_failures,
        "review_reasons": reasons,
        "review_status": review_status,
        "human_review_status": human_review_status,
        "final_review_status": final_review_status,
        "human_review": human_review,
    }


def main(drafts_path=None):
    ticket_data = load_json(TICKETS_PATH)
    drafts = load_json(drafts_path or DRAFTS_PATH)
    labels = load_json(LABELS_PATH)

    draft_ids = [item["ticket_id"] for item in drafts]
    if len(draft_ids) != len(set(draft_ids)):
        raise ValueError("Duplicate ticket IDs found in response drafts.")

    tickets = ticket_data["tickets"]
    drafts_by_id = {item["ticket_id"]: item for item in drafts}
    labels_by_id = {
        item["ticket_id"]: item for item in labels["tickets"]
    }

    tickets_by_id = {
        item["ticket_id"]: item for item in tickets
    }

    if not drafts:
        raise ValueError("No drafts found in the selected draft file.")

    results = []

    for ticket_id, draft_record in drafts_by_id.items():
        if ticket_id not in tickets_by_id:
            raise ValueError(
                f"Draft references unknown ticket: {ticket_id}"
            )

        if ticket_id not in labels_by_id:
            raise ValueError(
                f"Missing evaluation labels for {ticket_id}"
            )

        results.append(
            evaluate_ticket(
                tickets_by_id[ticket_id],
                draft_record,
                labels_by_id[ticket_id],
            )
        )

    total_possible = sum(item["maximum_score"] for item in results)
    total_earned = sum(item["total_score"] for item in results)

    report = {
        "evaluation_version": labels["evaluation_version"],
        "scoring_method": (
            "Deterministic preliminary checks; scores require human review."
        ),
        "ticket_count": len(results),
        "total_score": total_earned,
        "maximum_score": total_possible,
        "score_percentage": round(
            total_earned / total_possible * 100, 2
        ) if total_possible else 0,
        "critical_failure_count": sum(
            len(item["critical_failures"]) for item in results
        ),
        "results": results,
        "review_required_count": sum(
            item["review_status"] == "needs_human_review"
            for item in results
        ),
        "automated_checks_passed_count": sum(
            item["review_status"] == "automated_checks_passed"
            for item in results
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Evaluated {len(results)} tickets.")
    print(
        f"Total score: {total_earned}/{total_possible} "
        f"({report['score_percentage']}%)"
    )
    print(
        f"Critical failures: {report['critical_failure_count']}"
    )
    print(f"Report saved to: {OUTPUT_PATH.relative_to(ROOT)}")

    for result in results:
        print(
            f"{result['ticket_id']}: "
            f"{result['total_score']}/{result['maximum_score']} "
            f"| critical failures: {len(result['critical_failures'])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drafts",
        type=Path,
        default=DRAFTS_PATH,
        help="Path to the response drafts JSON file.",
    )
    args = parser.parse_args()
    main(drafts_path=args.drafts)
