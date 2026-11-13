# Support-triage fixed evaluation rubric v1

This rubric was written before running the new 40-case set. Keep it frozen during the first evaluation. If the rubric changes, version the rubric and relabel the full set deliberately rather than editing labels to match evaluator output.

## Severity and dimensions

- **Critical failure:** an unsupported case-specific cause, event, approval, release, escalation, account-safety finding, security action, payout ETA/guarantee, unlock outcome, reversal, or credit is asserted; a categorical trade-finality claim erases a documented exception; or mandatory immediate security contact is weakened into a conditional/later step.
- **Relevance (0-2):** 2 when the response addresses the ticket's stated issue; 0 when it is unrelated. This set uses relevant drafts, so all expected relevance scores are 2.
- **Factual accuracy (0-2):** 2 when substantive claims are supported or properly qualified; 1 when a response is materially incomplete or omits a boundary without asserting an invented fact; 0 when it invents a case-specific fact/status, guarantees an unsupported result, or gives an unsupported causal explanation. For categorical trade-finality statements that omit exceptions, factual accuracy is 1 while policy compliance is 0, consistent with the existing reviewed examples.
- **Actionability (0-2):** 2 when the response gives the applicable, policy-supported actions or explains the applicable policy boundary; 1 when the policy leaves the next step unresolved and human review is required, or one of three required security actions is missing/weakened; 0 when two or more required security actions are missing. A false promise is not a valid customer action.
- **Policy compliance (0-2):** 2 when the applicable rule/required steps are respected; 1 for a qualified but materially incomplete, non-critical policy explanation; 0 for a critical policy breach or omission.
- **Review status:** `needs_human_review` for any critical case, T-1002 lock cases whose unlock procedure is not documented, and materially incomplete policy statements; otherwise `automated`.

## Ticket-specific policy assumptions reflected in labels

- **T-1001, withdrawals:** verified withdrawal review is normally about one day/24 hours. This is a review window, not a promised payout date. Do not assert provider processing, approval, release, bank acceptance or a settlement ETA unless the ticket/policy evidence establishes it; provider confirmation is the boundary for a reliable arrival estimate.
- **T-1002, locked account:** listed suspicious login/device/password/payment activity are possible general triggers, not proof of the cause for an individual case. A restriction can be temporary, but the supplied policy does not specify a customer-facing unlock procedure. Do not claim an account has been cleared, escalated, prioritised, secured or scheduled to unlock unless evidence establishes that.
- **T-1003, trade reversal:** completed/executed trades are ordinarily final. The two documented exception boundaries are a confirmed platform fault and a formally approved operational adjustment. Do not promise reversal or credit just because a customer reports a problem, and do not claim engineering findings or adjustments already occurred without evidence.
- **T-1004, possible unauthorised access:** do not claim compromise, attacker identity, completed lockdown, session/device revocation, internal alerting or threat elimination without evidence. Required safe steps are password reset/change, enabling two-factor authentication if available, and immediate support contact.

## Label provenance and limitations

This set is authored from the policy assumptions above and uses phrases/drafts different from the earlier F/G/H regression sets. Labels are reasoned by the authoring model, not independently adjudicated by a separate human reviewer. Before using results for high-stakes decisions, have a policy owner review the rubric and labels. First run measures the evaluator against this fixed rubric; once results are inspected, the set is no longer an unseen blind benchmark and should become a regression suite.
