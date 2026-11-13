def customer_report_summary(message: str, subject: str = "") -> str:
    reported_text = str(message or "").strip()

    if not reported_text:
        reported_text = str(subject or "").strip()

    if not reported_text:
        return "The customer submitted a support request."

    return f'The customer reports that "{reported_text}"'