import unittest

from src.ticket_reporting import customer_report_summary


class TestCustomerReportSummary(unittest.TestCase):

    def test_attributes_customer_message(self):
        result = customer_report_summary(
            "My bank transfer has not arrived."
        )

        self.assertEqual(
            result,
            'The customer reports that "My bank transfer has not arrived."'
        )

    def test_falls_back_to_subject_when_message_is_empty(self):
        result = customer_report_summary(
            "",
            "Bank transfer not received"
        )

        self.assertEqual(
            result,
            'The customer reports that "Bank transfer not received"'
        )

    def test_handles_empty_message_and_subject(self):
        result = customer_report_summary("", "")

        self.assertEqual(
            result,
            "The customer submitted a support request."
        )

    def test_trims_whitespace(self):
        result = customer_report_summary(
            "  I cannot access my account.  "
        )

        self.assertEqual(
            result,
            'The customer reports that "I cannot access my account."'
        )


if __name__ == "__main__":
    unittest.main()
