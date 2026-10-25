from django.test import SimpleTestCase

from .ml_service import CATEGORY_MAPS, DEFAULT_INPUTS, get_credit_risk_service, resolve_report_path


FORM_INPUTS = {
    "age": 30,
    "income": 65000,
    "home_ownership": "MORTGAGE",
    "emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 12000,
    "int_rate": 11.5,
    "loan_percent_income": 0.18,
    "default_on_file": "N",
    "cred_hist_length": 6,
}


class CreditRiskServiceTests(SimpleTestCase):
    def test_category_maps_match_label_encoder_order(self):
        self.assertEqual(CATEGORY_MAPS["person_home_ownership"]["MORTGAGE"], 0)
        self.assertEqual(CATEGORY_MAPS["person_home_ownership"]["RENT"], 3)
        self.assertEqual(CATEGORY_MAPS["loan_intent"]["DEBTCONSOLIDATION"], 0)
        self.assertEqual(CATEGORY_MAPS["loan_intent"]["VENTURE"], 5)

    def test_default_profile_can_be_predicted(self):
        result = get_credit_risk_service().predict(DEFAULT_INPUTS)

        self.assertIn(result.label, {"Safe", "Default"})
        self.assertGreaterEqual(result.risk_probability, 0)
        self.assertLessEqual(result.risk_probability, 1)

    def test_report_route_only_allows_known_images(self):
        self.assertIsNone(resolve_report_path("../params.json"))
        self.assertIsNotNone(resolve_report_path("comparison_80_20_chart.png"))

    def test_index_page_renders(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Credit Risk Lab")

    def test_prediction_form_post_renders_result(self):
        response = self.client.post("/", FORM_INPUTS)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "KNN")
        self.assertContains(response, "LINEAR REGRESSION")
        self.assertContains(response, "LOGISTIC REGRESSION")
