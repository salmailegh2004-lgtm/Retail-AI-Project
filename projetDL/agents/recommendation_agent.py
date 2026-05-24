import json
from datetime import datetime
from agents.crewai_layer import CrewAILayer


class RecommendationAgent:
    def __init__(self, mode,
                 future_df=None, peaks=None, forecasting_summary=None,
                 detected_anomalies=None, anomaly_summary=None,
                 use_ollama=True,
                 ollama_model="llama3.2"):

        self.mode = mode
        self.future_df = future_df
        self.peaks = peaks
        self.forecasting_summary = forecasting_summary or {}
        self.detected_anomalies = detected_anomalies
        self.anomaly_summary = anomaly_summary or {}

        self.use_ollama = use_ollama
        self.ollama_model = ollama_model

        self.recommendations = []
        self.logs = []

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "RecommendationAgent",
            "action": action,
            "status": status,
            "details": details
        })

    def add_recommendation(self, category, priority, recommendation, reason):
        self.recommendations.append({
            "category": category,
            "priority": priority,
            "recommendation": recommendation,
            "reason": reason
        })

    def generate_forecasting_recommendations(self):
        trend = self.forecasting_summary["trend_analysis"]["trend"]
        change_pct = self.forecasting_summary["trend_analysis"]["change_percentage"]

        if trend == "Increasing demand":
            self.add_recommendation(
                "Demand Forecasting",
                "High",
                "Increase stock availability and prepare additional inventory.",
                f"The model predicts a demand increase of {change_pct:.2f}%."
            )
        elif trend == "Decreasing demand":
            self.add_recommendation(
                "Demand Forecasting",
                "Medium",
                "Reduce excessive stock preparation for the next period.",
                f"The model predicts a demand decrease of {change_pct:.2f}%."
            )
        else:
            self.add_recommendation(
                "Demand Forecasting",
                "Low",
                "Maintain current stock planning strategy.",
                "The predicted demand is stable."
            )

        if self.peaks is not None and len(self.peaks) > 0:
            peak_dates = self.peaks["date"].astype(str).tolist()

            self.add_recommendation(
                "Peak Demand",
                "High",
                "Prepare additional inventory before detected peak demand dates.",
                f"Future peak periods detected on: {peak_dates}."
            )

        inventory = self.forecasting_summary["inventory_requirements"]
        total_demand = inventory["total_predicted_demand"]
        margin = inventory["safety_margin"]

        self.add_recommendation(
            "Inventory Management",
            "High",
            f"Prepare inventory based on forecasted demand with a {int(margin * 100)}% safety margin.",
            f"The expected demand is approximately {total_demand:.0f} units during the forecast period."
        )

        promo = self.forecasting_summary.get("promotion_impact", {})

        if promo.get("available") and promo.get("correlation_onpromotion_sales") is not None:
            corr = promo["correlation_onpromotion_sales"]

            self.add_recommendation(
                "Promotion Strategy",
                "High" if corr > 0.5 else "Medium",
                "Use promotions carefully because they influence sales demand.",
                f"Promotion-sales correlation is {corr:.2f}."
            )

    def generate_anomaly_recommendations(self):
        anomaly_rate = self.anomaly_summary["anomaly_rate"]
        total_anomalies = self.anomaly_summary["total_anomalies_detected"]
        risk_distribution = self.anomaly_summary["risk_distribution"]
        anomaly_types = self.anomaly_summary["anomaly_type_distribution"]

        priority = "High" if anomaly_rate > 0.10 else "Medium"

        self.add_recommendation(
            "Anomaly Monitoring",
            priority,
            "Investigate abnormal sales and promotion behavior before making final stock decisions.",
            f"{total_anomalies} anomalies were detected with risk distribution: {risk_distribution}."
        )

        self.add_recommendation(
            "Operational Risk",
            priority,
            "Review the main anomaly categories to identify operational inconsistencies.",
            f"Detected anomaly types: {anomaly_types}."
        )

        self.add_recommendation(
            "Business Control",
            "High",
            "Validate suspicious periods before taking inventory or promotion decisions.",
            "Human review is required because abnormal behavior may indicate stock or promotion issues."
        )

    def generate_rule_based_recommendations(self):
        if self.mode == "forecasting":
            self.generate_forecasting_recommendations()

        elif self.mode == "anomaly":
            self.generate_anomaly_recommendations()

        else:
            raise ValueError("Invalid mode for recommendation agent.")

    def generate_crewai_business_insight(self):
        if not self.use_ollama:
            return "CrewAI/Ollama was not used."

        try:
            crewai_layer = CrewAILayer(model_name=self.ollama_model)

            insight = crewai_layer.run_recommendation_crew(
                forecasting_summary=self.forecasting_summary,
                anomaly_summary=self.anomaly_summary,
                rule_based_recommendations=self.recommendations
            )

            self.log_action(
                "crewai_business_insight",
                "success",
                "CrewAI generated business insight using Ollama."
            )

            return insight

        except Exception as e:
            self.log_action(
                "crewai_business_insight",
                "failed",
                str(e)
            )

            return "CrewAI/Ollama insight generation failed. Rule-based recommendations were used."

    def generate_action_plan(self):
        if self.mode == "forecasting":
            return {
                "stock_strategy": "Adjust stock based on forecasted demand and detected peak periods.",
                "promotion_strategy": "Use promotion impact analysis to prepare inventory before campaigns.",
                "decision_strategy": "Send forecasting recommendations to Human-in-the-Loop Agent for validation."
            }

        return {
            "risk_strategy": "Review detected anomalies before approving operational decisions.",
            "control_strategy": "Investigate suspicious periods and abnormal demand patterns.",
            "decision_strategy": "Send anomaly recommendations to Human-in-the-Loop Agent for validation."
        }

    def save_outputs(self, output_path="outputs/recommendation_report.json"):
        output = {
            "mode": self.mode,
            "recommendations": self.recommendations,
            "crewai_business_insight": self.crewai_business_insight,
            "ollama_business_insight": self.crewai_business_insight,
            "action_plan": self.generate_action_plan(),
            "logs": self.logs
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, default=str)

        return output

    def run(self):
        self.generate_rule_based_recommendations()
        self.crewai_business_insight = self.generate_crewai_business_insight()

        self.log_action(
            "run",
            "success",
            f"Recommendations generated for mode: {self.mode}"
        )

        return self.save_outputs()