import json
from crewai import Agent, Task, Crew, Process, LLM


class CrewAILayer:
    def __init__(self, model_name="llama3.2"):
        self.llm = LLM(
            model=f"ollama/{model_name}",
            base_url="http://localhost:11434"
        )

    def format_number(self, value):
        try:
            return f"{float(value):,.0f}"
        except Exception:
            return str(value)

    def format_percent(self, value):
        try:
            value = float(value)
            if abs(value) <= 1:
                return f"{value * 100:.2f}%"
            return f"{value:.2f}%"
        except Exception:
            return str(value)

    def extract_actions(self, recommendations):
        actions = []
        for rec in recommendations[:3]:
            actions.append(rec.get("recommendation", "Review this recommendation."))
        return actions

    def run_recommendation_crew(self, forecasting_summary, anomaly_summary, rule_based_recommendations):
        actions = self.extract_actions(rule_based_recommendations)

        if anomaly_summary and anomaly_summary.get("total_anomalies_detected") is not None:
            total_records = anomaly_summary.get("total_records_analyzed", "N/A")
            total_anomalies = anomaly_summary.get("total_anomalies_detected", "N/A")
            anomaly_rate = self.format_percent(anomaly_summary.get("anomaly_rate", 0))
            risk_distribution = anomaly_summary.get("risk_distribution", {})
            type_distribution = anomaly_summary.get("anomaly_type_distribution", {})

            main_type = max(type_distribution, key=type_distribution.get) if type_distribution else "unknown anomaly type"

            return f"""SUMMARY:
- The anomaly detection model identified {total_anomalies} anomalies out of {total_records} analyzed records.

KEY FINDINGS:
- The anomaly rate is {anomaly_rate}.
- The main anomaly type is {main_type}.
- Risk distribution: {risk_distribution}.

MAIN RISK:
- Abnormal sales or promotion behavior may lead to wrong inventory, promotion, or operational decisions.

RECOMMENDED ACTIONS:
- {actions[0] if len(actions) > 0 else "Investigate detected anomalies before making business decisions."}
- {actions[1] if len(actions) > 1 else "Review the main anomaly categories and affected dates."}
- {actions[2] if len(actions) > 2 else "Validate suspicious periods through human review."}
"""

        if forecasting_summary and forecasting_summary.get("trend_analysis") is not None:
            trend = forecasting_summary.get("trend_analysis", {})
            inventory = forecasting_summary.get("inventory_requirements", {})
            promo = forecasting_summary.get("promotion_impact", {})
            peaks = forecasting_summary.get("number_of_detected_future_peaks", 0)

            trend_name = trend.get("trend", "Unknown trend")
            change_pct = self.format_percent(trend.get("change_percentage", 0))
            total_demand = self.format_number(inventory.get("total_predicted_demand", "N/A"))
            recommended_inventory = self.format_number(inventory.get("total_recommended_inventory", "N/A"))
            safety_margin = self.format_percent(inventory.get("safety_margin", 0))
            promo_corr = promo.get("correlation_onpromotion_sales", None)

            promo_text = (
                f"Promotion-sales correlation is {promo_corr:.2f}."
                if isinstance(promo_corr, (int, float))
                else "Promotion impact was not available."
            )

            return f"""SUMMARY:
- The forecasting model predicts {trend_name.lower()} with {peaks} detected future peak periods.

KEY FINDINGS:
- Demand change over the forecast period is {change_pct}.
- Total predicted demand is approximately {total_demand} units.
- Recommended inventory level is {recommended_inventory} units with a {safety_margin} safety margin.

MAIN RISK:
- Even with stable overall demand, peak periods may create stockout risk if inventory is not prepared.

RECOMMENDED ACTIONS:
- {actions[0] if len(actions) > 0 else "Maintain stock planning based on forecasted demand."}
- {actions[1] if len(actions) > 1 else "Prepare additional inventory before detected peak dates."}
- {actions[2] if len(actions) > 2 else promo_text}
"""

        return """SUMMARY:
- No valid forecasting or anomaly results were provided.

KEY FINDINGS:
- No model summary was available.
- No business metrics were available.
- No recommendation context was available.

MAIN RISK:
- Business decisions cannot be supported without valid model outputs.

RECOMMENDED ACTIONS:
- Verify preprocessing output.
- Check model inference results.
- Re-run the selected pipeline mode.
"""

    def run_report_crew(self, final_report):
        mode = final_report.get("analysis_mode", "unknown")
        recommendations = final_report.get("business_recommendations", [])
        validation = final_report.get("human_validation", {})

        actions = self.extract_actions(recommendations)
        status = validation.get("final_status", "Unknown")

        if mode == "forecasting":
            forecasting = final_report.get("forecasting_results", {})
            trend = forecasting.get("trend_analysis", {})
            inventory = forecasting.get("inventory_requirements", {})

            trend_name = trend.get("trend", "Unknown trend")
            peaks = forecasting.get("detected_future_peaks", 0)
            total_demand = self.format_number(inventory.get("total_predicted_demand", "N/A"))
            recommended_inventory = self.format_number(inventory.get("total_recommended_inventory", "N/A"))

            return f"""EXECUTIVE SUMMARY:
- The forecasting analysis predicts {trend_name.lower()} and identifies {peaks} future peak periods.

MODEL RESULT:
- Total predicted demand is approximately {total_demand} units.
- Recommended inventory level is {recommended_inventory} units.

BUSINESS RECOMMENDATIONS:
- {actions[0] if len(actions) > 0 else "Maintain stock planning strategy."}
- {actions[1] if len(actions) > 1 else "Prepare additional inventory before detected peak periods."}
- {actions[2] if len(actions) > 2 else "Monitor promotion impact on sales demand."}

HUMAN VALIDATION:
- Human validation status: {status}.

FINAL DECISION:
- Apply the validated inventory and promotion recommendations before operational decisions.
"""

        if mode == "anomaly":
            anomaly = final_report.get("anomaly_detection_results", {})
            total_records = anomaly.get("total_records_analyzed", "N/A")
            total_anomalies = anomaly.get("total_anomalies_detected", "N/A")
            anomaly_rate = self.format_percent(anomaly.get("anomaly_rate", 0))
            type_distribution = anomaly.get("anomaly_type_distribution", {})

            main_type = max(type_distribution, key=type_distribution.get) if type_distribution else "unknown anomaly type"

            return f"""EXECUTIVE SUMMARY:
- The anomaly analysis detected abnormal retail behavior that requires business review.

MODEL RESULT:
- {total_anomalies} anomalies were detected out of {total_records} records.
- The anomaly rate is {anomaly_rate}.
- The main anomaly type is {main_type}.

BUSINESS RECOMMENDATIONS:
- {actions[0] if len(actions) > 0 else "Investigate abnormal sales and promotion behavior."}
- {actions[1] if len(actions) > 1 else "Review main anomaly categories."}
- {actions[2] if len(actions) > 2 else "Validate suspicious periods before decisions."}

HUMAN VALIDATION:
- Human validation status: {status}.

FINAL DECISION:
- Review detected anomalies before making inventory or promotion decisions.
"""

        return """EXECUTIVE SUMMARY:
- No valid analysis mode was found.

MODEL RESULT:
- No model result available.

BUSINESS RECOMMENDATIONS:
- Verify pipeline configuration.
- Re-run the selected analysis mode.
- Check generated logs.

HUMAN VALIDATION:
- Human validation status unavailable.

FINAL DECISION:
- Do not make business decisions until valid outputs are generated.
"""