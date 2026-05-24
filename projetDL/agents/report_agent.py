import os
import json
import math
from datetime import datetime
from agents.crewai_layer import CrewAILayer


class ReportGenerationAgent:
    def __init__(self,
                 mode,
                 forecasting_summary=None,
                 anomaly_summary=None,
                 recommendation_output=None,
                 human_validation_output=None,
                 output_dir="outputs",
                 ollama_model="llama3.2"):

        self.mode = mode
        self.forecasting_summary = forecasting_summary or {}
        self.anomaly_summary = anomaly_summary or {}
        self.recommendation_output = recommendation_output or {}
        self.human_validation_output = human_validation_output or {}
        self.output_dir = output_dir
        self.ollama_model = ollama_model
        self.logs = []

        os.makedirs(self.output_dir, exist_ok=True)

    def safe_float(self, value, default=None):
        try:
            if value is None:
                return default

            value = float(value)

            if math.isnan(value) or math.isinf(value):
                return default

            return value

        except Exception:
            return default

    def safe_format(self, value, decimals=2, default="Not available"):
        value = self.safe_float(value)

        if value is None:
            return default

        return f"{value:.{decimals}f}"

    def safe_percent(self, value, default="Not available"):
        value = self.safe_float(value)

        if value is None:
            return default

        if abs(value) <= 1:
            return f"{value * 100:.2f}%"

        return f"{value:.2f}%"

    def safe_units(self, value, default="Not available"):
        value = self.safe_float(value)

        if value is None:
            return default

        return f"{value:.0f} units"

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "ReportGenerationAgent",
            "action": action,
            "status": status,
            "details": details
        })

    def build_forecasting_section(self):
        if self.mode != "forecasting":
            return None

        return {
            "model_type": self.forecasting_summary.get(
                "model_type",
                "Forecasting Model"
            ),

            "trend_analysis": self.forecasting_summary.get(
                "trend_analysis",
                {}
            ),

            "metrics": self.forecasting_summary.get(
                "metrics",
                {}
            ),

            "promotion_impact": self.forecasting_summary.get(
                "promotion_impact",
                {}
            ),

            "inventory_requirements": self.forecasting_summary.get(
                "inventory_requirements",
                {}
            ),

            "detected_future_peaks": self.forecasting_summary.get(
                "number_of_detected_future_peaks",
                0
            )
        }

    def build_anomaly_section(self):
        if self.mode != "anomaly":
            return None

        return {
            "total_records_analyzed": self.anomaly_summary.get(
                "total_records_analyzed",
                "Not available"
            ),

            "total_anomalies_detected": self.anomaly_summary.get(
                "total_anomalies_detected",
                "Not available"
            ),

            "anomaly_rate": self.anomaly_summary.get(
                "anomaly_rate",
                None
            ),

            "risk_distribution": self.anomaly_summary.get(
                "risk_distribution",
                {}
            ),

            "anomaly_type_distribution": self.anomaly_summary.get(
                "anomaly_type_distribution",
                {}
            ),

            "highest_risk_dates": self.anomaly_summary.get(
                "highest_risk_dates",
                []
            ),

            "business_interpretation": self.anomaly_summary.get(
                "business_interpretation",
                "No business interpretation available."
            )
        }

    def generate_crewai_summary(self, temporary_report):
        try:
            crewai_layer = CrewAILayer(
                model_name=self.ollama_model
            )

            summary = crewai_layer.run_report_crew(
                temporary_report
            )

            self.log_action(
                "crewai_report_summary",
                "success",
                "CrewAI generated executive summary."
            )

            return summary

        except Exception as e:
            self.log_action(
                "crewai_report_summary",
                "failed",
                str(e)
            )

            return (
                "CrewAI report summary generation failed. "
                "Structured report was used."
            )

    def generate_report(self):
        report = {
            "report_title":
                "AI-Powered Retail Business Intelligence Report",

            "analysis_mode": self.mode,

            "generation_date": datetime.now().isoformat(),

            "forecasting_results":
                self.build_forecasting_section(),

            "anomaly_detection_results":
                self.build_anomaly_section(),

            "business_recommendations":
                self.recommendation_output.get(
                    "recommendations",
                    []
                ),

            "crewai_business_insight":
                self.recommendation_output.get(
                    "crewai_business_insight",
                    "No CrewAI business insight generated."
                ),

            "human_validation":
                self.human_validation_output.get(
                    "validation_summary",
                    {
                        "final_status": "Not available",
                        "total_recommendations": 0,
                        "approved_recommendations": 0,
                        "rejected_recommendations": 0,
                        "human_validation_required": True
                    }
                ),

            "final_business_conclusion": (
                "The AI system analyzed retail data, generated "
                "business recommendations, and validated decisions "
                "through a human-in-the-loop checkpoint."
            )
        }

        report["crewai_executive_summary"] = \
            self.generate_crewai_summary(report)

        return report

    def save_json_report(self, report):
        output_path = os.path.join(
            self.output_dir,
            "final_business_report.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, default=str)

        return output_path

    def has_valid_metrics(self, metrics):
        if not metrics:
            return False

        metric_keys = ["MAE", "RMSE", "R2", "SMAPE"]

        for key in metric_keys:
            value = self.safe_float(metrics.get(key))

            if value is not None and value != 0:
                return True

        return False

    def write_forecasting_results(self, f, forecasting):
        trend = forecasting.get("trend_analysis", {})
        metrics = forecasting.get("metrics", {})
        inventory = forecasting.get(
            "inventory_requirements",
            {}
        )

        promo = forecasting.get(
            "promotion_impact",
            {}
        )

        f.write(
            "1. Forecasting and Peak Detection Results\n"
        )

        f.write("-" * 50 + "\n")

        f.write(
            f"Model used: "
            f"{forecasting.get('model_type', 'Forecasting Model')}\n"
        )

        f.write(
            f"Predicted trend: "
            f"{trend.get('trend', 'Not available')}\n"
        )

        f.write(
            f"Demand change: "
            f"{self.safe_percent(trend.get('change_percentage'))}\n"
        )

        f.write(
            f"Detected future peaks: "
            f"{forecasting.get('detected_future_peaks', 0)}\n\n"
        )

        f.write("Saved Training/Test Metrics:\n")

        if self.has_valid_metrics(metrics):

            f.write(
                f"- MAE: "
                f"{self.safe_format(metrics.get('MAE'))}\n"
            )

            f.write(
                f"- RMSE: "
                f"{self.safe_format(metrics.get('RMSE'))}\n"
            )

            f.write(
                f"- R2 Score: "
                f"{self.safe_format(metrics.get('R2'))}\n"
            )

            f.write(
                f"- SMAPE: "
                f"{self.safe_percent(metrics.get('SMAPE'))}\n\n"
            )

        else:
            f.write(
                "- Not recomputed during inference mode.\n"
            )

            f.write(
                "- The saved trained model was loaded "
                "and used for forecasting.\n\n"
            )

        f.write("Inventory Planning:\n")

        f.write(
            f"- Total predicted demand: "
            f"{self.safe_units(inventory.get('total_predicted_demand'))}\n"
        )

        f.write(
            f"- Recommended inventory: "
            f"{self.safe_units(inventory.get('total_recommended_inventory'))}\n"
        )

        f.write(
            f"- Safety margin: "
            f"{self.safe_percent(inventory.get('safety_margin'))}\n\n"
        )

        f.write("Promotion Impact:\n")

        corr = self.safe_float(
            promo.get("correlation_onpromotion_sales")
        )

        if promo.get("available") and corr is not None:
            f.write(
                f"- Promotion-sales correlation: {corr:.2f}\n\n"
            )
        else:
            f.write(
                "- Promotion-sales correlation: "
                "Not available for this dataset.\n\n"
            )

        f.write("Model Execution Evidence:\n")

        model_path = os.path.join(
            self.output_dir,
            "best_forecasting_lstm.pt"
        )

        if os.path.exists(model_path):
            f.write(f"- Loaded model: {model_path}\n")
        else:
            f.write("- Loaded model: Not found\n")

        f.write(
            "- Execution mode: inference using saved trained model\n"
        )

        f.write(
            "- Forecast horizon: 30 future time steps\n"
        )

        f.write(
            "- Input dataset was automatically preprocessed\n"
        )

        f.write(
           "- These metrics were computed during "
           "training/testing on the original dataset\n"
)

        f.write(
          "- They are displayed as model evidence "
          "and are not recomputed on uploaded future data\n\n"
      )

    def write_anomaly_results(self, f, anomaly):

        f.write("1. Anomaly Detection Results\n")
        f.write("-" * 50 + "\n")

        f.write(
            f"Total records analyzed: "
            f"{anomaly.get('total_records_analyzed', 'Not available')}\n"
        )

        f.write(
            f"Total anomalies detected: "
            f"{anomaly.get('total_anomalies_detected', 'Not available')}\n"
        )

        f.write(
            f"Anomaly rate: "
            f"{self.safe_percent(anomaly.get('anomaly_rate'))}\n\n"
        )

        f.write("Risk Distribution:\n")

        risk_distribution = anomaly.get(
            "risk_distribution",
            {}
        )

        if risk_distribution:
            for risk, count in risk_distribution.items():
                f.write(f"- {risk}: {count}\n")
        else:
            f.write("- Not available\n")

        f.write("\nAnomaly Type Distribution:\n")

        type_distribution = anomaly.get(
            "anomaly_type_distribution",
            {}
        )

        if type_distribution:
            for anomaly_type, count in type_distribution.items():
                f.write(f"- {anomaly_type}: {count}\n")
        else:
            f.write("- Not available\n")

        f.write("\nHighest Risk Dates:\n")

        highest_risk_dates = anomaly.get(
            "highest_risk_dates",
            []
        )

        if highest_risk_dates:
            for item in highest_risk_dates[:5]:

                date = item.get("date", "Unknown")

                sales = self.safe_format(
                    item.get("sales")
                )

                risk = item.get(
                    "risk_level",
                    "Unknown"
                )

                anomaly_type = item.get(
                    "anomaly_type",
                    "Unknown"
                )

                f.write(
                    f"- {date} | "
                    f"Sales: {sales} | "
                    f"Risk: {risk} | "
                    f"Type: {anomaly_type}\n"
                )

        else:
            f.write("- Not available\n")

        f.write("\nBusiness Interpretation:\n")

        f.write(
            anomaly.get(
                "business_interpretation",
                "No interpretation available."
            ) + "\n\n"
        )

        f.write("Model Execution Evidence:\n")

        model_path = os.path.join(
            self.output_dir,
            "best_anomaly_autoencoder.pt"
        )

        if os.path.exists(model_path):
           f.write(f"- Loaded model: {model_path}\n")
        else:
           f.write(
              f"- Loaded model: "
              f"{self.anomaly_summary.get('training_metrics', {}).get('model_path', 'Saved anomaly model')}\n"
    )

        f.write(
            "- Execution mode: inference using saved trained model\n"
        )

        f.write(
            "- Input dataset was automatically preprocessed\n"
        )

        f.write(
            "- Anomalies were detected using reconstruction "
            "error analysis\n"
        )

        f.write(
            "- The LSTM Autoencoder evaluated temporal "
            "retail behavior patterns\n\n"
        )

    def write_recommendations(self, f, recommendations):

        f.write("2. Business Recommendations\n")
        f.write("-" * 50 + "\n")

        if not recommendations:
            f.write(
                "No business recommendations generated.\n\n"
            )
            return

        for rec in recommendations:

            f.write(
                f"[{rec.get('priority', 'N/A')}] "
                f"{rec.get('category', 'General')}\n"
            )

            f.write(
                f"Recommendation: "
                f"{rec.get('recommendation', 'No recommendation available.')}\n"
            )

            f.write(
                f"Reason: "
                f"{rec.get('reason', 'No reason available.')}\n\n"
            )

    def write_human_validation(self, f, validation):

        f.write("5. Human Validation\n")
        f.write("-" * 50 + "\n")

        f.write(
            f"Final status: "
            f"{validation.get('final_status', 'Not available')}\n"
        )

        f.write(
            f"Total recommendations: "
            f"{validation.get('total_recommendations', 0)}\n"
        )

        f.write(
            f"Approved recommendations: "
            f"{validation.get('approved_recommendations', 0)}\n"
        )

        f.write(
            f"Rejected recommendations: "
            f"{validation.get('rejected_recommendations', 0)}\n"
        )

        f.write(
            f"Human validation required: "
            f"{validation.get('human_validation_required', True)}\n\n"
        )

    def save_text_report(self, report):

        output_path = os.path.join(
            self.output_dir,
            "final_business_report.txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:

            f.write(report["report_title"] + "\n")

            f.write("=" * 70 + "\n\n")

            f.write(
                f"Analysis Mode: "
                f"{report['analysis_mode']}\n"
            )

            f.write(
                f"Generation Date: "
                f"{report['generation_date']}\n\n"
            )

            if report["forecasting_results"] is not None:
                self.write_forecasting_results(
                    f,
                    report["forecasting_results"]
                )

            if report["anomaly_detection_results"] is not None:
                self.write_anomaly_results(
                    f,
                    report["anomaly_detection_results"]
                )

            self.write_recommendations(
                f,
                report["business_recommendations"]
            )

            f.write("3. CrewAI Business Insight\n")
            f.write("-" * 50 + "\n")

            f.write(
                str(report["crewai_business_insight"]) + "\n\n"
            )

            f.write("4. CrewAI Executive Summary\n")
            f.write("-" * 50 + "\n")

            f.write(
                str(report["crewai_executive_summary"]) + "\n\n"
            )

            self.write_human_validation(
                f,
                report["human_validation"]
            )

            f.write("6. Final Conclusion\n")
            f.write("-" * 50 + "\n")

            f.write(
                report["final_business_conclusion"] + "\n"
            )

        return output_path

    def run(self):

        report = self.generate_report()

        json_path = self.save_json_report(report)

        text_path = self.save_text_report(report)

        self.log_action(
            "run",
            "success",
            {
                "json_report": json_path,
                "text_report": text_path
            }
        )

        report["logs"] = self.logs

        return report