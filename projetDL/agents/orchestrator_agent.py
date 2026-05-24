import os
import json
from datetime import datetime

from agents.preprocessing_agent import RetailPreprocessingAgent
from agents.forecasting_agent import ForecastingTrendAgent
from agents.anomaly_agent import DLAnomalyDetectionAgent
from agents.recommendation_agent import RecommendationAgent
from agents.human_loop import HumanInLoopAgent
from agents.report_agent import ReportGenerationAgent


class OrchestratorAgent:
    def __init__(self, input_path="data/store_sales", output_dir="outputs", mode="forecasting"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.mode = mode
        self.logs = []

        os.makedirs(self.output_dir, exist_ok=True)

    def log_action(self, agent, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "status": status,
            "details": details
        })

    def save_logs(self):
        with open(f"{self.output_dir}/orchestrator_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, default=str)

    def check_required_model_files(self):
        if self.mode == "forecasting":
            required_files = [
                f"{self.output_dir}/best_forecasting_lstm.pt",
                f"{self.output_dir}/forecasting_model_metadata.json",
                f"{self.output_dir}/forecasting_feature_scaler.pkl",
                f"{self.output_dir}/forecasting_target_scaler.pkl"
            ]

        elif self.mode == "anomaly":
            required_files = [
                f"{self.output_dir}/best_anomaly_lstm_autoencoder.pt",
                f"{self.output_dir}/anomaly_model_metadata.json",
                f"{self.output_dir}/anomaly_scaler.pkl"
            ]

        else:
            raise ValueError("Invalid mode. Choose 'forecasting' or 'anomaly'.")

        missing_files = [file for file in required_files if not os.path.exists(file)]

        if missing_files:
            raise FileNotFoundError(
                "Saved model files are missing. Train the model once first. Missing files: "
                + str(missing_files)
            )

    def run(self):
        try:
            # 1. Preprocessing
            self.log_action("OrchestratorAgent", "start_preprocessing", "running")

            preprocessing_agent = RetailPreprocessingAgent(
                input_path=self.input_path
            )

            clean_df, model_df = preprocessing_agent.run(
                clean_output_path=f"{self.output_dir}/clean_merged_data.csv",
                model_output_path=f"{self.output_dir}/model_ready_data.csv"
            )

            if clean_df is None or model_df is None:
                raise Exception("Preprocessing failed.")

            self.log_action("PreprocessingAgent", "run", "success")

            # Check if saved model exists
            self.check_required_model_files()

            future_df = None
            peaks = None
            forecasting_summary = None
            detected_anomalies = None
            anomaly_summary = None

            # 2. Forecasting inference
            if self.mode == "forecasting":
                self.log_action("OrchestratorAgent", "start_forecasting_inference", "running")

                forecast_agent = ForecastingTrendAgent(
                    input_path=f"{self.output_dir}/model_ready_data.csv",
                    window_size=30,
                    future_steps=30,
                    stock_safety_margin=0.15,
                    batch_size=16,
                    scaler_type="robust",
                    threshold_method="hybrid",
                    output_dir=self.output_dir,
                    show_plots=False
                )

                future_df, peaks, forecasting_summary = forecast_agent.run_inference()

                self.log_action("ForecastingTrendAgent", "inference", "success")

            # 3. Anomaly inference
            elif self.mode == "anomaly":
                self.log_action("OrchestratorAgent", "start_anomaly_inference", "running")

                anomaly_agent = DLAnomalyDetectionAgent(
                    input_path=f"{self.output_dir}/model_ready_data.csv",
                    window_size=30,
                    batch_size=50,
                    anomaly_percentile=98,
                    dropout_rate=0.2,
                    output_dir=self.output_dir,
                    show_plots=False
                )

                anomaly_results, detected_anomalies, anomaly_summary = anomaly_agent.run_inference()

                self.log_action("DLAnomalyDetectionAgent", "inference", "success")

            # 4. Recommendation Agent
            self.log_action("OrchestratorAgent", "start_recommendation", "running")

            recommendation_agent = RecommendationAgent(
                mode=self.mode,
                future_df=future_df,
                peaks=peaks,
                forecasting_summary=forecasting_summary,
                detected_anomalies=detected_anomalies,
                anomaly_summary=anomaly_summary,
                use_ollama=True,
                ollama_model="llama3.2"
            )

            recommendation_output = recommendation_agent.run()

            self.log_action("RecommendationAgent", "run", "success")

            # 5. Human-in-the-loop
            self.log_action("OrchestratorAgent", "start_human_validation", "running")

            human_agent = HumanInLoopAgent(
                recommendation_output=recommendation_output
            )

            human_validation_output = human_agent.run()

            self.log_action("HumanInLoopAgent", "run", "success")

            # 6. Report Generation
            self.log_action("OrchestratorAgent", "start_report_generation", "running")

            report_agent = ReportGenerationAgent(
                mode=self.mode,
                forecasting_summary=forecasting_summary,
                anomaly_summary=anomaly_summary,
                recommendation_output=recommendation_output,
                human_validation_output=human_validation_output,
                output_dir=self.output_dir,
                ollama_model="llama3.2"
            )

            final_report = report_agent.run()

            self.log_action("ReportGenerationAgent", "run", "success")

            pipeline_output = {
                "mode": self.mode,
                "execution_mode": "inference",
                "forecasting_summary": forecasting_summary,
                "anomaly_summary": anomaly_summary,
                "recommendation_output": recommendation_output,
                "human_validation_output": human_validation_output,
                "final_report": final_report,
                "orchestrator_logs": self.logs
            }

            with open(f"{self.output_dir}/pipeline_output.json", "w", encoding="utf-8") as f:
                json.dump(pipeline_output, f, indent=4, default=str)

            self.log_action("OrchestratorAgent", "pipeline_completed", "success")
            self.save_logs()

            print("Pipeline completed successfully.")
            return pipeline_output

        except Exception as e:
            self.log_action("OrchestratorAgent", "pipeline_failed", "failed", str(e))
            self.save_logs()
            print(f"Pipeline failed: {e}")
            return None