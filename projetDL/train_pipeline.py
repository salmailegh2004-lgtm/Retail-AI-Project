from agents.preprocessing_agent import RetailPreprocessingAgent
from agents.forecasting_agent import ForecastingTrendAgent
from agents.anomaly_agent import DLAnomalyDetectionAgent

RAW_DATA_PATH = "data/store_sales"
OUTPUT_DIR = "outputs"

def main():
    print("========== TRAINING PIPELINE STARTED ==========")

    preprocessing_agent = RetailPreprocessingAgent(
        input_path=RAW_DATA_PATH
    )

    clean_df, model_ready_df = preprocessing_agent.run(
        clean_output_path=f"{OUTPUT_DIR}/clean_merged_data.csv",
        model_output_path=f"{OUTPUT_DIR}/model_ready_data.csv"
    )

    if clean_df is None or model_ready_df is None:
        raise Exception("Preprocessing failed. Training stopped.")

    forecasting_agent = ForecastingTrendAgent(
        input_path=f"{OUTPUT_DIR}/model_ready_data.csv",
        window_size=30,
        future_steps=30,
        stock_safety_margin=0.15,
        batch_size=50,
        scaler_type="robust",
        threshold_method="hybrid",
        output_dir=OUTPUT_DIR,
        show_plots=False
    )

    forecasting_agent.run_train(
        epochs=35,
        lr=0.001
    )

    anomaly_agent = DLAnomalyDetectionAgent(
        input_path=f"{OUTPUT_DIR}/model_ready_data.csv",
        window_size=30,
        batch_size=50,
        anomaly_percentile=90,
        dropout_rate=0.2,
        output_dir=OUTPUT_DIR,
        show_plots=False
    )

    anomaly_agent.run_train(
        epochs=50,
        lr=0.001
    )

    print("========== TRAINING PIPELINE COMPLETED ==========")
    print("Models saved in outputs/")

if __name__ == "__main__":
    main()