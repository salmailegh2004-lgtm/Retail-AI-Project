from agents.orchestrator_agent import OrchestratorAgent

print("Choose analysis mode:")
print("1 - Forecasting and Peak Detection")
print("2 - Anomaly Detection")

choice = input("Enter your choice: ")

if choice == "1":
    mode = "forecasting"
elif choice == "2":
    mode = "anomaly"
else:
    print("Invalid choice. Defaulting to forecasting.")
    mode = "forecasting"

print("\nEnter dataset path.")
print("Example: data/store_sales")
print("Example: data/new_dataset")
print("You can also paste the full folder path.")

dataset_path = input("Dataset path: ").strip()

if dataset_path == "":
    dataset_path = "data/store_sales"

orchestrator = OrchestratorAgent(
    input_path=dataset_path,
    output_dir="outputs",
    mode=mode
)

pipeline_output = orchestrator.run()

if pipeline_output is not None:
    print("\nPipeline completed successfully.")
    print(f"Dataset used: {dataset_path}")
    print(f"Analysis mode: {pipeline_output['mode']}")
    print("\nGenerated files:")
    print("- outputs/final_business_report.txt")
    print("- outputs/final_business_report.json")
    print("- outputs/pipeline_output.json")
    print("- outputs/orchestrator_logs.json")
else:
    print("\nPipeline failed.")
    print("Check: outputs/orchestrator_logs.json")