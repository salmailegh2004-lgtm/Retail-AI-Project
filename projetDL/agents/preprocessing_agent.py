import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class RetailPreprocessingAgent:
    def __init__(self, input_path, freq="D"):
        self.input_path = input_path
        self.freq = freq
        self.logs = []
        self.dataframes = {}
        self.df = None
        self.main_file = None
        self.date_col = None
        self.target_col = None
        self.metadata = {}

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "RetailPreprocessingAgent",
            "action": action,
            "status": status,
            "details": details
        })

    def load_data(self):
        if os.path.isfile(self.input_path):
            name = os.path.basename(self.input_path)
            self.dataframes[name] = pd.read_csv(self.input_path)

        elif os.path.isdir(self.input_path):
            for file in os.listdir(self.input_path):
                if file.endswith(".csv"):
                    path = os.path.join(self.input_path, file)
                    self.dataframes[file] = pd.read_csv(path)
        else:
            raise ValueError("Input path must be a CSV file or a folder containing CSV files.")

        if not self.dataframes:
            raise ValueError("No CSV files found.")

        self.log_action("load_data", "success", list(self.dataframes.keys()))
        return self.dataframes

    def detect_date_column(self, df):
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "time", "timestamp", "day"]):
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().mean() > 0.7:
                    return col
        return None

    def detect_target_column(self, df):
        keywords = ["sales", "revenue", "amount", "quantity", "qty", "demand"]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scores = []

        for col in numeric_cols:
            score = 0
            col_lower = col.lower()

            if any(k in col_lower for k in keywords):
                score += 10

            if "id" in col_lower or col_lower in ["store_nbr", "product_id", "item_id"]:
                score -= 10

            if df[col].nunique() > 10:
                score += 2

            scores.append((col, score))

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def choose_main_dataframe(self):
        best_score = -1

        for name, df in self.dataframes.items():
            date_col = self.detect_date_column(df)
            target_col = self.detect_target_column(df)

            score = 0
            if date_col:
                score += 5
            if target_col:
                score += 5
            score += len(df) / 100000

            if score > best_score:
                best_score = score
                self.main_file = name

        if self.main_file is None:
            raise ValueError("Could not identify main dataset.")

        self.df = self.dataframes[self.main_file].copy()
        self.date_col = self.detect_date_column(self.df)
        self.target_col = self.detect_target_column(self.df)

        if self.date_col is None:
            raise ValueError("No valid date column detected.")

        if self.target_col is None:
            raise ValueError("No valid target column detected.")

        self.log_action("choose_main_dataframe", "success", {
            "main_file": self.main_file,
            "date_col": self.date_col,
            "target_col": self.target_col
        })

    def merge_secondary_files(self):
        merged = self.df.copy()

        allowed_keys = [
            "date", "store_nbr", "store_id",
            "product_id", "item_id", "family",
            "city", "state", "type", "category"
        ]

        for name, other_df in self.dataframes.items():
            if name == self.main_file:
                continue

            common_cols = list(set(merged.columns).intersection(set(other_df.columns)))
            useful_keys = [col for col in common_cols if col.lower() in allowed_keys]

            if not useful_keys:
                self.log_action("merge_secondary_files", "skipped", f"No useful keys for {name}")
                continue

            try:
                other_df = other_df.copy()

                for col in useful_keys:
                    if "date" in col.lower():
                        merged[col] = pd.to_datetime(merged[col], errors="coerce")
                        other_df[col] = pd.to_datetime(other_df[col], errors="coerce")

                other_df = other_df.drop_duplicates(subset=useful_keys)

                merged = merged.merge(
                    other_df,
                    on=useful_keys,
                    how="left",
                    suffixes=("", f"_{name.replace('.csv', '')}")
                )

                self.log_action("merge_secondary_files", "success", {
                    "file": name,
                    "keys": useful_keys
                })

            except Exception as e:
                self.log_action("merge_secondary_files", "failed", f"{name}: {e}")

        self.df = merged
        return self.df

    def clean_basic_data(self):
        df = self.df.copy()

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        before = len(df)

        df = df.dropna(subset=[self.date_col, self.target_col])
        df = df[df[self.target_col] >= 0]
        df = df.drop_duplicates()
        df = df.sort_values(self.date_col).reset_index(drop=True)

        self.df = df

        self.log_action("clean_basic_data", "success", f"Removed {before - len(df)} rows")
        return self.df

    def handle_missing_values(self):
        df = self.df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

        for col in numeric_cols:
            median_value = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median_value) else median_value)

        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown").astype(str)

        self.df = df

        self.log_action("handle_missing_values", "success", {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols
        })

        return self.df

    def create_time_features(self):
        df = self.df.copy()

        df["year"] = df[self.date_col].dt.year
        df["month"] = df[self.date_col].dt.month
        df["day"] = df[self.date_col].dt.day
        df["day_of_week"] = df[self.date_col].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        self.df = df

        self.log_action("create_time_features", "success", "Time features created")
        return self.df

    def build_model_ready_dataset(self):
        df = self.df.copy()

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        agg_dict = {}
        for col in numeric_cols:
            if col == self.target_col:
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "mean"

        model_ready_df = (
            df.groupby(self.date_col)
            .agg(agg_dict)
            .reset_index()
            .sort_values(self.date_col)
        )

        model_ready_df = model_ready_df.rename(
            columns={
                self.date_col: "date",
                self.target_col: "sales"
            }
        )

        model_ready_df = model_ready_df.fillna(0)

        self.metadata = {
            "date_col": "date",
            "target_col": "sales",
            "original_date_col": self.date_col,
            "original_target_col": self.target_col,
            "frequency": self.freq,
            "main_file": self.main_file,
            "model_ready_type": "daily_aggregated_time_series"
        }

        self.log_action(
            "build_model_ready_dataset",
            "success",
            f"Model-ready shape: {model_ready_df.shape}"
        )

        return model_ready_df

    def save_outputs(
        self,
        clean_output_path="outputs/clean_merged_data.csv",
        model_output_path="outputs/model_ready_data.csv",
        metadata_path="outputs/preprocessing_metadata.json",
        log_path="outputs/preprocessing_logs.json"
    ):
        clean_df = self.df.copy()
        model_ready_df = self.build_model_ready_dataset()

        clean_df.to_csv(clean_output_path, index=False)
        model_ready_df.to_csv(model_output_path, index=False)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, default=str)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, default=str)

        self.log_action("save_outputs", "success", {
            "clean_output": clean_output_path,
            "model_output": model_output_path
        })

        return clean_df, model_ready_df

    def run(self, clean_output_path="outputs/clean_merged_data.csv",
            model_output_path="outputs/model_ready_data.csv"):
        try:
            self.load_data()
            self.choose_main_dataframe()
            self.merge_secondary_files()
            self.clean_basic_data()
            self.handle_missing_values()
            self.create_time_features()

            clean_df, model_ready_df = self.save_outputs(
                clean_output_path=clean_output_path,
                model_output_path=model_output_path
            )

            self.log_action("run_pipeline", "success", "Preprocessing completed successfully")
            return clean_df, model_ready_df

        except Exception as e:
            self.log_action("run_pipeline", "failed", str(e))
            print(f"[ERROR] Preprocessing failed: {e}")
            return None, None