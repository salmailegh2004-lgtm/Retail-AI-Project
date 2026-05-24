import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class LSTMForecastingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class ForecastingTrendAgent:
    def __init__(
        self,
        input_path,
        window_size=30,
        future_steps=30,
        stock_safety_margin=0.15,
        batch_size=50,
        scaler_type="robust",
        threshold_method="mean_std",
        output_dir="outputs",
        show_plots=False
    ):
        self.input_path = input_path
        self.window_size = window_size
        self.future_steps = future_steps
        self.stock_safety_margin = stock_safety_margin
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.threshold_method = threshold_method
        self.output_dir = output_dir
        self.show_plots = show_plots

        os.makedirs(self.output_dir, exist_ok=True)

        self.date_col = "date"
        self.target_col = "sales"

        self.feature_scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler()
        self.target_scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler()

        self.model = None
        self.time_series = None
        self.feature_cols = []
        self.logs = []
        self.training_history = []
        self.peak_threshold = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = f"{self.output_dir}/best_forecasting_lstm.pt"
        self.feature_scaler_path = f"{self.output_dir}/forecasting_feature_scaler.pkl"
        self.target_scaler_path = f"{self.output_dir}/forecasting_target_scaler.pkl"
        self.metadata_path = f"{self.output_dir}/forecasting_model_metadata.json"

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "ForecastingTrendAgent",
            "action": action,
            "status": status,
            "details": details
        })

    def save_logs(self):
        with open(f"{self.output_dir}/forecasting_agent_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, default=str)

    def load_data(self):
        df = pd.read_csv(self.input_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.target_col])
        df = df.drop_duplicates()
        df = df.sort_values(self.date_col)

        if len(df) <= self.window_size + 10:
            raise ValueError("Not enough records for LSTM forecasting.")

        self.log_action("load_data", "success", f"Loaded shape: {df.shape}")
        return df

    def aggregate_by_date(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = "sum" if col == self.target_col else "mean"

        self.time_series = (
            df.groupby(self.date_col)
            .agg(agg_dict)
            .reset_index()
            .sort_values(self.date_col)
        )

        self.log_action("aggregate_by_date", "success", f"Shape: {self.time_series.shape}")
        return self.time_series

    def add_features(self, time_series):
        ts = time_series.copy()

        ts["day_of_week"] = ts[self.date_col].dt.dayofweek
        ts["month"] = ts[self.date_col].dt.month
        ts["day"] = ts[self.date_col].dt.day
        ts["is_weekend"] = ts["day_of_week"].isin([5, 6]).astype(int)

        ts["sin_week"] = np.sin(2 * np.pi * ts["day_of_week"] / 7)
        ts["cos_week"] = np.cos(2 * np.pi * ts["day_of_week"] / 7)
        ts["sin_month"] = np.sin(2 * np.pi * ts["month"] / 12)
        ts["cos_month"] = np.cos(2 * np.pi * ts["month"] / 12)

        for lag in [1, 7, 14, 30]:
            ts[f"sales_lag_{lag}"] = ts[self.target_col].shift(lag)

        for window in [7, 14, 30]:
            ts[f"sales_ma_{window}"] = (
                ts[self.target_col]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

            ts[f"sales_std_{window}"] = (
                ts[self.target_col]
                .shift(1)
                .rolling(window, min_periods=1)
                .std()
            )

        ts["sales_ema_7"] = ts[self.target_col].ewm(span=7, adjust=False).mean()
        ts["sales_ema_30"] = ts[self.target_col].ewm(span=30, adjust=False).mean()

        ts["sales_diff_1"] = ts[self.target_col].diff(1)
        ts["sales_pct_change"] = ts[self.target_col].pct_change()

        ts = ts.replace([np.inf, -np.inf], np.nan)
        ts = ts.fillna(0)

        self.time_series = ts
        self.log_action("feature_engineering", "success", "Forecasting features created")
        return ts

    def prepare_features_train(self, time_series):
        self.feature_cols = [
            col for col in time_series.columns
            if col not in [self.date_col, self.target_col]
            and pd.api.types.is_numeric_dtype(time_series[col])
        ]

        X_raw = time_series[self.feature_cols].values
        y_raw = time_series[[self.target_col]].values

        train_cutoff = max(int(len(time_series) * 0.70), 1)

        self.feature_scaler.fit(X_raw[:train_cutoff])
        self.target_scaler.fit(y_raw[:train_cutoff])

        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw)

        full_scaled = np.concatenate([y_scaled, X_scaled], axis=1)

        self.log_action("prepare_features_train", "success", {
            "features": self.feature_cols,
            "input_size": full_scaled.shape[1]
        })

        return full_scaled, y_scaled

    def prepare_features_inference(self, time_series):
        for col in self.feature_cols:
            if col not in time_series.columns:
                time_series[col] = 0

        X_raw = time_series[self.feature_cols].values
        y_raw = time_series[[self.target_col]].values

        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw)

        full_scaled = np.concatenate([y_scaled, X_scaled], axis=1)

        return full_scaled, y_scaled

    def create_sequences(self, full_scaled, y_scaled, dates):
        X, y, sequence_dates = [], [], []

        for i in range(len(full_scaled) - self.window_size):
            X.append(full_scaled[i:i + self.window_size])
            y.append(y_scaled[i + self.window_size])
            sequence_dates.append(dates.iloc[i + self.window_size])

        return np.array(X), np.array(y), sequence_dates

    def split_data(self, X, y, sequence_dates):
        n = len(X)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        return (
            X[:train_end],
            y[:train_end],
            X[train_end:val_end],
            y[train_end:val_end],
            X[val_end:],
            y[val_end:],
            sequence_dates[val_end:]
        )

    def train_model(self, X_train, y_train, X_val, y_val, epochs=35, lr=0.001, patience=10):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model = LSTMForecastingModel(
            input_size=X_train.shape[2],
            hidden_size=64,
            num_layers=1,
            output_size=1,
            dropout=0.1
        ).to(self.device)

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        best_val_loss = np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            self.model.eval()
            with torch.no_grad():
                if len(X_val) > 0:
                    val_predictions = self.model(X_val.to(self.device))
                    val_loss = criterion(val_predictions, y_val.to(self.device)).item()
                else:
                    val_loss = avg_train_loss

            scheduler.step(val_loss)

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": float(val_loss)
            })

            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        torch.save(self.model.state_dict(), self.model_path)

        self.log_action("train_model", "success", f"Model saved to {self.model_path}")

    def inverse_target(self, values):
        return self.target_scaler.inverse_transform(values).flatten()

    def predict_real_values(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()

        pred_real = self.inverse_target(pred_scaled)
        return np.clip(pred_real, 0, None)

    def compute_peak_threshold(self):
        historical_sales = self.time_series[self.target_col].values

        mean_std_threshold = np.mean(historical_sales) + np.std(historical_sales)
        percentile_threshold = np.percentile(historical_sales, 90)

        if self.threshold_method == "hybrid":
            threshold = max(mean_std_threshold, percentile_threshold)
            method = "hybrid max(percentile_90, mean + std)"
        else:
            threshold = mean_std_threshold
            method = "mean + standard deviation"

        self.peak_threshold = threshold
        return threshold, method

    def evaluate_model(self, X_test, y_test, test_dates):
        y_pred = self.predict_real_values(X_test)
        y_true = self.inverse_target(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

        smape = np.mean(
            2 * np.abs(y_pred - y_true)
            / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        ) * 100

        threshold, threshold_method = self.compute_peak_threshold()

        metrics = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "SMAPE": float(smape),
            "Peak_Threshold": float(threshold),
            "Threshold_Method": threshold_method
        }

        test_df = pd.DataFrame({
            "date": test_dates,
            "actual_sales": y_true,
            "predicted_sales": y_pred,
            "actual_peak": y_true >= threshold,
            "predicted_peak": y_pred >= threshold
        })

        self.log_action("evaluate_model", "success", metrics)
        return test_df, metrics

    def forecast_future(self, full_scaled):
        current_sequence = full_scaled[-self.window_size:].copy()
        future_scaled = []

        self.model.eval()

        for _ in range(self.future_steps):
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                next_scaled_sales = self.model(input_tensor).cpu().numpy()[0]

            future_scaled.append(next_scaled_sales)

            last_features = current_sequence[-1, 1:]
            next_row = np.concatenate([next_scaled_sales, last_features])
            current_sequence = np.vstack([current_sequence[1:], next_row])

        future_sales = self.inverse_target(np.array(future_scaled))
        return np.clip(future_sales, 0, None)

    def detect_future_peaks(self, future_sales):
        future_dates = pd.date_range(
            start=self.time_series[self.date_col].max() + pd.Timedelta(days=1),
            periods=self.future_steps,
            freq="D"
        )

        future_df = pd.DataFrame({
            "date": future_dates,
            "predicted_sales": future_sales
        })

        future_peak_threshold = min(
            np.percentile(future_df["predicted_sales"], 90),
            future_df["predicted_sales"].mean() + future_df["predicted_sales"].std()
        )

        future_df["peak_threshold"] = future_peak_threshold
        future_df["is_peak"] = future_df["predicted_sales"] >= future_peak_threshold

        future_df["recommended_inventory"] = (
            future_df["predicted_sales"] * (1 + self.stock_safety_margin)
        ).round()

        peaks = future_df[future_df["is_peak"]].copy()
        return future_df, peaks

    def analyze_trend(self, future_df):
        first_value = future_df["predicted_sales"].iloc[0]
        last_value = future_df["predicted_sales"].iloc[-1]

        change = last_value - first_value
        change_pct = (change / first_value) * 100 if first_value != 0 else 0

        if change_pct > 5:
            trend = "Increasing demand"
        elif change_pct < -5:
            trend = "Decreasing demand"
        else:
            trend = "Stable demand"

        future_df["demand_change_pct_from_start"] = (
            (future_df["predicted_sales"] - first_value) / first_value
        ) * 100 if first_value != 0 else 0

        trend_summary = {
            "first_predicted_sales": float(first_value),
            "last_predicted_sales": float(last_value),
            "change": float(change),
            "change_percentage": float(change_pct),
            "trend": trend
        }

        return future_df, trend_summary

    def analyze_promotion_impact(self):
        if "onpromotion" not in self.time_series.columns:
            return {
                "available": False,
                "message": "Promotion feature not found."
            }

        corr = self.time_series["onpromotion"].corr(self.time_series[self.target_col])

        return {
            "available": True,
            "correlation_onpromotion_sales": float(corr) if not pd.isna(corr) else None
        }

    def save_model_artifacts(self):
        with open(self.feature_scaler_path, "wb") as f:
            pickle.dump(self.feature_scaler, f)

        with open(self.target_scaler_path, "wb") as f:
            pickle.dump(self.target_scaler, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "window_size": self.window_size,
            "future_steps": self.future_steps,
            "stock_safety_margin": self.stock_safety_margin,
            "scaler_type": self.scaler_type,
            "threshold_method": self.threshold_method,
            "input_size": len(self.feature_cols) + 1,
            "model_path": self.model_path
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, default=str)

    def load_model_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Forecasting model not found: {self.model_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Forecasting metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_cols = metadata["feature_cols"]

        with open(self.feature_scaler_path, "rb") as f:
            self.feature_scaler = pickle.load(f)

        with open(self.target_scaler_path, "rb") as f:
            self.target_scaler = pickle.load(f)

        self.model = LSTMForecastingModel(
            input_size=metadata["input_size"],
            hidden_size=64,
            num_layers=1,
            output_size=1,
            dropout=0.1
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.log_action("load_model_artifacts", "success", "Forecasting model loaded")

    def save_outputs(self, future_df, peaks, test_df, summary):
        future_df.to_csv(f"{self.output_dir}/forecasting_results.csv", index=False)
        peaks.to_csv(f"{self.output_dir}/detected_peak_periods.csv", index=False)

        if test_df is not None:
            test_df.to_csv(f"{self.output_dir}/test_actual_vs_predicted.csv", index=False)

        with open(f"{self.output_dir}/forecasting_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        self.save_logs()

    def save_plot(self, filename):
        path = f"{self.output_dir}/{filename}"
        plt.tight_layout()
        plt.savefig(path)
        if self.show_plots:
            plt.show()
        plt.close()

    def plot_forecast(self, future_df, peaks):
        plt.figure(figsize=(12, 5))
        plt.plot(
            future_df["date"],
            future_df["predicted_sales"],
            marker="o",
            label="Future predicted sales"
        )

        plt.axhline(
            y=future_df["peak_threshold"].iloc[0],
            linestyle="--",
            label="Peak threshold"
        )

        if len(peaks) > 0:
            plt.scatter(
                peaks["date"],
                peaks["predicted_sales"],
                s=80,
                label="Detected peaks"
            )

        plt.title("Future Demand Forecasting & Peak Detection")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.legend()
        self.save_plot("forecasting_future_peaks.png")

    def plot_training_history(self):
        if not self.training_history:
            return

        history_df = pd.DataFrame(self.training_history)

        history_df.to_csv(
            f"{self.output_dir}/forecasting_training_history.csv",
            index=False
        )

        plt.figure(figsize=(10, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
        plt.title("Forecasting LSTM - Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        self.save_plot("forecasting_training_loss.png")
    
    def load_training_metrics(self):
        path = f"{self.output_dir}/forecasting_training_metrics.json"

        if not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_training_metrics(self, metrics):
        with open(
            f"{self.output_dir}/forecasting_training_metrics.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=4, default=str)
            
    def run_train(self, epochs=35, lr=0.001):
        df = self.load_data()
        time_series = self.aggregate_by_date(df)
        time_series = self.add_features(time_series)

        full_scaled, y_scaled = self.prepare_features_train(time_series)

        X, y, sequence_dates = self.create_sequences(
            full_scaled,
            y_scaled,
            time_series[self.date_col]
        )

        X_train, y_train, X_val, y_val, X_test, y_test, test_dates = self.split_data(
            X, y, sequence_dates
        )

        self.train_model(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)

        test_df, metrics = self.evaluate_model(X_test, y_test, test_dates)
        
        self.save_training_metrics(metrics)

        future_sales = self.forecast_future(full_scaled)
        future_df, peaks = self.detect_future_peaks(future_sales)
        future_df, trend_summary = self.analyze_trend(future_df)

        promotion_impact = self.analyze_promotion_impact()

        inventory_summary = {
            "safety_margin": self.stock_safety_margin,
            "total_predicted_demand": float(future_df["predicted_sales"].sum()),
            "total_recommended_inventory": float(future_df["recommended_inventory"].sum())
        }
        
        summary = {
            "mode": "train",
            "metrics": metrics,
            "trend_analysis": trend_summary,
            "promotion_impact": promotion_impact,
            "inventory_requirements": inventory_summary,
            "number_of_detected_future_peaks": int(len(peaks)),
            "model_type": "LSTM Forecasting Model",
            "model_saved_path": self.model_path
        }

        self.save_model_artifacts()
        self.save_outputs(future_df, peaks, test_df, summary)

        self.plot_forecast(future_df, peaks)
        self.plot_training_history()

        self.log_action("run_train", "success", "Training completed")
        self.save_logs()

        return future_df, peaks, summary

    def run_inference(self):
        self.load_model_artifacts()

        df = self.load_data()
        time_series = self.aggregate_by_date(df)
        time_series = self.add_features(time_series)

        full_scaled, _ = self.prepare_features_inference(time_series)

        future_sales = self.forecast_future(full_scaled)
        future_df, peaks = self.detect_future_peaks(future_sales)
        future_df, trend_summary = self.analyze_trend(future_df)

        promotion_impact = self.analyze_promotion_impact()

        inventory_summary = {
            "safety_margin": self.stock_safety_margin,
            "total_predicted_demand": float(future_df["predicted_sales"].sum()),
            "total_recommended_inventory": float(future_df["recommended_inventory"].sum())
        }
        training_metrics = self.load_training_metrics()
        summary = {
            "mode": "inference",
            "metrics": training_metrics,
            "trend_analysis": trend_summary,
            "promotion_impact": promotion_impact,
            "inventory_requirements": inventory_summary,
            "number_of_detected_future_peaks": int(len(peaks)),
            "model_type": "Loaded LSTM Forecasting Model",
            "model_loaded_path": self.model_path
        }

        self.save_outputs(future_df, peaks, None, summary)
        self.plot_forecast(future_df, peaks)

        self.log_action("run_inference", "success", "Inference completed")
        self.save_logs()

        return future_df, peaks, summary

    def run(self, epochs=35, lr=0.001):
        return self.run_train(epochs=epochs, lr=lr)