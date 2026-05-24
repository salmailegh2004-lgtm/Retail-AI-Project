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
from sklearn.preprocessing import MinMaxScaler


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=96, latent_size=32, num_layers=1, dropout_rate=0.2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.latent = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        last_hidden = encoded[:, -1, :]

        latent = self.latent(last_hidden)
        latent = self.dropout(latent)

        decoder_start = self.decoder_input(latent)
        repeated = decoder_start.unsqueeze(1).repeat(1, x.shape[1], 1)

        decoded, _ = self.decoder(repeated)
        reconstruction = self.output_layer(decoded)

        return reconstruction


class DLAnomalyDetectionAgent:
    def __init__(
        self,
        input_path,
        window_size=30,
        batch_size=50,
        anomaly_percentile=95,
        z_threshold=2.8,
        promotion_threshold=0.2,
        dropout_rate=0.2,
        output_dir="outputs",
        show_plots=False
    ):
        self.input_path = input_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.anomaly_percentile = anomaly_percentile
        self.z_threshold = z_threshold
        self.promotion_threshold = promotion_threshold
        self.dropout_rate = dropout_rate
        self.output_dir = output_dir
        self.show_plots = show_plots

        os.makedirs(self.output_dir, exist_ok=True)

        self.date_col = "date"
        self.target_col = "sales"

        self.scaler = MinMaxScaler()
        self.logs = []
        self.training_history = []

        self.model = None
        self.df = None
        self.daily_df = None
        self.results = None
        self.anomalies = None
        self.feature_cols = []

        self.model_path = f"{self.output_dir}/best_anomaly_lstm_autoencoder.pt"
        self.scaler_path = f"{self.output_dir}/anomaly_scaler.pkl"
        self.metadata_path = f"{self.output_dir}/anomaly_model_metadata.json"

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "DLAnomalyDetectionAgent",
            "action": action,
            "status": status,
            "details": details
        })

    def save_logs(self):
        with open(f"{self.output_dir}/anomaly_agent_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=4, default=str)

    def load_data(self):
        df = pd.read_csv(self.input_path)

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")

        df = df.dropna(subset=[self.date_col, self.target_col])
        df = df.sort_values(self.date_col)

        if len(df) <= self.window_size + 10:
            raise ValueError("Not enough records for anomaly detection.")

        self.df = df
        self.log_action("load_data", "success", f"Loaded shape: {df.shape}")
        return df

    def aggregate_by_date(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = "sum" if col == self.target_col else "mean"

        self.daily_df = (
            self.df.groupby(self.date_col)
            .agg(agg_dict)
            .reset_index()
            .sort_values(self.date_col)
        )

        self.log_action("aggregate_by_date", "success", f"Shape: {self.daily_df.shape}")
        return self.daily_df

    def create_features(self):
        df = self.daily_df.copy()

        df["sales_lag_1"] = df[self.target_col].shift(1)
        df["sales_lag_7"] = df[self.target_col].shift(7)

        df["sales_diff"] = df[self.target_col] - df["sales_lag_1"]
        df["sales_pct_change"] = df[self.target_col].pct_change()

        df["rolling_mean_7"] = df[self.target_col].rolling(7, min_periods=1).mean()
        df["rolling_std_7"] = df[self.target_col].rolling(7, min_periods=1).std()

        df["z_score"] = (
            (df[self.target_col] - df["rolling_mean_7"])
            / df["rolling_std_7"].replace(0, np.nan)
        )

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        self.daily_df = df
        self.log_action("create_features", "success", "Anomaly features created")
        return df

    def select_model_features_train(self):
        preferred_cols = [
            "sales",
            "onpromotion",
            "transactions",
            "dcoilwtico",
            "day_of_week",
            "is_weekend",
            "sales_lag_1",
            "sales_lag_7",
            "sales_diff",
            "sales_pct_change",
            "rolling_mean_7",
            "rolling_std_7",
            "z_score"
        ]

        self.feature_cols = [
            col for col in preferred_cols
            if col in self.daily_df.columns
            and pd.api.types.is_numeric_dtype(self.daily_df[col])
        ]

        if not self.feature_cols:
            raise ValueError("No numeric features available for anomaly detection.")

        self.log_action("select_model_features_train", "success", self.feature_cols)
        return self.feature_cols

    def select_model_features_inference(self):
        for col in self.feature_cols:
            if col not in self.daily_df.columns:
                self.daily_df[col] = 0

        return self.feature_cols

    def prepare_sequences_train(self):
        self.select_model_features_train()

        values = self.daily_df[self.feature_cols].values
        scaled_values = self.scaler.fit_transform(values)

        return self.create_sequences_from_scaled(scaled_values)

    def prepare_sequences_inference(self):
        self.select_model_features_inference()

        values = self.daily_df[self.feature_cols].values
        scaled_values = self.scaler.transform(values)

        return self.create_sequences_from_scaled(scaled_values)

    def create_sequences_from_scaled(self, scaled_values):
        X = []
        sequence_dates = []
        normal_sequence_mask = []

        for i in range(len(scaled_values) - self.window_size):
            sequence = scaled_values[i:i + self.window_size]
            target_row = self.daily_df.iloc[i + self.window_size]

            X.append(sequence)
            sequence_dates.append(target_row[self.date_col])

            is_normal_like = (
                abs(target_row["z_score"]) < self.z_threshold
                and abs(target_row["sales_pct_change"]) < 0.5
            )

            normal_sequence_mask.append(is_normal_like)

        X = np.array(X)

        if len(X) == 0:
            raise ValueError("No sequences created. Reduce window_size or provide more data.")

        self.sequence_dates = sequence_dates
        self.normal_sequence_mask = np.array(normal_sequence_mask)

        self.log_action("prepare_sequences", "success", f"Created sequences: {X.shape}")

        return X

    def train_autoencoder(self, X, epochs=50, lr=0.001, patience=8, min_delta=0.0001):
        X_tensor = torch.tensor(X, dtype=torch.float32)

        normal_X = X_tensor[self.normal_sequence_mask]

        if len(normal_X) < 100:
            normal_X = X_tensor

        train_size = max(int(len(normal_X) * 0.8), 1)

        X_train = normal_X[:train_size]
        X_val = normal_X[train_size:]

        if len(X_val) == 0:
            X_val = X_train

        train_loader = DataLoader(
            TensorDataset(X_train, X_train),
            batch_size=self.batch_size,
            shuffle=False
        )

        val_loader = DataLoader(
            TensorDataset(X_val, X_val),
            batch_size=self.batch_size,
            shuffle=False
        )

        input_size = X.shape[2]

        self.model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=96,
            latent_size=32,
            num_layers=1,
            dropout_rate=self.dropout_rate
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_losses = []

            for batch_X, _ in train_loader:
                reconstruction = self.model(batch_X)
                loss = criterion(reconstruction, batch_X)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, _ in val_loader:
                    reconstruction = self.model(batch_X)
                    val_loss = criterion(reconstruction, batch_X)
                    val_losses.append(val_loss.item())

            avg_val_loss = float(np.mean(val_losses))

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.final_loss = float(avg_train_loss)
        self.best_validation_loss = float(best_val_loss)

        torch.save(self.model.state_dict(), self.model_path)
        self.log_action("train_autoencoder", "success", f"Model saved to {self.model_path}")

    def compute_reconstruction_errors(self, X, use_training_threshold=True):
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = self.model(X_tensor).numpy()

        errors = np.mean((X - reconstructed) ** 2, axis=(1, 2))

        if use_training_threshold:
            normal_errors = errors[self.normal_sequence_mask]

            if len(normal_errors) < 100:
                normal_errors = errors

            threshold = np.percentile(normal_errors, self.anomaly_percentile)
            self.anomaly_threshold = float(threshold)

        else:
            saved_threshold = getattr(self, "anomaly_threshold", None)
            adaptive_threshold = np.percentile(errors, 95)

            if saved_threshold is None:
                threshold = adaptive_threshold
            else:
                threshold = max(saved_threshold, adaptive_threshold)

        results = pd.DataFrame({
            "date": self.sequence_dates,
            "reconstruction_error": errors,
            "anomaly_threshold": threshold,
            "autoencoder_anomaly": errors > threshold
        })

        return results

    def enrich_anomaly_results(self, results):
        df = self.daily_df.copy()
        enriched = results.merge(df, on="date", how="left")

        enriched["sales_spike"] = enriched["z_score"] >= self.z_threshold
        enriched["sales_drop"] = enriched["z_score"] <= -self.z_threshold

        if "onpromotion" in enriched.columns:
            enriched["promotion_anomaly"] = (
                (enriched["onpromotion"] > 0)
                & (enriched["sales_pct_change"] < -self.promotion_threshold)
            )
        else:
            enriched["promotion_anomaly"] = False

        enriched["is_anomaly"] = (
            enriched["autoencoder_anomaly"]
            | enriched["sales_spike"]
            | enriched["sales_drop"]
            | enriched["promotion_anomaly"]
        )

        def anomaly_type(row):
            if not row["is_anomaly"]:
                return "Normal"
            if row["sales_spike"]:
                return "Sudden Sales Spike"
            if row["sales_drop"]:
                return "Sudden Sales Drop"
            if row["promotion_anomaly"]:
                return "Suspicious Promotional Behavior"
            return "Inconsistent Demand Pattern"

        enriched["anomaly_type"] = enriched.apply(anomaly_type, axis=1)

        def risk_level(row):
            if not row["is_anomaly"]:
                return "Normal"

            score = 0
            if row["autoencoder_anomaly"]:
                score += 1
            if row["sales_spike"] or row["sales_drop"]:
                score += 1
            if row["promotion_anomaly"]:
                score += 1

            if score >= 3:
                return "High Risk"
            elif score == 2:
                return "Medium Risk"
            else:
                return "Low Risk"

        enriched["risk_level"] = enriched.apply(risk_level, axis=1)

        self.results = enriched
        self.anomalies = enriched[enriched["is_anomaly"]].copy()

        self.log_action("enrich_anomaly_results", "success", {
            "total_results": len(self.results),
            "total_anomalies": len(self.anomalies)
        })

        return self.results, self.anomalies

    def generate_summary(self, mode):
        anomaly_rate = float(len(self.anomalies) / len(self.results)) if len(self.results) > 0 else 0.0

        highest_risk_cols = ["date", "sales", "reconstruction_error", "risk_level", "anomaly_type"]
        available_cols = [col for col in highest_risk_cols if col in self.anomalies.columns]

        return {
            "mode": mode,
            "training_metrics": {
                "final_reconstruction_loss": getattr(self, "final_loss", None),
                "best_validation_loss": getattr(self, "best_validation_loss", None),
                "mean_reconstruction_error": float(self.results["reconstruction_error"].mean()),
                "max_reconstruction_error": float(self.results["reconstruction_error"].max()),
                "anomaly_threshold": float(self.results["anomaly_threshold"].iloc[0]),
                "anomaly_percentile_used": self.anomaly_percentile,
                "window_size": self.window_size,
                "batch_size": self.batch_size,
                "features_used": self.feature_cols,
                "model_path": self.model_path
            },
            "total_records_analyzed": int(len(self.results)),
            "total_anomalies_detected": int(len(self.anomalies)),
            "anomaly_rate": anomaly_rate,
            "risk_distribution": self.anomalies["risk_level"].value_counts().to_dict(),
            "anomaly_type_distribution": self.anomalies["anomaly_type"].value_counts().to_dict(),
            "highest_risk_dates": self.anomalies[
                self.anomalies["risk_level"].isin(["High Risk", "Medium Risk"])
            ][available_cols].head(10).to_dict(orient="records"),
            "business_interpretation": (
                "The LSTM Autoencoder learns normal temporal retail behavior. "
                "Anomalies are detected using reconstruction error and business rules."
            )
        }

    def save_model_artifacts(self):
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        metadata = {
            "feature_cols": self.feature_cols,
            "window_size": self.window_size,
            "input_size": len(self.feature_cols),
            "anomaly_threshold": self.anomaly_threshold,
            "anomaly_percentile": self.anomaly_percentile,
            "z_threshold": self.z_threshold,
            "promotion_threshold": self.promotion_threshold,
            "dropout_rate": self.dropout_rate,
            "model_path": self.model_path,
            "final_reconstruction_loss": getattr(self, "final_loss", None),
            "best_validation_loss": getattr(self, "best_validation_loss", None)
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, default=str)

    def load_model_artifacts(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Anomaly model not found: {self.model_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Anomaly metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_cols = metadata["feature_cols"]
        self.anomaly_threshold = metadata["anomaly_threshold"]
        self.final_loss = metadata.get("final_reconstruction_loss")
        self.best_validation_loss = metadata.get("best_validation_loss")

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model = LSTMAutoencoder(
            input_size=metadata["input_size"],
            hidden_size=96,
            latent_size=32,
            num_layers=1,
            dropout_rate=metadata["dropout_rate"]
        )

        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()

        self.log_action("load_model_artifacts", "success", "Anomaly model loaded")

    def save_outputs(self, summary):
        self.results.to_csv(
            f"{self.output_dir}/anomaly_detection_results.csv",
            index=False
        )

        self.anomalies.to_csv(
            f"{self.output_dir}/detected_anomalies.csv",
            index=False
        )

        with open(
            f"{self.output_dir}/anomaly_summary.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=4, default=str)

        with open(
            f"{self.output_dir}/anomaly_training_history.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(self.training_history, f, indent=4, default=str)

        self.save_logs()

    def save_plot(self, filename):
        path = f"{self.output_dir}/{filename}"

        plt.tight_layout()
        plt.savefig(path)

        if self.show_plots:
            plt.show()

        plt.close()

    def plot_anomalies(self):
        plt.figure(figsize=(12, 5))

        plt.plot(
            self.daily_df["date"],
            self.daily_df[self.target_col],
            label="Sales"
        )

        if len(self.anomalies) > 0:
            plt.scatter(
                self.anomalies["date"],
                self.anomalies[self.target_col],
                label="Detected anomalies"
            )

        plt.title("Sales Anomaly Detection using LSTM Autoencoder")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.legend()

        self.save_plot("anomaly_detected_points.png")

    def plot_training_history(self):
        if not self.training_history:
            return

        history_df = pd.DataFrame(self.training_history)

        history_df.to_csv(
            f"{self.output_dir}/anomaly_training_history.csv",
            index=False
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            history_df["epoch"],
            history_df["train_loss"],
            label="Train Loss"
        )

        plt.plot(
            history_df["epoch"],
            history_df["val_loss"],
            label="Validation Loss"
        )

        plt.title("Anomaly LSTM Autoencoder - Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        self.save_plot("anomaly_training_loss.png")

    def run_train(self, epochs=50, lr=0.001):
        self.load_data()

        self.aggregate_by_date()

        self.create_features()

        X = self.prepare_sequences_train()

        self.train_autoencoder(
            X,
            epochs=epochs,
            lr=lr
        )

        raw_results = self.compute_reconstruction_errors(
            X,
            use_training_threshold=True
        )

        results, anomalies = self.enrich_anomaly_results(raw_results)

        summary = self.generate_summary(mode="train")

        self.save_model_artifacts()

        self.save_outputs(summary)

        self.plot_anomalies()

        self.plot_training_history()

        self.log_action(
            "run_train",
            "success",
            "Training completed"
        )

        self.save_logs()

        return results, anomalies, summary

    def run_inference(self):
        self.load_model_artifacts()

        self.load_data()

        self.aggregate_by_date()

        self.create_features()

        X = self.prepare_sequences_inference()

        raw_results = self.compute_reconstruction_errors(
            X,
            use_training_threshold=False
        )

        results, anomalies = self.enrich_anomaly_results(raw_results)

        summary = self.generate_summary(mode="inference")

        self.save_outputs(summary)

        self.plot_anomalies()

        self.log_action(
            "run_inference",
            "success",
            "Inference completed"
        )

        self.save_logs()

        return results, anomalies, summary

    def run(self, epochs=50, lr=0.001):
        return self.run_train(
            epochs=epochs,
            lr=lr
        )