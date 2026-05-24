import os
import json
from datetime import datetime


class HumanInLoopAgent:
    """
    Human-in-the-Loop validation agent.

    Behaviour:
    - If a manual validation file already exists at `output_path`
      (written by the Streamlit UI), it is used directly — preserving
      all human decisions (Approved / Rejected / Needs Review) and comments.
    - Otherwise, it falls back to the automatic rule-based validation
      (High → careful review note, Medium → moderate, Low → auto-accept).

    This allows the Streamlit page to act as the real human review UI
    while keeping the pipeline fully automated when no UI input is present.
    """

    MANUAL_VALIDATION_PATH = "outputs/human_validation_report.json"

    def __init__(self, recommendation_output, output_path=None):
        self.recommendation_output = recommendation_output
        self.recommendations = recommendation_output.get("recommendations", [])
        self.output_path = output_path or self.MANUAL_VALIDATION_PATH

        self.validation_results = []
        self.logs = []

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_action(self, action, status, details=None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent":     "HumanInLoopAgent",
            "action":    action,
            "status":    status,
            "details":   details,
        })

    # ── Manual path ──────────────────────────────────────────────────────────

    def _load_manual_validation(self):
        """Return saved validation results if the file exists, else None."""
        if not os.path.exists(self.output_path):
            return None
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("validation_results", [])
            if results:
                self.log_action(
                    "load_manual_validation",
                    "success",
                    f"Loaded {len(results)} manually-reviewed decisions from {self.output_path}",
                )
                return results
        except Exception as e:
            self.log_action("load_manual_validation", "error", str(e))
        return None

    # ── Automatic fallback ────────────────────────────────────────────────────

    def _auto_validate(self, recommendation: dict) -> dict:
        priority = recommendation.get("priority", "Low")

        if priority == "High":
            comment = "High-priority recommendation — requires careful review before implementation."
        elif priority == "Medium":
            comment = "Recommendation accepted with moderate business impact."
        else:
            comment = "Low-risk recommendation automatically accepted."

        return {
            "category":        recommendation.get("category", "—"),
            "priority":        priority,
            "recommendation":  recommendation.get("recommendation", "—"),
            "approval_status": "Approved",
            "reviewer_comment": comment,
        }

    def _auto_validate_all(self):
        for rec in self.recommendations:
            self.validation_results.append(self._auto_validate(rec))

        self.log_action(
            "auto_validate_all",
            "success",
            f"{len(self.validation_results)} recommendations auto-validated (no manual input found).",
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    def _generate_summary(self) -> dict:
        approved      = sum(1 for r in self.validation_results if r["approval_status"] == "Approved")
        rejected      = sum(1 for r in self.validation_results if r["approval_status"] == "Rejected")
        needs_review  = sum(1 for r in self.validation_results if r["approval_status"] == "Needs Review")

        if rejected == 0 and needs_review == 0:
            final_status = "Validated"
        elif approved > 0:
            final_status = "Partially Validated"
        else:
            final_status = "Rejected"

        return {
            "total_recommendations":     len(self.validation_results),
            "approved_recommendations":  approved,
            "rejected_recommendations":  rejected,
            "needs_review":              needs_review,
            "human_validation_required": True,
            "final_status":              final_status,
        }

    # ── Persist ───────────────────────────────────────────────────────────────

    def _save_outputs(self) -> dict:
        output = {
            "validation_results":  self.validation_results,
            "validation_summary":  self._generate_summary(),
            "logs":                self.logs,
        }

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

        return output

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run human-in-the-loop validation.

        Priority order:
        1. Load manually-reviewed decisions from the Streamlit UI (if present).
        2. Fall back to automatic rule-based validation.
        """
        manual = self._load_manual_validation()

        if manual:
            self.validation_results = manual
            self.log_action(
                "run",
                "success",
                "Used manual human validation from UI.",
            )
        else:
            self._auto_validate_all()
            self.log_action(
                "run",
                "success",
                "Automatic validation completed (no UI input detected).",
            )

        return self._save_outputs()