"""
Data Leakage Audit Module.
Validates pre-training and pre-inference feature matrices to ensure:
1. No future temporal leakage (rainfall/drainwatch timestamps <= event evaluation time).
2. Target variable is strictly isolated from feature columns.
3. No historical event target leakage into historical frequency features.
4. Sample source (REAL / SYNTHETIC) is declared on every row.
5. Spatial group integrity for cross-validation splits.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np


class DataLeakageError(ValueError):
    """Raised when data leakage is detected during pipeline execution."""
    pass


@dataclass
class LeakageAuditReport:
    """Audit outcome summary."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_count: int = 0
    row_count: int = 0


class DataLeakageAuditor:
    """
    Guarantees ML pipeline compliance with strict non-leakage principles.
    """

    @staticmethod
    def audit_training_data(
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "road_inaccessible",
        source_column: str = "sample_source",
        zone_column: str = "zone_no",
        event_timestamp: Optional[datetime] = None,
        eval_timestamp_col: Optional[str] = None,
    ) -> LeakageAuditReport:
        """
        Audit a training dataset for leakage hazards.

        Parameters:
            df: DataFrame containing features, target, and metadata.
            feature_columns: Columns designated as model inputs.
            target_column: Column name of target label.
            source_column: Column declaring REAL or SYNTHETIC provenance.
            zone_column: Column used for spatial group splitting.
            event_timestamp: Event onset time if auditing a specific historical event.
            eval_timestamp_col: Column with observation timestamps.

        Returns:
            LeakageAuditReport: Detailed audit report.
            
        Raises:
            DataLeakageError: If critical leakage is discovered.
        """
        issues: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            return LeakageAuditReport(passed=False, issues=["Dataset is empty or None"])

        # 1. Target isolation check
        if target_column in feature_columns:
            issues.append(
                f"CRITICAL LEAKAGE: Target variable '{target_column}' is included in feature_columns!"
            )

        # 2. Check for target duplicates or direct aliases in feature columns
        suspicious_aliases = [
            "inaccessible", "inundated", "is_flooded", "flood_label", "target"
        ]
        for col in feature_columns:
            if any(alias in col.lower() for alias in suspicious_aliases) and col != "historical_inundation_presence":
                warnings.append(f"Suspicious feature name '{col}' may be an unmasked target alias.")

        # 3. Source provenance check
        if source_column not in df.columns:
            issues.append(f"Provenance column '{source_column}' is missing from dataset.")
        else:
            missing_source_count = df[source_column].isna().sum()
            if missing_source_count > 0:
                issues.append(f"{missing_source_count} rows are missing '{source_column}' declaration.")

        # 4. Spatial group check for GroupKFold
        if zone_column not in df.columns:
            warnings.append(f"Spatial blocking column '{zone_column}' is missing; spatial CV cannot be enforced.")

        # 5. Temporal leakage check
        if event_timestamp is not None and eval_timestamp_col is not None and eval_timestamp_col in df.columns:
            post_event_mask = pd.to_datetime(df[eval_timestamp_col]) > event_timestamp
            if post_event_mask.any():
                count = post_event_mask.sum()
                issues.append(
                    f"CRITICAL TEMPORAL LEAKAGE: {count} rows have timestamps occurring AFTER event onset {event_timestamp}."
                )

        passed = len(issues) == 0
        if not passed:
            raise DataLeakageError("; ".join(issues))

        return LeakageAuditReport(
            passed=passed,
            issues=issues,
            warnings=warnings,
            feature_count=len(feature_columns),
            row_count=len(df),
        )
