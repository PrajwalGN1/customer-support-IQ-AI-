"""
Hybrid Anomaly Detection System for SupportIQ.
Combines deterministic SLA rules, statistical IQR outlier detection, and optional Isolation Forest ML.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from app.config import settings
from app.data_loader import data_loader
from app.schemas import (
    AnomalyItem,
    AnomalyResponse,
    AnomalySeverity,
    AnomalySummary,
)
from app.utils import logger


class AnomalyDetector:
    """
    Hybrid Anomaly Detection Engine.
    Executes domain SLA rules, IQR statistical bounds, and Isolation Forest ML.
    """

    def __init__(self, loader=None):
        self.loader = loader or data_loader

    def detect_all(
        self,
        include_ml: bool = True,
        severity_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        method_filter: Optional[str] = None,
    ) -> AnomalyResponse:
        """
        Runs all configured anomaly detectors and returns aggregated results.
        """
        df = self.loader.get_data()
        ref_dt = self.loader.get_reference_timestamp()

        anomalies: List[AnomalyItem] = []

        # 1. Rule 1: Unresolved High/Critical tickets older than SLA (24h)
        anomalies.extend(self._detect_unresolved_sla_breaches(df, ref_dt))

        # 2. Rule 2: Critical tickets with excessive resolution time (> 12h)
        anomalies.extend(self._detect_critical_long_resolution(df))

        # 3. Statistical: IQR on resolution time
        anomalies.extend(self._detect_iqr_resolution_outliers(df))

        # 4. Optional ML: Isolation Forest
        if include_ml:
            anomalies.extend(self._detect_isolation_forest_anomalies(df))

        # Deduplicate identical ticket + anomaly_type entries
        unique_map = {}
        for a in anomalies:
            key = f"{a.ticket_id}_{a.anomaly_type}"
            if key not in unique_map:
                unique_map[key] = a
        deduped = list(unique_map.values())

        # Apply optional filters
        filtered = deduped
        if severity_filter:
            sf = severity_filter.upper().strip()
            filtered = [a for a in filtered if a.severity.value == sf]
        if type_filter:
            tf = type_filter.upper().strip()
            filtered = [a for a in filtered if a.anomaly_type.upper() == tf]
        if method_filter:
            mf = method_filter.upper().strip()
            filtered = [a for a in filtered if mf in a.detection_method.upper()]

        summary = self._build_summary(filtered)
        return AnomalyResponse(
            total=len(filtered),
            summary=summary,
            anomalies=filtered,
        )

    def _detect_unresolved_sla_breaches(self, df: pd.DataFrame, ref_dt: pd.Timestamp) -> List[AnomalyItem]:
        """Flag unresolved High or Critical tickets older than the SLA threshold (default 24h)."""
        items = []
        mask = (
            df["priority"].isin(["High", "Critical"])
            & df["status"].isin(["Open", "Escalated"])
            & df["created_at_dt"].notna()
        )
        candidates = df[mask].copy()
        if candidates.empty:
            return items

        candidates["age_hrs"] = (ref_dt - candidates["created_at_dt"]).dt.total_seconds() / 3600.0
        sla_hours = settings.SLA_HIGH_HOURS
        breached = candidates[candidates["age_hrs"] > sla_hours]

        for _, row in breached.iterrows():
            age_hrs = round(float(row["age_hrs"]), 1)
            sev = AnomalySeverity.CRITICAL if row["priority"] == "Critical" else AnomalySeverity.HIGH
            items.append(
                AnomalyItem(
                    ticket_id=str(row["ticket_id"]),
                    anomaly_type="UNRESOLVED_SLA_BREACH",
                    severity=sev,
                    reason=(
                        f"Unresolved {row['priority']} ticket open for {age_hrs} hours "
                        f"(exceeds {sla_hours}h SLA relative to {ref_dt.strftime('%Y-%m-%d %H:%M')})."
                    ),
                    metrics={
                        "age_hours": age_hrs,
                        "priority": row["priority"],
                        "status": row["status"],
                        "agent_id": row["agent_id"],
                        "sla_threshold_hours": sla_hours,
                    },
                    detection_method="RULE_SLA_BREACH",
                )
            )
        return items

    def _detect_critical_long_resolution(self, df: pd.DataFrame) -> List[AnomalyItem]:
        """Flag Critical tickets that took > SLA_CRITICAL_HOURS (12h) to resolve."""
        items = []
        sla_crit = settings.SLA_CRITICAL_HOURS
        mask = (
            (df["priority"] == "Critical")
            & (df["status"] == "Resolved")
            & (df["resolution_time_hrs"] > sla_crit)
        )
        breached = df[mask]

        for _, row in breached.iterrows():
            res_hrs = float(row["resolution_time_hrs"])
            items.append(
                AnomalyItem(
                    ticket_id=str(row["ticket_id"]),
                    anomaly_type="CRITICAL_EXCESSIVE_RESOLUTION",
                    severity=AnomalySeverity.CRITICAL,
                    reason=f"Critical ticket took {res_hrs:.1f} hours to resolve (exceeds {sla_crit}h Critical SLA threshold).",
                    metrics={
                        "resolution_time_hrs": res_hrs,
                        "priority": "Critical",
                        "agent_id": row["agent_id"],
                        "customer_rating": row.get("customer_rating"),
                        "sla_threshold_hours": sla_crit,
                    },
                    detection_method="RULE_CRITICAL_SLA",
                )
            )
        return items

    def _detect_iqr_resolution_outliers(self, df: pd.DataFrame) -> List[AnomalyItem]:
        """Flag tickets with resolution time exceeding Q3 + 1.5 * IQR."""
        items = []
        series = df["resolution_time_hrs"].dropna()
        if len(series) < 10:
            return items

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        upper_bound = float(q3 + settings.IQR_MULTIPLIER * iqr)

        outliers = df[df["resolution_time_hrs"] > upper_bound]

        for _, row in outliers.iterrows():
            val = float(row["resolution_time_hrs"])
            severity = AnomalySeverity.CRITICAL if val > (q3 + 3.0 * iqr) else AnomalySeverity.HIGH
            items.append(
                AnomalyItem(
                    ticket_id=str(row["ticket_id"]),
                    anomaly_type="LONG_RESOLUTION_TIME",
                    severity=severity,
                    reason=f"Resolution time of {val:.1f}h is a statistical outlier exceeding the upper bound of {upper_bound:.1f}h (Q3={q3:.1f}h, IQR={iqr:.1f}h).",
                    metrics={
                        "resolution_time_hrs": val,
                        "iqr_upper_bound": round(upper_bound, 2),
                        "q1": round(q1, 2),
                        "q3": round(q3, 2),
                        "agent_id": row["agent_id"],
                        "category": row["category"],
                    },
                    detection_method="IQR_STATISTICAL",
                )
            )
        return items

    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame) -> List[AnomalyItem]:
        """
        Multivariate machine learning anomaly detection.
        Learns non-linear interactions across response time, resolution time, priority, and rating.
        """
        items = []
        resolved = df[df["status"] == "Resolved"].copy()
        if len(resolved) < 30:
            return items

        # Feature preparation
        priority_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        resolved["priority_num"] = resolved["priority"].map(priority_map).fillna(1)
        resolved["resp_filled"] = resolved["response_time_hrs"].fillna(resolved["response_time_hrs"].median())
        resolved["resol_filled"] = resolved["resolution_time_hrs"].fillna(resolved["resolution_time_hrs"].median())
        resolved["rating_filled"] = resolved["customer_rating"].fillna(resolved["customer_rating"].median())

        features = resolved[["resp_filled", "resol_filled", "priority_num", "rating_filled"]]

        try:
            iso = IsolationForest(
                contamination=settings.ISOLATION_FOREST_CONTAMINATION,
                random_state=42,
                n_estimators=100,
            )
            preds = iso.fit_predict(features)
            scores = iso.decision_function(features)

            outlier_indices = np.where(preds == -1)[0]

            for idx in outlier_indices:
                row = resolved.iloc[idx]
                score = float(round(scores[idx], 3))
                items.append(
                    AnomalyItem(
                        ticket_id=str(row["ticket_id"]),
                        anomaly_type="MULTIVARIATE_OPERATIONAL_ANOMALY",
                        severity=AnomalySeverity.MEDIUM,
                        reason=(
                            f"Multivariate pattern anomaly detected by Isolation Forest (anomaly score: {score}). "
                            f"Response: {row['response_time_hrs']}h, Resolution: {row['resolution_time_hrs']}h, "
                            f"Priority: {row['priority']}, Rating: {row['customer_rating']}."
                        ),
                        metrics={
                            "anomaly_score": score,
                            "response_time_hrs": row["response_time_hrs"],
                            "resolution_time_hrs": row["resolution_time_hrs"],
                            "customer_rating": row["customer_rating"],
                            "priority": row["priority"],
                        },
                        detection_method="ISOLATION_FOREST_ML",
                    )
                )
        except Exception as e:
            logger.warning(f"Isolation Forest execution skipped due to: {e}")

        return items

    def _build_summary(self, anomalies: List[AnomalyItem]) -> AnomalySummary:
        """Calculate counts by severity, type, and detection method."""
        by_sev: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_method: Dict[str, int] = {}

        for a in anomalies:
            by_sev[a.severity.value] = by_sev.get(a.severity.value, 0) + 1
            by_type[a.anomaly_type] = by_type.get(a.anomaly_type, 0) + 1
            by_method[a.detection_method] = by_method.get(a.detection_method, 0) + 1

        return AnomalySummary(
            total_anomalies=len(anomalies),
            by_severity=by_sev,
            by_type=by_type,
            by_method=by_method,
        )


anomaly_detector = AnomalyDetector()
