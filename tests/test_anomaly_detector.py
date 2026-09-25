"""
Unit tests for anomaly_detector.py
Validates SLA breach rules, statistical IQR outlier identification, and ML anomaly detection.
"""
import pytest
from app.anomaly_detector import AnomalyDetector
from app.schemas import AnomalySeverity


@pytest.fixture
def detector():
    return AnomalyDetector()


def test_detect_all_returns_valid_structure(detector):
    response = detector.detect_all(include_ml=True)

    assert response.total > 0
    assert response.summary.total_anomalies == response.total
    assert "CRITICAL" in response.summary.by_severity
    assert "HIGH" in response.summary.by_severity
    assert len(response.anomalies) == response.total


def test_iqr_outliers_identified(detector):
    response = detector.detect_all(method_filter="IQR_STATISTICAL", include_ml=False)

    # 21 tickets have resolution_time_hrs > 48.15h (Q3 + 1.5*IQR)
    assert response.total == 21
    for a in response.anomalies:
        assert a.detection_method == "IQR_STATISTICAL"
        assert a.metrics["resolution_time_hrs"] > 48.15
        assert a.anomaly_type == "LONG_RESOLUTION_TIME"


def test_unresolved_sla_breach_rule(detector):
    response = detector.detect_all(type_filter="UNRESOLVED_SLA_BREACH", include_ml=False)

    # Unresolved High/Critical tickets older than 24h from dataset reference time
    assert response.total == 80
    for a in response.anomalies:
        assert a.metrics["priority"] in ["High", "Critical"]
        assert a.metrics["status"] in ["Open", "Escalated"]
        assert a.metrics["age_hours"] > 24.0


def test_critical_sla_excessive_resolution(detector):
    response = detector.detect_all(type_filter="CRITICAL_EXCESSIVE_RESOLUTION", include_ml=False)

    # Critical tickets resolved in > 12 hours
    assert response.total == 3
    ticket_ids = {a.ticket_id for a in response.anomalies}
    assert ticket_ids == {"TKT-238", "TKT-255", "TKT-446"}


def test_severity_filter(detector):
    crit_response = detector.detect_all(severity_filter="CRITICAL")

    assert crit_response.total > 0
    for a in crit_response.anomalies:
        assert a.severity == AnomalySeverity.CRITICAL


def test_isolation_forest_ml_detection(detector):
    ml_response = detector.detect_all(method_filter="ISOLATION_FOREST_ML")

    assert ml_response.total > 0
    for a in ml_response.anomalies:
        assert a.detection_method == "ISOLATION_FOREST_ML"
        assert "anomaly_score" in a.metrics
