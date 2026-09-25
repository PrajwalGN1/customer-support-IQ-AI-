"""
Integration tests for FastAPI REST API endpoints using TestClient.
Tests endpoints without requiring external API keys.
"""
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app

# Ensure mock provider is used during tests for deterministic offline execution
settings.LLM_PROVIDER = "mock"

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "SupportIQ"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_loaded"] is True
    assert data["records"] == 500


def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tickets"] == 500
    assert data["open_tickets"] == 111
    assert data["resolved_tickets"] == 327
    assert data["critical_tickets"] == 55


def test_agents_endpoint():
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 12  # AGT-01 to AGT-12


def test_ticket_detail_found():
    response = client.get("/tickets/TKT-001")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == "TKT-001"
    assert data["category"] == "General"


def test_ticket_detail_not_found():
    response = client.get("/tickets/TKT-NONEXISTENT")
    assert response.status_code == 404


def test_anomalies_endpoint():
    response = client.get("/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert "summary" in data
    assert "anomalies" in data


def test_anomalies_with_filters():
    response = client.get("/anomalies?severity=CRITICAL")
    assert response.status_code == 200
    data = response.json()
    for anom in data["anomalies"]:
        assert anom["severity"] == "CRITICAL"


def test_query_count_open():
    payload = {"question": "How many tickets are currently open?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "111" in data["answer"]
    assert data["result"] == 111
    assert data["metadata"]["rows_scanned"] == 500


def test_query_lowest_rating_agent():
    payload = {"question": "Which agent has the lowest average customer rating?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "AGT-08" in data["answer"]
    assert "3.48" in data["answer"]


def test_query_unsupported_out_of_domain():
    payload = {"question": "What is the company's annual revenue?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query_plan"]["is_supported"] is False
    assert "revenue" in data["answer"].lower() or "support ticket" in data["answer"].lower()


def test_query_validation_error():
    payload = {"question": ""}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
