# SupportIQ — AI-Powered Customer Support Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

SupportIQ is a production-grade prototype built for the **DOTMappers AI Engineer Technical Assessment**. It empowers support managers to query ticket analytics using natural language and automatically detects operational anomalies using hybrid deterministic rules, statistical IQR bounds, and unsupervised machine learning.

---

## 1. Overview
Managing customer support operations requires answering questions rapidly:
- *"How many critical tickets are unresolved?"*
- *"Which agent has the lowest average customer rating?"*
- *"Show all Critical tickets not resolved within 12 hours."*

Traditional solutions either require SQL knowledge or risk system security by allowing Large Language Models (LLMs) to execute arbitrary Python code. **SupportIQ separates natural-language understanding from execution**, enforcing strict Pydantic schemas and deterministic Pandas computation to guarantee data integrity, zero code-injection risks, and verifiable numerical calculations.

---

## 2. Problem Statement
Given a customer support ticket dataset (`support_tickets.csv` with 500 records), the system must:
1. Ingest and validate data integrity with robust handling of missing values and historical dates.
2. Answer natural-language queries accurately without hallucinating metrics.
3. Automatically flag operational anomalies (SLA breaches, extreme resolution times).
4. Expose functionality through both a REST API (FastAPI) and an interactive web interface (Streamlit).

---

## 3. Objectives
- **Security First:** Never allow LLM-generated code execution (`eval`, `exec`, or direct SQL).
- **Explainable & Deterministic:** Every calculation is performed by validated Python functions on Pandas.
- **Reliable Fallbacks:** Works with live Groq LLM API or deterministic MockLLMProvider for offline tests.
- **Production-Minded:** Complete unit & integration test coverage, Docker support, and comprehensive API documentation.

---

## 4. Key Features
- **Natural Language Query Engine:** Converts free-form questions into structured, validated query plans.
- **Out-of-Domain Guardrails:** Detects questions unrelated to support tickets (e.g. *"What is the company revenue?"*) and refuses safely without hallucinating.
- **Hybrid Anomaly Detection:**
  - **Rule-Based SLA Breaches:** Unresolved High/Critical tickets older than 24 hours.
  - **Rule-Based Critical SLA Breaches:** Critical tickets taking longer than 12 hours to resolve.
  - **Statistical Outlier Detection:** Interquartile Range (IQR) on resolution time ($Q3 + 1.5 \times IQR$).
  - **Multivariate ML Outliers:** Isolation Forest identifying non-linear anomalies across response time, resolution time, priority, and satisfaction ratings.
- **Dual Interface:** Full REST API with OpenAPI/Swagger docs + interactive Streamlit dashboard.

---

## 5. Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web Interface                    │
│   - Executive KPI Dashboard     - Natural Language Query Area   │
│   - Anomaly Inspection Table    - Dataset Distributions         │
└────────────────────────────────┬────────────────────────────────┘
                                 │ REST / Local
┌────────────────────────────────▼────────────────────────────────┐
│                       FastAPI REST Layer                        │
│   GET /health    POST /query    GET /anomalies    GET /stats    │
└─────────────────┬───────────────────────────────┬───────────────┘
                  │                               │
        Natural Language Query             Anomaly Request
                  │                               │
┌─────────────────▼───────────────┐ ┌─────────────▼───────────────┐
│       LLM Query Engine          │ │    Hybrid Anomaly Detector  │
│  - Groq LLM (Llama 3.3 70B)     │ │  - Rule SLA Breaches        │
│  - Structured JSON Prompt       │ │  - IQR Statistical Bounds   │
│  - MockLLMProvider (offline)    │ │  - Isolation Forest ML      │
└─────────────────┬───────────────┘ └─────────────┬───────────────┘
                  │ QueryPlan JSON                │
┌─────────────────▼───────────────┐               │
│       Pydantic Validation       │               │
│  - Column Whitelist Check       │               │
│  - Allowed Operations Check     │               │
└─────────────────┬───────────────┘               │
                  │ Validated QueryPlan           │
┌─────────────────▼───────────────────────────────▼───────────────┐
│              Deterministic Pandas Query Engine                  │
│   - Safe filtering, aggregations, groupings, listings           │
│   - Zero arbitrary code execution                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│            Cleaned In-Memory Dataset (500 records)              │
│            Cached via Singleton DataLoader                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Stack & Justification

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Stable, performant typing system, and rich data ecosystem. |
| **Pandas & NumPy** | In-Memory Computation | Vectorized performance on 500 records (< 5ms response); eliminates database setup overhead. |
| **FastAPI** | REST API Framework | High-performance asynchronous runtime with automatic Swagger docs and native Pydantic validation. |
| **Pydantic v2** | Data Validation | Strictly validates LLM output against allowed columns, operations, and ranges before execution. |
| **Groq API (Llama 3.3)** | LLM Provider | Ultra-low inference latency (< 500ms) with strong adherence to structured JSON schemas. |
| **Scikit-Learn** | Machine Learning | Isolation Forest for unsupervised multivariate operational anomaly detection. |
| **Streamlit** | Web Interface | Rapid delivery of a clean, responsive analytics dashboard for business stakeholders. |
| **Pytest & TestClient** | Testing Framework | Automated test coverage for data ingestion, query logic, anomaly rules, and API endpoints. |
| **Docker & Compose** | Containerization | Reproducible cross-platform deployment for API and UI services. |

---

## 7. Project Structure

```text
support-iq/
├── app/
│   ├── __init__.py           # Application package init and version
│   ├── config.py             # Centralized settings and flexible path resolution
│   ├── schemas.py            # Pydantic models for queries, plans, and anomalies
│   ├── data_loader.py        # Ingestion, validation, date parsing, and caching
│   ├── query_engine.py       # Deterministic safe Pandas query execution
│   ├── llm_engine.py         # Groq LLM planner and offline MockLLMProvider
│   ├── anomaly_detector.py   # Hybrid rules, IQR statistical, and ML detectors
│   ├── analytics.py          # KPI metrics, agent performance, and category summaries
│   ├── utils.py              # Logging, timing, JSON serialization, and date math
│   └── main.py               # FastAPI application with REST endpoints
│
├── data/
│   └── support_tickets.csv   # Source dataset (500 support tickets)
│
├── ui/
│   └── streamlit_app.py      # Streamlit web dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py   # Tests for CSV ingestion and schema validation
│   ├── test_query_engine.py  # Tests for query planning and execution logic
│   ├── test_anomaly_detector.py # Tests for SLA breaches, IQR, and ML detector
│   └── test_api.py           # Integration tests for FastAPI endpoints
│
├── docs/
│   └── WALKTHROUGH.md        # 30-minute interview guide and technical talking points
│
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules for Python and secrets
├── requirements.txt          # Pinned project dependencies
├── Dockerfile                # Multi-stage production container definition
├── docker-compose.yml        # Docker compose configuration for API + Streamlit
├── README.md                 # Complete system documentation
└── run.py                    # Unified CLI runner (api, ui, test, all)
```

---

## 8. Prerequisites
- **Python:** Version 3.11 or higher.
- **Git:** For cloning and version control.
- **Groq API Key:** (Optional for live LLM planning; get free key at [console.groq.com](https://console.groq.com)).
  - *Note:* The system automatically defaults to `MockLLMProvider` if no key is supplied, enabling 100% offline testing.
- **Docker:** (Optional) If running containerized.

---

## 9. Installation

1. **Clone repository and navigate to `support-iq`:**
   ```bash
   git clone <repo-url>
   cd support-iq
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure the following variables in `.env`:
```ini
# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Provider mode ('groq' for live LLM, 'mock' for deterministic offline testing)
LLM_PROVIDER=groq

# Data and Application Settings
DATA_PATH=data/support_tickets.csv
APP_ENV=development
LOG_LEVEL=INFO
API_PORT=8000
STREAMLIT_PORT=8501

# Anomaly Reference Time ('dataset_max' for historical records or 'now')
REFERENCE_TIME_MODE=dataset_max
```

---

## 11. Running the API

Start the FastAPI application via Uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Or via the unified CLI runner:
```bash
python run.py api
```

The REST API will be available at:
- **API Base:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

---

## 12. Running Streamlit UI

Start the Streamlit dashboard:
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```
Or via the unified CLI runner:
```bash
python run.py ui
```

Access the web interface at: `http://localhost:8501`.

---

## 13. Docker Deployment

Launch both the REST API and Streamlit UI with a single command:
```bash
docker compose up --build
```

- API container runs on port `8000`.
- Streamlit container runs on port `8501`.

To stop services:
```bash
docker compose down
```

---

## 14. API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health status, dataset stats, record count, and LLM configuration. |
| `POST` | `/query` | Natural-language query interface with structured execution. |
| `GET` | `/anomalies` | Hybrid anomaly detection with optional severity and method filters. |
| `GET` | `/stats` | High-level KPI aggregates and ticket status distributions. |
| `GET` | `/agents` | Agent workload, resolution counts, and customer ratings. |
| `GET` | `/tickets/{ticket_id}` | Detailed metadata for an individual ticket. |

---

## 15. Example Queries & 16. Real Dataset Outputs

All numerical values below are **real calculations computed directly from `support_tickets.csv`**:

### Query 1: *"How many tickets are currently open?"*
- **Query Plan:** `operation="count", filters={"status": "Open"}`
- **Calculated Result:** `111`
- **Answer:** *"There are 111 tickets matching status = 'Open'."*
  *(Note: 173 tickets are unresolved in total: 111 Open + 62 Escalated)*

### Query 2: *"Which agent has the lowest average customer rating?"*
- **Query Plan:** `operation="group_average", column="customer_rating", group_by="agent_id", sort_order="asc", limit=1`
- **Calculated Result:** `{"AGT-08": {"average": 3.48, "count": 25}}`
- **Answer:** *"AGT-08 has the lowest average customer rating with an average of 3.48 (across 25 rated tickets)."*

### Query 3: *"Which agent resolved the most tickets?"*
- **Query Plan:** `operation="group_count", group_by="agent_id", filters={"status": "Resolved"}, sort_order="desc", limit=1`
- **Calculated Result:** `{"AGT-09": 37, "AGT-12": 37}`
- **Answer:** *"AGT-09 and AGT-12 are tied with the highest count of 37 resolved tickets each."*

### Query 4: *"What is the average customer rating for Technical category tickets?"*
- **Query Plan:** `operation="average", column="customer_rating", filters={"category": "Technical"}`
- **Calculated Result:** `3.74`
- **Answer:** *"The average customer rating is 3.74 (computed across 109 resolved Technical records)."*

### Query 5: *"Show all Critical tickets not resolved within 12 hours."*
- **Query Plan:** `operation="list", filters={"priority": "Critical"}, filter_conditions=[{"field": "resolution_time_hrs", "operator": "gt", "value": 12.0}]`
- **Calculated Result:** `3 resolved tickets` (`TKT-238` with 53.4h, `TKT-255` with 66.6h, `TKT-446` with 60.6h) plus `31 unresolved Critical tickets`.

### Query 6: *"What is the company's annual revenue?"* (Out-of-Domain Guardrail)
- **Query Plan:** `is_supported=false`
- **Answer:** *"I cannot answer that because revenue is not contained in the support ticket dataset."*

---

## 17. LLM Query Planner & Safety Guardrails
To prevent arbitrary code execution, SupportIQ enforces a strict unidirectional pipeline:

```text
User Question
     ↓
LLM System Prompt (with column whitelists & valid operations)
     ↓
Structured JSON QueryPlan
     ↓
Pydantic Strict Validation (rejects unknown columns/operations)
     ↓
Safe Deterministic Pandas Execution
     ↓
Deterministic Response Template
```

**Guardrail Rules Enforced:**
1. No raw Python expressions (`exec`, `eval`) or arbitrary SQL.
2. Only approved columns (`VALID_COLUMNS`) and operations (`VALID_OPERATIONS`) are permitted.
3. Unsupported questions are rejected with an explicit explanation.

---

## 18. Anomaly Detection Engine

SupportIQ employs a **hybrid detection strategy**:

1. **Rule-Based SLA Breaches:**
   - Detects unresolved High/Critical tickets older than 24 hours relative to reference time.
   - Result: **80 tickets** flagged in the dataset.
2. **Rule-Based Critical SLA Breaches:**
   - Detects Critical tickets resolved in excess of 12 hours.
   - Result: **3 tickets** flagged (`TKT-238`, `TKT-255`, `TKT-446`).
3. **Statistical IQR Outliers:**
   - Evaluates resolution times for resolved tickets: $Q1 = 6.15\text{h}$, $Q3 = 22.95\text{h}$, $IQR = 16.80\text{h}$.
   - Upper bound: $Q3 + 1.5 \times IQR = 48.15\text{h}$.
   - Result: **21 tickets** flagged (highest: `TKT-108` at 119.7 hours).
4. **Multivariate ML (Isolation Forest):**
   - Evaluates non-linear relationships across response time, resolution time, priority level, and customer rating.
   - Result: **17 multivariate operational anomalies** identified.

---

## 19. Testing & Quality Assurance

Run the complete test suite via `pytest`:
```bash
pytest tests -v
```
Or via `run.py`:
```bash
python run.py test
```

### Test Suite Structure
- `tests/test_data_loader.py`: Schema validation, type parsing, missing value handling, and reference timestamp accuracy.
- `tests/test_query_engine.py`: Safe execution of counts, averages, group metrics, listings, filters, and guardrails.
- `tests/test_anomaly_detector.py`: Deterministic SLA rules, IQR upper bound validation, and ML anomaly detector.
- `tests/test_api.py`: FastAPI endpoints (`/health`, `/query`, `/anomalies`, `/stats`, `/agents`, `/tickets/{id}`).

*All tests execute using `MockLLMProvider`, requiring zero external API keys or network calls.*

---

## 20. Known Limitations
1. **In-Memory Scale:** Designed for CSV-scale datasets (< 100,000 rows). Datasets with millions of rows require a dedicated database engine.
2. **Historical Dataset Bounds:** The dataset covers January 2024 to March 2024. Queries referencing "today" use the dataset's maximum date (`2024-03-30`) unless `REFERENCE_TIME_MODE=now` is explicitly set.
3. **External LLM Dependency:** When using Groq, connectivity or quota limits affect live query planning (mitigated by automatic fallback to `MockLLMProvider`).

---

## 21. Future Improvements
1. **DuckDB / SQLAlchemy Integration:** Allow seamless transition from in-memory Pandas to SQL databases (PostgreSQL, Snowflake).
2. **Query Caching:** Redis caching for repetitive aggregation queries.
3. **Role-Based Access Control (RBAC):** JWT authentication for customer support supervisors and agents.
4. **Automated SLA Alerting:** Webhook integrations for Slack and PagerDuty when Critical SLA breaches occur.

---

## 22. 5-Minute Demo Sequence
1. Start the API: `python run.py api`
2. Start the UI: `python run.py ui`
3. Open `http://localhost:8501` to view live KPI cards (500 tickets, 111 open, 3.75 avg rating).
4. Click through the suggested natural-language question chips to observe instant answers and inspect JSON query plans.
5. Filter the Operational Anomalies table by `CRITICAL` severity to inspect SLA breaches.
6. Run `pytest tests` in the terminal to demonstrate 100% test pass rate.
