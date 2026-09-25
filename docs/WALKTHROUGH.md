# SupportIQ — 30-Minute Interview & Technical Walkthrough Guide

This document is prepared for the 30-minute post-submission technical interview and architecture walkthrough for the **AI Engineer role at DOTMappers IT Pvt. Ltd.**

---

## 1. Problem Overview
Customer support teams handle hundreds of tickets across various channels. Management needs real-time visibility into:
1. Operational metrics (open vs. resolved tickets, backlog status).
2. Agent performance and customer satisfaction.
3. Timely identification of SLA breaches and operational bottlenecks.

Support teams currently rely on static spreadsheets or complex BI tools that non-technical managers struggle to query. **SupportIQ** solves this by providing:
- Natural-language querying with zero risk of arbitrary code execution.
- Deterministic, verifiable analytics.
- Automated hybrid anomaly detection (SLA breaches, statistical outliers, and multi-feature ML anomalies).

---

## 2. System Architecture
```text
┌────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                     │
│    (KPI Dashboard, Ask AI Assistant, Anomaly Table)    │
└───────────────────────────┬────────────────────────────┘
                            │ REST / In-Memory
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI REST Layer                    │
│    Endpoints: /health, /query, /anomalies, /stats      │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
    Natural Language Query         Anomaly Request
               │                          │
┌──────────────▼─────────────┐ ┌──────────▼──────────────┐
│       LLM Query Engine     │ │ Hybrid Anomaly Detector │
│  - Groq LLM (Llama 3.3 70B)│ │  - Rule SLA Breaches    │
│  - MockLLMProvider (tests) │ │  - IQR Statistical      │
│  - JSON Schema Prompt      │ │  - Isolation Forest ML  │
└──────────────┬─────────────┘ └──────────┬──────────────┘
               │ QueryPlan JSON           │
┌──────────────▼─────────────┐            │
│     Pydantic Validation    │            │
│  - Column Whitelist        │            │
│  - Allowed Operations Only │            │
└──────────────┬─────────────┘            │
               │ Validated Plan           │
┌──────────────▼──────────────────────────▼──────────────┐
│           Deterministic Pandas Query Engine            │
│   - In-memory cached DataLoader                        │
│   - Exact aggregation, filtering, group metrics        │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          Support Tickets Dataset (500 records)         │
│          support-iq/data/support_tickets.csv           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Key Design Decisions & "Why" Rationales

### Why Groq?
- **Speed & Predictability:** Groq's LPU inference delivers near-instantaneous token generation (often sub-400ms for structured JSON), critical for conversational analytics.
- **Model Quality:** Llama-3.3-70B-Versatile excels at adhering strictly to structured JSON schemas without hallucinating schema fields.
- **Zero Cost/High Value:** Free tier availability allows cost-effective prototyping without infrastructure overhead.

### Why FastAPI?
- **Asynchronous & High-Performance:** Native async support, high concurrency, and minimal latency.
- **Pydantic Validation:** Request/response schemas are validated out of the box with clear 422 errors.
- **Automatic OpenAPI Docs:** Generates interactive Swagger (`/docs`) and ReDoc (`/redoc`) without extra effort.

### Why Pandas?
- **Dataset Scale (500 rows):** For 500 rows (~46 KB), introducing an external relational database (Postgres, SQLite) or distributed engine (Spark) adds operational overhead with zero latency benefit.
- **Vectorized Operations:** In-memory Pandas executes filtering and grouping in sub-millisecond times (~2-5ms).
- **Extensible Abstraction:** The `DataLoader` and `QueryEngine` interfaces isolate the data access layer, allowing seamless migration to DuckDB, SQLAlchemy, or Snowflake when data scales.

### Why Streamlit?
- **Rapid Prototyping:** Builds a clean, responsive analytics dashboard with live KPI cards, dataframes, and filters within hours.
- **Direct Integration:** Seamlessly interacts with the backend components and allows quick stakeholder demos.

---

## 4. Why NOT Let the LLM Directly Execute Python Code?
**This is the fundamental architectural principle of SupportIQ.**
1. **Security (Remote Code Execution):** Allowing an LLM to generate executable Python or SQL opens the system to prompt injection, sandbox escapes, and unauthorized data extraction or file system tampering.
2. **Reliability & Hallucination:** LLMs often generate subtly incorrect Pandas code (e.g., misinterpreting NaN in averages, improper group index handling, or fabricated method names).
3. **Auditability:** Every query in SupportIQ produces a serialized `QueryPlan` (Pydantic model) that is logged, inspected, and validated *before* a single line of computation runs.

---

## 5. How Query Validation & Guardrails Work
1. **Pydantic Strict Validation:** The LLM's JSON is parsed into a `QueryPlan`. Any column not in `VALID_COLUMNS` or any operation outside `VALID_OPERATIONS` throws a validation error.
2. **Out-of-Domain Guardrail:** Questions regarding revenue, salaries, weather, or company stock are intercepted either by the LLM system prompt or the planner guardrail, returning a polite explanation rather than fabricating answers.
3. **Deterministic Response Formatting:** Numerical answers are never calculated by the LLM. The exact values (e.g. `3.74`, `111`, `AGT-08`) are computed by Python and formatted using verifiable templates.

---

## 6. Data Integrity & Missing Values Handling
- **Dataset Domain Logic:** In customer support, open and escalated tickets **do not have** a resolution time or customer rating. In our dataset:
  - 327 resolved tickets have `resolution_time_hrs` and `customer_rating`.
  - 173 unresolved tickets (`Open` + `Escalated`) have `NaN` for both.
- **No Silent Corruption:** Rather than imputing 0 or mean into the raw dataset (which would severely distort average resolution time and rating), the ingestion layer preserves `NaN` and the query engine filters or skips missing values appropriately during aggregations.

---

## 7. Historical Dates & Reference Time
- **The 2024 vs 2026 Problem:** The dataset records are from January to March 2024. If ticket age is compared against the current system date in 2026, **every ticket would appear over 2 years old**, creating false anomalies!
- **Solution:** SupportIQ uses a dataset-aware reference timestamp (`2024-03-30 18:06:00`, the maximum ticket timestamp).
- **Configurability:** Controlled via `REFERENCE_TIME_MODE="dataset_max"` (default) or `"now"`.

---

## 8. Hybrid Anomaly Detection Explained
| Detector | Method | What it Detects | Dataset Results |
| :--- | :--- | :--- | :--- |
| **Rule 1: SLA Breach** | Deterministic Rule | Unresolved High/Critical tickets > 24h old | **80 tickets** |
| **Rule 2: Critical Long Resolution** | Deterministic Rule | Critical tickets resolved > 12h | **3 tickets** (TKT-238, TKT-255, TKT-446) |
| **Statistical: IQR Outliers** | Interquartile Range ($Q3 + 1.5 \times IQR$) | Unusually long resolution times ($> 48.15$ hours) | **21 tickets** (Max: TKT-108 with 119.7h) |
| **ML: Isolation Forest** | Unsupervised Tree Ensemble | Non-linear multivariate anomalies across response, resolution, priority, and rating | **17 tickets** |

---

## 9. Interviewer Q&A Cheat Sheet

### Q1: "What happens if the Groq API key is invalid or Groq experiences downtime?"
> **Answer:** The application has a layered fallback architecture. The `get_llm_provider()` factory detects missing credentials and gracefully switches to `MockLLMProvider`, ensuring the system remains completely functional for standard queries. In the API layer, network failures to Groq return a clean `503 Service Unavailable` with a user-friendly error message, never crashing or leaking tracebacks.

### Q2: "How would you scale this to 10 million tickets?"
> **Answer:**
> 1. **Storage & Engine:** Replace Pandas with DuckDB (for analytical queries on Parquet) or Snowflake/ClickHouse.
> 2. **Caching:** Introduce a Redis cache layer for frequent aggregate queries (e.g. open ticket count).
> 3. **Batch Anomaly Detection:** Run IQR and Isolation Forest as scheduled background jobs (e.g., Celery/Airflow), storing flagged anomalies in an indexed PostgreSQL table.
> 4. **API:** Scale FastAPI horizontally behind an AWS ALB or Nginx reverse proxy.

### Q3: "Why did you use both Rule-based and Machine Learning anomaly detection?"
> **Answer:**
> Rules provide **guaranteed compliance** for business SLAs (e.g. Critical tickets must be addressed within 12 hours). However, rules miss complex, non-linear relationships—such as a ticket with a 15-minute response time that inexplicably received a 1-star rating and required 45 hours to resolve. Machine Learning (Isolation Forest) catches these multidimensional outliers without requiring hardcoded thresholds.

---

## 10. 5-Minute Live Demo Script
1. **Show Health:** Open `http://localhost:8000/health` (demonstrate 500 records loaded, healthy status).
2. **Launch UI:** Open Streamlit dashboard at `http://localhost:8501`.
3. **KPI Inspection:** Point out 500 total tickets, 111 open, 62 escalated, 327 resolved, 3.75 avg rating.
4. **Natural Language Queries:**
   - Click *"How many tickets are currently open?"* -> Shows **111 tickets**.
   - Click *"Which agent has the lowest average customer rating?"* -> Shows **AGT-08 (3.48 average)**.
   - Click *"What is the company's annual revenue?"* -> Shows the **guardrail response** preventing hallucination.
5. **Inspect Anomaly Table:** Filter by `CRITICAL` severity and show TKT-238, TKT-255, and TKT-446 exceeding resolution SLAs.
6. **Show Test Suite:** Run `pytest tests` (all 18+ tests passing in < 2 seconds).
