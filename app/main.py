"""
FastAPI REST API application for SupportIQ.
Exposes health checks, natural-language query planning & execution, and anomaly detection.
"""
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.analytics import analytics_engine
from app.anomaly_detector import anomaly_detector
from app.config import settings
from app.data_loader import data_loader
from app.llm_engine import get_llm_provider
from app.query_engine import query_engine
from app.schemas import (
    AnomalyResponse,
    ExecutionMetadata,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TicketDetail,
    TicketSummaryStats,
)
from app.utils import logger, sanitize_for_json, time_execution


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset on startup to ensure in-memory caching."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    try:
        data_loader.load_data()
        logger.info(f"Dataset successfully loaded with {data_loader.record_count} records.")
    except Exception as e:
        logger.error(f"Startup error loading dataset: {e}")
    yield


# Initialize FastAPI application
app = FastAPI(
    title="SupportIQ REST API",
    description=(
        "AI-Powered Customer Support Intelligence System. "
        "Allows natural language questions about customer support tickets and detects operational anomalies."
    ),
    version=settings.PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for local dev and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service metadata and navigation."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "anomalies": "/anomalies",
            "stats": "/stats",
            "agents": "/agents",
            "documentation": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health check endpoint.
    Reports dataset status, row count, LLM configuration, and reference time mode.
    """
    llm_configured = bool(settings.GROQ_API_KEY) or settings.LLM_PROVIDER == "mock"
    ref_time = data_loader.get_reference_timestamp()

    return HealthResponse(
        status="healthy" if data_loader.is_loaded else "degraded",
        project=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        dataset_loaded=data_loader.is_loaded,
        records=data_loader.record_count,
        llm_provider=settings.LLM_PROVIDER,
        llm_configured=llm_configured,
        reference_time_mode=settings.REFERENCE_TIME_MODE,
        reference_time=ref_time.strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.post("/query", response_model=QueryResponse, tags=["Analytics & AI"])
def ask_question(payload: QueryRequest):
    """
    Process natural-language questions about the support ticket dataset.
    Architecture:
      User question -> LLM -> Structured JSON QueryPlan -> Pydantic Validation -> Safe Pandas Execution -> Deterministic Response.
    """
    logger.info(f"Incoming query request: '{payload.question}'")

    with time_execution() as timer:
        # Step 1: LLM Query Planning
        try:
            planner = get_llm_provider()
            plan = planner.generate_plan(payload.question)
        except Exception as e:
            logger.error(f"LLM planner error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Unable to process query via LLM: {str(e)}",
            )

        # Step 2: Safe deterministic execution
        try:
            raw_result, meta, answer = query_engine.execute_plan(plan)
        except Exception as e:
            logger.error(f"Execution error on plan: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query execution error: {str(e)}",
            )

    execution_metadata = ExecutionMetadata(
        execution_time_ms=timer["elapsed_ms"],
        rows_scanned=meta.get("rows_scanned", data_loader.record_count),
        matched_rows=meta.get("matched_rows", 0),
        operation=plan.operation.value,
        llm_provider=settings.LLM_PROVIDER,
    )

    return QueryResponse(
        question=payload.question,
        answer=answer,
        query_plan=plan,
        result=raw_result,
        metadata=execution_metadata,
        error=None if plan.is_supported else plan.unsupported_reason,
    )


@app.get("/anomalies", response_model=AnomalyResponse, tags=["Anomalies"])
def get_anomalies(
    severity: Optional[str] = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    anomaly_type: Optional[str] = Query(None, description="Filter by type: UNRESOLVED_SLA_BREACH, LONG_RESOLUTION_TIME, etc."),
    detection_method: Optional[str] = Query(None, description="Filter by method: RULE_SLA_BREACH, IQR_STATISTICAL, ISOLATION_FOREST_ML"),
    include_ml: bool = Query(True, description="Whether to include multivariate ML Isolation Forest anomalies"),
):
    """
    Detect operational support anomalies using hybrid rules, IQR statistics, and optional Isolation Forest ML.
    """
    try:
        response = anomaly_detector.detect_all(
            include_ml=include_ml,
            severity_filter=severity,
            type_filter=anomaly_type,
            method_filter=detection_method,
        )
        return response
    except Exception as e:
        logger.error(f"Anomaly detection failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute anomalies: {str(e)}",
        )


@app.get("/stats", response_model=TicketSummaryStats, tags=["Analytics & AI"])
def get_summary_statistics():
    """Get high-level summary KPIs and ticket status breakdowns."""
    try:
        return analytics_engine.get_kpi_summary()
    except Exception as e:
        logger.error(f"Statistics computation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/agents", tags=["Analytics & AI"])
def get_agent_metrics():
    """Get agent workload, resolution counts, and customer satisfaction leaderboard."""
    try:
        return analytics_engine.get_agent_performance()
    except Exception as e:
        logger.error(f"Agent metrics computation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=TicketDetail, tags=["Tickets"])
def get_ticket_detail(ticket_id: str):
    """Look up a specific ticket by ID."""
    df = data_loader.get_data()
    clean_id = ticket_id.strip().upper()
    match = df[df["ticket_id"].str.upper() == clean_id]

    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found.",
        )

    row = match.iloc[0].to_dict()
    return TicketDetail(
        ticket_id=str(row["ticket_id"]),
        created_at=str(row["created_at"]),
        category=str(row["category"]),
        priority=str(row["priority"]),
        status=str(row["status"]),
        response_time_hrs=float(row["response_time_hrs"]),
        resolution_time_hrs=float(row["resolution_time_hrs"]) if pd.notna(row.get("resolution_time_hrs")) else None,
        agent_id=str(row["agent_id"]),
        customer_rating=float(row["customer_rating"]) if pd.notna(row.get("customer_rating")) else None,
        issue_summary=str(row["issue_summary"]),
    )
