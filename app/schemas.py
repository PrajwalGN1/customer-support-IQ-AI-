"""
Pydantic schemas and domain models for SupportIQ.
Enforces strict schema validation to prevent arbitrary execution or hallucinated parameters.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from app.config import settings


class QueryOperation(str, Enum):
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    GROUP_COUNT = "group_count"
    GROUP_AVERAGE = "group_average"
    LIST = "list"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"


class FilterCondition(BaseModel):
    field: str
    operator: FilterOperator = FilterOperator.EQ
    value: Any

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in settings.VALID_COLUMNS:
            raise ValueError(f"Invalid filter field '{v}'. Allowed fields: {settings.VALID_COLUMNS}")
        return clean


class DateRangeFilter(BaseModel):
    field: str = "created_at"
    start: Optional[str] = None
    end: Optional[str] = None
    preset: Optional[str] = None  # e.g., 'today', 'yesterday', 'this_week', 'this_month'

    @field_validator("field")
    @classmethod
    def validate_date_field(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean != "created_at":
            raise ValueError(f"Only 'created_at' supports date range filtering. Given: '{v}'")
        return clean


class QueryPlan(BaseModel):
    """
    Deterministic query plan structured output from the LLM.
    Strictly checked against column whitelists and allowed operations.
    """
    operation: Optional[QueryOperation] = QueryOperation.COUNT

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, v: Any) -> Any:
        if v is None or str(v).lower() in ("null", "none", ""):
            return QueryOperation.COUNT
        return v
    column: Optional[str] = None
    group_by: Optional[str] = None
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    filter_conditions: Optional[List[FilterCondition]] = Field(default_factory=list)
    date_range: Optional[DateRangeFilter] = None
    sort_by: Optional[str] = None
    sort_order: Optional[SortOrder] = SortOrder.DESC
    limit: Optional[int] = Field(default=None, ge=1, le=500)
    is_supported: bool = True
    unsupported_reason: Optional[str] = None

    @field_validator("column")
    @classmethod
    def validate_column(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().lower()
        if clean not in settings.VALID_COLUMNS:
            raise ValueError(f"Invalid target column '{v}'. Allowed: {settings.VALID_COLUMNS}")
        return clean

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip().lower()
        if clean not in settings.CATEGORICAL_COLUMNS and clean not in settings.VALID_COLUMNS:
            raise ValueError(f"Invalid group_by column '{v}'. Allowed: {settings.CATEGORICAL_COLUMNS}")
        return clean

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not v:
            return {}
        cleaned = {}
        for key, val in v.items():
            k_clean = key.strip().lower()
            if k_clean not in settings.VALID_COLUMNS:
                raise ValueError(f"Invalid filter field '{key}'. Allowed: {settings.VALID_COLUMNS}")
            cleaned[k_clean] = val
        return cleaned


# =====================================================================
# API Request & Response Schemas
# =====================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="Natural language support analytics question")


class ExecutionMetadata(BaseModel):
    execution_time_ms: float
    rows_scanned: int
    matched_rows: int
    operation: str
    llm_provider: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    query_plan: QueryPlan
    result: Any
    metadata: ExecutionMetadata
    error: Optional[str] = None


class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyItem(BaseModel):
    ticket_id: str
    anomaly_type: str
    severity: AnomalySeverity
    reason: str
    metrics: Dict[str, Any]
    detection_method: str  # e.g., 'IQR', 'RULE_SLA_BREACH', 'ISOLATION_FOREST'


class AnomalySummary(BaseModel):
    total_anomalies: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    by_method: Dict[str, int]


class AnomalyResponse(BaseModel):
    total: int
    summary: AnomalySummary
    anomalies: List[AnomalyItem]


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    dataset_loaded: bool
    records: int
    llm_provider: str
    llm_configured: bool
    reference_time_mode: str
    reference_time: str


class TicketSummaryStats(BaseModel):
    total_tickets: int
    open_tickets: int
    escalated_tickets: int
    resolved_tickets: int
    critical_tickets: int
    avg_customer_rating: Optional[float]
    avg_resolution_time_hrs: Optional[float]
    avg_response_time_hrs: Optional[float]
    categories: Dict[str, int]
    priorities: Dict[str, int]
    statuses: Dict[str, int]


class TicketDetail(BaseModel):
    ticket_id: str
    created_at: str
    category: str
    priority: str
    status: str
    response_time_hrs: float
    resolution_time_hrs: Optional[float]
    agent_id: str
    customer_rating: Optional[float]
    issue_summary: str
