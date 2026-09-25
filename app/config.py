"""
Centralized Configuration for SupportIQ.
Loads environment variables safely and defines default runtime parameters.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env if found in root or current directory
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()


def resolve_data_path() -> Path:
    """Find the dataset path across typical working directories."""
    env_path = os.getenv("DATA_PATH")
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
        candidate_paths.append(BASE_DIR / env_path)

    # Standard fallback paths
    candidate_paths.extend([
        BASE_DIR / "data" / "support_tickets.csv",
        BASE_DIR.parent / "support_tickets.csv",
        Path("data/support_tickets.csv"),
        Path("support_tickets.csv"),
    ])

    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p.resolve()

    # Default to base_dir/data/support_tickets.csv even if not yet created
    return (BASE_DIR / "data" / "support_tickets.csv").resolve()


class Settings:
    """Application settings and constants."""
    # Project Info
    PROJECT_NAME: str = "SupportIQ"
    PROJECT_VERSION: str = "1.0.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # Data Path
    DATA_PATH: Path = resolve_data_path()

    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # 'groq' or 'mock'
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # Anomaly Detection & Reference Time
    # 'dataset_max' prevents 2024 records from falsely breaching 2026 current time
    REFERENCE_TIME_MODE: str = os.getenv("REFERENCE_TIME_MODE", "dataset_max")
    MANUAL_REFERENCE_TIME: Optional[str] = os.getenv("MANUAL_REFERENCE_TIME", None)
    
    # SLA & Threshold Defaults (hours)
    SLA_CRITICAL_HOURS: float = float(os.getenv("SLA_CRITICAL_HOURS", "12.0"))
    SLA_HIGH_HOURS: float = float(os.getenv("SLA_HIGH_HOURS", "24.0"))
    IQR_MULTIPLIER: float = float(os.getenv("IQR_MULTIPLIER", "1.5"))
    ISOLATION_FOREST_CONTAMINATION: float = float(os.getenv("ISOLATION_FOREST_CONTAMINATION", "0.05"))

    # Valid schema definitions
    VALID_COLUMNS = [
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "agent_id",
        "customer_rating",
        "issue_summary",
    ]

    NUMERIC_COLUMNS = [
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating",
    ]

    CATEGORICAL_COLUMNS = [
        "category",
        "priority",
        "status",
        "agent_id",
    ]

    VALID_CATEGORIES = ["General", "Billing", "Technical"]
    VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]
    VALID_STATUSES = ["Open", "Escalated", "Resolved"]


settings = Settings()
