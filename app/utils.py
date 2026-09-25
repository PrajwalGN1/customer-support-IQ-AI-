"""
Utility functions for SupportIQ: logging, timing, date math, and serialization.
"""
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from app.config import settings

# Setup standard logger
logger = logging.getLogger("support_iq")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


@contextmanager
def time_execution():
    """Context manager to measure execution time in milliseconds."""
    start = time.perf_counter()
    res = {"elapsed_ms": 0.0}
    try:
        yield res
    finally:
        res["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas objects into standard Python primitives for JSON encoding."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(round(obj, 4))
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (int, str, bool)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


def resolve_relative_date_range(
    preset: str,
    ref_dt: datetime,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Resolve natural language date presets relative to a given reference timestamp.
    Useful for historical datasets where 'today' or 'this week' relates to the latest data point.
    """
    p = preset.lower().strip().replace(" ", "_")
    if p in ("today", "this_day"):
        start = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = ref_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    elif p in ("yesterday", "last_day"):
        prev = ref_dt - timedelta(days=1)
        start = prev.replace(hour=0, minute=0, second=0, microsecond=0)
        end = prev.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    elif p in ("this_week", "current_week"):
        start = (ref_dt - timedelta(days=ref_dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = (start + timedelta(days=6)).replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        return start, end
    elif p in ("last_week", "previous_week"):
        this_week_start = (ref_dt - timedelta(days=ref_dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(microseconds=1)
        return start, end
    elif p in ("this_month", "current_month"):
        start = ref_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # approximate month end
        next_month = (start + timedelta(days=32)).replace(day=1)
        end = next_month - timedelta(microseconds=1)
        return start, end
    elif p in ("last_month", "previous_month"):
        first_of_this_month = ref_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_of_this_month - timedelta(microseconds=1)
        start = (end.replace(day=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, end
    return None, None
