"""
Data ingestion and validation layer for SupportIQ.
Loads, validates, cleans, and caches the customer support ticket dataset.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from app.config import settings
from app.utils import logger


class DataValidationError(Exception):
    """Raised when the dataset fails schema or data integrity validation."""
    pass


class DataLoader:
    """
    Singleton data ingestion manager.
    Caches the cleaned DataFrame in memory to avoid repeated disk reads.
    """
    _instance: Optional["DataLoader"] = None
    _df: Optional[pd.DataFrame] = None
    _raw_df: Optional[pd.DataFrame] = None
    _loaded_at: Optional[datetime] = None
    _malformed_records: List[Dict] = []
    _dataset_path: Optional[Path] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data_path: Optional[Path] = None):
        target_path = data_path or settings.DATA_PATH
        if self._df is None or self._dataset_path != target_path:
            self.load_data(target_path)

    def load_data(self, data_path: Optional[Path] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and validate the CSV dataset from disk.
        """
        if self._df is not None and not force_reload:
            return self._df

        path = Path(data_path or settings.DATA_PATH).resolve()
        if not path.exists():
            raise DataValidationError(f"Dataset file not found at: {path}")

        logger.info(f"Loading dataset from: {path}")
        try:
            raw_df = pd.read_csv(path)
        except Exception as e:
            raise DataValidationError(f"Failed to read CSV file '{path}': {str(e)}")

        if raw_df.empty:
            raise DataValidationError(f"Dataset at '{path}' is empty.")

        cleaned_df, malformed = self._validate_and_clean(raw_df)

        self._raw_df = raw_df
        self._df = cleaned_df
        self._malformed_records = malformed
        self._loaded_at = datetime.now(timezone.utc)
        self._dataset_path = path

        logger.info(
            f"Successfully ingested {len(cleaned_df)} records. "
            f"Identified {len(malformed)} malformed/anomalous rows."
        )
        return self._df

    def _validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Validate columns, types, ranges, and formats."""
        cleaned = df.copy()

        # 1. Normalize column names (strip whitespace and lower-case)
        cleaned.columns = [c.strip().lower() for c in cleaned.columns]

        # 2. Check for required columns
        missing_cols = set(settings.VALID_COLUMNS) - set(cleaned.columns)
        if missing_cols:
            raise DataValidationError(f"Dataset is missing required columns: {sorted(list(missing_cols))}")

        malformed = []

        # 3. Clean string columns
        for col in ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]:
            cleaned[col] = cleaned[col].astype(str).str.strip()

        # 4. Parse Datetime
        try:
            cleaned["created_at_dt"] = pd.to_datetime(cleaned["created_at"], errors="coerce")
            invalid_dates = cleaned[cleaned["created_at_dt"].isna()]
            if not invalid_dates.empty:
                for idx, row in invalid_dates.iterrows():
                    malformed.append({
                        "row_index": idx,
                        "ticket_id": row.get("ticket_id"),
                        "issue": f"Invalid created_at timestamp: {row.get('created_at')}"
                    })
        except Exception as e:
            raise DataValidationError(f"Error parsing datetime column 'created_at': {e}")

        # 5. Parse Numeric Columns
        for num_col in settings.NUMERIC_COLUMNS:
            cleaned[num_col] = pd.to_numeric(cleaned[num_col], errors="coerce")

        # 6. Validate Categorical Consistency
        invalid_priorities = cleaned[~cleaned["priority"].isin(settings.VALID_PRIORITIES)]
        if not invalid_priorities.empty:
            for idx, row in invalid_priorities.iterrows():
                malformed.append({
                    "row_index": idx,
                    "ticket_id": row.get("ticket_id"),
                    "issue": f"Unrecognized priority '{row.get('priority')}'"
                })

        invalid_statuses = cleaned[~cleaned["status"].isin(settings.VALID_STATUSES)]
        if not invalid_statuses.empty:
            for idx, row in invalid_statuses.iterrows():
                malformed.append({
                    "row_index": idx,
                    "ticket_id": row.get("ticket_id"),
                    "issue": f"Unrecognized status '{row.get('status')}'"
                })

        # 7. Check for duplicate ticket IDs
        duplicates = cleaned[cleaned.duplicated(subset=["ticket_id"], keep=False)]
        if not duplicates.empty:
            logger.warning(f"Detected {len(duplicates)} duplicate ticket_ids in dataset.")

        return cleaned, malformed

    def get_data(self) -> pd.DataFrame:
        """Returns the in-memory processed DataFrame."""
        if self._df is None:
            self.load_data()
        return self._df.copy()

    def get_raw_data(self) -> pd.DataFrame:
        """Returns the original unmutated raw DataFrame."""
        if self._raw_df is None:
            self.load_data()
        return self._raw_df.copy()

    def get_reference_timestamp(self) -> pd.Timestamp:
        """
        Determines the reference timestamp for age calculations and relative date filters.
        If REFERENCE_TIME_MODE is 'dataset_max', uses the maximum created_at date in the dataset.
        Otherwise falls back to datetime.now().
        """
        df = self.get_data()
        if settings.MANUAL_REFERENCE_TIME:
            try:
                return pd.to_datetime(settings.MANUAL_REFERENCE_TIME)
            except Exception:
                pass

        if settings.REFERENCE_TIME_MODE == "dataset_max":
            max_dt = df["created_at_dt"].max()
            if pd.notna(max_dt):
                return max_dt

        return pd.Timestamp.now()

    @property
    def record_count(self) -> int:
        return len(self._df) if self._df is not None else 0

    @property
    def malformed_records(self) -> List[Dict]:
        return list(self._malformed_records)

    @property
    def is_loaded(self) -> bool:
        return self._df is not None


# Global access singleton
data_loader = DataLoader()
