"""
Safe, deterministic query engine for SupportIQ.
Executes Pydantic-validated QueryPlan objects against the Pandas dataset.
Strictly disallows raw Python, SQL, or shell code execution.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.config import settings
from app.data_loader import data_loader
from app.schemas import (
    FilterCondition,
    FilterOperator,
    QueryOperation,
    QueryPlan,
    SortOrder,
)
from app.utils import logger, resolve_relative_date_range, sanitize_for_json


class QueryExecutionError(Exception):
    """Raised when query execution encounters an unrecoverable logic or schema error."""
    pass


class QueryEngine:
    """
    Deterministic query executor.
    Applies filters, date ranges, aggregations, groupings, sorting, and limits.
    """

    def __init__(self, loader=None):
        self.loader = loader or data_loader

    def execute_plan(self, plan: QueryPlan) -> Tuple[Any, Dict[str, Any], str]:
        """
        Executes a QueryPlan against the underlying DataFrame.
        Returns:
            - raw_result: Python native object / list / dict / scalar
            - metadata: rows_scanned, matched_rows, etc.
            - formatted_answer: Deterministic human-readable explanation
        """
        if not plan.is_supported:
            reason = plan.unsupported_reason or "That question cannot be answered using the available ticket dataset."
            return None, {"rows_scanned": 0, "matched_rows": 0, "operation": "unsupported"}, reason

        df = self.loader.get_data()
        total_scanned = len(df)

        # 1. Apply Filters
        filtered_df, filter_notes = self._apply_filters(df, plan)
        matched_rows = len(filtered_df)

        metadata = {
            "rows_scanned": total_scanned,
            "matched_rows": matched_rows,
            "operation": plan.operation.value,
            "filter_notes": filter_notes,
        }

        # Handle zero-match edge case cleanly
        if matched_rows == 0:
            msg = "No tickets matched the requested criteria."
            if "date_filtered" in filter_notes:
                min_d = df["created_at_dt"].min().strftime("%Y-%m-%d")
                max_d = df["created_at_dt"].max().strftime("%Y-%m-%d")
                msg += f" Note: This dataset covers historical records between {min_d} and {max_d}."
            return None, metadata, msg

        # 2. Execute Operation
        raw_result, formatted_answer = self._run_operation(filtered_df, plan)

        return sanitize_for_json(raw_result), metadata, formatted_answer

    def _apply_filters(self, df: pd.DataFrame, plan: QueryPlan) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply simple dictionary filters, advanced filter conditions, and date ranges."""
        res = df.copy()
        notes = {}

        # A. Simple dictionary filters
        if plan.filters:
            for field, val in plan.filters.items():
                field_lower = field.lower().strip()
                if field_lower not in res.columns:
                    raise QueryExecutionError(f"Filter column '{field}' does not exist in dataset.")

                if isinstance(val, (list, set, tuple)):
                    # Case-insensitive matching for string values
                    val_clean = [str(v).lower() for v in val]
                    mask = res[field_lower].astype(str).str.lower().isin(val_clean)
                    res = res[mask]
                elif val is not None:
                    str_val = str(val).lower()
                    mask = res[field_lower].astype(str).str.lower() == str_val
                    res = res[mask]

        # B. Advanced filter conditions
        if plan.filter_conditions:
            for cond in plan.filter_conditions:
                f = cond.field.lower()
                op = cond.operator
                val = cond.value

                if f not in res.columns:
                    raise QueryExecutionError(f"Filter condition field '{f}' is invalid.")

                series = res[f]

                if op == FilterOperator.EQ:
                    res = res[series.astype(str).str.lower() == str(val).lower()]
                elif op == FilterOperator.NEQ:
                    res = res[series.astype(str).str.lower() != str(val).lower()]
                elif op == FilterOperator.IN:
                    v_list = [str(x).lower() for x in (val if isinstance(val, list) else [val])]
                    res = res[series.astype(str).str.lower().isin(v_list)]
                elif op == FilterOperator.NOT_IN:
                    v_list = [str(x).lower() for x in (val if isinstance(val, list) else [val])]
                    res = res[~series.astype(str).str.lower().isin(v_list)]
                elif op in (FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE):
                    numeric_s = pd.to_numeric(series, errors="coerce")
                    num_val = float(val)
                    if op == FilterOperator.GT:
                        res = res[numeric_s > num_val]
                    elif op == FilterOperator.GTE:
                        res = res[numeric_s >= num_val]
                    elif op == FilterOperator.LT:
                        res = res[numeric_s < num_val]
                    elif op == FilterOperator.LTE:
                        res = res[numeric_s <= num_val]
                elif op == FilterOperator.CONTAINS:
                    res = res[series.astype(str).str.contains(str(val), case=False, na=False)]

        # C. Date range filter
        if plan.date_range:
            dr = plan.date_range
            ref_dt = self.loader.get_reference_timestamp()
            start_dt, end_dt = None, None

            if dr.preset:
                start_dt, end_dt = resolve_relative_date_range(dr.preset, ref_dt)
                notes["date_preset"] = dr.preset

            if dr.start:
                try:
                    start_dt = pd.to_datetime(dr.start)
                except Exception as e:
                    logger.warning(f"Could not parse start date '{dr.start}': {e}")
            if dr.end:
                try:
                    end_dt = pd.to_datetime(dr.end)
                except Exception as e:
                    logger.warning(f"Could not parse end date '{dr.end}': {e}")

            if start_dt is not None:
                res = res[res["created_at_dt"] >= start_dt]
                notes["date_filtered"] = True
            if end_dt is not None:
                res = res[res["created_at_dt"] <= end_dt]
                notes["date_filtered"] = True

        return res, notes

    def _run_operation(self, df: pd.DataFrame, plan: QueryPlan) -> Tuple[Any, str]:
        """Execute the specific computation and format the answer deterministically."""
        op = plan.operation

        if op == QueryOperation.COUNT:
            count_val = len(df)
            filter_desc = self._summarize_filters(plan)
            if filter_desc:
                ans = f"There are {count_val} tickets matching {filter_desc}."
            else:
                ans = f"There are {count_val} total tickets in the dataset."
            return count_val, ans

        elif op in (QueryOperation.AVERAGE, QueryOperation.MIN, QueryOperation.MAX, QueryOperation.SUM):
            col = plan.column
            if not col or col not in settings.NUMERIC_COLUMNS:
                raise QueryExecutionError(f"Operation '{op.value}' requires a numeric column in {settings.NUMERIC_COLUMNS}. Given: '{col}'")

            series = df[col].dropna()
            if series.empty:
                return None, f"No numerical data available in '{col}' for the selected records."

            if op == QueryOperation.AVERAGE:
                val = float(round(series.mean(), 2))
                ans = f"The average {col.replace('_', ' ')} is {val:.2f} (computed across {len(series)} records)."
            elif op == QueryOperation.MIN:
                val = float(round(series.min(), 2))
                ans = f"The minimum {col.replace('_', ' ')} is {val:.2f}."
            elif op == QueryOperation.MAX:
                val = float(round(series.max(), 2))
                ans = f"The maximum {col.replace('_', ' ')} is {val:.2f}."
            elif op == QueryOperation.SUM:
                val = float(round(series.sum(), 2))
                ans = f"The sum of {col.replace('_', ' ')} is {val:.2f}."
            return val, ans

        elif op == QueryOperation.GROUP_COUNT:
            grp_col = plan.group_by
            if not grp_col or grp_col not in settings.VALID_COLUMNS:
                raise QueryExecutionError(f"Operation 'group_count' requires a valid group_by column in {settings.CATEGORICAL_COLUMNS}.")

            counts = df[grp_col].value_counts()
            if plan.sort_order == SortOrder.ASC:
                counts = counts.sort_values(ascending=True)
            else:
                counts = counts.sort_values(ascending=False)

            if plan.limit:
                counts = counts.head(plan.limit)

            result_dict = counts.to_dict()
            top_k = list(result_dict.items())
            if top_k:
                top_entity, top_count = top_k[0]
                ans = f"{top_entity} has the highest count with {top_count} tickets. (Group count by {grp_col}: {result_dict})"
            else:
                ans = f"Group count by {grp_col}: {result_dict}"
            return result_dict, ans

        elif op == QueryOperation.GROUP_AVERAGE:
            grp_col = plan.group_by
            target_col = plan.column or "customer_rating"

            if not grp_col or grp_col not in settings.VALID_COLUMNS:
                raise QueryExecutionError(f"Operation 'group_average' requires a valid group_by column in {settings.CATEGORICAL_COLUMNS}.")
            if target_col not in settings.NUMERIC_COLUMNS:
                raise QueryExecutionError(f"Target column '{target_col}' is not a valid numeric column for group_average.")

            valid_df = df.dropna(subset=[target_col])
            if valid_df.empty:
                return {}, f"No valid numerical values in '{target_col}' for group average."

            grouped = valid_df.groupby(grp_col)[target_col].agg(["mean", "count"])
            is_asc = plan.sort_order == SortOrder.ASC
            grouped = grouped.sort_values("mean", ascending=is_asc)

            if plan.limit:
                grouped = grouped.head(plan.limit)

            res_dict = {
                idx: {
                    "average": round(float(row["mean"]), 2),
                    "count": int(row["count"]),
                }
                for idx, row in grouped.iterrows()
            }

            if res_dict:
                first_k = list(res_dict.keys())[0]
                first_v = res_dict[first_k]["average"]
                adj = "lowest" if is_asc else "highest"
                ans = f"{first_k} has the {adj} average {target_col.replace('_', ' ')} with an average of {first_v:.2f}."
            else:
                ans = f"Group average for {target_col} by {grp_col} calculated."

            return res_dict, ans

        elif op == QueryOperation.LIST:
            target_df = df.copy()

            if plan.sort_by and plan.sort_by in target_df.columns:
                is_asc = plan.sort_order == SortOrder.ASC
                target_df = target_df.sort_values(plan.sort_by, ascending=is_asc)

            limit = plan.limit or 20
            target_df = target_df.head(limit)

            output_cols = [c for c in settings.VALID_COLUMNS if c in target_df.columns]
            records = target_df[output_cols].to_dict(orient="records")

            ans = f"Showing {len(records)} tickets matching the criteria."
            return records, ans

        raise QueryExecutionError(f"Unsupported query operation: '{op}'")

    def _summarize_filters(self, plan: QueryPlan) -> str:
        """Helper to create human-readable filter explanation."""
        parts = []
        if plan.filters:
            for k, v in plan.filters.items():
                if isinstance(v, list):
                    parts.append(f"{k} in {v}")
                else:
                    parts.append(f"{k} = '{v}'")
        if plan.filter_conditions:
            for cond in plan.filter_conditions:
                parts.append(f"{cond.field} {cond.operator.value} {cond.value}")
        return " and ".join(parts) if parts else ""


query_engine = QueryEngine()
