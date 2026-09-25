"""
Analytics helper module for SupportIQ.
Computes aggregated metrics, distributions, and agent leaderboards.
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from app.data_loader import data_loader
from app.schemas import TicketSummaryStats
from app.utils import sanitize_for_json


class AnalyticsEngine:
    """Computes summary statistics and visualizations for the API and dashboard."""

    def __init__(self, loader=None):
        self.loader = loader or data_loader

    def get_kpi_summary(self) -> TicketSummaryStats:
        """Returns executive KPI statistics."""
        df = self.loader.get_data()

        status_counts = df["status"].value_counts().to_dict()
        priority_counts = df["priority"].value_counts().to_dict()
        category_counts = df["category"].value_counts().to_dict()

        avg_rating = df["customer_rating"].dropna().mean()
        avg_resol = df["resolution_time_hrs"].dropna().mean()
        avg_resp = df["response_time_hrs"].dropna().mean()

        return TicketSummaryStats(
            total_tickets=len(df),
            open_tickets=status_counts.get("Open", 0),
            escalated_tickets=status_counts.get("Escalated", 0),
            resolved_tickets=status_counts.get("Resolved", 0),
            critical_tickets=priority_counts.get("Critical", 0),
            avg_customer_rating=round(float(avg_rating), 2) if pd.notna(avg_rating) else None,
            avg_resolution_time_hrs=round(float(avg_resol), 2) if pd.notna(avg_resol) else None,
            avg_response_time_hrs=round(float(avg_resp), 2) if pd.notna(avg_resp) else None,
            categories=category_counts,
            priorities=priority_counts,
            statuses=status_counts,
        )

    def get_agent_performance(self) -> List[Dict[str, Any]]:
        """Returns detailed agent performance metrics."""
        df = self.loader.get_data()

        records = []
        for agent_id, group in df.groupby("agent_id"):
            total = len(group)
            resolved = len(group[group["status"] == "Resolved"])
            open_cnt = len(group[group["status"] == "Open"])
            escalated = len(group[group["status"] == "Escalated"])

            ratings = group["customer_rating"].dropna()
            avg_rating = round(float(ratings.mean()), 2) if not ratings.empty else None

            resols = group["resolution_time_hrs"].dropna()
            avg_resol = round(float(resols.mean()), 1) if not resols.empty else None

            resps = group["response_time_hrs"].dropna()
            avg_resp = round(float(resps.mean()), 1) if not resps.empty else None

            records.append({
                "agent_id": agent_id,
                "total_tickets": total,
                "resolved": resolved,
                "open": open_cnt,
                "escalated": escalated,
                "avg_rating": avg_rating,
                "avg_resolution_hrs": avg_resol,
                "avg_response_hrs": avg_resp,
            })

        records.sort(key=lambda x: (x["avg_rating"] is None, x["avg_rating"]))
        return sanitize_for_json(records)

    def get_category_metrics(self) -> Dict[str, Any]:
        """Category breakdowns and average customer rating."""
        df = self.loader.get_data()
        res = {}
        for cat, group in df.groupby("category"):
            res[cat] = {
                "total": len(group),
                "resolved": len(group[group["status"] == "Resolved"]),
                "avg_rating": round(float(group["customer_rating"].dropna().mean()), 2),
                "avg_resolution_hrs": round(float(group["resolution_time_hrs"].dropna().mean()), 1),
            }
        return sanitize_for_json(res)


analytics_engine = AnalyticsEngine()
