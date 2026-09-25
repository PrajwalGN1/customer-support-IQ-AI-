"""
Unit tests for query_engine.py
Validates safe deterministic execution of counts, aggregations, groupings, listings, and filters.
"""
import pytest
from app.query_engine import QueryEngine, QueryExecutionError
from app.schemas import (
    FilterCondition,
    FilterOperator,
    QueryOperation,
    QueryPlan,
    SortOrder,
)


@pytest.fixture
def engine():
    return QueryEngine()


def test_count_all_tickets(engine):
    plan = QueryPlan(operation=QueryOperation.COUNT)
    result, meta, answer = engine.execute_plan(plan)

    assert result == 500
    assert meta["matched_rows"] == 500
    assert "500" in answer


def test_count_open_tickets(engine):
    plan = QueryPlan(
        operation=QueryOperation.COUNT,
        filters={"status": "Open"},
    )
    result, meta, answer = engine.execute_plan(plan)

    assert result == 111
    assert meta["matched_rows"] == 111
    assert "111" in answer


def test_count_unresolved_tickets_list_filter(engine):
    plan = QueryPlan(
        operation=QueryOperation.COUNT,
        filters={"status": ["Open", "Escalated"]},
    )
    result, meta, answer = engine.execute_plan(plan)

    assert result == 173
    assert meta["matched_rows"] == 173


def test_average_rating_technical_tickets(engine):
    plan = QueryPlan(
        operation=QueryOperation.AVERAGE,
        column="customer_rating",
        filters={"category": "Technical"},
    )
    result, meta, answer = engine.execute_plan(plan)

    assert result == pytest.approx(3.74, rel=1e-2)
    assert meta["matched_rows"] == 152
    assert "3.74" in answer


def test_group_average_lowest_rated_agent(engine):
    plan = QueryPlan(
        operation=QueryOperation.GROUP_AVERAGE,
        column="customer_rating",
        group_by="agent_id",
        sort_order=SortOrder.ASC,
        limit=1,
    )
    result, meta, answer = engine.execute_plan(plan)

    assert "AGT-08" in result
    assert result["AGT-08"]["average"] == pytest.approx(3.48, rel=1e-2)
    assert "AGT-08" in answer
    assert "3.48" in answer


def test_group_count_most_resolved_agent(engine):
    plan = QueryPlan(
        operation=QueryOperation.GROUP_COUNT,
        group_by="agent_id",
        filters={"status": "Resolved"},
        sort_order=SortOrder.DESC,
        limit=1,
    )
    result, meta, answer = engine.execute_plan(plan)

    # Both AGT-09 and AGT-12 have 37 resolved tickets
    top_agent = list(result.keys())[0]
    assert result[top_agent] == 37
    assert top_agent in ["AGT-09", "AGT-12"]


def test_filter_condition_numeric_comparison(engine):
    plan = QueryPlan(
        operation=QueryOperation.LIST,
        filters={"priority": "Critical", "status": "Resolved"},
        filter_conditions=[
            FilterCondition(field="resolution_time_hrs", operator=FilterOperator.GT, value=12.0)
        ],
        sort_by="resolution_time_hrs",
        sort_order=SortOrder.DESC,
    )
    result, meta, answer = engine.execute_plan(plan)

    assert isinstance(result, list)
    assert len(result) == 3
    # IDs should be TKT-255, TKT-446, TKT-238
    ticket_ids = {r["ticket_id"] for r in result}
    assert ticket_ids == {"TKT-255", "TKT-446", "TKT-238"}


def test_unsupported_plan_handling(engine):
    plan = QueryPlan(
        operation=QueryOperation.COUNT,
        is_supported=False,
        unsupported_reason="Revenue is not tracked in support tickets.",
    )
    result, meta, answer = engine.execute_plan(plan)

    assert result is None
    assert "Revenue is not tracked" in answer


def test_zero_match_handling(engine):
    plan = QueryPlan(
        operation=QueryOperation.COUNT,
        filters={"category": "Technical", "priority": "Critical", "status": "NonExistentStatus"},
    )
    result, meta, answer = engine.execute_plan(plan)

    assert result is None
    assert "No tickets matched" in answer
