"""
Validation script for the 7 core assessment questions.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.llm_engine import get_llm_provider
from app.query_engine import query_engine
from app.config import settings

# Use Mock provider for deterministic offline run
settings.LLM_PROVIDER = "mock"

questions = [
    "How many tickets are currently open?",
    "Which agent has the lowest average customer rating?",
    "Which agent resolved the most tickets?",
    "What is the average customer rating for Technical tickets?",
    "Show all Critical tickets not resolved within 12 hours.",
    "Are there any anomalies in resolution times?",
    "What is the company's annual revenue?",
]

planner = get_llm_provider()
print("=" * 70)
print("=== SUPPORTIQ CORE QUESTIONS VERIFICATION ===")
print("=" * 70 + "\n")

for i, q in enumerate(questions, 1):
    plan = planner.generate_plan(q)
    result, meta, answer = query_engine.execute_plan(plan)
    print(f"Q{i}: \"{q}\"")
    print(f"  Plan: operation={plan.operation.value}, is_supported={plan.is_supported}")
    if plan.filters:
        print(f"  Filters: {plan.filters}")
    if plan.filter_conditions:
        print(f"  Conditions: {[c.model_dump() for c in plan.filter_conditions]}")
    print(f"  Answer: {answer}")
    if isinstance(result, list):
        print(f"  Result summary: {len(result)} records returned.")
    elif isinstance(result, dict):
        print(f"  Result summary: {result}")
    else:
        print(f"  Result value: {result}")
    print("-" * 70)
