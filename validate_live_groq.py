"""
End-to-end validation of the 7 core questions using LIVE GROQ LLM API.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.llm_engine import GroqLLMProvider
from app.query_engine import query_engine

print("=" * 75)
print(f"=== SUPPORTIQ LIVE GROQ VERIFICATION (Model: {settings.GROQ_MODEL}) ===")
print("=" * 75 + "\n")

planner = GroqLLMProvider(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)

questions = [
    "How many tickets are currently open?",
    "Which agent has the lowest average customer rating?",
    "Which agent resolved the most tickets?",
    "What is the average customer rating for Technical tickets?",
    "Show all Critical tickets not resolved within 12 hours.",
    "Are there any anomalies in resolution times?",
    "What is the company's annual revenue?",
]

all_passed = True

for i, q in enumerate(questions, 1):
    print(f"Q{i}: \"{q}\"")
    try:
        plan = planner.generate_plan(q)
        result, meta, answer = query_engine.execute_plan(plan)
        print(f"  [Plan] Operation: {plan.operation.value}, Supported: {plan.is_supported}")
        if plan.filters:
            print(f"  [Filters] {plan.filters}")
        if plan.filter_conditions:
            print(f"  [Conditions] {[c.model_dump() for c in plan.filter_conditions]}")
        print(f"  [Answer] {answer}")
        if isinstance(result, list):
            print(f"  [Result] {len(result)} records returned.")
        elif isinstance(result, dict):
            print(f"  [Result] {result}")
        else:
            print(f"  [Result] {result}")
        print("  Status: SUCCESS")
    except Exception as e:
        print(f"  Status: FAILED -> {e}")
        all_passed = False
    print("-" * 75)

if all_passed:
    print("\nALL 7 CORE QUESTIONS PROCESSED SUCCESSFULLY VIA LIVE GROQ API!")
else:
    print("\nSOME QUESTIONS FAILED.")
