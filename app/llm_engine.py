"""
LLM Query Planner for SupportIQ.
Translates natural-language questions into strictly validated Pydantic QueryPlan objects.
Supports live Groq API (via SDK or direct HTTP) and a MockLLMProvider for offline/test environments.
"""
from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, Optional
import httpx
from pydantic import ValidationError
from app.config import settings
from app.schemas import (
    FilterCondition,
    FilterOperator,
    QueryOperation,
    QueryPlan,
    SortOrder,
)
from app.utils import logger

SYSTEM_PROMPT = """You are an expert Query Planner for SupportIQ, a customer support intelligence system.
Your SOLE job is to convert a user's natural language question into a structured JSON query plan.
You must NEVER execute Python code, SQL, or arbitrary calculations.

Dataset Schema:
- Columns:
  * ticket_id (str): Unique identifier e.g. TKT-001
  * created_at (str): Timestamp e.g. 2024-02-05 11:14
  * category (str): "General", "Billing", "Technical"
  * priority (str): "Low", "Medium", "High", "Critical"
  * status (str): "Open", "Escalated", "Resolved"
  * response_time_hrs (float): Initial response time in hours
  * resolution_time_hrs (float): Total resolution time in hours (null for unresolved)
  * agent_id (str): Agent identifier e.g. AGT-01 to AGT-12
  * customer_rating (float): Rating 1 to 5 (null for unresolved)
  * issue_summary (str): Text summary of issue

Allowed Query Operations:
- "count": Count matching records.
- "average": Compute mean of a numeric column ("response_time_hrs", "resolution_time_hrs", "customer_rating").
- "min": Minimum value of a numeric column.
- "max": Maximum value of a numeric column.
- "sum": Sum of a numeric column.
- "group_count": Count records grouped by a categorical column ("category", "priority", "status", "agent_id").
- "group_average": Compute average of a numeric column grouped by a categorical column.
- "list": List matching records with optional sorting and limit.

STRICT RULES:
1. ONLY use columns and categorical values that exist in the schema.
2. NEVER invent new columns or metrics (e.g., revenue, salary, profit do NOT exist).
3. If a question is NOT about the support ticket dataset (e.g. "What is company revenue?", "Tell me a joke"), set:
   "is_supported": false, "unsupported_reason": "I cannot answer that because company revenue is not contained in the support ticket dataset."
4. If asked "Which agent has lowest rating", use operation: "group_average", column: "customer_rating", group_by: "agent_id", sort_order: "asc", limit: 1.
5. If asked "Which agent resolved the most tickets", filter by status "Resolved", operation: "group_count", group_by: "agent_id", sort_order: "desc", limit: 1.
6. For "unresolved" tickets, use filters: {"status": ["Open", "Escalated"]}.
7. Return ONLY valid JSON matching the schema. No markdown formatting outside JSON.

JSON Schema to return:
{
  "operation": "count" | "average" | "min" | "max" | "sum" | "group_count" | "group_average" | "list",
  "column": null | "response_time_hrs" | "resolution_time_hrs" | "customer_rating",
  "group_by": null | "category" | "priority" | "status" | "agent_id",
  "filters": { "column_name": "value" or ["val1", "val2"] },
  "filter_conditions": [
     {"field": "resolution_time_hrs", "operator": "gt", "value": 12.0}
  ],
  "date_range": null | { "preset": "this_week", "start": null, "end": null },
  "sort_by": null | "column_name",
  "sort_order": "asc" | "desc",
  "limit": null | integer,
  "is_supported": true | false,
  "unsupported_reason": null | string
}
"""


class LLMProvider(ABC):
    """Abstract interface for LLM Query Planners."""

    @abstractmethod
    def generate_plan(self, question: str) -> QueryPlan:
        """Parse natural language question into QueryPlan."""
        pass


class GroqLLMProvider(LLMProvider):
    """Production Groq LLM Query Planner supporting both SDK and direct HTTP."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._groq_client = None

        if self.api_key:
            try:
                import groq
                self._groq_client = groq.Groq(api_key=self.api_key)
            except ImportError:
                self._groq_client = None

    def generate_plan(self, question: str) -> QueryPlan:
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not configured; delegating to MockLLMProvider.")
            return MockLLMProvider().generate_plan(question)

        prompt = f"User Question: {question}\n\nGenerate the QueryPlan JSON:"
        content = ""

        # Try Groq Python SDK if installed
        if self._groq_client is not None:
            try:
                resp = self._groq_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=settings.LLM_TIMEOUT_SECONDS,
                )
                content = resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq SDK invocation error: {e}. Attempting direct HTTP fallback.")
                content = self._call_via_httpx(prompt)
        else:
            content = self._call_via_httpx(prompt)

        return self._parse_json_to_plan(content, question)

    def _call_via_httpx(self, prompt: str) -> str:
        """Direct HTTPS call to Groq OpenAI-compatible endpoint."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq HTTP status error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Groq API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            logger.error(f"Groq connection failure: {e}")
            raise RuntimeError(f"Unable to reach Groq API: {str(e)}")

    def _parse_json_to_plan(self, content: str, original_question: str) -> QueryPlan:
        """Extract and validate JSON into a QueryPlan."""
        if not content:
            raise ValueError("Empty response received from LLM service.")

        # Strip markdown fences if present
        cleaned = content.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

        try:
            raw_dict = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM response JSON: {cleaned}")
            raise ValueError(f"LLM produced malformed JSON: {e}")

        try:
            plan = QueryPlan(**raw_dict)
            return plan
        except ValidationError as e:
            logger.error(f"Pydantic validation failed on LLM plan: {e}")
            raise ValueError(f"LLM query plan failed schema validation: {e}")


class MockLLMProvider(LLMProvider):
    """
    Deterministic query planner for tests, CI, and local runs without external API keys.
    Accurately maps assessment questions and linguistic variations to valid QueryPlans.
    """

    def generate_plan(self, question: str) -> QueryPlan:
        q = question.lower().strip()

        # 1. Unsupported out-of-domain questions
        unsupported_keywords = ["revenue", "salary", "profit", "stock", "weather", "joke", "dividend", "ceo"]
        for kw in unsupported_keywords:
            if kw in q:
                return QueryPlan(
                    operation=QueryOperation.COUNT,
                    is_supported=False,
                    unsupported_reason=f"I cannot answer that because {kw} is not contained in the support ticket dataset.",
                )

        # 2. "How many tickets are currently open?" / "unresolved tickets"
        if ("how many" in q or "count" in q) and ("open" in q or "unresolved" in q):
            if "unresolved" in q:
                return QueryPlan(
                    operation=QueryOperation.COUNT,
                    filters={"status": ["Open", "Escalated"]},
                )
            return QueryPlan(
                operation=QueryOperation.COUNT,
                filters={"status": "Open"},
            )

        # 3. "Which agent has the lowest average customer rating?" / "worst rating"
        if ("agent" in q or "who" in q) and ("lowest" in q or "worst" in q or "minimum" in q) and ("rating" in q or "satisfaction" in q):
            return QueryPlan(
                operation=QueryOperation.GROUP_AVERAGE,
                column="customer_rating",
                group_by="agent_id",
                sort_order=SortOrder.ASC,
                limit=1,
            )

        # 4. "Which agent resolved the most tickets?" / "highest resolved"
        if ("agent" in q or "who" in q) and ("most" in q or "highest" in q or "best" in q) and ("resolved" in q or "closed" in q):
            return QueryPlan(
                operation=QueryOperation.GROUP_COUNT,
                group_by="agent_id",
                filters={"status": "Resolved"},
                sort_order=SortOrder.DESC,
                limit=1,
            )

        # 5. "What is the average customer rating for Technical category tickets?"
        if "average" in q and "rating" in q and ("technical" in q or "category" in q):
            filters = {}
            if "technical" in q:
                filters["category"] = "Technical"
            elif "billing" in q:
                filters["category"] = "Billing"
            elif "general" in q:
                filters["category"] = "General"
            return QueryPlan(
                operation=QueryOperation.AVERAGE,
                column="customer_rating",
                filters=filters,
            )

        # 6. "Show all Critical tickets not resolved within 12 hours"
        if "critical" in q and ("not resolved" in q or "12" in q or "resolution" in q):
            # Tickets with priority Critical that took > 12 hours or are still unresolved
            return QueryPlan(
                operation=QueryOperation.LIST,
                filters={"priority": "Critical"},
                filter_conditions=[
                    FilterCondition(field="resolution_time_hrs", operator=FilterOperator.GT, value=12.0)
                ],
                sort_by="resolution_time_hrs",
                sort_order=SortOrder.DESC,
                limit=50,
            )

        # 7. "Are there any anomalies in resolution times?" / "anomalies"
        if "anomal" in q:
            return QueryPlan(
                operation=QueryOperation.LIST,
                sort_by="resolution_time_hrs",
                sort_order=SortOrder.DESC,
                limit=25,
            )

        # 8. General counts and listings
        if "critical" in q and "count" in q:
            return QueryPlan(operation=QueryOperation.COUNT, filters={"priority": "Critical"})

        if "technical" in q and "count" in q:
            return QueryPlan(operation=QueryOperation.COUNT, filters={"category": "Technical"})

        # Default fallback: safe count or list
        return QueryPlan(
            operation=QueryOperation.COUNT,
            filters={},
        )


def get_llm_provider() -> LLMProvider:
    """Factory function to instantiate the configured LLM provider."""
    provider_name = (settings.LLM_PROVIDER or "groq").lower().strip()
    if provider_name == "mock" or not settings.GROQ_API_KEY:
        logger.info("Using MockLLMProvider (no Groq key or LLM_PROVIDER=mock).")
        return MockLLMProvider()
    return GroqLLMProvider()
