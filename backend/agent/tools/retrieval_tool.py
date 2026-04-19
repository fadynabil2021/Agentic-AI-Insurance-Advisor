"""
Retrieval Tool Node — queries Pinecone vector store for relevant packages,
rules, and knowledge snippets.
Embedding model: Google text-embedding-004 via Gemini API.
"""
import asyncio
from agent.state import AgentState
from clients.langfuse_client import safe_create_span
from config import settings


def get_compatible_budgets(budget: str) -> list[str]:
    """Expand budget tier to compatible tiers for metadata filter."""
    mapping = {
        "low":    ["low", "medium"],
        "medium": ["low", "medium", "high"],
        "high":   ["medium", "high"],
    }
    return mapping.get(budget.lower(), ["low", "medium", "high"])


def build_metadata_filter(entities: dict) -> dict | None:
    """Build Pinecone metadata filter from extracted entities."""
    filters: list[dict] = []

    budget = entities.get("budget")
    if budget:
        compatible = get_compatible_budgets(budget)
        filters.append({"budget_tier": {"$in": compatible}})

    if len(filters) == 0:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def build_retrieval_query(entities: dict) -> str:
    """Construct a natural-language retrieval query from extracted entities."""
    parts = []
    if entities.get("industry"):
        parts.append(f"{entities['industry']} industry")
    if entities.get("region"):
        parts.append(f"region {entities['region']}")
    if entities.get("budget"):
        parts.append(f"{entities['budget']} budget")
    if entities.get("priority"):
        parts.append(f"priority {entities['priority']}")
    if not parts:
        parts = ["insurance plan recommendation"]
    return "insurance plan for " + " ".join(parts)


def merge_retrieval_results(packages: list, rules: list, snippets: list) -> list[dict]:
    """Merge results from multiple Pinecone queries into a flat list."""
    merged = []

    # Add package documents
    for match in packages:
        metadata = match.get("metadata", {})
        merged.append({
            "type": "package",
            "id": match.get("id", ""),
            "content": metadata.get("content", ""),
            "name": metadata.get("name", match.get("id", "")),
            "network": metadata.get("network", "B"),
            "price_range": metadata.get("price_range", [0, 0]),
            "coverage": metadata.get("coverage", "Medium"),
            "budget_tier": metadata.get("budget_tier", "medium"),
            "metadata": metadata,
            "score": match.get("score", 0),
        })

    # Add benchmark rules
    for match in rules:
        metadata = match.get("metadata", {})
        merged.append({
            "type": "rule",
            "id": match.get("id", ""),
            "content": metadata.get("content", ""),
            "metadata": metadata,
            "score": match.get("score", 0),
        })

    # Add knowledge snippets
    for match in snippets:
        metadata = match.get("metadata", {})
        merged.append({
            "type": "snippet",
            "id": match.get("id", ""),
            "content": metadata.get("content", ""),
            "metadata": metadata,
            "score": match.get("score", 0),
        })

    return merged


async def retrieval_tool_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Node 4: Query Pinecone for relevant packages and rules.
    Uses cosine similarity + metadata filters.
    """
    span = safe_create_span(langfuse_trace, "retrieval_tool", {
        "entities": state.get("extracted_entities"),
    })

    state["tools_used"].append("retrieval_tool")

    # Track retries
    current_retry = state.get("retry_count", 0)
    if current_retry > 0 or "retrieval_tool" in state["tools_used"][:-1]:
        state["retry_count"] = current_retry + 1

    entities = state.get("extracted_entities") or {}
    query = build_retrieval_query(entities)
    state["retrieval_query"] = query

    # Build metadata filter
    meta_filter = build_metadata_filter(entities)

    async def _do_query(gemini_client, pinecone_index):
        """Query Pinecone with Gemini embeddings."""
        if pinecone_index is None:
            return {"packages": [], "rules": [], "snippets": []}

        # Generate embedding for query
        try:
            query_embedding = await gemini_client.embed(query)
        except Exception:
            return {"packages": [], "rules": [], "snippets": []}

        results = {}

        # Query packages namespace
        try:
            packages_result = await query_vectors_async(
                pinecone_index,
                query_vector=query_embedding,
                top_k=3,
                namespace="packages",
                filter_dict=meta_filter,
            )
            results["packages"] = packages_result.get("matches", []) if packages_result else []
        except Exception:
            results["packages"] = []

        # Query benchmark_rules namespace
        try:
            rules_result = await query_vectors_async(
                pinecone_index,
                query_vector=query_embedding,
                top_k=5,
                namespace="benchmark_rules",
            )
            results["rules"] = rules_result.get("matches", []) if rules_result else []
        except Exception:
            results["rules"] = []

        # Query knowledge_snippets namespace
        try:
            snippets_result = await query_vectors_async(
                pinecone_index,
                query_vector=query_embedding,
                top_k=3,
                namespace="knowledge_snippets",
            )
            results["snippets"] = snippets_result.get("matches", []) if snippets_result else []
        except Exception:
            results["snippets"] = []

        return results

    # Get clients
    from clients.gemini_client import GeminiClient
    from clients.pinecone_client import get_pinecone_client, get_or_create_index, query_vectors_async

    gemini_client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
        timeout=settings.GEMINI_TIMEOUT,
    )

    pinecone_client = get_pinecone_client(settings.PINECONE_API_KEY)
    pinecone_index = None

    if pinecone_client:
        pinecone_index = await get_or_create_index(
            pinecone_client,
            settings.PINECONE_INDEX_NAME,
            dimension=3072,  # gemini-embedding-001 dimension
        )

    try:
        raw = await _do_query(gemini_client, pinecone_index)
        merged = merge_retrieval_results(
            raw.get("packages", []),
            raw.get("rules", []),
            raw.get("snippets", []),
        )
    except Exception as e:
        state["execution_trace"].append(f"retrieval_tool: ERROR — {e}")
        state["retrieval_results"] = []
        if span:
            try:
                span.end(output={"error": str(e)})
            except Exception:
                pass
        return state

    state["retrieval_results"] = merged
    pkg_count = sum(1 for r in merged if r["type"] == "package")
    rule_count = sum(1 for r in merged if r["type"] == "rule")

    if not merged:
        state["execution_trace"].append(
            f"retrieval_tool: WARNING — no results found for query='{query}'"
        )
    else:
        state["execution_trace"].append(
            f"retrieval_tool: returned {len(merged)} documents "
            f"(packages={pkg_count}, rules={rule_count})"
        )

    if span:
        try:
            span.end(output={"total": len(merged), "packages": pkg_count, "rules": rule_count})
        except Exception:
            pass

    return state
