"""
Retrieval Tool Node — queries ChromaDB vector store for relevant packages,
rules, and knowledge snippets.
Embedding model: nomic-embed-text via Ollama.
"""
import asyncio
from agent.state import AgentState
from clients.langfuse_client import safe_create_span
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from config import settings


def get_compatible_budgets(budget: str) -> list[str]:
    """Expand budget tier to compatible tiers for metadata filter."""
    mapping = {
        "low":    ["low", "medium"],
        "medium": ["low", "medium", "high"],
        "high":   ["medium", "high"],
    }
    return mapping.get(budget.lower(), ["low", "medium", "high"])


def build_metadata_filter(entities: dict) -> dict:
    """Build ChromaDB metadata $where filter from extracted entities."""
    filters: list[dict] = []

    budget = entities.get("budget")
    if budget:
        compatible = get_compatible_budgets(budget)
        filters.append({"budget_tier": {"$in": compatible}})

    if len(filters) == 0:
        return {}
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


def merge_retrieval_results(packages: dict, rules: dict, snippets: dict) -> list[dict]:
    """Merge results from multiple collections into a flat list."""
    merged = []

    # Add package documents
    if packages and packages.get("ids") and packages["ids"][0]:
        for i, doc_id in enumerate(packages["ids"][0]):
            doc = packages["documents"][0][i] if packages.get("documents") else ""
            meta = packages["metadatas"][0][i] if packages.get("metadatas") else {}
            merged.append({
                "type": "package",
                "id": doc_id,
                "content": doc,
                "name": meta.get("name", doc_id),
                "network": meta.get("network", "B"),
                "price_range": meta.get("price_range", [0, 0]),
                "coverage": meta.get("coverage", "Medium"),
                "budget_tier": meta.get("budget_tier", "medium"),
                "metadata": meta,
            })

    # Add benchmark rules
    if rules and rules.get("ids") and rules["ids"][0]:
        for i, doc_id in enumerate(rules["ids"][0]):
            doc = rules["documents"][0][i] if rules.get("documents") else ""
            meta = rules["metadatas"][0][i] if rules.get("metadatas") else {}
            merged.append({
                "type": "rule",
                "id": doc_id,
                "content": doc,
                "metadata": meta,
            })

    # Add knowledge snippets
    if snippets and snippets.get("ids") and snippets["ids"][0]:
        for i, doc_id in enumerate(snippets["ids"][0]):
            doc = snippets["documents"][0][i] if snippets.get("documents") else ""
            meta = snippets["metadatas"][0][i] if snippets.get("metadatas") else {}
            merged.append({
                "type": "snippet",
                "id": doc_id,
                "content": doc,
                "metadata": meta,
            })

    return merged


async def retrieval_tool_node(state: AgentState, langfuse_trace) -> AgentState:
    """
    Node 4: Query ChromaDB for relevant packages and rules.
    Uses cosine similarity + metadata filters.
    """
    span = safe_create_span(langfuse_trace, "retrieval_tool", {
        "entities": state.get("extracted_entities"),
    })

    state["tools_used"].append("retrieval_tool")

    # Track retries: first invocation is retry_count=0, subsequent are 1, 2, ...
    # This ensures MAX_RETRIES=2 allows exactly 2 retries after the initial call.
    current_retry = state.get("retry_count", 0)
    if current_retry > 0 or "retrieval_tool" in state["tools_used"][:-1]:
        state["retry_count"] = current_retry + 1
    # else: first call, keep retry_count at 0

    entities = state.get("extracted_entities") or {}
    query = build_retrieval_query(entities)
    state["retrieval_query"] = query

    # Build metadata filter
    meta_filter = build_metadata_filter(entities)

    def _do_query():
        # Reuse singleton client from chroma_client module (avoids per-call overhead)
        from clients.chroma_client import get_chroma_client
        client = get_chroma_client()
        ef = OllamaEmbeddingFunction(
            model_name=settings.OLLAMA_EMBED_MODEL,
            url=f"{settings.OLLAMA_HOST}/api/embeddings",
        )
        results = {}

        # Query packages collection
        try:
            pkg_col = client.get_collection("packages", embedding_function=ef)
            pkg_kwargs: dict = {"query_texts": [query], "n_results": 3}
            if meta_filter:
                pkg_kwargs["where"] = meta_filter
            results["packages"] = pkg_col.query(**pkg_kwargs)
        except Exception:
            results["packages"] = {}

        # Query benchmark_rules collection
        try:
            rules_col = client.get_collection("benchmark_rules", embedding_function=ef)
            results["rules"] = rules_col.query(query_texts=[query], n_results=5)
        except Exception:
            results["rules"] = {}

        # Query knowledge_snippets collection
        try:
            snip_col = client.get_collection("knowledge_snippets", embedding_function=ef)
            results["snippets"] = snip_col.query(query_texts=[query], n_results=3)
        except Exception:
            results["snippets"] = {}

        return results

    try:
        raw = await asyncio.to_thread(_do_query)
        merged = merge_retrieval_results(
            raw.get("packages", {}),
            raw.get("rules", {}),
            raw.get("snippets", {}),
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
