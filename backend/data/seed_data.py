"""
ChromaDB seed script — populates all 4 knowledge base collections.
Run via: docker compose run --rm backend python data/seed_data.py
Or with --reset flag to wipe and re-seed.
"""
import os
import sys
import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ─── Package Catalog (3 packages) ────────────────────────────────────────────

PACKAGES = [
    {
        "id": "pkg_basic",
        "document": (
            "Basic Insurance Plan: Network C provider network. "
            "Price range 4000-5000 SAR per employee annually. "
            "Coverage level: basic medical, emergency, and outpatient services. "
            "Suitable for low-risk industries with budget-conscious employers. "
            "Best for retail and education sectors with small employee counts."
        ),
        "metadata": {
            "name": "Basic",
            "network": "C",
            "price_range": "[4000, 5000]",
            "coverage": "Basic",
            "budget_tier": "low",
            "type": "package",
        },
    },
    {
        "id": "pkg_standard",
        "document": (
            "Standard Insurance Plan: Network B provider network. "
            "Price range 6000-7500 SAR per employee annually. "
            "Coverage level: medium — includes inpatient, outpatient, dental, and optical. "
            "Recommended for medium-risk industries. "
            "Well-suited for construction, manufacturing, and retail companies. "
            "Balanced cost-to-coverage ratio for most business profiles."
        ),
        "metadata": {
            "name": "Standard",
            "network": "B",
            "price_range": "[6000, 7500]",
            "coverage": "Medium",
            "budget_tier": "medium",
            "type": "package",
        },
    },
    {
        "id": "pkg_premium",
        "document": (
            "Premium Insurance Plan: Network A provider network — top-tier hospitals. "
            "Price range 9000-12000 SAR per employee annually. "
            "Coverage level: high — includes all Standard benefits plus maternity, "
            "chronic disease, specialist referrals, and international coverage. "
            "Recommended for high-risk industries such as healthcare. "
            "Suitable for companies with high dependents ratio or premium requirements."
        ),
        "metadata": {
            "name": "Premium",
            "network": "A",
            "price_range": "[9000, 12000]",
            "coverage": "High",
            "budget_tier": "high",
            "type": "package",
        },
    },
]

# ─── Benchmark Rules (12 rules) ───────────────────────────────────────────────

BENCHMARK_RULES = [
    {
        "id": "rule_industry_healthcare",
        "document": (
            "Healthcare industry risk classification: HIGH. "
            "Healthcare companies require minimum Network B coverage. "
            "Basic plan is not suitable for healthcare due to high clinical risk exposure. "
            "Premium is recommended when dependents ratio exceeds 0.50."
        ),
        "metadata": {"rule_type": "industry_risk", "industry": "healthcare", "risk": "high"},
    },
    {
        "id": "rule_industry_construction",
        "document": (
            "Construction industry risk classification: MEDIUM. "
            "Construction companies benefit from Standard plan with occupational coverage. "
            "Basic plan is acceptable for small construction firms with low claim history. "
            "Premium is typically over-specification for standard construction workforce."
        ),
        "metadata": {"rule_type": "industry_risk", "industry": "construction", "risk": "medium"},
    },
    {
        "id": "rule_industry_retail",
        "document": (
            "Retail industry risk classification: MEDIUM-LOW. "
            "Retail companies typically have standard health risks. "
            "Standard plan is most appropriate for retail workforce. "
            "Premium may be over-specification unless executive benefits are required."
        ),
        "metadata": {"rule_type": "industry_risk", "industry": "retail", "risk": "medium-low"},
    },
    {
        "id": "rule_region_riyadh",
        "document": (
            "Riyadh regional cost pressure: HIGH (score 3). "
            "Riyadh has the highest medical cost index in Saudi Arabia. "
            "Premium plans in Riyadh carry the highest total cost burden. "
            "Medium-budget companies in Riyadh should prioritize Standard over Premium."
        ),
        "metadata": {"rule_type": "region_cost", "region": "riyadh", "cost_pressure": 3},
    },
    {
        "id": "rule_region_dammam",
        "document": (
            "Dammam regional cost pressure: MEDIUM (score 2). "
            "Dammam has moderate medical costs, between Riyadh and Jeddah. "
            "Standard plan is recommended for medium-budget companies in Dammam. "
            "Premium is viable for high-budget or high-risk industries."
        ),
        "metadata": {"rule_type": "region_cost", "region": "dammam", "cost_pressure": 2},
    },
    {
        "id": "rule_region_jeddah",
        "document": (
            "Jeddah regional cost pressure: LOW (score 1). "
            "Jeddah has the lowest regional medical cost index. "
            "Basic and Standard plans are viable for most industries in Jeddah. "
            "Cost savings from lower regional rates make Standard highly attractive."
        ),
        "metadata": {"rule_type": "region_cost", "region": "jeddah", "cost_pressure": 1},
    },
    {
        "id": "rule_budget_low",
        "document": (
            "Low budget constraint: recommend Basic or Standard plans only. "
            "Premium plans are not compatible with low-budget profiles. "
            "Budget-constrained companies should prioritize Basic unless industry risk requires Standard."
        ),
        "metadata": {"rule_type": "budget", "budget": "low"},
    },
    {
        "id": "rule_budget_medium",
        "document": (
            "Medium budget constraint: Standard is the primary recommendation. "
            "Basic is acceptable if cost minimization is the primary priority. "
            "Premium is possible but may strain budget in high-cost regions like Riyadh."
        ),
        "metadata": {"rule_type": "budget", "budget": "medium"},
    },
    {
        "id": "rule_budget_high",
        "document": (
            "High budget: Standard and Premium plans are both viable. "
            "Premium recommended for high-risk industries or companies requiring Network A hospitals. "
            "Basic is not recommended for high-budget profiles."
        ),
        "metadata": {"rule_type": "budget", "budget": "high"},
    },
    {
        "id": "rule_cheapest",
        "document": (
            "Cheapest acceptable option scoring rule: "
            "When the customer requests cheapest acceptable option, apply cost-priority scoring. "
            "Basic gets +20 bonus if industry risk allows. "
            "Standard gets neutral treatment. "
            "Premium receives -30 penalty regardless of other factors."
        ),
        "metadata": {"rule_type": "priority", "priority": "cheapest"},
    },
    {
        "id": "rule_dependents_ratio",
        "document": (
            "Dependents ratio rule: when dependents ratio exceeds 0.50, "
            "Basic plan is penalized by 15 points due to increased family benefits cost. "
            "Standard is recommended as minimum when dependents ratio is high. "
            "Premium should be evaluated when dependents ratio exceeds 0.70."
        ),
        "metadata": {"rule_type": "dependents", "threshold": 0.5},
    },
    {
        "id": "rule_compare",
        "document": (
            "Package comparison rules: "
            "When comparing Standard vs Premium: evaluate on 6 dimensions — "
            "network quality, price range, coverage level, score, budget fit, risk fit. "
            "Standard typically wins on cost efficiency; Premium wins on coverage depth. "
            "Retail and construction industries favor Standard unless specific network A requirements exist."
        ),
        "metadata": {"rule_type": "comparison"},
    },
]

# ─── Knowledge Snippets (4 snippets) ─────────────────────────────────────────

KNOWLEDGE_SNIPPETS = [
    {
        "id": "snip_network_a",
        "document": (
            "Network A (Premium): Top-tier hospital network. "
            "Includes all major private hospitals in Saudi Arabia. "
            "Shortest referral times, highest specialist access. "
            "Required for healthcare industry workers with clinical exposure."
        ),
        "metadata": {"type": "snippet", "topic": "network_a"},
    },
    {
        "id": "snip_network_b",
        "document": (
            "Network B (Standard): Mid-tier hospital network. "
            "Covers all major public hospitals and selected private clinics. "
            "Appropriate for medium-risk industries. "
            "Good balance of coverage breadth and cost efficiency."
        ),
        "metadata": {"type": "snippet", "topic": "network_b"},
    },
    {
        "id": "snip_network_c",
        "document": (
            "Network C (Basic): Entry-level provider network. "
            "Government and community hospitals only. "
            "Limited specialist access. "
            "Suitable for low-risk, budget-conscious employers in low-cost regions."
        ),
        "metadata": {"type": "snippet", "topic": "network_c"},
    },
    {
        "id": "snip_saudi_market",
        "document": (
            "Saudi Arabian health insurance market context: "
            "CCHI (Council of Cooperative Health Insurance) mandates employer-provided health insurance. "
            "Compliance requires minimum Basic plan coverage. "
            "Regional pricing varies significantly: Riyadh > Dammam > Jeddah. "
            "Industry risk classification directly affects mandated coverage minimums."
        ),
        "metadata": {"type": "snippet", "topic": "market_context"},
    },
]

# ─── Customer Profiles (3 profiles) ──────────────────────────────────────────

CUSTOMER_PROFILES = [
    {
        "id": "profile_healthcare_riyadh",
        "document": (
            "Sample profile: Healthcare company in Riyadh, 500 employees, medium budget, "
            "dependents ratio 0.45. Recommended: Standard Plan, Network B. "
            "High industry risk drives minimum Network B requirement. "
            "Riyadh cost pressure makes Premium difficult on medium budget."
        ),
        "metadata": {
            "industry": "healthcare", "region": "riyadh",
            "budget": "medium", "recommended_plan": "Standard"
        },
    },
    {
        "id": "profile_construction_jeddah",
        "document": (
            "Sample profile: Construction company in Jeddah, 150 employees, low budget, "
            "priority: cheapest acceptable. Recommended: Basic Plan, Network C. "
            "Medium risk industry + lowest cost region + budget constraint → Basic viable."
        ),
        "metadata": {
            "industry": "construction", "region": "jeddah",
            "budget": "low", "recommended_plan": "Basic"
        },
    },
    {
        "id": "profile_retail_dammam",
        "document": (
            "Sample profile: Retail company in Dammam, 200 employees, medium budget. "
            "Comparing Standard vs Premium. Recommended: Standard. "
            "Medium-low risk industry does not justify Premium cost in Dammam."
        ),
        "metadata": {
            "industry": "retail", "region": "dammam",
            "budget": "medium", "recommended_plan": "Standard"
        },
    },
]


def seed(chroma_host: str = "localhost", chroma_port: int = 8001, reset: bool = False):
    print(f"[seed] Connecting to ChromaDB at {chroma_host}:{chroma_port}...")
    client = chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collections_data = [
        ("packages",          PACKAGES,          "3 documents"),
        ("benchmark_rules",   BENCHMARK_RULES,   "12 documents"),
        ("knowledge_snippets", KNOWLEDGE_SNIPPETS, "4 documents"),
        ("customer_profiles", CUSTOMER_PROFILES, "3 documents"),
    ]

    # Initialize embedding function
    embedding_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_host = os.environ.get("OLLAMA_HOST", f"http://{chroma_host}:11434")
    # For Docker: ollama_host is 'http://ollama:11434'
    if chroma_host == "chromadb":
        ollama_host = "http://ollama:11434"
        
    ef = OllamaEmbeddingFunction(
        model_name=embedding_model,
        url=f"{ollama_host}/api/embeddings",
    )

    for col_name, docs, desc in collections_data:
        if reset:
            try:
                client.delete_collection(col_name)
                print(f"[seed] Deleted existing collection: {col_name}")
            except Exception:
                pass

        try:
            col = client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=ef,
            )
        except Exception as e:
            print(f"[seed] ERROR creating collection {col_name}: {e}")
            continue

        # Check if already seeded
        existing = col.count()
        if existing > 0 and not reset:
            print(f"[seed] Collection '{col_name}' already has {existing} docs — skipping (use --reset to force)")
            continue

        ids = [d["id"] for d in docs]
        documents = [d["document"] for d in docs]
        metadatas = [d["metadata"] for d in docs]

        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[seed] Created collection: {col_name} ({desc})")

    print("[seed] Done. All collections seeded successfully.")

    # Verify
    print("\n[seed] Verification:")
    for col_name, _, _ in collections_data:
        try:
            col = client.get_collection(col_name)
            print(f"  {col_name}: {col.count()} documents")
        except Exception as e:
            print(f"  {col_name}: ERROR — {e}")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8001"))
    # In Docker: host=chromadb, port=8000 (internal)
    if host == "chromadb":
        port = 8000
    seed(chroma_host=host, chroma_port=port, reset=reset_flag)
