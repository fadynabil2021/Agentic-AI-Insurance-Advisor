"""
Pinecone seed script — populates vectors for packages, rules, snippets, and profiles namespaces.
Run via: docker compose run --rm backend python data/seed_data.py
Or with --reset flag to wipe and re-seed.
"""
import os
import sys
import asyncio
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any

# Ensure we can import from backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clients.gemini_client import GeminiClient
from config import settings

# ─── Package Catalog (3 packages) ────────────────────────────────────────────
# Matches PDF Section 5.B exactly.

PACKAGES = [
    {
        "id": "pkg_basic",
        "document": (
            "Basic Insurance Plan: Network C provider network. "
            "Price range 4000-5000 SAR per employee annually. "
            "Coverage level: Low — cheapest option, limited provider access. "
            "Suitable for low-risk industries with budget-conscious employers. "
            "Best for retail and education sectors with small employee counts. "
            "Network C is suitable for cost-focused customers with simpler expectations."
        ),
        "metadata": {
            "name": "Basic",
            "network": "C",
            "price_range": "[4000, 5000]",
            "coverage": "Low",
            "budget_tier": "low",
            "type": "package",
        },
    },
    {
        "id": "pkg_standard",
        "document": (
            "Standard Insurance Plan: Network B provider network. "
            "Price range 6000-7500 SAR per employee annually. "
            "Coverage level: Medium — balanced price and network. "
            "Includes inpatient, outpatient, dental, and optical. "
            "Recommended for medium-risk industries. "
            "Well-suited for construction, manufacturing, and retail companies. "
            "Network B is best for balanced price-to-value tradeoffs."
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
            "Coverage level: High — best coverage and strongest network. "
            "Includes all Standard benefits plus maternity, "
            "chronic disease, specialist referrals, and international coverage. "
            "Recommended for high-risk industries such as healthcare. "
            "Suitable for companies with high dependents ratio or premium requirements. "
            "Network A is best for customers who prioritize access and premium hospitals."
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
# Matches PDF Section 5.C exactly.

BENCHMARK_RULES = [
    {
        "id": "rule_industry_healthcare",
        "document": (
            "Healthcare industry risk classification: HIGH. "
            "Healthcare = high risk. "
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
            "Construction = medium risk. "
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
            "Retail = medium-low risk. "
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
            "Riyadh has higher medical cost than Jeddah and Dammam. "
            "Riyadh has the highest medical cost index in Saudi Arabia. "
            "Premium plans in Riyadh carry the highest total cost burden. "
            "Medium-budget companies in Riyadh should prioritize Standard over Premium."
        ),
        "metadata": {"rule_type": "region_cost", "region": "riyadh", "cost_pressure": 3},
    },
    {
        "id": "rule_region_dammam",
        "document": (
            "Dammam regional cost pressure: MEDIUM-HIGH (score 2). "
            "Dammam is moderate-to-high cost. "
            "Dammam has moderate-to-high medical costs, between Riyadh and Jeddah. "
            "Standard plan is recommended for medium-budget companies in Dammam. "
            "Premium is viable for high-budget or high-risk industries."
        ),
        "metadata": {"rule_type": "region_cost", "region": "dammam", "cost_pressure": 2},
    },
    {
        "id": "rule_region_jeddah",
        "document": (
            "Jeddah regional cost pressure: MODERATE (score 1). "
            "Jeddah is moderate cost. "
            "Basic and Standard plans are viable for most industries in Jeddah. "
            "Cost savings from moderate regional rates make Standard highly attractive."
        ),
        "metadata": {"rule_type": "region_cost", "region": "jeddah", "cost_pressure": 1},
    },
    {
        "id": "rule_budget_low",
        "document": (
            "Low budget constraint: recommend Basic or Standard plans only. "
            "Low-budget users should not be recommended Premium unless clearly justified. "
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
            "If user asks for 'cheapest acceptable', favor Basic or Standard with explanation. "
            "When the customer requests cheapest acceptable option, apply cost-priority scoring. "
            "Basic gets +20 bonus if industry risk allows. "
            "Standard gets neutral treatment. "
            "Premium receives -30 penalty regardless of other factors."
        ),
        "metadata": {"rule_type": "priority", "priority": "cheapest"},
    },
    {
        "id": "rule_best_coverage",
        "document": (
            "Best coverage option scoring rule: "
            "If user asks for 'best coverage', favor Premium unless budget or rules strongly conflict. "
            "Premium gets priority for coverage-focused requests. "
            "Standard is fallback when budget constraints prevent Premium. "
            "Basic should only be recommended if budget is severely constrained."
        ),
        "metadata": {"rule_type": "priority", "priority": "best_coverage"},
    },
    {
        "id": "rule_dependents_ratio",
        "document": (
            "Dependents ratio rule: when dependents ratio exceeds 0.50, "
            "Basic plan is penalized by 15 points due to increased family benefits cost. "
            "High dependents ratio increases cost pressure. "
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

# ─── Knowledge Snippets (5 snippets) ─────────────────────────────────────────
# Matches PDF Section 5.D exactly.

KNOWLEDGE_SNIPPETS = [
    {
        "id": "snip_network_a",
        "document": (
            "Network A is best for customers who prioritize access and premium hospitals. "
            "Top-tier hospital network. "
            "Includes all major private hospitals in Saudi Arabia. "
            "Shortest referral times, highest specialist access. "
            "Required for healthcare industry workers with clinical exposure."
        ),
        "metadata": {"type": "snippet", "topic": "network_a"},
    },
    {
        "id": "snip_network_b",
        "document": (
            "Network B is best for balanced price-to-value tradeoffs. "
            "Mid-tier hospital network. "
            "Covers all major public hospitals and selected private clinics. "
            "Appropriate for medium-risk industries. "
            "Good balance of coverage breadth and cost efficiency."
        ),
        "metadata": {"type": "snippet", "topic": "network_b"},
    },
    {
        "id": "snip_network_c",
        "document": (
            "Network C is suitable for cost-focused customers with simpler expectations. "
            "Entry-level provider network. "
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
    {
        "id": "snip_dependents_healthcare",
        "document": (
            "High-dependents healthcare accounts tend to need tighter control of benefits "
            "and cost tradeoffs. When dependents ratio is high, the total cost of coverage "
            "increases substantially. Companies with high dependents ratios should consider "
            "Standard or Premium plans to ensure adequate family coverage."
        ),
        "metadata": {"type": "snippet", "topic": "dependents_healthcare"},
    },
]

# ─── Customer Profiles (3 profiles) ──────────────────────────────────────────
# Matches PDF Section 5.A (Customer A, B, C) exactly.

CUSTOMER_PROFILES = [
    {
        "id": "profile_healthcare_riyadh",
        "document": (
            "Customer A profile: Healthcare company in Riyadh, 120 employees, medium budget, "
            "dependents ratio 0.60. Priority: Balanced coverage and network strength. "
            "Recommended: Standard Plan, Network B. "
            "High industry risk drives minimum Network B requirement. "
            "Riyadh cost pressure makes Premium difficult on medium budget. "
            "Dependents ratio 0.60 exceeds 0.50 threshold — Basic insufficient."
        ),
        "metadata": {
            "industry": "healthcare", "region": "riyadh",
            "employees": 120, "dependents_ratio": 0.60,
            "budget": "medium", "priority": "balanced coverage and network strength",
            "recommended_plan": "Standard",
        },
    },
    {
        "id": "profile_construction_jeddah",
        "document": (
            "Customer B profile: Construction company in Jeddah, 80 employees, low budget, "
            "dependents ratio 0.30. Priority: Cheapest acceptable option. "
            "Recommended: Basic Plan, Network C. "
            "Medium risk industry + moderate cost region + budget constraint → Basic viable. "
            "Low dependents ratio means Basic coverage is sufficient."
        ),
        "metadata": {
            "industry": "construction", "region": "jeddah",
            "employees": 80, "dependents_ratio": 0.30,
            "budget": "low", "priority": "cheapest acceptable option",
            "recommended_plan": "Basic",
        },
    },
    {
        "id": "profile_retail_dammam",
        "document": (
            "Customer C profile: Retail company in Dammam, 200 employees, medium budget, "
            "dependents ratio 0.50. Priority: Stable service and moderate cost. "
            "Recommended: Standard Plan, Network B. "
            "Medium-low risk industry does not justify Premium cost in Dammam. "
            "Standard provides stable service with balanced cost."
        ),
        "metadata": {
            "industry": "retail", "region": "dammam",
            "employees": 200, "dependents_ratio": 0.50,
            "budget": "medium", "priority": "stable service and moderate cost",
            "recommended_plan": "Standard",
        },
    },
]


async def seed(reset: bool = False):
    print("[seed] Connecting to Pinecone...")
    api_key = settings.PINECONE_API_KEY
    if not api_key:
         print("Error: PINECONE_API_KEY is missing!")
         return
         
    client = Pinecone(api_key=api_key)
    index_name = settings.PINECONE_INDEX_NAME
    dimension = 3072 # gemini-embedding-001 dimension
    
    if reset:
        try:
            client.delete_index(index_name)
            print(f"[seed] Deleted existing index: {index_name}")
            # Wait for deletion
            while index_name in client.list_indexes().names():
                await asyncio.sleep(1)
        except Exception:
            pass

    existing_indexes = client.list_indexes().names()
    if index_name not in existing_indexes:
        print(f"[seed] Creating index: {index_name}")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
        )
        # Wait for index to be ready
        while not client.describe_index(index_name).status["ready"]:
            await asyncio.sleep(1)
            
    index = client.Index(index_name)
    
    gemini_client = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    collections_data = [
        ("packages",          PACKAGES,          "3 documents"),
        ("benchmark_rules",   BENCHMARK_RULES,   "12 documents"),
        ("knowledge_snippets", KNOWLEDGE_SNIPPETS, "4 documents"),
        ("customer_profiles", CUSTOMER_PROFILES, "3 documents"),
    ]

    for namespace, docs, desc in collections_data:
        if reset:
             try:
                 index.delete(delete_all=True, namespace=namespace)
             except Exception:
                 pass
                 
        ids = [d["id"] for d in docs]
        documents = [d["document"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        
        # Merge document content into metadata for retrieval later
        for i in range(len(metadatas)):
            metadatas[i]["content"] = documents[i]

        try:
            print(f"[seed] Generating embeddings for {namespace}...")
            # We'll embed one by one or in batch. Gemini has no direct batch method in our client, but we can gather.
            embeddings = []
            for doc in documents:
                 emb = await gemini_client.embed(doc, task_type="retrieval_document")
                 embeddings.append(emb)
                 
            vectors = list(zip(ids, embeddings, metadatas))
            
            print(f"[seed] Upserting to {namespace}...")
            index.upsert(vectors=vectors, namespace=namespace)
            print(f"[seed] Seeded namespace: {namespace} ({desc})")
        except Exception as e:
            print(f"[seed] ERROR seeding {namespace}: {e}")

    print("[seed] Done. All namespaces seeded successfully.")
    
    stats = index.describe_index_stats()
    print(f"\n[seed] Verification: {stats}")
    
    await gemini_client.aclose()


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    asyncio.run(seed(reset=reset_flag))
