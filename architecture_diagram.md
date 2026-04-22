# Architecture Diagram — Agentic Insurance Advisor

## System Overview

```mermaid
%%{init: {"flowchart": {"curve": "basis"}}}%%
flowchart TD
    U([👤 User Request]) --> A

    subgraph API["🌐 FastAPI Backend — Railway"]
        A[POST /api/v1/query]
    end

    subgraph CACHE["⚡ Upstash Redis"]
        RC[(Cache\nLookup)]
        RS[(Cache\nStore)]
    end

    A --> RC
    RC -- HIT --> AR([✅ Cached Response])
    RC -- MISS --> LF

    subgraph OBS["📊 Langfuse Cloud"]
        LF[Create Trace]
    end

    LF --> G1

    subgraph GRAPH["🔷 LangGraph StateGraph — Stateful Orchestration"]
        direction TB

        G1["① Intent Parser\n(Gemini LLM)\nExtract: query_type, industry,\nregion, budget, priority, dependents"]

        G1 --> G2

        G2["② Validator\n(Deterministic Python)\nCheck required fields present"]

        G2 -- complete --> G3
        G2 -- incomplete --> G5
        G2 -- unsupported --> G6

        G3["③ Planner\n(Static lookup + Gemini fallback)\nSelect ordered tool execution plan"]

        G3 --> G4

        G4["④ Retrieval Tool\n🔧 TOOL\n(Gemini Embeddings + Pinecone)\nQuery: packages, rules, snippets namespaces"]

        G4 --> G4b

        G4b["⑤ Scoring Tool\n🔧 TOOL\n(Pure Deterministic Python)\nRules: budget, industry, region,\npriority, dependents, conflict detection\nOutput: ranked packages + confidence"]

        G4b -- recommend --> G7
        G4b -- compare --> G4c
        G4b -- retry --> G4
        G4b -- fallback --> G6

        G4c["⑥ Comparison Tool\n🔧 TOOL\n(Deterministic + Gemini narrative)\n6-dimension matrix: network, price,\ncoverage, score, budget_fit, risk_fit"]

        G4c --> G7

        G5["Clarifier\n(Template-based)\nGenerate clarification question"]

        G6["Fallback\n(Static templates)\nGraceful failure message"]

        G7["⑦ Output Composer\n(Deterministic structure + LLM narrative)\nBuild: recommendation, reasoning[3-5],\nexecution_trace, fallback_or_risk_note"]

        G7 --> G8

        G5 --> G8
        G6 --> G8

        G8["⑧ Evaluator\n(5 deterministic + 2 LLM-judge dims)\nScores: task_success, grounding,\nhallucination, confidence_calibration,\ntrace_integrity, tool_coverage"]

        G8 --> END([END])
    end

    subgraph EXTERNAL["☁️ External Services"]
        GEMINI["🤖 Google Gemini API\nModel: gemma-4-31b-it\n• Intent parsing\n• Planning (edge cases)\n• Comparison narrative\n• Hallucination judge"]

        PINE["🌲 Pinecone\nVector DB\nNamespaces:\n• packages (3)\n• benchmark_rules (13)\n• knowledge_snippets (5)\n• customer_profiles (3)"]

        LANGFUSE["📈 Langfuse Cloud\nObservability\n• Per-node spans\n• Eval scores logged\n• Hallucination flags"]
    end

    G1 -.->|"LLM call"| GEMINI
    G4 -.->|"embed + query"| GEMINI
    G4 -.->|"cosine search"| PINE
    G4c -.->|"narrative"| GEMINI
    G8 -.->|"grounding judge"| GEMINI
    G8 -.->|"log scores"| LANGFUSE

    END --> RS
    RS --> RESP([📦 Structured Response\nJSON: plan, recommendation,\nreasoning, confidence,\nexecution_trace, eval_scores])
```

---

## State Flow

```mermaid
stateDiagram-v2
    [*] --> IntentParser : user_request
    IntentParser --> Validator : query_type + entities + missing_fields

    state Validator {
        [*] --> CheckFields
        CheckFields --> Complete : all required fields present
        CheckFields --> Incomplete : missing fields
        CheckFields --> Unsupported : outside domain
    }

    Complete --> Planner
    Incomplete --> Clarifier
    Unsupported --> Fallback

    Planner --> RetrievalTool : execution_plan

    state RetrievalTool {
        [*] --> EmbedQuery
        EmbedQuery --> QueryPinecone
        QueryPinecone --> MergeResults
    }

    RetrievalTool --> ScoringTool : retrieval_results

    state ScoringTool {
        [*] --> ApplyRules
        ApplyRules --> BudgetRule
        ApplyRules --> IndustryRule
        ApplyRules --> RegionRule
        ApplyRules --> PriorityRule
        ApplyRules --> DependentsRule
        ApplyRules --> ConflictRule
        BudgetRule --> ComputeConfidence
        IndustryRule --> ComputeConfidence
        RegionRule --> ComputeConfidence
        PriorityRule --> ComputeConfidence
        DependentsRule --> ComputeConfidence
        ConflictRule --> ComputeConfidence
    }

    ScoringTool --> ComparisonTool : compare query
    ScoringTool --> OutputComposer : recommend query
    ScoringTool --> RetrievalTool : retry (empty results)
    ScoringTool --> Fallback : no results after retries

    ComparisonTool --> OutputComposer : comparison_matrix
    Clarifier --> Evaluator
    Fallback --> Evaluator
    OutputComposer --> Evaluator : recommendation + reasoning

    Evaluator --> [*] : eval_scores + final AgentState
```

---

## Data Model

### AgentState (TypedDict — shared across all nodes)

| Field | Type | Set By |
|-------|------|--------|
| `user_request` | `str` | API input |
| `thread_id` | `str` | API input |
| `query_type` | `QueryType` | Intent Parser |
| `extracted_entities` | `dict` | Intent Parser |
| `missing_fields` | `List[str]` | Validator |
| `execution_plan` | `List[str]` | Planner |
| `tools_used` | `List[str]` | Each tool |
| `retrieval_results` | `List[dict]` | Retrieval Tool |
| `retrieval_query` | `str` | Retrieval Tool |
| `scoring_results` | `List[dict]` | Scoring Tool |
| `scoring_breakdown` | `dict` | Scoring Tool |
| `comparison_matrix` | `dict` | Comparison Tool |
| `recommendation` | `dict` | Output Composer |
| `reasoning` | `List[str]` | Output Composer |
| `confidence` | `ConfidenceLevel` | Scoring Tool |
| `execution_trace` | `List[str]` | All nodes |
| `fallback_or_risk_note` | `str` | Output Composer / Fallback |
| `eval_scores` | `dict` | Evaluator |
| `retry_count` | `int` | Retrieval Tool |
| `error` | `dict` | Various nodes |
| `requires_clarification` | `bool` | Validator / Clarifier |
| `clarification_question` | `str` | Clarifier |

---

## Output Format (PDF Section 10 — Strict)

```json
{
  "user_request": "...",
  "plan": {
    "steps": ["retrieve relevant package knowledge", "apply scoring rules", "compose final recommendation"]
  },
  "tools_used": ["retrieval_tool", "scoring_tool"],
  "recommendation": {
    "plan_name": "Standard",
    "network": "B",
    "price_range": [6000, 7500]
  },
  "reasoning": [
    "driver 1",
    "driver 2",
    "driver 3"
  ],
  "confidence": "medium-high",
  "execution_trace": [
    "intent_parser: query_type=recommend, entities={...}",
    "planner: deterministic plan selected",
    "retrieval_tool: returned 11 documents",
    "scoring_tool: top package='Standard' score=90",
    "output_composer: generated 3 reasoning bullets",
    "evaluator: task_success=1.0, grounding=1.0"
  ],
  "fallback_or_risk_note": "..."
}
```
