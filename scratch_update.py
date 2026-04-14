import re

with open("PRD_Mission3_Agentic_AI.md", "r") as f:
    prd = f.read()

# Graph edges update
prd = prd.replace(
    'graph.add_edge("clarifier", END)  # Returns clarification; user must resubmit\ngraph.add_edge("fallback", END)',
    'graph.add_edge("clarifier", "evaluator")\ngraph.add_edge("fallback", "evaluator")'
)

# Langfuse trace config note
prd = prd.replace(
    '# Compile with in-memory checkpointing (upgrade to SQLite saver for persistence)\ncheckpointer = MemorySaver()\ncompiled_graph = graph.compile(checkpointer=checkpointer)',
    '# Compile with in-memory checkpointing\ncheckpointer = MemorySaver()\ncompiled_graph = graph.compile(checkpointer=checkpointer)\n# Note: Langfuse trace is injected via config["configurable"]["langfuse_trace"]'
)

# Scoring tool node update
scoring_start = prd.find('**Scoring Rules (directly from PDF Section 5C):**')
scoring_end = prd.find('### 7.6 Comparison Tool Node')

new_scoring = """**Scoring Rules and Constants:**
Constants are defined in `scoring_constants.py` to cleanly decouple weighted penalties from the logic.

**Scoring Function:**

```python
def scoring_tool_node(state: AgentState, langfuse_trace) -> AgentState:
    # ... Create span via langfuse_trace ...
    
    for pkg in packages:
        # Score calculation applies flat weights from scoring_constants.py
        # priority matching uses longest-match-first to avoid dict key ordering bugs
        
        # Determine confidence using _compute_confidence(top_score, second_score):
        # Base confidence from absolute score:
        #  >= 80 -> HIGH
        #  >= 60 -> MEDIUM_HIGH
        #  >= 40 -> MEDIUM
        #  < 40  -> LOW
        # THEN factor in score gap to #2. If the gap < 10 (CONFIDENCE_GAP_NARROW), downgrade confidence by one tier.
    
    # ... End span ...
    return state
```

---

"""

prd = prd[:scoring_start] + new_scoring + prd[scoring_end:]

with open("PRD_Mission3_Agentic_AI.md", "w") as f:
    f.write(prd)

################# technical_report.md ##################

with open("technical_report.md", "r") as f:
    tr = f.read()

tr = tr.replace(
    '`intent_parser → validator → [planner | clarifier | fallback] → retrieval_tool → scoring_tool → [comparison_tool] → output_composer → evaluator`',
    '`intent_parser → validator → [planner | clarifier | fallback]`\n`clarifier / fallback → evaluator`\n`planner → retrieval_tool → scoring_tool → [comparison_tool] → output_composer → evaluator`'
)

# Execution model
tr = tr.replace(
    '- Langfuse span creation is non-blocking (fire-and-forget with try/except)',
    '- Langfuse span creation is non-blocking (fire-and-forget with try/except)\n- **Dependency Injection:** Langfuse trace context is injected via LangGraph `config["configurable"]["langfuse_trace"]`. This avoids polluting the `AgentState` with non-msgpack-serializable objects which would break `MemorySaver` checkpointing.'
)

# Scoring Tool section
tr = tr.replace(
    '100 and a breakdown of which rules were applied.',
    '100 and a breakdown of which rules were applied. *Refactoring Note:* The scoring engine includes gap-aware, weighted confidence calculations (confidence drops if the score margin between the top recommendation and the runner-up is too narrow) and uses longest-match-first for text-based priority matching.'
)

with open("technical_report.md", "w") as f:
    f.write(tr)

print("Updated PRD and TR successfully.")
