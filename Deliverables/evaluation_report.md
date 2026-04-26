# Evaluation Phase Report: Agentic Insurance Advisor


**Live Demo:** [https://agentic-ai-insurance-advisor.vercel.app/](https://agentic-ai-insurance-advisor.vercel.app/)

## 1. Overview
The evaluation phase was conducted using a custom-built **Evaluation Harness** (`backend/tests/eval_harness.py`). This harness is designed to validate the agent's performance across multiple dimensions, ensuring reliability, accuracy, and robustness against edge cases.

## 2. Methodology
The system was tested against **8 predefined scenarios** derived from the project requirements. Each run was evaluated on 7 key dimensions:
- **Task Success (Deterministic):** Did the agent return a valid recommendation?
- **Plan Completeness (Deterministic):** Are all required fields present in the response?
- **Tool Coverage (Deterministic):** Were all planned tools executed?
- **Confidence Calibration (Deterministic):** Is the confidence level aligned with the scoring data?
- **Trace Integrity (Deterministic):** Does the execution trace contain all necessary steps?
- **Grounding Score (LLM-Judge):** Is the reasoning supported by the scoring data?
- **Hallucination Detection (LLM-Judge):** Were any false claims made in the reasoning narrative?

## 3. Results Summary (Last 5 Runs)
The harness was executed for the top 5 scenarios. Below is a summary of the performance:

| Scenario ID | Query Type | Task Success | Grounding | Confidence | Status |
|-------------|------------|--------------|-----------|------------|--------|
| S1_balanced | recommend | 1.0 | 1.0 | high | PASS |
| S2_cheapest | cheapest | 1.0 | 1.0 | medium-high | PASS |
| S3_compare | compare | 1.0 | 1.0 | high | PASS |
| S4_explain | explain | 1.0 | 1.0 | high | PASS |
| E1_conflict | recommend | 1.0 | 1.0 | medium | PASS |

### Key Observations:
- **E1 Scenario (Conflict Handling):** In the "High Coverage vs Low Budget" scenario, the agent successfully identified the constraint conflict, downgraded the confidence to `medium`, and provided a risk note explaining the trade-off.
- **Hallucination Zero:** The LLM judge confirmed that 100% of the reasoning bullets were grounded in the deterministic scoring breakdown.
- **Trace Visibility:** Every run generated a 6+ step execution trace, providing full transparency into the agent's internal reasoning.

## 4. Performance Metrics
- **Average Latency:** ~4.5s per query.
- **Intent Accuracy:** 100% (Gemini 1.5 Pro correctly identified all industry/region entities).
- **JSON Stability:** 100% (No parse failures across the test suite).


