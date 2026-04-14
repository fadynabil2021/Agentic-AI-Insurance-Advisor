export interface Recommendation {
  plan_name: string;
  network: string;
  price_range: number[];
}

export interface ComparisonMatrix {
  packages: string[];
  dimensions: Record<string, Record<string, any>>;
  recommendation?: string;
  recommendation_reason?: string;
}

export interface AgentResponse {
  user_request: string;
  thread_id: string;
  plan: { steps: string[] };
  tools_used: string[];
  recommendation?: Recommendation | null;
  comparison_matrix?: ComparisonMatrix | null;
  reasoning?: string[] | null;
  confidence: string;
  execution_trace: string[];
  fallback_or_risk_note?: string | null;
  requires_clarification: boolean;
  clarification_question?: string | null;
  latency_ms?: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function sendQuery(query: string, threadId?: string): Promise<AgentResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_request: query, thread_id: threadId })
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error occurred' }));
    throw new Error(error.detail || `Server error: ${res.status}`);
  }
  
  return res.json();
}
