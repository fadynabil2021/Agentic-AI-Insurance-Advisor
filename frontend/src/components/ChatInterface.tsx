"use client";

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, AlertCircle } from 'lucide-react';
import { sendQuery, AgentResponse } from '@/lib/api';
import RecommendationCard from './RecommendationCard';
import ComparisonTable from './ComparisonTable';

interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  data?: AgentResponse;
  error?: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await sendQuery(userMsg.content, threadId);
      if (!threadId) setThreadId(res.thread_id);

      const agentMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: '',
        data: res
      };
      setMessages(prev => [...prev, agentMsg]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: '',
        error: err.message || 'Failed to connect to the advisor.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* Message List Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {messages.length === 0 && (
          <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Bot size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
            <h2 style={{ color: '#fff', marginBottom: '0.5rem' }}>Welcome to the Insurance Advisor</h2>
            <p>Ask for a recommendation for an industry domain in Saudi Arabia</p>
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id} className="animate-fade-in" style={{ display: 'flex', gap: '1rem', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
            <div style={{ 
              width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: msg.role === 'user' ? 'var(--primary)' : 'var(--panel-bg)',
              border: '1px solid var(--panel-border)',
              flexShrink: 0
            }}>
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} color="var(--accent)" />}
            </div>
            
            <div style={{ 
              maxWidth: '80%', padding: '1rem', borderRadius: '1rem',
              background: msg.role === 'user' ? 'var(--user-msg)' : 'none',
              border: msg.role === 'user' ? '1px solid var(--primary)' : 'none',
              color: msg.role === 'user' ? '#fff' : 'var(--text-main)',
              borderTopRightRadius: msg.role === 'user' ? '0.25rem' : '1rem',
              borderTopLeftRadius: msg.role === 'agent' ? '0.25rem' : '1rem',
            }}>
              {msg.role === 'user' && <p>{msg.content}</p>}
              
              {msg.error && (
                <div style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertCircle size={18} /> {msg.error}
                </div>
              )}

              {msg.data && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
                  
                  {msg.data.requires_clarification && msg.data.clarification_question && (
                    <div style={{ background: 'var(--bot-msg)', padding: '1rem', borderRadius: '0.5rem', borderLeft: '4px solid #f59e0b' }}>
                      <p style={{ whiteSpace: 'pre-wrap' }}>{msg.data.clarification_question}</p>
                    </div>
                  )}

                  {msg.data.recommendation && (
                    <RecommendationCard 
                      rec={msg.data.recommendation} 
                      reasoning={msg.data.reasoning || []} 
                      confidence={msg.data.confidence}
                    />
                  )}

                  {msg.data.comparison_matrix && (
                    <ComparisonTable matrix={msg.data.comparison_matrix} />
                  )}

                  {msg.data.fallback_or_risk_note && !msg.data.requires_clarification && (
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                      <strong>Note:</strong> {msg.data.fallback_or_risk_note}
                    </div>
                  )}

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
                    <span>Latency: {msg.data.latency_ms}ms</span>
                    <span>Tools used: {msg.data.tools_used.join(', ')}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div style={{ display: 'flex', gap: '1rem' }} className="animate-pulse-slow">
            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--panel-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Loader2 size={20} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
            </div>
            <div style={{ padding: '1rem', color: 'var(--text-muted)' }}>Thinking... executing LangGraph tools...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ padding: '1.5rem', borderTop: '1px solid var(--panel-border)', background: 'rgba(15, 23, 42, 0.8)' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '1rem' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your insurance query here..."
            disabled={isLoading}
            style={{
              flex: 1, padding: '1rem 1.5rem', borderRadius: '2rem', border: '1px solid var(--panel-border)',
              background: 'rgba(255, 255, 255, 0.05)', color: '#fff', fontSize: '1rem',
              outline: 'none', transition: 'border-color 0.2s'
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--panel-border)'}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            style={{
              padding: '0 1.5rem', borderRadius: '2rem', background: 'var(--primary)', color: '#fff',
              border: 'none', cursor: (!input.trim() || isLoading) ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 'bold',
              opacity: (!input.trim() || isLoading) ? 0.5 : 1, transition: 'background 0.2s'
            }}
            onMouseOver={(e) => { if (input.trim() && !isLoading) e.currentTarget.style.background = 'var(--primary-hover)'; }}
            onMouseOut={(e) => e.currentTarget.style.background = 'var(--primary)'}
          >
            <Send size={18} /> Send
          </button>
        </form>
      </div>
    </div>
  );
}
