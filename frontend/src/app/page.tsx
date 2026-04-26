import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{
        padding: '1.5rem 2rem',
        borderBottom: '1px solid var(--panel-border)',
        background: 'rgba(26, 15, 10, 0.6)',
        backdropFilter: 'blur(8px)'
      }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: '600', color: 'var(--text-main)' }}>
          Agentic AI Insurance Advisor
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
          Powered by LangGraph & Gemini 1.5 Pro
        </p>
      </header>

      <div style={{ flex: 1, display: 'flex', padding: '2rem', gap: '2rem', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)' }}>
          <ChatInterface />
        </div>
      </div>
    </main>
  );
}
