import { Recommendation } from '@/lib/api';
import { Shield, Settings, CheckCircle2 } from 'lucide-react';

export default function RecommendationCard({ 
  rec, 
  reasoning, 
  confidence 
}: { 
  rec: Recommendation; 
  reasoning: string[]; 
  confidence: string;
}) {
  const formatPrice = (arr: number[]) => {
    if (!arr || arr.length < 2) return "0";
    return `${arr[0].toLocaleString()} – ${arr[1].toLocaleString()} SAR`;
  };

  const getConfidenceColor = () => {
    switch(confidence) {
      case 'high': return 'var(--accent)';
      case 'medium-high': return '#3b82f6';
      case 'medium': return '#f59e0b';
      case 'low': return '#ef4444';
      default: return 'var(--text-muted)';
    }
  };

  return (
    <div style={{ 
      background: 'rgba(30, 41, 59, 1)', 
      borderRadius: '0.75rem', 
      border: '1px solid var(--panel-border)',
      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{ 
        background: 'linear-gradient(to right, rgba(99, 102, 241, 0.2), rgba(16, 185, 129, 0.1))',
        padding: '1.5rem',
        borderBottom: '1px solid var(--panel-border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center'
      }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fff' }}>
            <Shield size={24} color="var(--primary)" /> {rec.plan_name} Plan
          </h3>
          <p style={{ margin: '0.5rem 0 0', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Network {rec.network} • {formatPrice(rec.price_range)} / yr
          </p>
        </div>
        <div style={{ 
          background: 'rgba(0,0,0,0.2)', padding: '0.5rem 1rem', borderRadius: '2rem',
          display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem'
        }}>
          <span style={{ color: 'var(--text-muted)' }}>Confidence:</span>
          <span style={{ color: getConfidenceColor(), fontWeight: 'bold', textTransform: 'capitalize' }}>
            {confidence}
          </span>
        </div>
      </div>

      {/* Body: Reasoning */}
      <div style={{ padding: '1.5rem' }}>
        <h4 style={{ margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fff' }}>
          <Settings size={18} /> Why this plan?
        </h4>
        <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: '0.875rem' }}>
          {reasoning.map((reason, idx) => (
            <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', color: 'var(--text-main)' }}>
              <CheckCircle2 size={18} color="var(--accent)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span style={{ lineHeight: 1.5 }}>{reason}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
