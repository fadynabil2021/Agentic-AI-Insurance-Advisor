import { ComparisonMatrix } from '@/lib/api';

export default function ComparisonTable({ matrix }: { matrix: ComparisonMatrix }) {
  const packages = matrix.packages;
  const dimensions = Object.keys(matrix.dimensions);

  return (
    <div style={{ 
      background: 'rgba(30, 41, 59, 1)', 
      borderRadius: '0.75rem', 
      border: '1px solid var(--panel-border)',
      overflow: 'hidden'
    }}>
      <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--panel-border)' }}>
        <h3 style={{ margin: 0, color: '#fff' }}>Plan Comparison</h3>
        {matrix.recommendation_reason && (
          <p style={{ margin: '0.5rem 0 0', color: 'var(--accent)' }}>
            <strong>Verdict:</strong> {matrix.recommendation_reason}
          </p>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
              <th style={{ padding: '1rem', borderBottom: '1px solid var(--panel-border)', color: 'var(--text-muted)' }}>Feature</th>
              {packages.map(pkg => (
                <th key={pkg} style={{ 
                  padding: '1rem', borderBottom: '1px solid var(--panel-border)', borderLeft: '1px solid var(--panel-border)',
                  color: pkg === matrix.recommendation ? 'var(--accent)' : '#fff'
                }}>
                  {pkg} Plan {pkg === matrix.recommendation && '★'}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dimensions.map(dim => (
              <tr key={dim} style={{ borderBottom: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '1rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                  {dim.replace(/_/g, ' ')}
                </td>
                {packages.map(pkg => {
                  const val = matrix.dimensions[dim][pkg];
                  return (
                    <td key={`${dim}-${pkg}`} style={{ padding: '1rem', borderLeft: '1px solid var(--panel-border)' }}>
                      {typeof val === 'number' ? (dim === 'score' ? `${val}/100` : val) : val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
