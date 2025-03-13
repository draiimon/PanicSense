export const sentimentColors = {
  'Panic': {
    background: '#ef4444',
    light: '#fef2f2',
    border: '#fee2e2',
    text: '#b91c1c',
    badgeBg: '#fee2e2',
    badgeText: '#ef4444',
  },
  'Fear/Anxiety': {
    background: '#f97316',
    light: '#fff7ed',
    border: '#ffedd5',
    text: '#c2410c',
    badgeBg: '#ffedd5',
    badgeText: '#f97316',
  },
  'Disbelief': {
    background: '#8b5cf6',
    light: '#f5f3ff',
    border: '#ede9fe',
    text: '#6d28d9',
    badgeBg: '#ede9fe',
    badgeText: '#8b5cf6',
  },
  'Resilience': {
    background: '#10b981',
    light: '#ecfdf5',
    border: '#d1fae5',
    text: '#047857',
    badgeBg: '#d1fae5',
    badgeText: '#10b981',
  },
  'Neutral': {
    background: '#6b7280',
    light: '#f9fafb',
    border: '#f3f4f6',
    text: '#4b5563',
    badgeBg: '#f3f4f6',
    badgeText: '#6b7280',
  }
};

export const chartColors = [
  '#ef4444', // Panic - Red
  '#f97316', // Fear/Anxiety - Orange
  '#8b5cf6', // Disbelief - Purple
  '#10b981', // Resilience - Green
  '#6b7280'  // Neutral - Gray
];

export function getSentimentColor(sentiment: string): string {
  switch (sentiment) {
    case 'Panic':
      return '#ef4444';
    case 'Fear/Anxiety':
      return '#f97316';
    case 'Disbelief':
      return '#8b5cf6';
    case 'Resilience':
      return '#10b981';
    case 'Neutral':
    default:
      return '#6b7280';
  }
}

export function getSentimentBadgeClasses(sentiment: string): string {
  switch (sentiment) {
    case 'Panic':
      return 'bg-red-100 text-red-600';
    case 'Fear/Anxiety':
      return 'bg-orange-100 text-orange-600';
    case 'Disbelief':
      return 'bg-purple-100 text-purple-600';
    case 'Resilience':
      return 'bg-green-100 text-green-600';
    case 'Neutral':
    default:
      return 'bg-slate-100 text-slate-600';
  }
}
