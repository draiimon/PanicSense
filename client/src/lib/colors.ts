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

// Disaster type colors as per user specification
export const disasterTypeColors = {
  'Flood': '#3b82f6',      // Blue
  'Typhoon': '#6b7280',    // Gray
  'Fire': '#f97316',       // Orange
  'Volcano': '#ef4444',    // Red
  'Volcanic Eruption': '#ef4444', // Red (same as Volcano)
  'Earthquake': '#92400e', // Brown
  'Landslide': '#78350f',  // Dark Brown
  'Default': '#6b7280'     // Neutral color for other disaster types
};

export const chartColors = [
  '#ef4444', // Panic - Red
  '#f97316', // Fear/Anxiety - Orange
  '#8b5cf6', // Disbelief - Purple
  '#10b981', // Resilience - Green
  '#6b7280'  // Neutral - Gray
];

export function getSentimentColor(sentiment: string | null): string {
  if (!sentiment) return '#6b7280'; // Default gray for null

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

export function getSentimentBadgeClasses(sentiment: string | null): string {
  if (!sentiment) return 'bg-slate-100 text-slate-600'; // Default for null
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

/**
 * Get color for disaster type according to user specifications:
 * - Flood: Blue
 * - Typhoon: Gray
 * - Fire: Orange
 * - Volcanic Eruptions: Red
 * - Earthquake: Brown
 * - Landslide: Dark Brown
 * - Others: Neutral gray
 */
export function getDisasterTypeColor(disasterType: string | null): string {
  if (!disasterType) return disasterTypeColors.Default;
  
  // Normalize the input by converting to lowercase
  const normalizedType = disasterType.toLowerCase();
  
  // Check for each disaster type, including variations
  if (normalizedType.includes('flood')) return disasterTypeColors.Flood;
  if (normalizedType.includes('typhoon') || normalizedType.includes('storm') || normalizedType.includes('bagyo')) return disasterTypeColors.Typhoon;
  if (normalizedType.includes('fire') || normalizedType.includes('sunog')) return disasterTypeColors.Fire;
  if (normalizedType.includes('volcano') || normalizedType.includes('volcanic') || normalizedType.includes('eruption') || normalizedType.includes('bulkan')) return disasterTypeColors.Volcano;
  if (normalizedType.includes('earthquake') || normalizedType.includes('quake') || normalizedType.includes('lindol')) return disasterTypeColors.Earthquake;
  if (normalizedType.includes('landslide') || normalizedType.includes('mudslide')) return disasterTypeColors.Landslide;
  
  // Default color for other disaster types
  return disasterTypeColors.Default;
}
