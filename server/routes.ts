import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";
import { pythonService } from "./python-service";
import { insertSentimentPostSchema, insertAnalyzedFileSchema } from "@shared/schema";
import { EventEmitter } from 'events';

// Configure multer for file uploads with improved performance
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // Increased to 50MB for faster batch processing
  },
  fileFilter: (req, file, cb) => {
    if (file.originalname.toLowerCase().endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// Enhanced upload progress tracking with better performance
const uploadProgressMap = new Map<string, {
  processed: number;
  total: number;
  stage: string;
  timestamp: number;
  error?: string;
}>();

// Enhanced disaster detection patterns with comprehensive synonyms and related terms
const disasterPatterns = {
  Earthquake: {
    primaryIndicators: [
      'earthquake', 'lindol', 'quake', 'magnitude', 'aftershock', 'tremor',
      'seismic', 'fault line', 'epicenter', 'temblor', 'lumindol', 'yumanig',
      'ground shaking', 'earth movement', 'shake', 'rumble', 'umaalog',
      'nayayanig', 'pagyanig', 'lindol na malakas', 'intensity', 'dahan-dahan'
    ],
    tagalogIndicators: [
      'lindol', 'lumindol', 'yumanig', 'paglindol', 'pagyanig', 'yanig',
      'alog', 'umaalog', 'nayayanig', 'gumalaw', 'gumugalaw', 'nayanig',
      'paggalaw ng lupa', 'paglindol ng lupa', 'mayaning', 'nagalaw'
    ],
    impactWords: [
      'damage', 'collapse', 'rubble', 'crack', 'fallen', 'trapped', 'gumuho', 'nasira',
      'tumba', 'natumba', 'bumagsak', 'nabagsak', 'wasak', 'nawasak', 'bitak',
      'nabitak', 'sira', 'nasira', 'basag', 'nabasag', 'guho', 'naguho'
    ],
    intensityWords: [
      'strong', 'powerful', 'massive', 'malakas', 'intensity', 'violent', 'severe',
      'malaki', 'napakalakas', 'matindi', 'sobrang lakas', 'nakakatakot',
      'nakakabahala', 'hindi makagalaw', 'hindi makatayo'
    ],
    weight: 2.5
  },
  Flood: {
    primaryIndicators: [
      'flood', 'baha', 'tubig', 'water level', 'rising', 'overflow',
      'submerged', 'inundated', 'deluge', 'flash flood', 'flooding',
      'high water', 'water rise', 'flood warning', 'flooded areas'
    ],
    tagalogIndicators: [
      'baha', 'bumabaha', 'pagbaha', 'binabaha', 'daluyong', 'lumagpas',
      'tumataas ang tubig', 'lumubog', 'nalubog', 'nakalubog', 'apaw',
      'umapaw', 'nagapaw', 'tubig-baha', 'rumaragasa', 'lumalaki'
    ],
    impactWords: [
      'stranded', 'evacuation', 'rescue', 'trapped', 'submerged', 'nasira', 'naipit',
      'lubog', 'nalubog', 'natrap', 'hindi makalabas', 'naipit', 'rescue',
      'saklolo', 'tulong', 'walang makapasok', 'walang madaanan'
    ],
    intensityWords: [
      'deep', 'rising', 'severe', 'malalim', 'lubog', 'mataas', 'lumalalim',
      'mabilis', 'rumaragasa', 'malakas', 'hindi makatawid', 'hanggang tuhod',
      'hanggang baywang', 'hanggang dibdib', 'hanggang leeg'
    ],
    weight: 2.3
  },
  Typhoon: {
    primaryIndicators: [
      'typhoon', 'bagyo', 'storm', 'cyclone', 'hurricane', 'winds', 'signal',
      'tropical depression', 'tropical storm', 'super typhoon', 'storm surge',
      'low pressure', 'weather disturbance', 'heavy rain', 'gale warning'
    ],
    tagalogIndicators: [
      'bagyo', 'unos', 'bagyong', 'hanging', 'ulan', 'habagat', 'malakas na hangin',
      'signal no', 'pag-ulan', 'daluyong', 'hampas ng alon', 'malalakas na alon',
      'pagbugso ng hangin', 'hanging habagat', 'ulap', 'kulog', 'kidlat'
    ],
    impactWords: [
      'damage', 'blown', 'destroyed', 'evacuation', 'landfall', 'storm surge',
      'flying debris', 'toppled', 'uprooted', 'blown away', 'nasira',
      'natangay', 'nagiba', 'nawala', 'lumipad', 'natumba'
    ],
    intensityWords: [
      'strong', 'intense', 'super', 'powerful', 'malakas', 'malaks na hangin',
      'napakalakas', 'matindi', 'rumaragasa', 'walang hinto', 'tuloy-tuloy',
      'hindi humihina', 'lumalakas pa', 'pabagsik nang pabagsik'
    ],
    weight: 2.4
  },
  Landslide: {
    primaryIndicators: [
      'landslide', 'mudslide', 'rockslide', 'avalanche', 'soil erosion',
      'ground collapse', 'earth movement', 'debris flow', 'soil movement',
      'ground failure', 'slope failure', 'mass movement'
    ],
    tagalogIndicators: [
      'pagguho', 'guho', 'pagguho ng lupa', 'nagtunaw', 'nagbagsak',
      'pagdaloy ng lupa', 'pagkadurog', 'pagkatibag', 'pagkatumba',
      'pagtunaw ng lupa', 'pagkaanod', 'pagbagsak ng lupa'
    ],
    impactWords: [
      'buried', 'trapped', 'blocked', 'covered', 'natabunan', 'naipit',
      'nabulid', 'nalibing', 'natakpan', 'natabunan', 'hindi makita',
      'nawawala', 'nalibing', 'hindi mahanap', 'nawalan ng bakas'
    ],
    intensityWords: [
      'massive', 'major', 'significant', 'malaking', 'malawak', 'grabe',
      'napakalaki', 'napakabilis', 'hindi mapigilan', 'tuloy-tuloy',
      'walang tigil', 'patuloy', 'lumalaki pa'
    ],
    weight: 2.2
  },
  Volcanic: {
    primaryIndicators: [
      'volcano', 'volcanic', 'eruption', 'ash fall', 'lava', 'magma',
      'pyroclastic flow', 'volcanic activity', 'phreatic', 'crater',
      'volcanic ash', 'steam emission', 'sulfur dioxide', 'alert level'
    ],
    tagalogIndicators: [
      'bulkan', 'bulkang', 'pagputok', 'abo', 'lahar', 'nagputok',
      'pumutok', 'sumabog', 'nagbubuga', 'nagbuga', 'umuusok',
      'naglalavang', 'nagbabaga', 'umalingasaw'
    ],
    impactWords: [
      'evacuation', 'danger zone', 'alert level', 'active', 'ashfall',
      'covered in ash', 'respiratory', 'masks needed', 'zero visibility',
      'contaminated', 'health hazard', 'toxic', 'acid rain'
    ],
    intensityWords: [
      'explosive', 'major', 'intense', 'violent', 'malakas', 'matindi',
      'mapanganib', 'mapaminsala', 'nakamamatay', 'nakakalason',
      'hindi humihinto', 'lumalala', 'tumitindi'
    ],
    weight: 2.4
  },
  Fire: {
    primaryIndicators: [
      'fire', 'blaze', 'flames', 'burning', 'smoke', 'inferno',
      'conflagration', 'wildfire', 'forest fire', 'bush fire',
      'fire outbreak', 'combustion', 'fire alarm', 'fire alert'
    ],
    tagalogIndicators: [
      'sunog', 'apoy', 'nasusunog', 'nagliliyab', 'nagkakasunog',
      'nagliliyab', 'nag-aapoy', 'usok', 'nag-uusok', 'nasunog',
      'natupok', 'nagliliyab', 'nagbabaga'
    ],
    impactWords: [
      'burned', 'gutted', 'destroyed', 'damage', 'casualties',
      'evacuation', 'rescue', 'trapped', 'smoke inhalation',
      'property damage', 'total damage', 'ashes'
    ],
    intensityWords: [
      'massive', 'huge', 'intense', 'raging', 'uncontrolled',
      'spreading', 'wild', 'fierce', 'devastating', 'malakas',
      'malaki', 'lumalaki', 'kumakalat', 'hindi mapigilan'
    ],
    weight: 2.3
  }
};

// Enhanced emotion detection with more sophisticated patterns (This section remains largely unchanged)
const emotionPatterns = {
  Panic: {
    keywords: [
      'panic', 'terrified', 'horrified', 'scared', 'frightened', 'takot', 'natatakot',
      'emergency', 'help', 'tulong', 'evacuate', 'evacuating', 'run', 'flee', 'escape',
      'trapped', 'stuck', 'nasukol', 'naiipit', 'hindi makalabas', 'can\'t get out',
      'SOS', 'mayday', 'danger', 'dangerous', 'delikado', 'mapanganib'
    ],
    intensifiers: ['very', 'really', 'extremely', 'sobra', 'grabe', 'napaka', 'super'],
    contextual: ['need immediate', 'right now', 'quickly', 'urgent', 'emergency'],
    weight: 2.0
  },
  'Fear/Anxiety': {
    keywords: [
      'fear', 'worried', 'anxious', 'nervous', 'kabado', 'nag-aalala', 'balisa',
      'concerned', 'scared', 'afraid', 'natatakot', 'kinakabahan', 'nangangamba',
      'uncertain', 'unsure', 'hindi sigurado', 'dread', 'warning', 'babala',
      'incoming', 'approaching', 'papalapit', 'threatening', 'threat', 'banta'
    ],
    intensifiers: ['getting', 'becoming', 'more', 'increasing', 'growing', 'lumalakas'],
    contextual: ['might', 'could', 'possibly', 'baka', 'siguro', 'posible'],
    weight: 1.5
  },
  'Disbelief': {
    keywords: [
      'unbelievable', 'impossible', 'hindi kapani-paniwala', 'di makapaniwala',
      'shocked', 'stunned', 'nagulat', 'nagugulat', 'cannot believe', 'di matanggap',
      'how could', 'why would', 'bakit ganun', 'paano nangyari', 'unexpected',
      'hindi inaasahan', 'surprising', 'nakakagulat', 'grabe'
    ],
    intensifiers: ['totally', 'completely', 'absolutely', 'lubos', 'sobrang'],
    contextual: ['never thought', 'first time', 'unprecedented', 'unusual'],
    weight: 1.3
  },
  'Resilience': {
    keywords: [
      'strong', 'brave', 'hope', 'malakas', 'matapang', 'pag-asa', 'kakayanin',
      'survive', 'overcome', 'lalaban', 'fight', 'recover', 'rebuild', 'help',
      'support', 'tulong', 'together', 'sama-sama', 'bayanihan', 'community',
      'volunteers', 'rescue', 'saved', 'safe', 'ligtas', 'evacuated', 'shelter'
    ],
    intensifiers: ['will', 'shall', 'must', 'dapat', 'kailangan', 'always'],
    contextual: ['we can', 'we will', 'kaya natin', 'magtulungan', 'unity'],
    weight: 1.8
  },
  'Neutral': {
    keywords: [
      'information', 'update', 'announcement', 'balita', 'impormasyon', 'advisory',
      'report', 'status', 'situation', 'current', 'kasalukuyan', 'official',
      'notice', 'alert', 'bulletin', 'news', 'reported', 'according'
    ],
    intensifiers: ['please', 'kindly', 'pakiusap', 'paki'],
    contextual: ['as of', 'currently', 'ngayon', 'latest'],
    weight: 1.0
  }
};

// Contextual disaster indicators for better detection (This section remains largely unchanged)
const disasterContexts = {
  Earthquake: {
    primaryIndicators: ['earthquake', 'lindol', 'quake', 'magnitude', 'aftershock', 'tremor'],
    locationIndicators: ['epicenter', 'fault line', 'ground', 'building', 'structure'],
    intensityWords: ['strong', 'powerful', 'massive', 'malakas', 'devastating'],
    weight: 2.0
  },
  Flood: {
    primaryIndicators: ['flood', 'baha', 'tubig', 'water level', 'rising', 'overflow'],
    locationIndicators: ['street', 'road', 'area', 'community', 'river', 'dam'],
    intensityWords: ['deep', 'rising', 'severe', 'malalim', 'lumalalim'],
    weight: 1.8
  },
  Typhoon: {
    primaryIndicators: ['typhoon', 'bagyo', 'storm', 'wind', 'rain', 'signal'],
    locationIndicators: ['eye', 'path', 'track', 'landfall', 'coastal', 'area'],
    intensityWords: ['strong', 'intense', 'super', 'powerful', 'malakas'],
    weight: 1.9
  }
};

// Function to analyze text content with advanced pattern matching
function analyzeTextContent(text: string) {
  const textLower = text.toLowerCase();
  const words = textLower.split(/\s+/);

  // Advanced contextual patterns for disaster detection
  const contextualPatterns = {
    emergency: {
      keywords: ['emergency', 'urgent', 'immediately', 'asap', 'quick', 'agad', 'importante'],
      weight: 1.5
    },
    damage: {
      keywords: ['damage', 'destroyed', 'broken', 'collapsed', 'sira', 'wasak', 'guho'],
      weight: 1.3
    },
    danger: {
      keywords: ['dangerous', 'threat', 'warning', 'careful', 'delikado', 'ingat', 'bantay'],
      weight: 1.4
    },
    help: {
      keywords: ['help', 'rescue', 'save', 'assist', 'tulong', 'saklolo', '救助'],
      weight: 1.6
    },
    impact: {
      keywords: ['affected', 'victims', 'displaced', 'evacuated', 'nasalanta', 'biktima'],
      weight: 1.2
    }
  };

  // Calculate contextual scores
  let contextScore = 0;
  let contextReasons = [];

  Object.entries(contextualPatterns).forEach(([context, pattern]) => {
    pattern.keywords.forEach(keyword => {
      if (textLower.includes(keyword)) {
        contextScore += pattern.weight;
        contextReasons.push(`${context}: ${keyword}`);
      }
    });
  });

  // Enhanced disaster patterns with multi-word recognition
  const enhancedDisasterPatterns = {
    Earthquake: {
      primaryPatterns: [
        'earthquake hit', 'magnitude', 'seismic activity', 'ground shaking',
        'lindol na malakas', 'lumindol', 'aftershocks reported', 'tremors felt'
      ],
      secondaryPatterns: [
        'fault line', 'epicenter', 'seismograph', 'tectonic', 'geological',
        'may lindol', 'yumanig', 'pagyanig'
      ],
      locationPatterns: [
        'ground', 'building', 'structure', 'foundation', 'road', 'walls'
      ],
      weight: 2.5
    },
    Flood: {
      primaryPatterns: [
        'flood warning', 'water level rising', 'heavy rainfall', 'overflow',
        'baha sa', 'bumabaha', 'tubig tumataas', 'flash flood'
      ],
      secondaryPatterns: [
        'drainage', 'river swelling', 'dam', 'water system', 'flood control',
        'pagbaha', 'lumalaki ang ilog', 'mataas na tubig'
      ],
      locationPatterns: [
        'street', 'road', 'area', 'community', 'river', 'creek', 'kalsada'
      ],
      weight: 2.3
    },
    Typhoon: {
      primaryPatterns: [
        'typhoon warning', 'storm signal', 'heavy winds', 'cyclone approaching',
        'bagyo papasok', 'malakas na hangin', 'storm surge', 'super typhoon'
      ],
      secondaryPatterns: [
        'rainfall', 'wind speed', 'eye of the storm', 'weather system',
        'ulan', 'hangin', 'daluyong', 'signal number'
      ],
      locationPatterns: [
        'coast', 'region', 'province', 'city', 'area', 'lalawigan', 'baybayin'
      ],
      weight: 2.4
    },
    // Add more disaster types...
  };

  // Advanced disaster detection with context awareness
  let maxDisasterScore = 0;
  let detectedDisaster = null;
  let disasterConfidence = 0;
  let detectionExplanation = [];

  Object.entries(enhancedDisasterPatterns).forEach(([disasterType, patterns]) => {
    let score = 0;
    let evidence = [];

    // Check for primary patterns (highest weight)
    patterns.primaryPatterns.forEach(pattern => {
      if (textLower.includes(pattern.toLowerCase())) {
        score += 3.0 * patterns.weight;
        evidence.push(`Strong indicator: "${pattern}"`);
      }
    });

    // Check for secondary patterns (medium weight)
    patterns.secondaryPatterns.forEach(pattern => {
      if (textLower.includes(pattern.toLowerCase())) {
        score += 2.0 * patterns.weight;
        evidence.push(`Supporting evidence: "${pattern}"`);
      }
    });

    // Check for location context (adds credibility)
    patterns.locationPatterns.forEach(pattern => {
      if (textLower.includes(pattern.toLowerCase())) {
        score += 1.0 * patterns.weight;
        evidence.push(`Location context: "${pattern}"`);
      }
    });

    // Apply contextual boosting
    if (contextScore > 0) {
      score *= (1 + (contextScore * 0.2));
      evidence.push(`Context boost: ${contextReasons.join(', ')}`);
    }

    if (score > maxDisasterScore) {
      maxDisasterScore = score;
      detectedDisaster = disasterType;
      disasterConfidence = Math.min(score / 10, 1);
      detectionExplanation = evidence;
    }
  });

  return {
    contextualScore: contextScore,
    contextReasons,
    disaster: detectedDisaster,
    confidence: disasterConfidence,
    explanation: detectionExplanation,
    isEmergency: contextScore > 2.0
  };
}


// Update the main emotion analysis function to use the new analysis
function analyzeEmotionWithContext(text: string, providedDisasterType: string | null | undefined): {
  emotion: string;
  confidence: number;
  explanation: string;
  detectedDisaster?: {
    type: string;
    confidence: number;
  };
} {
  // Perform advanced text analysis
  const textAnalysis = analyzeTextContent(text);

  // Use either provided disaster type or detected one
  const disasterInfo = providedDisasterType ?
    { type: providedDisasterType, confidence: 1.0 } :
    { type: textAnalysis.disaster, confidence: textAnalysis.confidence };

  // Calculate emotion scores with context awareness
  const emotionScores = calculateEmotionScores(text, textAnalysis);

  // Apply disaster context boosting
  if (disasterInfo.type || textAnalysis.isEmergency) {
    emotionScores.Fear = (emotionScores.Fear || 0) * 1.5;
    emotionScores.Panic = (emotionScores.Panic || 0) * 1.8;
  }

  // Find dominant emotion
  const dominantEmotion = Object.entries(emotionScores)
    .reduce((prev, curr) => curr[1] > prev[1] ? curr : prev);

  // Generate detailed explanation
  const explanation = generateEmotionExplanation(
    dominantEmotion[0],
    emotionScores,
    textAnalysis,
    disasterInfo
  );

  return {
    emotion: dominantEmotion[0],
    confidence: dominantEmotion[1],
    explanation,
    detectedDisaster: disasterInfo.type ? {
      type: disasterInfo.type,
      confidence: disasterInfo.confidence
    } : undefined
  };
}

// Helper function to calculate emotion scores
function calculateEmotionScores(text: string, analysis: any): Record<string, number> {
  const textLower = text.toLowerCase();
  let scores: Record<string, number> = {};
  let explanations: Record<string, string[]> = {};

  // Calculate base emotion scores with context
  for (const [emotion, pattern] of Object.entries(emotionPatterns)) {
    let score = 0;
    let reasons: string[] = [];

    // Check keywords
    pattern.keywords.forEach(keyword => {
      const matches = (textLower.match(new RegExp(keyword, 'g')) || []).length;
      if (matches > 0) {
        score += matches * pattern.weight;
        reasons.push(`Found "${keyword}" ${matches} time(s)`);
      }
    });

    // Check intensifiers
    pattern.intensifiers.forEach(intensifier => {
      const matches = (textLower.match(new RegExp(intensifier, 'g')) || []).length;
      if (matches > 0) {
        score += matches * 0.5 * pattern.weight;
        reasons.push(`Intensifier "${intensifier}" present`);
      }
    });

    // Check contextual patterns
    pattern.contextual.forEach(context => {
      const matches = (textLower.match(new RegExp(context, 'g')) || []).length;
      if (matches > 0) {
        score += matches * 0.7 * pattern.weight;
        reasons.push(`Contextual pattern "${context}" found`);
      }
    });

    scores[emotion] = score;
    explanations[emotion] = reasons;
  }

  // Apply disaster context boost from analyzeTextContent
  if (analysis.disaster) {
    const disasterContext = disasterContexts[analysis.disaster as keyof typeof disasterContexts] || {}; 
    if (disasterContext) {
        disasterContext.primaryIndicators.forEach(indicator => {
          if (textLower.includes(indicator)) {
            scores['Fear/Anxiety'] *= 1.2;
            scores['Panic'] *= 1.3;
          }
        });

        disasterContext.intensityWords.forEach(word => {
          if (textLower.includes(word)) {
            scores['Fear/Anxiety'] *= 1.1;
            scores['Panic'] *= 1.2;
          }
        });
      }
  }
    
  return scores;
}

// Helper function to generate detailed explanation
function generateEmotionExplanation(
  emotion: string,
  scores: Record<string, number>,
  analysis: any,
  disasterInfo: { type: string | null, confidence: number }
): string {
  const lines = [
    `Primary Emotion: ${emotion}`,
    `Context Analysis: ${analysis.contextReasons.join(', ')}`,
  ];

  if (disasterInfo.type) {
    lines.push(`Disaster Context: ${disasterInfo.type} (${(disasterInfo.confidence * 100).toFixed(1)}% confidence)`);
    lines.push(`Supporting Evidence: ${analysis.explanation.join(', ')}`);
  }

  if (analysis.isEmergency) {
    lines.push('Emergency Situation Detected - Emotional intensity adjusted accordingly');
  }

  return lines.join('\n');
}

// Helper function to generate disaster events from sentiment posts
const generateDisasterEvents = async (posts: any[]): Promise<void> => {
    if (posts.length === 0) return;

    // Group posts by day to identify patterns
    const postsByDay: {[key: string]: {
        posts: any[],
        count: number,
        sentiments: {[key: string]: number}
    }} = {};

    // Group posts by day (YYYY-MM-DD)
    for (const post of posts) {
        const day = new Date(post.timestamp).toISOString().split('T')[0];

        if (!postsByDay[day]) {
            postsByDay[day] = {
                posts: [],
                count: 0,
                sentiments: {}
            };
        }

        postsByDay[day].posts.push(post);
        postsByDay[day].count++;

        // Count sentiment occurrences
        const sentiment = post.sentiment;
        postsByDay[day].sentiments[sentiment] = (postsByDay[day].sentiments[sentiment] || 0) + 1;
    }

    // Process each day with sufficient posts (at least 3)
    for (const [day, data] of Object.entries(postsByDay)) {
        if (data.count < 3) continue;

        // Find dominant sentiment
        let maxCount = 0;
        let dominantSentiment: string | null = null;

        for (const [sentiment, count] of Object.entries(data.sentiments)) {
            if (count > maxCount) {
                maxCount = count;
                dominantSentiment = sentiment;
            }
        }

        // Extract disaster type and location from text content
        const texts = data.posts.map(p => p.text.toLowerCase());
        let disasterType = null;
        let location = null;

        // Enhanced disaster type detection with more variations
        const disasterKeywords = {
            "Earthquake": ['lindol', 'earthquake', 'quake', 'tremor', 'lumindol', 'yugto', 'lindol na malakas', 'paglindol'],
            "Flood": ['baha', 'flood', 'pagbaha', 'pagbabaha', 'bumaha', 'tubig', 'binaha', 'napabaha', 'flash flood'],
            "Typhoon": ['bagyo', 'typhoon', 'storm', 'cyclone', 'hurricane', 'bagyong', 'unos', 'habagat', 'super typhoon'],
            "Fire": ['sunog', 'fire', 'nasunog', 'burning', 'apoy', 'silab', 'nagkasunog', 'wildfire', 'forest fire'],
            "Volcanic Eruption": ['bulkan', 'volcano', 'eruption', 'ash fall', 'lava', 'ashfall', 'bulkang', 'pumutok', 'sumabog'],
            "Landslide": ['landslide', 'pagguho', 'guho', 'mudslide', 'rockslide', 'avalanche', 'pagguho ng lupa', 'collapsed'],
            "Tsunami": ['tsunami', 'tidal wave', 'daluyong', 'alon', 'malalaking alon']
        };

        // Check each disaster type in texts
        for (const [type, keywords] of Object.entries(disasterKeywords)) {
            if (texts.some(text => keywords.some(keyword => text.includes(keyword)))) {
                disasterType = type;
                break;
            }
        }

        // Enhanced location detection with more Philippine locations
        const locations = [
            'Manila', 'Quezon City', 'Cebu', 'Davao', 'Mindanao', 'Luzon',
            'Visayas', 'Palawan', 'Boracay', 'Baguio', 'Bohol', 'Iloilo',
            'Batangas', 'Zambales', 'Pampanga', 'Bicol', 'Leyte', 'Samar',
            'Pangasinan', 'Tarlac', 'Cagayan', 'Bulacan', 'Cavite', 'Laguna',
            'Rizal', 'Marikina', 'Makati', 'Pasig', 'Taguig', 'Pasay', 'Mandaluyong',
            'Parañaque', 'Caloocan', 'Valenzuela', 'Muntinlupa', 'Malabon', 'Navotas',
            'San Juan', 'Las Piñas', 'Pateros', 'Nueva Ecija', 'Benguet', 'Albay',
            'Catanduanes', 'Sorsogon', 'Camarines Sur', 'Camarines Norte', 'Marinduque'
        ];

        // Try to find locations in text more aggressively
        for (const text of texts) {
            const textLower = text.toLowerCase();
            for (const loc of locations) {
                if (textLower.includes(loc.toLowerCase())) {
                    location = loc;
                    break;
                }
            }
            if (location) break;
        }

        if (disasterType && (location || dominantSentiment)) {
            // Create the disaster event
            await storage.createDisasterEvent({
                name: `${disasterType} Incident on ${new Date(day).toLocaleDateString()}`,
                description: `Based on ${data.count} social media reports. Sample content: ${data.posts[0].text}`,
                timestamp: new Date(day),
                location,
                type: disasterType,
                sentimentImpact: dominantSentiment || undefined
            });
        }
    }
};


export async function registerRoutes(app: Express): Promise<Server> {
  // Add the SSE endpoint inside registerRoutes
  app.get('/api/upload-progress/:sessionId', (req: Request, res: Response) => {
    const sessionId = req.params.sessionId;

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    const sendProgress = () => {
      const progress = uploadProgressMap.get(sessionId);
      if (progress) {
        const now = Date.now();
        if (now - progress.timestamp >= 50) { // Increased frequency for smoother updates
          res.write(`data: ${JSON.stringify({
            processed: progress.processed,
            total: progress.total,
            stage: progress.stage,
            percentage: Math.round((progress.processed / progress.total) * 100),
            error: progress.error
          })}\n\n`);
          progress.timestamp = now;
        }
      }
    };

    const progressInterval = setInterval(sendProgress, 50);

    req.on('close', () => {
      clearInterval(progressInterval);
      uploadProgressMap.delete(sessionId);
    });
  });

  // Authentication Routes
  app.post('/api/auth/signup', async (req: Request, res: Response) => {
    try {
      const { username, password, email, fullName } = req.body;

      // Check if user already exists
      const existingUser = await storage.getUserByUsername(username);
      if (existingUser) {
        return res.status(400).json({ error: "Username already taken" });
      }

      // Create new user
      const user = await storage.createUser({
        username,
        password,
        email,
        fullName,
        role: 'user'
      });

      // Create session
      const token = await storage.createSession(user.id);

      res.json({ token });
    } catch (error) {
      res.status(500).json({
        error: "Failed to create user",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.post('/api/auth/login', async (req: Request, res: Response) => {
    try {
      const { username, password } = req.body;
      const user = await storage.loginUser({ username, password });

      if (!user) {
        return res.status(401).json({ error: "Invalid credentials" });
      }

      const token = await storage.createSession(user.id);
      res.json({ token });
    } catch (error) {
      res.status(500).json({
        error: "Login failed",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.get('/api/auth/me', async (req: Request, res: Response) => {
    try {
      const token = req.headers.authorization?.split(' ')[1];
      if (!token) {
        return res.status(401).json({ error: "No token provided" });
      }

      const user = await storage.validateSession(token);
      if (!user) {
        return res.status(401).json({ error: "Invalid or expired token" });
      }

      // Don't send password in response
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error) {
      res.status(500).json({
        error: "Failed to get user info",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Helper function to generate disaster events from sentiment posts

  // Get all sentiment posts
  app.get('/api/sentiment-posts', async (req: Request, res: Response) => {
    try {
      const posts = await storage.getSentimentPosts();
      res.json(posts);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get sentiment posts by file id
  app.get('/api/sentiment-posts/file/:fileId', async (req: Request, res: Response) => {
    try {
      const fileId = parseInt(req.params.fileId);
      const posts = await storage.getSentimentPostsByFileId(fileId);
      res.json(posts);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch sentiment posts" });
    }
  });

  // Get all disaster events
  app.get('/api/disaster-events', async (req: Request, res: Response) => {
    try {
      const events = await storage.getDisasterEvents();
      res.json(events);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch disaster events" });
    }
  });

  // Get all analyzed files
  app.get('/api/analyzed-files', async (req: Request, res: Response) => {
    try {
      const files = await storage.getAnalyzedFiles();
      res.json(files);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch analyzed files" });
    }
  });

  // Get specific analyzed file
  app.get('/api/analyzed-files/:id', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      const file = await storage.getAnalyzedFile(id);

      if (!file) {
        return res.status(404).json({ error: "Analyzed file not found" });
      }

      res.json(file);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch analyzed file" });
    }
  });

  // Update file upload endpoint with enhanced emotion analysis
  app.post('/api/upload-csv', upload.single('file'), async (req: Request, res: Response) => {
    let sessionId: string | undefined;
    let updateProgress: ((processed: number, stage: string, error?: string) => void) | undefined;

    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      sessionId = req.headers['x-session-id'] as string;
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }

      const fileBuffer = req.file.buffer;
      const originalFilename = req.file.originalname;
      const fileContent = fileBuffer.toString('utf-8');
      const totalRecords = fileContent.split('\n').length - 1;

      // Initialize progress tracking
      uploadProgressMap.set(sessionId, {
        processed: 0,
        total: totalRecords,
        stage: 'Starting analysis',
        timestamp: Date.now()
      });

      updateProgress = (processed: number, stage: string, error?: string) => {
        if (sessionId) {
          const progress = uploadProgressMap.get(sessionId);
          if (progress) {
            progress.processed = processed;
            progress.stage = stage;
            progress.timestamp = Date.now();
            progress.error = error;
          }
        }
      };

      // Process the CSV with enhanced emotion analysis
      const { data, storedFilename, recordCount } = await pythonService.processCSV(
        fileBuffer,
        originalFilename,
        (processed: number, stage: string) => {
          updateProgress?.(processed, `Analyzing emotions in raw data: ${stage}`);
        }
      );

      // Save the analyzed file record
      const analyzedFile = await storage.createAnalyzedFile(
        insertAnalyzedFileSchema.parse({
          originalName: originalFilename,
          storedName: storedFilename,
          recordCount: recordCount,
          evaluationMetrics: data.metrics
        })
      );

      // Enhanced processing of sentiment posts with disaster detection
      const sentimentPosts = await Promise.all(
        data.results.map(async (result: any) => {
          // Then analyze emotion with disaster context
          const emotionAnalysis = analyzeEmotionWithContext(
            result.text,
            result.disasterType
          );

          return storage.createSentimentPost(
            insertSentimentPostSchema.parse({
              text: result.text,
              timestamp: new Date(result.timestamp),
              source: `CSV Upload: ${originalFilename}`,
              language: result.language,
              sentiment: emotionAnalysis.emotion,
              confidence: emotionAnalysis.confidence,
              explanation: `${emotionAnalysis.explanation}`,
              location: result.location || null,
              disasterType: emotionAnalysis.detectedDisaster?.type || null,
              fileId: analyzedFile.id
            })
          );
        })
      );

      // Prioritize disaster event generation for uploaded data
      await generateDisasterEvents(sentimentPosts);

      // Final progress update
      updateProgress?.(totalRecords, 'Analysis complete');

      res.json({
        file: analyzedFile,
        posts: sentimentPosts,
        metrics: {
          ...data.metrics,
          emotionBreakdown: sentimentPosts.reduce((acc: Record<string, number>, post) => {
            acc[post.sentiment] = (acc[post.sentiment] || 0) + 1;
            return acc;
          }, {}),
          averageConfidence: sentimentPosts.reduce((sum, post) => sum + post.confidence, 0) / sentimentPosts.length
        }
      });
    } catch (error) {
      console.error("Error processing CSV:", error);
      if (sessionId && updateProgress) {
        updateProgress(0, 'Error', error instanceof Error ? error.message : String(error));
      }
      res.status(500).json({
        error: "Failed to process CSV file",
        details: error instanceof Error ? error.message : String(error)
      });
    } finally {
      // Cleanup progress tracking after 5 seconds
      if (sessionId) {
        setTimeout(() => {
          if (sessionId) {  // Additional check to satisfy TypeScript
            uploadProgressMap.delete(sessionId);
          }
        }, 5000);
      }
    }
  });

  // Analyze text (single or batch)
  app.post('/api/analyze-text', async (req: Request, res: Response) => {
    try {
      const { text, texts, source = 'Manual Input' } = req.body;

      if (!text && (!texts || !Array.isArray(texts) || texts.length === 0)) {
        return res.status(400).json({ error: "No text provided. Send either 'text' or 'texts' array in the request body" });
      }

      // Process single text
      if (text) {
        const result = await pythonService.analyzeSentiment(text);
        const emotionAnalysis = analyzeEmotionWithContext(text, result.disasterType ?? null);

        const sentimentPost = await storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: emotionAnalysis.emotion,
            confidence: emotionAnalysis.confidence,
            explanation: emotionAnalysis.explanation,
            location: result.location || null,
            disasterType: result.disasterType || null,
            fileId: null
          })
        );

        return res.json({
          post: sentimentPost,
          analysis: emotionAnalysis
        });
      }

      // Process multiple texts with enhanced analysis
      const processResults = await Promise.all(texts.map(async (textItem: string) => {
        const result = await pythonService.analyzeSentiment(textItem);
        const emotionAnalysis = analyzeEmotionWithContext(textItem, result.disasterType ?? null);

        const post = await storage.createSentimentPost(
          insertSentimentPostSchema.parse({
            text: textItem,
            timestamp: new Date(),
            source,
            language: result.language,
            sentiment: emotionAnalysis.emotion,
            confidence: emotionAnalysis.confidence,
            explanation: emotionAnalysis.explanation,
            location: result.location || null,
            disasterType: result.disasterType || null,
            fileId: null
          })
        );

        return {
          post,
          analysis: emotionAnalysis
        };
      }));

      // Process uploaded file data with priority
      if (source.includes('CSV') || source.includes('Upload')) {
        await generateDisasterEvents(processResults.map(r => r.post));
      }

      res.json({
        results: processResults,
        summary: {
          totalAnalyzed: processResults.length,
          emotionBreakdown: processResults.reduce((acc: Record<string, number>, curr) => {
            const emotion = curr.analysis.emotion;
            acc[emotion] = (acc[emotion] || 0) + 1;
            return acc;
          }, {}),
          averageConfidence: processResults.reduce((sum, curr) => sum + curr.analysis.confidence, 0) / processResults.length
        }
      });
    } catch (error) {
      res.status(500).json({
        error: "Failed to analyze text",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Delete all data endpoint
  app.delete('/api/delete-all-data', async (req: Request, res: Response) => {
    try {
      // Delete all data
      await storage.deleteAllData();

      res.json({
        success: true,
        message: "All data has been deleted successfully"
      });
    } catch (error) {
      res.status(500).json({
        error: "Failed to delete all data",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Delete specific sentiment post endpoint
  app.delete('/api/sentiment-posts/:id', async (req: Request, res: Response) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ error: "Invalid post ID" });
      }

      await storage.deleteSentimentPost(id);

      res.json({
        success: true,
        message: `Sentiment post with ID ${id} has been deleted successfully`
      });
    } catch (error) {
      res.status(500).json({
        error: "Failed to delete sentiment post",
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}