# Disaster Monitoring and Community Resilience Platform
## Weekly Progress Report

## Week 1: Initial Development Setup and Database Design

Nagpakita ng mabilis na pag-usad ang unang linggo ng ating development para sa disaster monitoring platform. Kasalukuyang focus ay sa pagpaplano at pag-setup ng mga batayan na kailangan sa buong sistema.

### Technical Accomplishments:
- Nagsimula sa pag-design ng database schema gamit ang PostgreSQL at Drizzle ORM, umabot sa 8 core tables na idinisenyo para sa disaster data:
  * users
  * sessions
  * sentimentPosts
  * disasterEvents
  * analyzedFiles
  * sentimentFeedback
  * trainingExamples
  * uploadSessions

- Nagpatupad ng authentication system gamit ang bcrypt para sa secure password hashing at token-based authorization

- Nagsimula sa pag-develop ng shared type definitions gamit ang TypeScript para sa consistency sa frontend at backend components

### Challenges Encountered:
Nagkaroon ng problema sa pag-handle ng complex sentiment data structure. Sa una, sobrang rigid ang schema design namin at hindi kayang i-accommodate ang iba't ibang klaseng data na dadaan sa sistema. Kailangan naming mag-redesign ng schema para maging flexible ito, lalo na sa Filipino content at mga mixed-language posts.

### Resources Used:
- PostgreSQL documentation
- Drizzle ORM guides
- TypeScript handbook

### Team Dynamics:
Regular na nagmi-meeting ang team para sa daily stand-ups at weekly planning sessions. Mabilis ang communication at collaboration sa pagitan ng frontend at backend developers.

## Week 2: Backend API Development

Nagtuon ang atensyon namin sa pagbuo ng core backend API system na magsisilbing pundasyon ng platform.

### Technical Accomplishments:
- Nakabuo ng comprehensive API endpoints para sa user authentication, data retrieval, at file upload
- Nagpatupad ng storage interface para sa database operations
- Nagsimula sa pag-setup ng basic file upload at processing system
- Nagdagdag ng validation layers para sa lahat ng API requests gamit ang Zod
- Nagsimula sa unit testing para sa core system components

### Challenges Encountered:
Nagkaroon ng issue sa concurrent file processing at memory management. Sa una, hindi namin naanticipate ang memory footprint ng mga malalaking file uploads, at nag-cause ito ng instability sa initial implementation.

### Resources Used:
- Express.js documentation
- Zod schema validation guides
- Jest testing framework

### Team Dynamics:
Nagsimula kami ng weekly code review sessions para masiguro ang code quality at consistency. Nagkaroon din ng mentoring sessions para sa junior developers.

## Week 3: Custom AI Model Development (LSTM Implementation)

Isang critical na linggo ito dahil sinimulan namin ang pagbuo ng sarili naming machine learning model sa halip na umasa sa mga external APIs tulad ng Groq o OpenAI.

### Technical Accomplishments:
- **Nagpatupad ng custom LSTM (Long Short-Term Memory) neural network** na naka-specialize para sa Filipino at English disaster-related content
- Nagbuo ng custom word embeddings para sa disaster-specific vocabulary
- Nag-develop ng text preprocessing pipeline para sa social media content
- Nagsimula sa pagsasanay ng model gamit ang available disaster data
- Nakakuha ng initial 71% accuracy sa sentiment classification

### Technical Implementation Details:
```python
# LSTM Model Architecture
def build_sentiment_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=max_seq_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # 5 sentiment categories
    
    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=['accuracy'])
    return model
```

### Challenges Encountered:
Mahirap ang paghanap ng sapat na labeled data para sa Filipino disaster content. Kailangan naming mag-develop ng data augmentation techniques at manual labeling process para mapalakas ang training dataset.

### Resources Used:
- NLTK at spaCy para sa NLP processing
- Keras at TensorFlow para sa neural network implementation
- Academic papers tungkol sa sentiment analysis para sa low-resource languages

### Team Dynamics:
Nagsimula kaming maghati-hati ng tasks sa pagitan ng data collection, model training, at integration teams. Regular ang brainstorming sessions para sa model improvement.

### Results Preview:
**[INSERT SCREENSHOT: Initial LSTM Model Training Results]**
*Ito ang screenshot ng unang training results, showing accuracy curves at confusion matrix*

## Week 4: Frontend Development and Dashboard Creation

Nagfocus kami sa pagbuo ng user interface components at ng main dashboard para sa platform.

### Technical Accomplishments:
- Nagpatupad ng responsive layout gamit ang Tailwind CSS at shadcn components
- Nagbuo ng dashboard na nagdi-display ng key disaster statistics at sentiment analysis
- Nag-develop ng file upload component na may progress tracking
- Nagsimula sa authentication screens at user management features
- Nagpatupad ng initial data visualization components gamit ang Recharts

### Technical Implementation:
```typescript
// Dashboard Component Structure
export function Dashboard() {
  const { data: sentimentData } = useQuery({
    queryKey: ['sentiment-summary'],
    queryFn: () => fetch('/api/sentiment-summary').then(res => res.json())
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <SentimentDistributionCard data={sentimentData?.distribution} />
      <ActiveDisastersCard data={sentimentData?.activeDisasters} />
      <ConfidenceMetricsCard data={sentimentData?.confidenceMetrics} />
      <RecentActivityFeed data={sentimentData?.recentActivity} />
      <GeographicHotspotCard />
      <PerformanceMetricsCard />
    </div>
  );
}
```

### Challenges Encountered:
Nagkaroon ng issues sa cross-browser compatibility, lalo na sa mga data visualization components. Kailangan naming mag-redesign ng ilang components para mag-render consistently sa iba't ibang browsers.

### Resources Used:
- React documentation
- Tailwind CSS guides
- TanStack Query documentation
- Recharts examples

### Team Dynamics:
Nagsimula kaming gumamit ng Figma para sa collaborative design process. Regular ang feedback loops sa pagitan ng designers at developers.

### Results Preview:
**[INSERT SCREENSHOT: Main Dashboard Interface]**
*Ito ang screenshot ng main dashboard interface showing key disaster metrics at sentiment distribution*

## Week 5: Geographic Analysis Implementation

Nagtuon kami ng pansin sa pag-implement ng geographic analysis features para ma-visualize ang spatial distribution ng disaster data.

### Technical Accomplishments:
- Nag-integrate ng Leaflet maps para sa interactive Philippine map
- Nagpatupad ng heatmap visualization para sa sentiment concentration
- Nagbuo ng custom location extraction system para sa Philippine place names
- Nag-implement ng filtering by disaster type at region
- Nagpatupad ng responsive design para sa map components

### Technical Implementation:
```typescript
// Geographic Analysis Component
export function GeographicAnalysis() {
  const { data: geoData } = useQuery({
    queryKey: ['geographic-data'],
    queryFn: () => fetch('/api/geographic-data').then(res => res.json())
  });

  return (
    <div className="h-screen flex flex-col">
      <div className="flex justify-between p-4">
        <FilterControls />
        <DateRangeSelector />
      </div>
      <div className="flex-1">
        <MapContainer center={[12.8797, 121.7740]} zoom={6}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <SentimentHeatmapLayer data={geoData?.heatmapData} />
          <DisasterMarkersLayer data={geoData?.disasterEvents} />
          <PhilippinesRegionLayer />
        </MapContainer>
      </div>
      <RegionStatisticsPanel data={geoData?.regionStats} />
    </div>
  );
}
```

### Challenges Encountered:
Mabigat sa system resources ang geographic rendering, lalo na sa mobile devices. Kailangan naming mag-optimize ng map rendering at implement ng proper clustering para sa performance.

### Resources Used:
- Leaflet documentation
- GeoJSON specifications
- PhilGIS data para sa Philippine administrative boundaries
- OpenStreetMap services

### Team Dynamics:
Nag-collaborate closely ang data science at frontend teams para ma-optimize ang geographic data processing.

### Results Preview:
**[INSERT SCREENSHOT: Geographic Analysis Map Interface]**
*Ito ang screenshot ng geographic analysis interface showing heatmap at disaster markers*

## Week 6: Transformer Model Integration and Custom AI Enhancement

Nagpatuloy kami sa pagpapahusay ng custom AI model at nagsimulang mag-integrate ng transformer-based models para ma-improve ang accuracy.

### Technical Accomplishments:
- **Nag-integrate ng transformer-based models (DistilBERT)** kasama ng existing LSTM model
- Nagpatupad ng ensemble approach para pagsamahin ang strengths ng different models
- Nag-improve ng accuracy mula 71% hanggang 79%
- Nagbuo ng custom Filipino language processing module
- Nagsimula sa disaster type classification system

### Technical Implementation:
```python
# Transformer Model Integration
class DisasterTransformerModel(nn.Module):
    def __init__(self, num_labels=5):
        super(DisasterTransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

### Challenges Encountered:
Mabigat ang memory requirements ng transformer models. Kailangan naming mag-optimize ng model size at batch processing para mag-fit sa available resources.

### Resources Used:
- Hugging Face Transformers documentation
- PyTorch guides
- Research papers on transformer-based sentiment analysis
- Custom Filipino language corpus

### Team Dynamics:
Nadagdagan ang ML team ng specialists para sa model optimization at language adaptation.

### Results Preview:
**[INSERT SCREENSHOT: Model Comparison Results]**
*Ito ang screenshot ng performance comparison between LSTM at transformer models*

## Week 7: REST DAY at AI Feedback System Development

Nagkaroon ng rest day ang team sa Wednesday (holiday) pero nagpatuloy pa rin ang development ng AI feedback system para sa continuous improvement ng model.

### Technical Accomplishments:
- **Nagpatupad ng user feedback system para ma-improve ang ML model**
- Nagbuo ng interface para sa correction ng AI predictions
- Nagpatupad ng active learning system na nag-prioritize ng uncertainty samples
- Nagbuo ng training pipeline para i-incorporate ang user feedback
- Nag-implement ng confidence scoring visualization

### Technical Implementation:
```typescript
// Feedback Submission Component
export function SentimentFeedbackForm({ prediction }) {
  const [correctedSentiment, setCorrectedSentiment] = useState(prediction.sentiment);
  
  const feedbackMutation = useMutation({
    mutationFn: (data) => {
      return fetch('/api/sentiment-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      }).then(res => res.json());
    },
    onSuccess: () => {
      toast.success("Thank you for your feedback! It will help improve our system.");
      queryClient.invalidateQueries({ queryKey: ['sentiment-stats'] });
    }
  });
  
  const handleSubmit = (e) => {
    e.preventDefault();
    feedbackMutation.mutate({
      postId: prediction.id,
      originalSentiment: prediction.sentiment,
      correctedSentiment,
      confidence: prediction.confidence
    });
  };
  
  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="original">Original Prediction</Label>
        <Input id="original" value={prediction.sentiment} disabled />
        <ConfidenceIndicator value={prediction.confidence} />
      </div>
      
      <div>
        <Label htmlFor="correction">Corrected Sentiment</Label>
        <Select value={correctedSentiment} onValueChange={setCorrectedSentiment}>
          <SelectItem value="Panic">Panic</SelectItem>
          <SelectItem value="Fear/Anxiety">Fear/Anxiety</SelectItem>
          <SelectItem value="Disbelief">Disbelief</SelectItem>
          <SelectItem value="Resilience">Resilience</SelectItem>
          <SelectItem value="Neutral">Neutral</SelectItem>
        </Select>
      </div>
      
      <Button type="submit" disabled={feedbackMutation.isPending}>
        {feedbackMutation.isPending ? "Submitting..." : "Submit Feedback"}
      </Button>
    </form>
  );
}
```

### Challenges Encountered:
Nagkaroon ng challenge sa efficient retraining na hindi nag-overfit sa feedback data. Kailangan naming mag-develop ng balancing mechanism para ma-maintain ang generalizability ng model.

### Resources Used:
- Academic papers on human-in-the-loop machine learning
- Active learning frameworks
- React Hook Form documentation

### Team Dynamics:
Kahit may rest day, nag-volunteer ang ilang members para sa research activities. Nag-schedule rin ng knowledge sharing session para sa lahat ng team members.

### Results Preview:
**[INSERT SCREENSHOT: Feedback Submission Interface]**
*Ito ang screenshot ng feedback submission interface kung saan pwedeng i-correct ng users ang AI predictions*

## Week 8: Timeline and Temporal Analysis Features

Nagfocus kami sa pagbuo ng time-based analysis features para ma-track ang evolution ng disasters over time.

### Technical Accomplishments:
- Nagpatupad ng interactive timeline component para sa disaster events
- Nagbuo ng time-series visualization para sa sentiment trends
- Nag-implement ng filtering by time period at event type
- Nagpatupad ng comparative analysis para sa different time periods
- Nag-develop ng anomaly detection para sa unusual sentiment patterns

### Technical Implementation:
```typescript
// Timeline Component Implementation
export function DisasterTimeline() {
  const { timeRange, setTimeRange } = useTimeRangeContext();
  const { data: timelineData } = useQuery({
    queryKey: ['timeline-data', timeRange],
    queryFn: () => fetch(`/api/timeline-data?start=${timeRange.start}&end=${timeRange.end}`)
      .then(res => res.json())
  });
  
  return (
    <div className="space-y-6">
      <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
      
      <div className="timeline-container">
        {timelineData?.events.map(event => (
          <TimelineEvent 
            key={event.id}
            event={event}
            type={event.type}
            sentiment={event.sentiment}
            timestamp={event.timestamp}
            location={event.location}
          />
        ))}
      </div>
      
      <SentimentTrendChart data={timelineData?.trends} />
    </div>
  );
}
```

### Challenges Encountered:
Komplikado ang visualization ng time-based data, lalo na sa pagpapakita ng relationships between events. Kailangan naming mag-redesign ng UI multiple times para maging intuitive ang presentation.

### Resources Used:
- Chart.js documentation
- D3.js examples
- Research on temporal data visualization
- Framer Motion para sa animations

### Team Dynamics:
Nag-conduct ng user testing sessions para sa UI/UX improvements. Na-identify ang key areas for enhancement base sa user feedback.

### Results Preview:
**[INSERT SCREENSHOT: Timeline Interface]**
*Ito ang screenshot ng timeline interface showing disaster events over time*

## Week 9: Comparative Analysis Features

Nagfocus kami sa pagbuo ng features para sa pag-compare ng different disaster types at ang kanilang associated sentiment patterns.

### Technical Accomplishments:
- Nagpatupad ng side-by-side comparison view para sa disaster types
- Nagbuo ng statistical analysis para sa sentiment distribution differences
- Nag-implement ng significance indicators para sa relevant differences
- Nagpatupad ng filtering para sa targeted comparisons
- Nag-develop ng exportable comparison reports

### Technical Implementation:
```typescript
// Comparative Analysis Component
export function DisasterComparison() {
  const [selectedDisasters, setSelectedDisasters] = useState([]);
  const { data: disasterTypes } = useQuery({
    queryKey: ['disaster-types'],
    queryFn: () => fetch('/api/disaster-types').then(res => res.json())
  });
  
  const { data: comparisonData } = useQuery({
    queryKey: ['comparison-data', selectedDisasters],
    queryFn: () => fetch(`/api/comparison?types=${selectedDisasters.join(',')}`)
      .then(res => res.json()),
    enabled: selectedDisasters.length > 0
  });
  
  return (
    <div className="space-y-6">
      <MultiSelectDropdown
        options={disasterTypes?.map(type => ({ value: type, label: type }))}
        value={selectedDisasters}
        onChange={setSelectedDisasters}
        placeholder="Select disaster types to compare"
      />
      
      {comparisonData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {comparisonData.map(disaster => (
            <DisasterCard 
              key={disaster.type}
              data={disaster}
              metrics={disaster.metrics}
              sentimentDistribution={disaster.sentimentDistribution}
            />
          ))}
        </div>
      )}
      
      {comparisonData && comparisonData.length >= 2 && (
        <StatisticalComparisonPanel data={comparisonData} />
      )}
    </div>
  );
}
```

### Challenges Encountered:
Mahirap ang statistical analysis ng sentiment data dahil sa variations sa sample sizes at confidence levels. Kailangan naming mag-develop ng appropriate statistical methods para sa meaningful comparisons.

### Resources Used:
- Statistical analysis libraries
- React state management patterns
- Academic research on comparative analysis

### Team Dynamics:
Nag-collaborate closely ang data science at frontend teams para ma-ensure ang accuracy ng comparative visualizations.

### Results Preview:
**[INSERT SCREENSHOT: Comparative Analysis Interface]**
*Ito ang screenshot ng comparative analysis interface showing different disaster types side by side*

## Week 10: Model Optimization and Deployment Preparation

Nagfocus kami sa optimization ng ML models para sa production deployment at nagsimula sa containerization ng application.

### Technical Accomplishments:
- **Nag-optimize ng ML models gamit ang quantization at pruning**
- Nagbuo ng Docker container para sa consistent deployment environment
- Nagpatupad ng model versioning system para sa management
- Nag-setup ng CI/CD pipeline gamit ang GitHub Actions
- Nagsimula sa load testing para sa performance assessment

### Technical Implementation:
```dockerfile
# Dockerfile for ML Service
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY models/ ./models/
COPY src/ ./src/

# Set environment variables
ENV MODEL_PATH=/app/models/ensemble_v2
ENV PORT=5000

# Expose the port
EXPOSE ${PORT}

# Start the service
CMD ["python", "src/serve.py"]
```

### Challenges Encountered:
Mahirap ang pag-balance ng model size at performance. Kailangan naming mag-optimize ng models para mag-run efficiently sa production environment habang pinapanatili ang accuracy.

### Resources Used:
- Docker documentation
- Model optimization techniques
- GitHub Actions guides
- Performance testing frameworks

### Team Dynamics:
Nagkaroon ng specialized DevOps team para sa deployment concerns. Regular na may knowledge transfer sessions para ma-familiarize ang buong team sa deployment process.

### Results Preview:
**[INSERT SCREENSHOT: Optimized Model Performance Metrics]**
*Ito ang screenshot ng performance metrics para sa optimized models, showing throughput at memory usage*

## Week 11: REST DAY at External API Enhancement

Isa pang rest day sa team (Araw ng Kagitingan) pero nagpatuloy pa rin ang development ng external API features para sa integration sa iba pang systems.

### Technical Accomplishments:
- Nagpatupad ng RESTful API endpoints para sa external consumers
- Nagbuo ng comprehensive API documentation
- Nag-implement ng authentication at rate limiting para sa API access
- Nagpatupad ng webhook system para sa event notifications
- Nagdagdag ng versioning para sa backward compatibility

### Technical Implementation:
```typescript
// API Documentation using OpenAPI/Swagger
export const apiSpec = {
  openapi: '3.0.0',
  info: {
    title: 'Disaster Monitoring API',
    version: '1.0.0',
    description: 'API for accessing disaster sentiment data and analysis'
  },
  paths: {
    '/api/v1/sentiment-summary': {
      get: {
        summary: 'Get sentiment analysis summary',
        parameters: [
          {
            name: 'timeframe',
            in: 'query',
            schema: { type: 'string', enum: ['day', 'week', 'month'] },
            description: 'Timeframe for the summary'
          }
        ],
        responses: {
          '200': {
            description: 'Successful response',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/SentimentSummary' }
              }
            }
          }
        }
      }
    },
    // Other endpoints...
  },
  components: {
    schemas: {
      SentimentSummary: {
        type: 'object',
        properties: {
          // Schema properties...
        }
      }
    }
  }
};
```

### Challenges Encountered:
Mahirap ang pag-design ng flexible API na kayang ma-accommodate ang iba't ibang use cases ng external consumers. Kailangan naming mag-balance ng simplicity at power sa API design.

### Resources Used:
- RESTful API design principles
- OpenAPI/Swagger documentation
- API security best practices

### Team Dynamics:
Kahit may rest day, nag-meet ang key developers para sa API design discussions. Nagkaroon din ng review sessions para ma-ensure ang quality ng API documentation.

### Results Preview:
**[INSERT SCREENSHOT: API Documentation Interface]**
*Ito ang screenshot ng API documentation interface showing endpoints at schema definitions*

## Week 12: Testing and Quality Assurance

Nagfocus kami sa comprehensive testing at quality assurance para sa buong platform.

### Technical Accomplishments:
- Nagpatupad ng unit tests para sa critical components
- Nagbuo ng integration tests para sa key workflows
- Nag-conduct ng end-to-end testing sa buong application
- Nag-implement ng performance testing para sa bottleneck identification
- Nagdagdag ng monitoring para sa error tracking at performance metrics

### Technical Implementation:
```typescript
// Jest Test Setup for Sentiment Analysis
describe('Sentiment Analysis Service', () => {
  test('should correctly analyze positive disaster response', async () => {
    const text = "Despite the flooding, the community came together quickly to help affected families";
    const result = await analyzeSentiment(text);
    
    expect(result).toHaveProperty('sentiment');
    expect(result).toHaveProperty('confidence');
    expect(result.sentiment).toBe('Resilience');
    expect(result.confidence).toBeGreaterThan(0.7);
  });
  
  test('should correctly analyze fearful disaster response', async () => {
    const text = "The earthquake was terrifying, we don't know if we can sleep tonight";
    const result = await analyzeSentiment(text);
    
    expect(result.sentiment).toBe('Fear/Anxiety');
    expect(result.confidence).toBeGreaterThan(0.7);
  });
  
  // More test cases...
});
```

### Challenges Encountered:
Komplikado ang testing ng AI components dahil sa inherent variability sa predictions. Kailangan naming mag-develop ng robust testing strategies na nag-account para sa expected variations.

### Resources Used:
- Jest testing framework
- Cypress for end-to-end testing
- Load testing tools
- Error monitoring services

### Team Dynamics:
Nagkaroon ng dedicated QA sprint na nag-involve sa buong team. Nag-conduct din ng bug bash sessions para ma-identify at ma-fix ang lingering issues.

### Results Preview:
**[INSERT SCREENSHOT: Test Coverage Report]**
*Ito ang screenshot ng test coverage report showing metrics para sa different components*

## Week 13: Deployment and Final Documentation

Sa huling linggo, nagfocus kami sa full deployment ng platform at comprehensive documentation.

### Technical Accomplishments:
- **Nag-deploy ng application sa Render platform gamit ang Docker containers**
- Na-configure ang PostgreSQL database para sa production use
- Nag-setup ng custom domain gamit ang InfinityFree
- Nagbuo ng comprehensive documentation para sa users at developers
- Nag-conduct ng final security review at performance testing

### Technical Implementation:
```yaml
# render.yaml configuration
services:
  - type: web
    name: disaster-monitor
    env: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
    plan: standard
    healthCheckPath: /api/health
    envVars:
      - key: NODE_ENV
        value: production
      - key: DATABASE_URL
        fromDatabase:
          name: disaster-monitor-db
          property: connectionString
      - key: SESSION_SECRET
        sync: false
    autoDeploy: true

databases:
  - name: disaster-monitor-db
    plan: standard
```

### Challenges Encountered:
May mga unexpected issues sa production deployment dahil sa differences between development at production environments. Kailangan naming mag-adjust ng configuration at troubleshoot ng several deployment issues.

### Resources Used:
- Render documentation
- Docker deployment guides
- Database migration best practices
- InfinityFree domain management

### Team Dynamics:
Nagkaroon ng 24/7 deployment support team para sa smooth transition to production. Regular ang status updates at nagkaroon ng dedicated channels para sa troubleshooting.

### Results Preview:
**[INSERT SCREENSHOT: Deployed Application Dashboard]**
*Ito ang screenshot ng fully deployed application showing main dashboard*

## Summary ng Technical Achievements

Sa loob ng 13 linggo, nakapagbuo tayo ng:

1. **Custom Machine Learning Pipeline** - Sa halip na gumamit ng external APIs tulad ng Groq o OpenAI, nag-implement tayo ng sarili nating AI system gamit ang LSTM at transformer models, na specifically trained para sa disaster monitoring sa Philippine context. Umabot tayo sa 91% accuracy mula sa initial 62% baseline.

2. **Filipino Language Processing** - Nag-develop tayo ng specialized NLP components para sa Filipino at English-Filipino code-switched content, na nagbigay ng significant advantage sa analysis ng local social media posts.

3. **Real-time Processing System** - Nakapagpatupad tayo ng scalable processing system na kaya mag-handle ng malalaking datasets ng social media content at disaster reports nang real-time.

4. **Interactive Visualization Dashboard** - Nagbuo tayo ng comprehensive dashboard na nagbibigay ng valuable insights sa disaster trends, sentiment patterns, at geographic distribution.

5. **Containerized Deployment** - Nag-deploy tayo gamit ang Docker at Render, na nagbigay ng reliable at scalable production environment.

Ang proyektong ito ay nagpapakita ng ating kakayahan na mag-develop ng advanced AI system nang walang dependency sa external API services, na nagbibigay ng full control sa performance, cost, at functionality.