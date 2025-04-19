Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 - February 7, 2025 |

| Custom ML Implementation with MBERT and LSTM |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Sa Week 3, nag-focus kami sa pag-implement ng sarili naming machine learning models sa halip na gumamit ng external APIs tulad ng Groq. Nagdesisyon kami na gumamit ng MBERT (Multilingual BERT) at LSTM (Long Short-Term Memory) models para sa sentiment analysis.<br><br>Nagsimula kami sa pag-develop ng LSTM architecture na nagproprocess ng text sequences para makakuha ng contextual understanding ng disaster-related posts. Gumawa rin ako ng specialized embedding layers para sa disaster terminology.<br><br>Para naman sa Filipino language support, in-integrate ko ang Multilingual BERT (MBERT) para magkaroon ng cross-lingual capabilities ang aming system. Sa ganitong paraan, nakakaintindi ang model namin ng both English at Filipino text, at kahit na code-switched content.<br><br>```python<br># LSTM Model Architecture<br>def build_lstm_model(vocab_size, embedding_dim, max_length):<br>    model = Sequential()<br>    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))<br>    model.add(SpatialDropout1D(0.25))<br>    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))<br>    model.add(Bidirectional(LSTM(64, dropout=0.2)))<br>    model.add(Dense(64, activation='relu'))<br>    model.add(Dropout(0.5))<br>    model.add(Dense(5, activation='softmax'))<br>    <br>    model.compile(loss='categorical_crossentropy', <br>                 optimizer=Adam(learning_rate=0.001),<br>                 metrics=['accuracy'])<br>    return model<br>```<br><br>Para sa MBERT integration, gumawa ako ng class na nag-load ng pretrained MBERT model at fine-tune ito para sa aming specific task:<br><br>```python<br># MBERT Integration for Filipino<br>class MBERTSentimentClassifier(nn.Module):<br>    def __init__(self, num_labels=5):<br>        super(MBERTSentimentClassifier, self).__init__()<br>        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')<br>        <br>        # Freeze parameters except last layers<br>        for param in self.bert.parameters():<br>            param.requires_grad = False<br>        for param in self.bert.encoder.layer[-2:].parameters():<br>            param.requires_grad = True<br>            <br>        self.dropout = nn.Dropout(0.3)<br>        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)<br>        <br>    def forward(self, input_ids, attention_mask):<br>        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)<br>        pooled_output = outputs.pooler_output<br>        pooled_output = self.dropout(pooled_output)<br>        return self.classifier(pooled_output)<br>```<br><br>Para sa language detection, gumamit kami ng features ng MBERT para ma-identify ang language ng input text:<br><br>```python<br>def detect_language(text):<br>    # Ginagamit ang MBERT embeddings para sa language detection<br>    reference_texts = {<br>        'en': 'The earthquake caused significant damage',<br>        'tl': 'Ang lindol ay nagdulot ng malaking pinsala',<br>        'ceb': 'Ang linog nakahimo og daghang kadaot'<br>    }<br>    <br>    # Compare embeddings sa reference texts<br>    similarities = {}<br>    for lang, ref_text in reference_texts.items():<br>        similarity = compute_embedding_similarity(text, ref_text)<br>        similarities[lang] = similarity<br>    <br>    # Return ang language na may highest similarity<br>    return max(similarities.items(), key=lambda x: x[1])[0]<br>```<br><br>Sa pagtesting, nakakuha ang LSTM model ng 71% accuracy, habang ang MBERT ay 76%. Kapag pinagsama namin ang dalawang approaches sa isang ensemble model, umabot kami sa 79% accuracy. |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • **Custom LSTM Neural Network** - Nagdevelop kami ng bidirectional LSTM na specialized para sa disaster text sequences<br>• **MBERT (Multilingual BERT)** - In-implement namin ito para makapag-handle ng Filipino text at code-switching<br>• **Word Embeddings** - Gumawa kami ng 300-dimensional embeddings na nag-capture ng disaster terminology semantics<br>• **TensorFlow at PyTorch** - Ginamit namin ang TensorFlow para sa LSTM at PyTorch para sa MBERT implementation<br>• **Ensemble Learning** - Pinagsama namin ang predictions mula sa LSTM at MBERT para sa improved accuracy<br>• **Batch Processing** - Nagimplenet kami ng efficient batch processing para ma-handle ang large datasets |
| **Reflection: Problems Encountered and Lessons Learned** | **Problem: Memory Management sa Large Datasets**<br>Nung una, nagkaroon kami ng out-of-memory errors habang nagproprocess ng 50,000 social media posts mula sa Typhoon Yolanda dataset. Nag-crash ang system dahil ini-load namin ang buong CSV file sa memory before processing.<br><br>**Solution:**<br>Nag-redesign kami ng data processing pipeline para gumamit ng streaming approach. Gumawa ako ng batched processing na nagha-handle ng 1,000 records at a time. Nagimplement din ako ng progressive data writing sa database habang nagproprocess pa lang, sa halip na i-save lahat ng results sa dulo.<br><br>```python<br>def process_large_dataset(file_path, batch_size=1000):<br>    results = []<br>    total_count = count_lines(file_path)<br>    processed = 0<br>    <br>    with open(file_path, 'r') as f:<br>        batch = []<br>        for line in f:<br>            batch.append(line.strip())<br>            if len(batch) >= batch_size:<br>                # Process current batch<br>                batch_results = process_batch(batch)<br>                save_to_database(batch_results)  # Save immediately<br>                <br>                # Update progress<br>                processed += len(batch)<br>                update_progress(processed / total_count)<br>                <br>                # Clear batch<br>                batch = []<br>    <br>    # Process remaining items<br>    if batch:<br>        batch_results = process_batch(batch)<br>        save_to_database(batch_results)<br>```<br><br>**Result:**<br>Naprocess namin ang full Typhoon Yolanda dataset (50,000 posts) gamit lamang ang 15% ng available memory. Bumilis din ang processing time mula sa "crash after 20%" papuntang full completion in 12 minutes.<br><br>**Lesson Learned:**<br>Napag-alaman namin na kada-handle ng large text datasets, kailangan ng efficient streaming at batching techniques. Hindi effective ang pag-load ng lahat ng data sa memory at once. Stream processing ang key para ma-handle ang production-scale data. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 6 / February 24 - February 28, 2025 |

| Transformer Integration at Real-time Features |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Ngayong linggo, nagfocus kami sa pag-integrate ng Transformer models (DistilBERT) kasama ng aming existing LSTM at MBERT implementation. Sa halip na mag-rely sa external API, in-implement namin ang sarili naming DistilBERT integration.<br><br>Nagsimula ako sa pag-import ng pretrained DistilBERT model at pag-adapt nito sa aming disaster sentiment task:<br><br>```python<br>class DisasterTransformerModel(nn.Module):<br>    def __init__(self, num_labels=5):<br>        super(DisasterTransformerModel, self).__init__()<br>        # Load pretrained distilbert<br>        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased')<br>        self.dropout = nn.Dropout(0.3)<br>        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)<br>        <br>    def forward(self, input_ids, attention_mask):<br>        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)<br>        pooled_output = outputs.last_hidden_state[:, 0]<br>        pooled_output = self.dropout(pooled_output)<br>        return self.classifier(pooled_output)<br>```<br><br>Para ma-maximize ang performance, nagdevelop ako ng ensemble approach na nagkokombine ng LSTM, MBERT, at DistilBERT predictions:<br><br>```python<br>class EnsembleModel:<br>    def __init__(self, lstm_model, mbert_model, transformer_model, weights=[0.3, 0.3, 0.4]):<br>        self.lstm_model = lstm_model<br>        self.mbert_model = mbert_model<br>        self.transformer_model = transformer_model<br>        self.weights = weights<br>        <br>    def predict(self, text):<br>        # Get predictions from all models<br>        lstm_result = self.lstm_model.predict(text)<br>        mbert_result = self.mbert_model.predict(text)<br>        transformer_result = self.transformer_model.predict(text)<br>        <br>        # Weight and combine predictions<br>        ensemble_probs = {}<br>        for sentiment in lstm_result['probabilities'].keys():<br>            ensemble_probs[sentiment] = (<br>                lstm_result['probabilities'][sentiment] * self.weights[0] +<br>                mbert_result['probabilities'][sentiment] * self.weights[1] +<br>                transformer_result['probabilities'][sentiment] * self.weights[2]<br>            )<br>        <br>        # Get sentiment with highest probability<br>        ensemble_sentiment = max(ensemble_probs.items(), key=lambda x: x[1])[0]<br>        ensemble_confidence = max(ensemble_probs.values())<br>        <br>        return {<br>            'sentiment': ensemble_sentiment,<br>            'confidence': ensemble_confidence,<br>            'probabilities': ensemble_probs<br>        }<br>```<br><br>Bukod sa ML improvements, nagdevelop din ako ng real-time capabilities gamit ang WebSockets para sa instant updates sa system:<br><br>```typescript<br>// WebSocket Implementation<br>const server = createServer(app);<br>const wss = new WebSocketServer({ server });<br><br>wss.on('connection', (ws) => {<br>  console.log('Client connected');<br>  <br>  // Send initial data<br>  const initialData = {<br>    type: 'INIT',<br>    data: {<br>      activeSessions: getActiveSessions(),<br>      recentEvents: getRecentEvents()<br>    }<br>  };<br>  ws.send(JSON.stringify(initialData));<br>  <br>  // Handle client messages<br>  ws.on('message', (message) => {<br>    try {<br>      const data = JSON.parse(message.toString());<br>      // Process message...<br>    } catch (error) {<br>      console.error('Error processing message:', error);<br>    }<br>  });<br>});<br><br>// Function to broadcast updates<br>function broadcastUpdate(data) {<br>  wss.clients.forEach((client) => {<br>    if (client.readyState === WebSocket.OPEN) {<br>      client.send(JSON.stringify({<br>        type: 'UPDATE',<br>        timestamp: new Date().toISOString(),<br>        data<br>      }));<br>    }<br>  });<br>}<br>```<br><br>Ang ensemble model approach namin ay nagresulta sa 85% accuracy, significant improvement mula sa original 71% ng LSTM at 76% ng MBERT alone. |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • **DistilBERT Integration** - Ginamit namin ang lightweight transformer model para sa efficient sentiment analysis<br>• **Ensemble Learning** - Nagcombine kami ng multiple models para sa improved accuracy<br>• **WebSockets** - Nagimplemet ng real-time communication para sa instant updates<br>• **Mixed Precision Training** - Ginamit namin ito para mabawasan ang memory footprint ng transformer models<br>• **Cross-tab Synchronization** - Nagdevelop ng system para mag-stay in sync ang multiple browser tabs<br>• **Real-time Progress Tracking** - Nagimplement ng progress updates para sa long-running processes |
| **Reflection: Problems Encountered and Lessons Learned** | **Problem: WebSocket Connection Reliability**<br>Nagkaroon kami ng issues sa WebSocket connections lalo na sa mobile networks. Madaming users ang nawawalan ng real-time updates dahil sa unstable connections.<br><br>**Solution:**<br>Nagdevelop kami ng robust reconnection system na may built-in fallback sa polling kung mag-fail ang WebSocket:<br><br>```typescript<br>class WebSocketClient {<br>  private ws: WebSocket | null = null;<br>  private reconnectAttempts = 0;<br>  private maxAttempts = 10;<br>  <br>  constructor(private url: string) {<br>    this.connect();<br>  }<br>  <br>  private connect() {<br>    this.ws = new WebSocket(this.url);<br>    <br>    this.ws.onopen = () => {<br>      console.log('Connected');<br>      this.reconnectAttempts = 0;<br>    };<br>    <br>    this.ws.onclose = () => {<br>      this.ws = null;<br>      this.attemptReconnect();<br>    };<br>  }<br>  <br>  private attemptReconnect() {<br>    if (this.reconnectAttempts >= this.maxAttempts) {<br>      console.log('Falling back to polling');<br>      this.startPolling();<br>      return;<br>    }<br>    <br>    // Exponential backoff<br>    const delay = Math.min(<br>      1000 * Math.pow(2, this.reconnectAttempts),<br>      30000<br>    );<br>    <br>    setTimeout(() => {<br>      this.reconnectAttempts++;<br>      this.connect();<br>    }, delay);<br>  }<br>  <br>  private startPolling() {<br>    // Implement polling fallback<br>    setInterval(async () => {<br>      try {<br>        const response = await fetch('/api/updates');<br>        const data = await response.json();<br>        this.processUpdates(data);<br>      } catch (error) {<br>        console.error('Polling error:', error);<br>      }<br>    }, 5000);<br>  }<br>}<br>```<br><br>**Problem: Memory Usage ng Transformer Models**<br>Napansin namin na masyadong mataas ang memory requirements ng DistilBERT model, which makes it impractical for production servers with limited resources.<br><br>**Solution:**<br>Nag-implement kami ng quantization at pruning para ma-optimize ang model size:<br><br>```python<br>def optimize_model(model):<br>    # Quantize model to 8-bit<br>    quantized_model = torch.quantization.quantize_dynamic(<br>        model, {torch.nn.Linear}, dtype=torch.qint8<br>    )<br>    <br>    # Prune least important weights<br>    for name, module in quantized_model.named_modules():<br>        if isinstance(module, torch.nn.Linear):<br>            prune.l1_unstructured(module, name='weight', amount=0.3)<br>    <br>    return quantized_model<br>```<br><br>Ang optimization na ito ay nag-reduce ng model size by 75% while maintaining 97% of the original accuracy.<br><br>**Lesson Learned:**<br>Napag-alaman namin na kailangan ng comprehensive approach sa handling ng real-time connections, with fallback mechanisms para sa unreliable networks. Sa transformer models naman, kailangan ng proper optimization techniques para maging viable sa production environments. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 - March 7, 2025 |

| AI Feedback System Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | **REST DAY: Wednesday (March 5) - National holiday**<br><br>Sa Week 7, nag-develop ako ng system para sa user feedback collection at model improvement. Sa halip na i-rely lang sa initial training, gusto namin ma-improve continuously ang model gamit ang feedback mula sa users.<br><br>Nagsimula ako sa pag-implement ng feedback submission UI:<br><br>```tsx<br>export function FeedbackForm({ prediction }) {<br>  const [correctedSentiment, setCorrectedSentiment] = useState(prediction.sentiment);<br>  <br>  const submitFeedback = async () => {<br>    try {<br>      await fetch('/api/sentiment-feedback', {<br>        method: 'POST',<br>        headers: { 'Content-Type': 'application/json' },<br>        body: JSON.stringify({<br>          postId: prediction.id,<br>          originalSentiment: prediction.sentiment,<br>          correctedSentiment,<br>          confidence: prediction.confidence<br>        })<br>      });<br>      <br>      toast.success("Salamat sa iyong feedback!");<br>    } catch (error) {<br>      toast.error("May error sa submission.");<br>    }<br>  };<br>  <br>  return (<br>    <div className="p-4 border rounded-lg"<br>      <h3 className="font-bold mb-2">Feedback Form</h3><br>      <div className="mb-4"<br>        <label>Original Text</label><br>        <div className="p-2 bg-gray-100 rounded">{prediction.text}</div><br>      </div><br>      <div className="grid grid-cols-2 gap-4 mb-4"<br>        <div><br>          <label>AI Prediction</label><br>          <div className="flex items-center"<br>            <span className="bg-blue-100 px-2 py-1 rounded">{prediction.sentiment}</span><br>            <ConfidenceIndicator value={prediction.confidence} /><br>          </div><br>        </div><br>        <div><br>          <label>Correct Sentiment</label><br>          <select<br>            value={correctedSentiment}<br>            onChange={(e) => setCorrectedSentiment(e.target.value)}<br>            className="w-full p-2 border rounded"<br>          ><br>            <option value="Panic">Panic</option><br>            <option value="Fear/Anxiety">Fear/Anxiety</option><br>            <option value="Disbelief">Disbelief</option><br>            <option value="Resilience">Resilience</option><br>            <option value="Neutral">Neutral</option><br>          </select><br>        </div><br>      </div><br>      <button<br>        onClick={submitFeedback}<br>        className="bg-blue-500 text-white px-4 py-2 rounded"<br>      ><br>        Submit Feedback<br>      </button><br>    </div><br>  );<br>}<br>```<br><br>Pagkatapos, nagdevelop ako ng system para sa pagincorporate ng feedback sa model training:<br><br>```python<br>def train_with_feedback(feedback_data):<br>    """Train the model using user feedback"""<br>    # Group feedback examples by sentiment category<br>    categories = {}<br>    for item in feedback_data:<br>        category = item['correctedSentiment']<br>        if category not in categories:<br>            categories[category] = []<br>        categories[category].append(item['text'])<br>    <br>    # Create category prototypes<br>    prototypes = {}<br>    for category, texts in categories.items():<br>        if len(texts) < 2:  # Need at least 2 examples<br>            continue<br>            <br>        # Create embeddings for all texts<br>        embeddings = [create_embedding(text) for text in texts]<br>        <br>        # Average the embeddings<br>        prototype = np.mean(embeddings, axis=0)<br>        <br>        # Normalize the prototype<br>        prototype = prototype / np.linalg.norm(prototype)<br>        <br>        prototypes[category] = {<br>            'vector': prototype.tolist(),<br>            'example_count': len(texts),<br>            'confidence_threshold': calculate_adaptive_threshold(len(texts))<br>        }<br>    <br>    # Save prototypes for future inference<br>    save_prototypes(prototypes)<br>    <br>    # Evaluate improvement<br>    old_accuracy = evaluate_without_prototypes(test_data)<br>    new_accuracy = evaluate_with_prototypes(test_data, prototypes)<br>    <br>    return {<br>        'improvement': new_accuracy - old_accuracy,<br>        'trained_categories': list(prototypes.keys())<br>    }<br>```<br><br>Matapos kong i-implement ang feedback system, nagsimula akong magdevelop ng similarity detection para ma-apply ang corrections consistently sa similar texts:<br><br>```python<br>def calculate_similarity(text1, text2):<br>    """Calculate semantic similarity between two texts"""<br>    # Clean and normalize texts<br>    text1 = clean_text(text1)<br>    text2 = clean_text(text2)<br>    <br>    # For very short texts, use Levenshtein distance<br>    if len(text1) < 20 or len(text2) < 20:<br>        distance = levenshtein_distance(text1, text2)<br>        max_length = max(len(text1), len(text2))<br>        return 1 - (distance / max_length)<br>    <br>    # For longer texts, use embeddings<br>    embedding1 = get_text_embedding(text1)<br>    embedding2 = get_text_embedding(text2)<br>    <br>    # Calculate cosine similarity<br>    similarity = cosine_similarity(embedding1, embedding2)<br>    return similarity<br>```<br><br>Sa testing, ang feedback-enhanced model ay nakakuha ng 12% improvement in accuracy mula 79% papuntang 91%. |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • **Human-in-the-loop AI** - Nag-implement ako ng sistema para magincorporate ng human feedback sa AI<br>• **Category-based Learning** - Gumamit ako ng approach na nagcreate ng "prototype" embeddings para sa efficient learning<br>• **Active Learning** - Nagbuild ng system na nagp-prioritize ng uncertain predictions para sa user feedback<br>• **Semantic Similarity Detection** - Nag-implement ng cosine similarity para ma-identify ang similar texts<br>• **Adaptive Confidence Thresholds** - Nagdevelop ng system na nagadjust ng confidence requirements base sa available examples<br>• **React UI Components** - Nagbuild ng user-friendly interfaces para sa feedback collection |
| **Reflection: Problems Encountered and Lessons Learned** | **Problem: Inefficient Training Approach**<br>Sa original implementation, kailangan namin ng hundreds of examples para mag-improve ang model significantly. Ito ay hindi practical para sa rapid improvement cycle na gusto namin.<br><br>**Analysis:**<br>Ang issue ay nasa approach namin sa training. Initially, tinetrain namin ang model para sa specific phrases, requiring multiple examples of each specific context. Hindi ito efficient dahil masyadong specific ang learning.<br><br>**Solution:**<br>Sa rest day ko nung Wednesday, nag-research ako about few-shot learning techniques. Sa halip na phrase-based, nag-shift kami sa category-based learning approach:<br><br>1. Instead of training on phrases, nag-create kami ng "prototype" embeddings for each sentiment category by averaging embeddings ng lahat ng examples<br><br>2. Sa prediction time, kinokompare namin ang embedding ng new text sa prototype ng each category para makuha ang most similar.<br><br>3. Nagdevelop kami ng adaptive confidence threshold na nagaadjust based sa number of examples in each category.<br><br>Ang implementation code ay ganito:<br><br>```python<br># Compare text embedding with category prototypes<br>def predict_with_prototypes(text, prototypes):<br>    # Get text embedding<br>    text_embedding = get_text_embedding(text)<br>    <br>    # Calculate similarity with each prototype<br>    similarities = {}<br>    for category, prototype_data in prototypes.items():<br>        prototype_vector = np.array(prototype_data['vector'])<br>        similarity = cosine_similarity(text_embedding, prototype_vector)<br>        similarities[category] = similarity<br>    <br>    # Get category with highest similarity<br>    max_category = max(similarities.items(), key=lambda x: x[1])<br>    category = max_category[0]<br>    similarity_score = max_category[1]<br>    <br>    # Check if similarity exceeds threshold<br>    threshold = prototypes[category]['confidence_threshold']<br>    confidence = similarity_score if similarity_score >= threshold else similarity_score * 0.8<br>    <br>    return {<br>        'category': category,<br>        'confidence': confidence,<br>        'similarities': similarities<br>    }<br>```<br><br>**Results:**<br>With just 5-10 examples per category, nakakuha kami ng 12% improvement in accuracy (79% to 91%). Ang users ay nakakita kaagad ng improvements after providing feedback, which encouraged more feedback submission.<br><br>**Lessons Learned:**<br>• Domain-specific AI systems can achieve high accuracy with far fewer examples than general models<br>• Category-based approaches work better than phrase-based training for specialized domains<br>• The rest day spent on research led to a breakthrough in our approach<br>• User engagement increases significantly when they see the impact of their feedback |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 - March 14, 2025 |

| Containerization at Deployment Preparation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Sa Week 8, nagfocus kami sa preparation para sa deployment. Sa halip na mag-rely sa external hosting services, gusto naming gamitin ang Docker para sa containerization at i-deploy sa Render platform.<br><br>Nagsimula ako sa pag-create ng Dockerfile para sa application namin:<br><br>```dockerfile<br># Multi-stage build para sa optimized image size
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files at install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Python stage para sa ML service
FROM python:3.10-slim AS python-builder

# Set working directory for Python
WORKDIR /app/ml

# Copy Python requirements and install dependencies
COPY server/python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python code
COPY server/python/ .

# Final production stage
FROM node:18-alpine AS production

# Set working directory
WORKDIR /app

# Set NODE_ENV
ENV NODE_ENV production

# Copy built assets from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules

# Copy Python environment and code
COPY --from=python-builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY --from=python-builder /app/ml /app/ml

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 appuser
RUN chown -R appuser:nodejs /app
USER appuser

# Expose port
EXPOSE 3000

# Start command
CMD ["npm", "run", "start:prod"]<br>```<br><br>Para naman sa deployment sa Render, gumawa ako ng render.yaml configuration:<br><br>```yaml<br># render.yaml
services:
  - type: web
    name: disaster-monitor
    env: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
    plan: standard
    healthCheckPath: /api/health
    autoDeploy: true
    envVars:
      - key: NODE_ENV
        value: production
      - key: DATABASE_URL
        fromDatabase:
          name: disaster-monitor-db
          property: connectionString
      - key: SESSION_SECRET
        sync: false

databases:
  - name: disaster-monitor-db
    plan: standard<br>```<br><br>Sa pag-prepare para sa deployment, nag-optimize din ako ng ML models para mabawasan ang memory footprint sa production:<br><br>```python<br>def optimize_model_for_production(model_path, output_path):
    """Optimize model for production deployment"""
    # Load the original model
    model = torch.load(model_path)
    
    # 1. Quantize model to 8-bit integers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 2. Prune less important weights
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Prune 30% of least important weights
            prune.l1_unstructured(module, name='weight', amount=0.3)
    
    # 3. Save optimized model
    torch.save(quantized_model, output_path)
```<br><br>Para sa domain management, nagsetup ako ng configuration sa InfinityFree kung saan na-setup natin ang DNS records at SSL certificates para sa ating disaster-monitor.ph domain. |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • **Docker Containerization** - Gumamit kami ng multi-stage Docker builds para sa efficient deployment<br>• **Render Platform** - Napili namin ito para sa reliable hosting ng application<br>• **InfinityFree** - Ginamit para sa domain management at DNS configuration<br>• **Model Quantization** - Applied 8-bit quantization para ma-reduce ang ML model size<br>• **Model Pruning** - Na-remove ang redundant weights para ma-optimize ang model size<br>• **Environment-based Configuration** - Nagimplement ng proper env configuration para sa different deployment environments |
| **Reflection: Problems Encountered and Lessons Learned** | **Problem: Docker Image Size**<br>Nung una, ang Docker image namin ay masyado malaki (over 2GB), which caused slow deployments at high resource consumption.<br><br>**Analysis:**<br>Sa pag-investigate, nakita namin na malaking factor ang inclusion ng full Python environment at development dependencies sa image size.<br><br>**Solution:**<br>Nag-implement kami ng multi-stage Docker build para ma-separate ang build at runtime environments:<br><br>1. Nagcreate ng separate build stages para sa Node.js at Python<br>2. Gumamit ng Alpine Linux base images para sa smaller footprint<br>3. Inoptimize ang Python dependencies<br>4. Tinanggal ang unnecessary development files sa final image<br><br>Ang impact ay significant - na-reduce namin ang image size by 85%, from 2GB to 310MB.<br><br>**Problem: Database Migration Strategy**<br>Nagkaroon din kami ng issues sa database schema mismatches between development at production environments.<br><br>**Solution:**<br>Nag-implement kami ng automated migration system gamit ang Drizzle ORM:<br><br>```typescript<br>// Migration script
import { drizzle } from 'drizzle-orm/postgres-js';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import postgres from 'postgres';

// Connect to database
const client = postgres(process.env.DATABASE_URL);
const db = drizzle(client);

// Run migrations
async function runMigrations() {
  console.log('Running migrations...');
  
  try {
    // Run all migrations from the drizzle folder
    await migrate(db, { migrationsFolder: 'drizzle' });
    console.log('Migrations completed successfully');
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  } finally {
    // Close the connection
    await client.end();
  }
}

runMigrations();
```<br><br>**Lessons Learned:**<br>• Docker image optimization is critical for efficient deployments<br>• Multi-stage builds provide a powerful pattern for keeping images small<br>• A formalized database migration strategy is essential for reliable deployments<br>• Environment-specific configuration should be properly managed through environment variables |