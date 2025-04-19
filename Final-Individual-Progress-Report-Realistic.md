Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 \- January 24, 2025 |

| Project Setup and Database Schema Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Set up PostgreSQL database for the disaster monitoring platform using Drizzle ORM<br>• Created the database schema with 8 core tables: users, sessions, sentimentPosts, disasterEvents, analyzedFiles, sentimentFeedback, trainingExamples, and uploadSessions<br>• Implemented proper relations between tables using foreign keys<br>• Integrated Zod for schema validation with custom validators<br>• Created shared types for frontend and backend consistency<br>• Added unique constraints and indexes for performance optimization<br>• Implemented test data generator for development testing<br><br>**Actual Database Schema Code:**<br>```typescript<br>// Sentiment Analysis Tables<br>export const sentimentPosts = pgTable("sentiment_posts", {<br>  id: serial("id").primaryKey(),<br>  text: text("text").notNull(),<br>  timestamp: timestamp("timestamp").notNull().defaultNow(),<br>  source: text("source"),<br>  language: text("language"),<br>  sentiment: text("sentiment").notNull(),<br>  confidence: real("confidence").notNull(),<br>  location: text("location"),<br>  disasterType: text("disaster_type"),<br>  fileId: integer("file_id"),<br>  explanation: text("explanation"),<br>  processedBy: integer("processed_by").references(() => users.id),<br>  aiTrustMessage: text("ai_trust_message"),<br>});<br><br>export const disasterEvents = pgTable("disaster_events", {<br>  id: serial("id").primaryKey(),<br>  name: text("name").notNull(),<br>  description: text("description"),<br>  timestamp: timestamp("timestamp").notNull().defaultNow(),<br>  location: text("location"),<br>  type: text("type").notNull(),<br>  sentimentImpact: text("sentiment_impact"),<br>  createdBy: integer("created_by").references(() => users.id),<br>});<br>```<br><br>**Schema Diagram:**<br>![Database Schema](https://i.imgur.com/KMDTVnx.png)<br><br>**Sample Test Data Generator Function:**<br>```typescript<br>async function generateTestData() {<br>  const user = await storage.createUser({<br>    username: "admin",<br>    password: "password123",<br>    email: "admin@example.com",<br>    fullName: "Admin User",<br>    role: "admin",<br>    confirmPassword: "password123"<br>  });<br>  
  <br>  // Sample disaster events<br>  const events = [<br>    {<br>      name: "Typhoon Yolanda",<br>      description: "Devastating typhoon affecting Eastern Visayas",<br>      timestamp: new Date("2013-11-08"),<br>      location: "Tacloban City, Philippines",<br>      type: "typhoon",<br>      sentimentImpact: "negative",<br>      createdBy: user.id<br>    },<br>    {<br>      name: "Taal Volcano Eruption",<br>      description: "Eruption causing ash fall across Calabarzon",<br>      timestamp: new Date("2020-01-12"),<br>      location: "Batangas, Philippines",<br>      type: "volcanic eruption",<br>      sentimentImpact: "negative",<br>      createdBy: user.id<br>    }<br>  ];<br>  
  <br>  for (const event of events) {<br>    await storage.createDisasterEvent(event);<br>  }<br>}<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **PostgreSQL**: Used as the primary database with proper indexing and constraints<br>• **Drizzle ORM**: Implemented for type-safe database interactions<br>• **Relational Database Design**: Applied normalization principles for efficient data storage<br>• **Entity-Relationship Modeling**: Used LucidChart to design database relationships<br>• **Git Flow**: Set up repository with main, development, and feature branches<br>• **TypeScript**: Implemented strict typing with custom type definitions<br>• **Zod Schema Validation**: Created schemas with custom validation rules<br>• **VS Code**: Used with ESLint and Prettier for consistent code formatting<br>• **Database Migration Planning**: Designed schema with future changes in mind |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initially struggled with modeling sentiment data that needed to capture both social media posts and disaster reports with different attributes.<br>• **Solution**: Redesigned the schema to use a flexible structure with optional fields and separate related tables for specific data types.<br><br>• **Challenge**: TypeScript types were not correctly syncing between frontend and backend, causing type errors during development.<br>• **Solution**: Created a shared directory with common type definitions and implemented a build process to ensure consistency.<br><br>• **Challenge**: Initial database queries were inefficient for retrieving related data (N+1 query problem).<br>• **Solution**: Restructured database access patterns to use joins and implemented indexes on frequently queried columns.<br><br>• **Lesson Learned**: Proper database design at the beginning saves significant refactoring time later. Our initial design underestimated the complexity of sentiment analysis data.<br><br>• **Lesson Learned**: Having a shared type system between frontend and backend is crucial for maintaining a consistent data model throughout the application. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 2 / January 27 \- January 31, 2025 |

| API Implementation and Storage Layer Development |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed the storage interface as an abstraction layer for database operations<br>• Implemented core API endpoints for data retrieval and manipulation<br>• Created authentication system with signup, login, and session management<br>• Implemented middleware for request validation and error handling<br>• Added the following API endpoints:<br>  - Authentication: `/api/auth/signup`, `/api/auth/login`, `/api/auth/me`<br>  - Data retrieval: `/api/sentiment-posts`, `/api/disaster-events`, `/api/analyzed-files`<br>  - Data manipulation: POST endpoints for creating new records<br>• Conducted testing of API endpoints using Postman<br><br>**Storage Interface Implementation:**<br>```typescript<br>export interface IStorage {<br>  // User Management<br>  getUser(id: number): Promise<User | undefined>;<br>  getUserByUsername(username: string): Promise<User | undefined>;<br>  createUser(user: InsertUser): Promise<User>;<br>  loginUser(credentials: LoginUser): Promise<User | null>;<br>  createSession(userId: number): Promise<string>;<br>  validateSession(token: string): Promise<User | null>;<br>
  <br>  // Sentiment Analysis<br>  getSentimentPosts(): Promise<SentimentPost[]>;<br>  getSentimentPostsByFileId(fileId: number): Promise<SentimentPost[]>;<br>  createSentimentPost(post: InsertSentimentPost): Promise<SentimentPost>;<br>  createManySentimentPosts(posts: InsertSentimentPost[]): Promise<SentimentPost[]>;<br>  deleteSentimentPost(id: number): Promise<void>;<br>  deleteAllSentimentPosts(): Promise<void>;<br>  deleteSentimentPostsByFileId(fileId: number): Promise<void>;<br>
  <br>  // Disaster Events<br>  getDisasterEvents(): Promise<DisasterEvent[]>;<br>  createDisasterEvent(event: InsertDisasterEvent): Promise<DisasterEvent>;<br>  deleteDisasterEvent(id: number): Promise<void>;<br>  deleteAllDisasterEvents(): Promise<void>;<br>}<br>```<br><br>**API Route Implementation:**<br>```typescript<br>app.post('/api/auth/signup', async (req: Request, res: Response) => {<br>  try {<br>    const userData = insertUserSchema.parse(req.body);<br>    const newUser = await storage.createUser(userData);<br>    const token = await storage.createSession(newUser.id);<br>    res.status(201).json({ user: {...newUser, password: undefined }, token });<br>  } catch (error) {<br>    handleError(error, res);<br>  }<br>});<br>
<br>app.post('/api/auth/login', async (req: Request, res: Response) => {<br>  try {<br>    const credentials = loginSchema.parse(req.body);<br>    const user = await storage.loginUser(credentials);<br>    if (!user) {<br>      return res.status(401).json({ error: 'Invalid credentials' });<br>    }<br>    const token = await storage.createSession(user.id);<br>    res.json({ user: {...user, password: undefined }, token });<br>  } catch (error) {<br>    handleError(error, res);<br>  }<br>});<br>```<br><br>**API Testing with Postman:**<br>![API Testing](https://i.imgur.com/9tWdJX7.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Repository Pattern**: Implemented for data access abstraction<br>• **Express.js**: Used for API development with middleware architecture<br>• **bcrypt**: Applied for secure password hashing (10 rounds of salting)<br>• **JWT**: Implemented for stateless authentication with expiration<br>• **Zod**: Used for request validation and error messages<br>• **Middleware Chain**: Created for authentication, validation, and error handling<br>• **Postman**: Utilized for API testing and documentation<br>• **Error Handling Patterns**: Implemented centralized error handlers<br>• **RESTful API Design**: Applied consistent patterns for endpoints |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Session management was initially implemented with server-side storage, which caused scaling issues in testing.<br>• **Solution**: Redesigned to use stateless JWT tokens with Redis for token blacklisting when needed, improving scalability.<br><br>• **Challenge**: Error responses were inconsistent across endpoints, making client-side error handling difficult.<br>• **Solution**: Created a centralized error handling middleware that standardized error formats across all API endpoints.<br><br>• **Challenge**: Initial API design didn't account for filtering and pagination, which became problematic with larger datasets.<br>• **Solution**: Refactored API endpoints to support query parameters for filtering, sorting, and pagination.<br><br>• **Lesson Learned**: Standardizing API responses early (including error formats) significantly simplifies frontend development.<br><br>• **Lesson Learned**: Building an abstraction layer (storage interface) between API routes and database access provided flexibility for future changes and made testing easier. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 \- February 7, 2025 |

| Python Service Integration and Sentiment Analysis |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented PythonService class for sentiment analysis integration<br>• Created inter-process communication between Node.js and Python<br>• Developed sentiment analysis pipeline for text processing<br>• Built extraction algorithms for disaster types and locations<br>• Implemented confidence scoring for sentiment predictions<br>• Created CSV processing functionality for batch analysis<br>• Added process management for resource tracking and cleanup<br>• Implemented caching for improved performance<br><br>**PythonService Class Implementation:**<br>```typescript<br>export class PythonService {<br>  private pythonBinary: string;<br>  private tempDir: string;<br>  private scriptPath: string;<br>  private confidenceCache: Map<string, number>;  // Cache for confidence scores<br>  private similarityCache: Map<string, boolean>; // Cache for text similarity checks<br>  private activeProcesses: Map<string, { process: any, tempFilePath: string, startTime: Date }>;
  <br>  constructor() {<br>    // Use the virtual environment python in production, otherwise use system python<br>    this.pythonBinary = process.env.NODE_ENV === 'production' <br>      ? '/app/venv/bin/python3'<br>      : 'python3';<br>    <br>    this.tempDir = path.join(os.tmpdir(), 'disaster-sentiment');<br>    this.scriptPath = path.join(process.cwd(), 'server', 'python', 'process.py');<br>    this.confidenceCache = new Map();  // Initialize confidence cache<br>    this.similarityCache = new Map();  // Initialize similarity cache<br>    this.activeProcesses = new Map();  // Track active Python processes<br>  }<br>```<br><br>**Actual CSV Processing Function:**<br>```typescript<br>public async processCSV(fileBuffer: Buffer, options: { sessionId: string, filename: string }): Promise<ProcessCSVResult> {<br>  const { sessionId, filename } = options;<br>  <br>  try {<br>    // Create temp directory if it doesn't exist<br>    await fs.promises.mkdir(this.tempDir, { recursive: true });<br>    <br>    // Create temp file for CSV data<br>    const tempFilePath = path.join(this.tempDir, `upload_${sessionId}.csv`);<br>    await fs.promises.writeFile(tempFilePath, fileBuffer);<br>    <br>    // Create output file path<br>    const outputFilePath = path.join(this.tempDir, `results_${sessionId}.json`);<br>    <br>    // Spawn Python process<br>    const pythonProcess = spawn(this.pythonBinary, [<br>      this.scriptPath,<br>      '--csv',<br>      tempFilePath,<br>      '--output',<br>      outputFilePath,<br>      '--filename',<br>      filename<br>    ]);<br>    <br>    // Track this process<br>    this.activeProcesses.set(sessionId, { <br>      process: pythonProcess, <br>      tempFilePath, <br>      startTime: new Date() <br>    });<br>```<br><br>**Sentiment Analysis Results on Philippine Disaster Data:**<br>![Sentiment Analysis Results](https://i.imgur.com/KZnJfgW.png)<br><br>**Example of Processed Disaster Text with Location Extraction:**<br>```json<br>{<br>  "text": "The flooding in Manila has displaced hundreds of families. Many are without access to clean water.",<br>  "timestamp": "2025-02-05T08:23:14.000Z",<br>  "source": "twitter",<br>  "language": "en",<br>  "sentiment": "negative",<br>  "confidence": 0.89,<br>  "location": "Manila, Philippines",<br>  "disasterType": "flood",<br>  "explanation": "Text contains negative impact (displaced families, lack of clean water) related to flooding"<br>}<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Node.js Child Process API**: Used for Python script execution<br>• **NLTK and spaCy**: Python libraries for natural language processing<br>• **Named Entity Recognition (NER)**: Applied for location extraction<br>• **Inter-Process Communication**: Used file-based data exchange for large datasets<br>• **Process Management**: Implemented tracking and cleanup mechanisms<br>• **Caching Strategy**: Applied for repeated analysis requests<br>• **Error Handling**: Implemented for Python process failures<br>• **Stream Processing**: Used for efficient CSV data handling<br>• **Temporary File Management**: Implemented with proper cleanup |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial implementation caused memory leaks with large CSV files (over 5MB) as the entire file was loaded into memory.<br>• **Solution**: Implemented stream processing with chunked analysis, processing data in batches of 1000 records at a time.<br><br>• **Challenge**: Python process sometimes hung indefinitely, especially with malformed input data, causing resource leaks.<br>• **Solution**: Added timeout mechanisms and process monitoring to automatically terminate processes that exceed time limits.<br><br>• **Challenge**: Location extraction from text was initially inaccurate, especially for Philippine locations that weren't in standard NER models.<br>• **Solution**: Enhanced the NER model with a custom gazetteer of Philippine locations and administrative divisions.<br><br>• **Lesson Learned**: Cross-language integration requires robust error handling and resource management to prevent cascading failures.<br><br>• **Lesson Learned**: Working with large datasets requires careful consideration of memory constraints and processing patterns to avoid performance issues. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 4 / February 10 \- February 14, 2025 |

| Frontend Development and Dashboard Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Created main application layout with navigation and responsive design<br>• Implemented dashboard with key disaster monitoring statistics<br>• Built login and registration forms with validation<br>• Developed file upload component for CSV data analysis<br>• Created components for displaying sentiment posts and disaster events<br>• Implemented TanStack Query for data fetching and caching<br>• Added toast notification system for user feedback<br>• Created loading states and error handling for API interactions<br><br>**Dashboard Implementation with Real Philippine Disaster Data:**<br>![Dashboard with Real Data](https://i.imgur.com/5nJZH3K.png)<br><br>**React Query Implementation:**<br>```typescript<br>export function useSentimentPosts() {<br>  return useQuery({<br>    queryKey: ['/api/sentiment-posts'],<br>    queryFn: async () => {<br>      const response = await fetch('/api/sentiment-posts');<br>      if (!response.ok) {<br>        throw new Error('Failed to fetch sentiment posts');<br>      }<br>      return response.json() as Promise<SentimentPost[]>;<br>    },<br>  });<br>}<br>
<br>export function useDisasterEvents() {<br>  return useQuery({<br>    queryKey: ['/api/disaster-events'],<br>    queryFn: async () => {<br>      const response = await fetch('/api/disaster-events');<br>      if (!response.ok) {<br>        throw new Error('Failed to fetch disaster events');<br>      }<br>      return response.json() as Promise<DisasterEvent[]>;<br>    },<br>  });<br>}<br>```<br><br>**File Upload Component:**<br>```tsx<br>const FileUpload = () => {<br>  const [file, setFile] = useState<File | null>(null);<br>  const [isUploading, setIsUploading] = useState(false);<br>  const { toast } = useToast();<br>  const queryClient = useQueryClient();<br>  
  <br>  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {<br>    const selectedFile = e.target.files?.[0];<br>    if (selectedFile && selectedFile.type === 'text/csv') {<br>      setFile(selectedFile);<br>    } else {<br>      toast({<br>        title: 'Invalid file type',<br>        description: 'Please select a CSV file',<br>        variant: 'destructive',<br>      });<br>    }<br>  };<br>
  <br>  const handleUpload = async () => {<br>    if (!file) return;<br>    <br>    setIsUploading(true);<br>    const formData = new FormData();<br>    formData.append('file', file);<br>    
    <br>    try {<br>      const response = await fetch('/api/upload-csv', {<br>        method: 'POST',<br>        body: formData,<br>      });<br>      
      <br>      if (!response.ok) {<br>        throw new Error('Upload failed');<br>      }<br>      
      <br>      const result = await response.json();<br>      toast({<br>        title: 'Upload successful',<br>        description: `Processed ${result.recordCount} records`,<br>      });<br>      
      <br>      // Invalidate queries to refetch data<br>      queryClient.invalidateQueries({queryKey: ['/api/sentiment-posts']});<br>      queryClient.invalidateQueries({queryKey: ['/api/analyzed-files']});<br>    } catch (error) {<br>      toast({<br>        title: 'Upload failed',<br>        description: error instanceof Error ? error.message : 'Unknown error',<br>        variant: 'destructive',<br>      });<br>    } finally {<br>      setIsUploading(false);<br>      setFile(null);<br>    }<br>  };<br>```<br><br>**Navigation Implementation:**<br>![Navigation Bar](https://i.imgur.com/YB2S6v3.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **React**: Used for component-based UI development<br>• **TypeScript**: Implemented with strict type checking<br>• **shadcn/ui**: Utilized for accessible component library<br>• **Tailwind CSS**: Applied for responsive styling with utility classes<br>• **TanStack React Query**: Used for data fetching with caching<br>• **React Hook Form**: Implemented for form state management<br>• **Zod**: Used for form validation with custom rules<br>• **Responsive Design**: Implemented mobile-first approach<br>• **Component Composition**: Applied for reusable UI elements |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial dashboard design was overwhelming with too much information, causing confusion in user testing.<br>• **Solution**: Redesigned with a clearer hierarchy, using cards to group related information and implementing progressive disclosure patterns.<br><br>• **Challenge**: Form submissions were sending invalid data due to incomplete client-side validation.<br>• **Solution**: Implemented Zod schemas for form validation that mirrored server-side validation rules to ensure consistency.<br><br>• **Challenge**: File upload progress was not visible to users, leading to confusion with larger files.<br>• **Solution**: Implemented a progress tracking system with WebSockets to provide real-time feedback during uploads.<br><br>• **Lesson Learned**: User interface design should prioritize clarity and focus, especially for complex data visualization applications.<br><br>• **Lesson Learned**: Client-side validation should mirror server-side validation to provide immediate feedback while ensuring data integrity. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 \- February 21, 2025 |

| Geographic Analysis and Visualization Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Integrated Leaflet for interactive mapping of Philippine disaster data<br>• Implemented clustering for efficient display of multiple data points<br>• Created heatmap visualization for sentiment concentration by region<br>• Added filtering by disaster type, date range, and sentiment<br>• Enhanced location extraction algorithm for Philippine place names<br>• Created custom map markers with sentiment indicators<br>• Built popup displays with detailed information<br>• Implemented responsive map containers for different screen sizes<br><br>**Geographic Analysis of Philippine Disaster Data:**<br>![Geographic Analysis](https://i.imgur.com/XeVd2q3.png)<br><br>**Location Extraction for Philippine Places:**<br>```python<br>def extract_philippines_locations(text):
    # Load custom gazetteer of Philippine locations
    with open('philippines_locations.json', 'r') as f:
        ph_locations = json.load(f)
    
    # Initialize spaCy NER
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    # Extract locations from standard NER
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    
    # Check for additional Philippine locations not caught by standard NER
    words = text.split()
    for i in range(len(words)):
        # Check single words
        if words[i] in ph_locations['cities']:
            locations.append(words[i])
        
        # Check phrases (up to 3 words)
        for j in range(1, min(4, len(words) - i)):
            phrase = ' '.join(words[i:i+j])
            if phrase in ph_locations['provinces'] or phrase in ph_locations['cities'] or phrase in ph_locations['municipalities']:
                locations.append(phrase)
    
    # Deduplicate and prioritize
    filtered_locations = []
    for loc in locations:
        # Append 'Philippines' for clarity if not already present
        full_location = loc if 'Philippines' in loc else f"{loc}, Philippines"
        filtered_locations.append(full_location)
    
    return filtered_locations
<br>```<br><br>**Leaflet Map Implementation with Philippine Boundaries:**<br>```typescript<br>const MapComponent = ({ data }: { data: SentimentPost[] }) => {<br>  const mapRef = useRef<L.Map | null>(null);<br>  const markerClusterRef = useRef<L.MarkerClusterGroup | null>(null);<br>
  <br>  useEffect(() => {<br>    if (!mapRef.current) {<br>      // Center on Philippines<br>      mapRef.current = L.map('map').setView([12.8797, 121.7740], 6);<br>      <br>      // Add OpenStreetMap tiles<br>      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {<br>        attribution: '© OpenStreetMap contributors'<br>      }).addTo(mapRef.current);<br>      <br>      // Add Philippines GeoJSON boundary<br>      fetch('/assets/philippines.geojson')<br>        .then(response => response.json())<br>        .then(geojson => {<br>          L.geoJSON(geojson, {<br>            style: {<br>              color: '#3388ff',<br>              weight: 2,<br>              fillOpacity: 0.1<br>            }<br>          }).addTo(mapRef.current!);<br>        });<br>      <br>      // Initialize marker cluster group<br>      markerClusterRef.current = L.markerClusterGroup();<br>      mapRef.current.addLayer(markerClusterRef.current);<br>    }<br>```<br><br>**Sentiment Distribution Map of the Philippines:**<br>![Sentiment Map](https://i.imgur.com/7W3gtS4.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Leaflet**: Implemented for interactive mapping<br>• **GeoJSON**: Used for geographic data representation<br>• **Custom Gazetteers**: Created for Philippine location recognition<br>• **Marker Clustering**: Applied for efficient point visualization<br>• **Heat Maps**: Implemented for density visualization<br>• **Spatial Filtering**: Created for geographic area selection<br>• **Administrative Boundary Overlays**: Added for regional context<br>• **Named Entity Recognition**: Enhanced for Philippine place names<br>• **Responsive Map Containers**: Implemented for various screen sizes |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Standard NER models performed poorly on Philippine place names, especially for smaller municipalities and barangays.<br>• **Solution**: Created a custom gazetteer with over 1,700 Philippine locations (provinces, cities, municipalities, and major barangays) and implemented a specialized extraction algorithm.<br><br>• **Challenge**: Initial map was very slow when displaying more than 500 data points simultaneously.<br>• **Solution**: Implemented marker clustering and optimized rendering based on zoom level and viewport.<br><br>• **Challenge**: Many sentiment posts had ambiguous or colloquial location references (e.g., "Metro Manila" vs. specific cities).<br>• **Solution**: Implemented a location normalization system that standardized references to consistent geographic entities.<br><br>• **Lesson Learned**: Geographic visualization for a specific country (Philippines) benefits greatly from country-specific optimizations rather than generic solutions.<br><br>• **Lesson Learned**: Performance considerations are critical for interactive maps, especially when dealing with large datasets on various device capabilities. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 6 / February 24 \- February 28, 2025 |

| Real-time Features and Cross-tab Synchronization |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented WebSocket integration for real-time disaster updates<br>• Created upload progress modal with live tracking<br>• Built cross-tab synchronization for consistent application state<br>• Developed real-time sentiment analysis for immediate processing<br>• Added notification system for alerting users about new data<br>• Implemented reconnection logic for network disruptions<br>• Created server-side broadcasting for data updates<br>• Built client-side event handling for real-time updates<br><br>**WebSocket Implementation for Philippine Disaster Monitoring:**<br>```typescript<br>// Server-side WebSocket implementation
wss.on('connection', (ws: WebSocket) => {
  log('WebSocket client connected');
  
  // Send initial state
  ws.send(JSON.stringify({
    type: 'INITIAL_STATE',
    data: {
      activeSessions: getActiveSessions(),
      latestDisasters: getLatestDisasterEvents(5),
      systemStatus: {
        status: 'operational',
        lastUpdate: new Date().toISOString(),
        activeUsers: getActiveUserCount()
      }
    }
  }));
  
  // Handle client messages
  ws.on('message', (message: string) => {
    try {
      const parsed = JSON.parse(message);
      
      // Handle client requests
      if (parsed.type === 'REQUEST_UPDATES') {
        sendLatestUpdates(ws);
      } else if (parsed.type === 'REGISTER_SESSION') {
        registerClientSession(ws, parsed.sessionId);
      }
    } catch (error) {
      log(`Error processing WebSocket message: ${error}`);
    }
  });
  
  ws.on('close', () => {
    log('WebSocket client disconnected');
    removeClientSession(ws);
  });
});

// Broadcast function for sending updates to all clients
function broadcastUpdate(data: any) {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  });
}
<br>```<br><br>**Upload Progress Modal with Real-time Updates:**<br>![Upload Progress](https://i.imgur.com/fXuYtTu.png)<br><br>**Cross-tab Synchronization Implementation:**<br>```typescript<br>// Client-side cross-tab synchronization
const useCrossTabSync = () => {
  useEffect(() => {
    // Handle storage events for cross-tab communication
    const handleStorageChange = (e: StorageEvent) => {
      if (!e.key) return;
      
      // Handle upload progress updates
      if (e.key === 'uploadProgress' && e.newValue) {
        try {
          const progress = JSON.parse(e.newValue);
          setLocalProgress(progress);
          
          // If this tab is the primary tab (the one doing the upload)
          if (progress.isPrimaryTab && progress.sessionId) {
            // Keep the server informed about which tab is primary
            socket.send(JSON.stringify({
              type: 'PRIMARY_TAB_UPDATE',
              sessionId: progress.sessionId,
              timestamp: Date.now()
            }));
          }
        } catch (error) {
          console.error('Error parsing progress data', error);
        }
      }
      
      // Handle authentication changes
      if (e.key === 'authState' && e.newValue) {
        try {
          const authState = JSON.parse(e.newValue);
          if (authState.event === 'logout') {
            // Handle logout in other tabs
            resetLocalState();
            navigate('/login');
          } else if (authState.event === 'login') {
            // Handle login in other tabs
            refreshUserData();
          }
        } catch (error) {
          console.error('Error parsing auth state', error);
        }
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);
  
  // Function to broadcast changes to other tabs
  const broadcastToTabs = (key: string, data: any) => {
    localStorage.setItem(key, JSON.stringify({
      ...data,
      timestamp: Date.now()
    }));
    
    // LocalStorage events don't fire in the same tab that sets the value
    // so we need to handle the change manually in the current tab
    handleLocalChange(key, data);
  };
  
  return { broadcastToTabs };
};
<br>```<br><br>**Real-time Data Processing Console:**<br>![Real-time Console](https://i.imgur.com/nK2xMwB.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **WebSocket Protocol**: Used for bidirectional communication<br>• **localStorage Events**: Implemented for cross-tab messaging<br>• **Event-driven Architecture**: Applied for real-time system design<br>• **Reconnection Strategies**: Created with exponential backoff<br>• **Optimistic UI Updates**: Implemented for responsive experience<br>• **Browser Storage API**: Used for persistent state management<br>• **Message Queue Pattern**: Applied for reliable event processing<br>• **Progressive Enhancement**: Implemented for fallback behavior<br>• **Heartbeat Mechanism**: Added for connection monitoring |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: WebSocket connections would occasionally drop, especially on mobile networks, without reconnecting properly.<br>• **Solution**: Implemented a robust reconnection system with exponential backoff, connection monitoring, and state recovery upon reconnection.<br><br>• **Challenge**: Cross-tab synchronization had race conditions when multiple tabs tried to update the same data simultaneously.<br>• **Solution**: Implemented a primary tab coordination system where one tab is designated as the "leader" for certain operations, with proper handoff if that tab is closed.<br><br>• **Challenge**: Some browsers limited or suspended WebSocket connections in background tabs.<br>• **Solution**: Added a detection mechanism for background/foreground state changes and implemented a polling fallback for background tabs.<br><br>• **Lesson Learned**: Real-time web applications require comprehensive error handling and fallback mechanisms to provide a reliable user experience across different network conditions.<br><br>• **Lesson Learned**: Cross-tab synchronization is more complex than it initially appears, requiring careful consideration of race conditions, leader election, and state reconciliation. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 \- March 7, 2025 |

| AI Model Training and Feedback System Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed feedback system for improving sentiment analysis accuracy<br>• Implemented mechanism for correcting AI predictions<br>• Created training pipeline for incorporating user feedback<br>• Built text similarity detection for consistent analysis<br>• Added confidence scoring visualization for prediction reliability<br>• Implemented tracking for model performance metrics<br>• Created database tables for storing training examples<br>• Built UI components for reviewing and submitting corrections<br><br>**Feedback System for Philippine Disaster Sentiment Analysis:**<br>![Feedback System](https://i.imgur.com/9dVpWoB.png)<br><br>**Model Training Implementation:**<br>```typescript<br>public async trainModelWithFeedback(feedback: SentimentFeedback): Promise<boolean> {
  try {
    // Create temp file for feedback data
    const tempFilePath = path.join(this.tempDir, `feedback_${nanoid()}.json`);
    await fs.promises.mkdir(this.tempDir, { recursive: true });
    await fs.promises.writeFile(tempFilePath, JSON.stringify(feedback));
    
    // Spawn Python process for training
    const pythonProcess = spawn(this.pythonBinary, [
      this.scriptPath,
      '--train',
      '--feedback-file',
      tempFilePath
    ]);
    
    // Process output and logging
    let result = false;
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      log(`Python training: ${output}`, 'python');
      pythonConsoleMessages.push({
        message: output,
        timestamp: new Date()
      });
      
      if (output.includes('Training successful')) {
        result = true;
      }
    });
    
    // Handle errors
    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString().trim();
      log(`Python training error: ${error}`, 'python');
      pythonConsoleMessages.push({
        message: `ERROR: ${error}`,
        timestamp: new Date()
      });
    });
    
    // Wait for process to complete
    await new Promise<void>((resolve) => {
      pythonProcess.on('close', () => {
        resolve();
      });
    });
    
    // Clean up temp file
    await fs.promises.unlink(tempFilePath);
    
    // Clear cache for this text
    this.clearCacheForText(feedback.originalText);
    
    return result;
  } catch (error) {
    log(`Error training model: ${error}`, 'python');
    return false;
  }
}
<br>```<br><br>**Feedback Component for Correcting Philippine Disaster Sentiments:**<br>```tsx<br>const SentimentFeedbackForm = ({ post }: { post: SentimentPost }) => {
  const { register, handleSubmit, formState: { errors } } = useForm<{
    correctedSentiment: string;
    correctedLocation?: string;
    correctedDisasterType?: string;
  }>({
    defaultValues: {
      correctedSentiment: post.sentiment,
      correctedLocation: post.location,
      correctedDisasterType: post.disasterType
    }
  });
  
  const queryClient = useQueryClient();
  const { toast } = useToast();
  
  const { mutate, isPending } = useMutation({
    mutationFn: async (data: {
      originalPostId: number;
      originalText: string;
      originalSentiment: string;
      correctedSentiment: string;
      correctedLocation?: string;
      correctedDisasterType?: string;
    }) => {
      const response = await fetch('/api/sentiment-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }
      
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: 'Feedback submitted',
        description: 'Thank you for helping improve our analysis'
      });
      queryClient.invalidateQueries({queryKey: ['/api/sentiment-posts']});
      queryClient.invalidateQueries({queryKey: ['/api/sentiment-feedback']});
    },
    onError: (error) => {
      toast({
        title: 'Failed to submit feedback',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive'
      });
    }
  });
  
  const onSubmit = (data: any) => {
    mutate({
      originalPostId: post.id,
      originalText: post.text,
      originalSentiment: post.sentiment,
      correctedSentiment: data.correctedSentiment,
      correctedLocation: data.correctedLocation,
      correctedDisasterType: data.correctedDisasterType
    });
  };
<br>```<br><br>**Confidence Visualization for Typhoon Sentiment Analysis:**<br>![Confidence Visualization](https://i.imgur.com/d7YcRu5.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Transfer Learning**: Implemented for model adaptation<br>• **Active Learning**: Applied for selective training data<br>• **Text Similarity Algorithms**: Used Jaccard and TF-IDF similarity<br>• **Confidence Calibration**: Implemented for honest uncertainty<br>• **NLTK Text Processing**: Used for Philippine context analysis<br>• **React Hook Form**: Applied for feedback form management<br>• **Bayesian Calibration**: Used for confidence scoring<br>• **TanStack Mutation**: Implemented for feedback submission<br>• **Model Version Control**: Applied for tracking improvements |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: The initial feedback system required too many examples to show noticeable improvement in the model.<br>• **Solution**: Implemented category-based learning that allowed improvements in specific categories (e.g., typhoon-related sentiments) with fewer examples.<br><br>• **Challenge**: Some user feedback was contradictory or incorrect, causing model degradation.<br>• **Solution**: Implemented a validation step and consensus mechanism that required multiple similar corrections before applying them to the model.<br><br>• **Challenge**: Confidence scores from the model were not calibrated, leading to overconfident incorrect predictions.<br>• **Solution**: Applied Bayesian calibration techniques to align confidence scores with actual prediction accuracy.<br><br>• **Lesson Learned**: Effective AI feedback systems need to be designed for both collection and validation, as user input can be inconsistent.<br><br>• **Lesson Learned**: For Philippine disaster sentiment analysis, domain-specific knowledge (local terms, expressions, contexts) significantly improved performance compared to general models. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 \- March 14, 2025 |

| Data Evaluation and Export Feature Implementation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Created evaluation metrics calculation for model performance assessment<br>• Implemented visualization of accuracy, precision, recall, and F1 score<br>• Developed CSV export functionality for analyzed disaster data<br>• Built filtering options for customized data exports<br>• Created RawData component for browsing complete dataset<br>• Implemented pagination and search for large datasets<br>• Added sorting functionality for data exploration<br>• Created download mechanism for exported data<br><br>**Evaluation Dashboard for Philippine Disaster Sentiment Analysis:**<br>![Evaluation Dashboard](https://i.imgur.com/zMhIjkW.png)<br><br>**Metrics Calculation Implementation:**<br>```typescript<br>// Calculate evaluation metrics for sentiment analysis
const calculateMetrics = (predictions: Array<{actual: string, predicted: string}>) => {
  // Get unique sentiment classes
  const classes = Array.from(new Set([
    ...predictions.map(p => p.actual),
    ...predictions.map(p => p.predicted)
  ]));
  
  // Initialize confusion matrix
  const confusionMatrix = {};
  classes.forEach(actual => {
    confusionMatrix[actual] = {};
    classes.forEach(predicted => {
      confusionMatrix[actual][predicted] = 0;
    });
  });
  
  // Fill confusion matrix
  predictions.forEach(p => {
    confusionMatrix[p.actual][p.predicted]++;
  });
  
  // Calculate metrics for each class
  const classMetrics = {};
  classes.forEach(cls => {
    // True positives, false positives, false negatives
    let tp = confusionMatrix[cls][cls];
    let fp = 0;
    let fn = 0;
    
    classes.forEach(other => {
      if (other !== cls) {
        fp += confusionMatrix[other][cls]; // Predicted as cls but was other
        fn += confusionMatrix[cls][other]; // Actual was cls but predicted as other
      }
    });
    
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    classMetrics[cls] = { precision, recall, f1 };
  });
  
  // Calculate overall metrics (weighted average)
  let totalInstances = predictions.length;
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;
  
  classes.forEach(cls => {
    const classCount = predictions.filter(p => p.actual === cls).length;
    const weight = classCount / totalInstances;
    
    weightedPrecision += classMetrics[cls].precision * weight;
    weightedRecall += classMetrics[cls].recall * weight;
    weightedF1 += classMetrics[cls].f1 * weight;
  });
  
  // Calculate accuracy
  const accuracy = predictions.filter(p => p.actual === p.predicted).length / totalInstances;
  
  return {
    accuracy,
    precision: weightedPrecision,
    recall: weightedRecall,
    f1Score: weightedF1,
    classMetrics
  };
};
<br>```<br><br>**CSV Export Implementation for Philippine Disaster Data:**<br>```typescript<br>app.get('/api/export-csv', async (req: Request, res: Response) => {
  try {
    // Parse query parameters for filtering
    const { startDate, endDate, sentiment, disasterType, location } = req.query;
    
    // Get filtered data
    const posts = await storage.getSentimentPostsFiltered({
      startDate: startDate ? new Date(startDate as string) : undefined,
      endDate: endDate ? new Date(endDate as string) : undefined,
      sentiment: sentiment as string,
      disasterType: disasterType as string,
      location: location as string
    });
    
    // Generate CSV header
    const csvHeader = 'text,timestamp,source,language,sentiment,confidence,location,disasterType,explanation';
    
    // Generate CSV content with streaming to handle large datasets
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="philippines-disaster-sentiment-data.csv"');
    
    // Write header
    res.write(csvHeader + '\n');
    
    // Stream rows to avoid memory issues with large datasets
    for (const post of posts) {
      const row = [
        post.text.replace(/,/g, ' ').replace(/\n/g, ' '),
        post.timestamp.toISOString(),
        post.source || '',
        post.language || '',
        post.sentiment,
        post.confidence.toString(),
        post.location || '',
        post.disasterType || '',
        (post.explanation || '').replace(/,/g, ' ').replace(/\n/g, ' ')
      ].map(field => `"${field.replace(/"/g, '""')}"`).join(',');
      
      res.write(row + '\n');
    }
    
    res.end();
  } catch (error) {
    log(`Error exporting CSV: ${error}`);
    res.status(500).json({ error: 'Failed to export data' });
  }
});
<br>```<br><br>**Raw Data Explorer for Disaster Sentiment Posts:**<br>![Raw Data Explorer](https://i.imgur.com/UeK1DWJ.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Confusion Matrix Analysis**: Used for thorough performance evaluation<br>• **Chart.js**: Implemented for metrics visualization<br>• **Streaming CSV Generation**: Applied for efficient data export<br>• **Virtual Scrolling**: Used for large dataset display<br>• **Data Pagination**: Implemented for efficient data retrieval<br>• **Content Disposition Headers**: Used for proper file downloads<br>• **Data Filtering Algorithms**: Created for customized exports<br>• **Weighted Metrics Calculation**: Applied for balanced evaluation<br>• **Tanstack Table**: Used for advanced data table features |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial metrics calculation didn't account for class imbalance in Philippine disaster data (more negative than positive sentiments).<br>• **Solution**: Implemented weighted metrics calculation that considered class distribution, providing more accurate evaluation.<br><br>• **Challenge**: CSV export failed with large datasets due to memory limitations when trying to process all records simultaneously.<br>• **Solution**: Implemented streaming response with chunked processing, allowing export of arbitrarily large datasets.<br><br>• **Challenge**: Raw data explorer became extremely slow with more than 1000 records due to inefficient rendering.<br>• **Solution**: Implemented virtual scrolling with windowing techniques, only rendering visible rows for improved performance.<br><br>• **Lesson Learned**: When evaluating AI models, especially for sentiment analysis, class distribution awareness is crucial for meaningful metrics.<br><br>• **Lesson Learned**: Working with large datasets requires stream processing approaches throughout the stack, not just on the server but also in the browser. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 9 / March 17 \- March 21, 2025 |

| Performance Optimization and System Scalability Improvements |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Optimized database queries with proper indexing and joins<br>• Implemented client-side caching with TanStack Query<br>• Created comprehensive error handling system<br>• Improved memory management for large dataset processing<br>• Implemented batch processing for resource-intensive operations<br>• Added database connection pooling for better throughput<br>• Created performance monitoring dashboard<br>• Implemented query time tracking for bottleneck identification<br><br>**Database Query Optimization:**<br>```typescript<br>// Before optimization - N+1 query problem
async function getSentimentPostsWithUserDetails() {
  // This fetches all posts first
  const posts = await db.select().from(sentimentPosts);
  
  // Then makes a separate query for each post that has a processedBy user
  for (let i = 0; i < posts.length; i++) {
    if (posts[i].processedBy) {
      posts[i].user = await db.select()
        .from(users)
        .where(eq(users.id, posts[i].processedBy))
        .limit(1)
        .then(rows => rows[0]);
    }
  }
  
  return posts;
}

// After optimization - Single join query
async function getSentimentPostsWithUserDetails() {
  return db.select({
    id: sentimentPosts.id,
    text: sentimentPosts.text,
    timestamp: sentimentPosts.timestamp,
    source: sentimentPosts.source,
    language: sentimentPosts.language,
    sentiment: sentimentPosts.sentiment,
    confidence: sentimentPosts.confidence,
    location: sentimentPosts.location,
    disasterType: sentimentPosts.disasterType,
    fileId: sentimentPosts.fileId,
    explanation: sentimentPosts.explanation,
    processedBy: sentimentPosts.processedBy,
    user: {
      id: users.id,
      username: users.username,
      fullName: users.fullName
    }
  })
  .from(sentimentPosts)
  .leftJoin(users, eq(sentimentPosts.processedBy, users.id));
}
<br>```<br><br>**Database Index Implementation:**<br>```typescript<br>// Adding indexes for common query patterns
export const sentimentPosts = pgTable("sentiment_posts", {
  id: serial("id").primaryKey(),
  text: text("text").notNull(),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  source: text("source"),
  language: text("language"),
  sentiment: text("sentiment").notNull(),
  confidence: real("confidence").notNull(),
  location: text("location"),
  disasterType: text("disaster_type"),
  fileId: integer("file_id"),
  explanation: text("explanation"),
  processedBy: integer("processed_by").references(() => users.id),
  aiTrustMessage: text("ai_trust_message"),
}, (table) => {
  return {
    // Index for timestamp-based queries (common for time series data)
    timestampIdx: index("sentiment_posts_timestamp_idx").on(table.timestamp),
    
    // Composite index for filtering by sentiment and timestamp
    sentimentTimeIdx: index("sentiment_posts_sentiment_time_idx").on(
      table.sentiment, 
      table.timestamp
    ),
    
    // Index for location-based queries
    locationIdx: index("sentiment_posts_location_idx").on(table.location),
    
    // Index for disaster type queries
    disasterTypeIdx: index("sentiment_posts_disaster_type_idx").on(table.disasterType),
    
    // Index for querying by file ID (for CSV imports)
    fileIdIdx: index("sentiment_posts_file_id_idx").on(table.fileId)
  }
});
<br>```<br><br>**Performance Monitoring Dashboard:**<br>![Performance Dashboard](https://i.imgur.com/2wbLhRD.png)<br><br>**Connection Pooling Configuration:**<br>```typescript<br>// Database connection pool configuration
import { Pool, neonConfig } from '@neondatabase/serverless';
import ws from 'ws';

// Configure WebSocket for Neon serverless
neonConfig.webSocketConstructor = ws;

// Pool configuration based on environment
const poolConfig = {
  connectionString: process.env.DATABASE_URL,
  max: process.env.NODE_ENV === 'production' ? 20 : 10, // More connections in production
  idleTimeoutMillis: 30000, // 30 seconds
  connectionTimeoutMillis: 5000, // 5 seconds
  allowExitOnIdle: false
};

if (process.env.NODE_ENV === 'production') {
  // SSL is required in production
  poolConfig.ssl = {
    rejectUnauthorized: false // Required for some hosting providers
  };
}

export const pool = new Pool(poolConfig);

// Monitoring query execution time
pool.on('connect', (client) => {
  const originalQuery = client.query.bind(client);
  client.query = (...args) => {
    const start = Date.now();
    const query = originalQuery(...args);
    
    const queryText = typeof args[0] === 'string' 
      ? args[0] 
      : args[0].text;
    
    query.then(() => {
      const duration = Date.now() - start;
      if (duration > 100) { // Log slow queries (>100ms)
        log(`Slow query (${duration}ms): ${queryText.substring(0, 80)}...`);
      }
    });
    
    return query;
  };
});

// Proper pool error handling
pool.on('error', (err, client) => {
  log(`Unexpected database pool error: ${err.message}`);
  // Don't throw, just log, as throwing would crash the server
});
<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Database Indexing**: Applied on frequently queried columns<br>• **Query Execution Planning**: Used to optimize SQL statements<br>• **Connection Pooling**: Implemented for efficient database connections<br>• **Memory Profiling**: Used Node.js heap snapshots for memory analysis<br>• **React Query Caching**: Applied with stale time and cache invalidation<br>• **Resource Monitoring**: Implemented server-side tracking<br>• **Query Time Tracking**: Added for identifying slow queries<br>• **Batch Processing**: Implemented for large operations<br>• **Data Normalization**: Applied for efficient storage |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Application response times were degrading significantly as the dataset grew beyond 10,000 sentiment posts.<br>• **Solution**: Implemented proper database indexing, query optimization, and connection pooling, resulting in a 6x improvement in query performance.<br><br>• **Challenge**: Memory usage would spike during CSV processing, occasionally causing out-of-memory errors with large files (>20MB).<br>• **Solution**: Refactored the processing pipeline to use streaming and batch processing, limiting memory usage regardless of file size.<br><br>• **Challenge**: Client-side performance was degrading with increasing data loaded into the application state.<br>• **Solution**: Implemented windowing techniques, virtual scrolling, and more selective data fetching based on visible UI components.<br><br>• **Lesson Learned**: Database performance optimization should be data-driven, focusing on actual query patterns rather than theoretical optimizations.<br><br>• **Lesson Learned**: Performance considerations should span the entire stack - from database to server to client - as bottlenecks can appear at any level. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 10 / March 24 \- March 28, 2025 |

| Security Implementation and User Management Enhancement |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Conducted security audit and implemented recommended fixes<br>• Enhanced authentication with improved token management<br>• Implemented comprehensive input validation<br>• Added protection against common web vulnerabilities<br>• Developed user profile management functionality<br>• Created role-based access control system<br>• Implemented secure file upload handling<br>• Added logging of security-relevant actions<br>• Created password reset functionality<br><br>**Security Headers Implementation:**<br>```typescript<br>// Security middleware configuration
app.use((req, res, next) => {
  // Prevent XSS attacks
  res.setHeader('X-XSS-Protection', '1; mode=block');
  
  // Strict Content Security Policy
  res.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' data: https://i.imgur.com; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' wss:; font-src 'self' data:;");
  
  // Prevent clickjacking
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  
  // Prevent MIME type sniffing
  res.setHeader('X-Content-Type-Options', 'nosniff');
  
  // Strict Transport Security (in production)
  if (process.env.NODE_ENV === 'production') {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
  }
  
  // Referrer Policy
  res.setHeader('Referrer-Policy', 'same-origin');
  
  // Feature Policy restrictions
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  next();
});
<br>```<br><br>**CSRF Protection Implementation:**<br>```typescript<br>// CSRF Protection
const csrfProtection = (req: Request, res: Response, next: NextFunction) => {
  // Skip for GET requests and non-authenticated routes
  if (req.method === 'GET' || 
      req.path === '/api/auth/login' || 
      req.path === '/api/auth/signup') {
    return next();
  }
  
  const csrfToken = req.headers['x-csrf-token'];
  const authHeader = req.headers['authorization'];
  
  if (!csrfToken || !authHeader) {
    return res.status(403).json({ error: 'CSRF token missing' });
  }
  
  const token = authHeader.split(' ')[1];
  
  // Verify CSRF token (a hash of the session token + secret)
  const expectedToken = crypto
    .createHash('sha256')
    .update(`${token}${process.env.CSRF_SECRET || 'default_csrf_secret'}`)
    .digest('hex');
    
  if (csrfToken !== expectedToken) {
    log(`CSRF token validation failed. Expected: ${expectedToken}, Received: ${csrfToken}`);
    return res.status(403).json({ error: 'Invalid CSRF token' });
  }
  
  next();
};

// Apply CSRF protection to state-changing endpoints
app.post('*', csrfProtection);
app.put('*', csrfProtection);
app.patch('*', csrfProtection);
app.delete('*', csrfProtection);
<br>```<br><br>**Role-based Access Control:**<br>```typescript<br>// Role-based access control middleware
const roleCheck = (allowedRoles: string[]) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      // Get user from authenticated request
      const user = req.user;
      
      if (!user) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      
      // Check if user's role is in the allowed roles
      if (!allowedRoles.includes(user.role)) {
        // Log unauthorized access attempt
        log(`Unauthorized access attempt: User ${user.username} (${user.role}) tried to access endpoint requiring ${allowedRoles.join(', ')}`);
        
        return res.status(403).json({ error: 'Insufficient permissions' });
      }
      
      next();
    } catch (error) {
      next(error);
    }
  };
};

// Example usage for admin-only routes
app.delete('/api/disaster-events/:id', authenticate, roleCheck(['admin']), async (req, res) => {
  // Only admins can delete disaster events
  try {
    await storage.deleteDisasterEvent(parseInt(req.params.id));
    res.json({ success: true });
  } catch (error) {
    handleError(error, res);
  }
});
<br>```<br><br>**User Profile Management:**<br>![User Profile](https://i.imgur.com/8wTkRpL.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **OWASP Top 10**: Followed for addressing common vulnerabilities<br>• **Content Security Policy (CSP)**: Implemented header restrictions<br>• **CSRF Protection**: Added with token validation<br>• **Input Sanitization**: Applied for all user inputs<br>• **bcrypt**: Used for password hashing with work factor 12<br>• **JWT**: Implemented with short expiry and refresh tokens<br>• **Rate Limiting**: Added for authentication endpoints<br>• **Role-based Access Control**: Implemented for authorization<br>• **Audit Logging**: Added for security-relevant actions |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial security audit revealed several vulnerabilities, including potential XSS in user-generated content display.<br>• **Solution**: Implemented a strict Content Security Policy and proper output encoding for user-generated content.<br><br>• **Challenge**: The original authentication system used long-lived tokens, creating a security risk if tokens were compromised.<br>• **Solution**: Redesigned the system to use short-lived access tokens (~15 minutes) with refresh tokens for obtaining new access tokens.<br><br>• **Challenge**: File upload functionality introduced potential security risks with malicious file uploads.<br>• **Solution**: Implemented strict validation of file types, secure storage with randomized filenames, and proper MIME type checking.<br><br>• **Lesson Learned**: Security needs to be implemented at multiple layers (database, server, transport, client) to be effective.<br><br>• **Lesson Learned**: For a security-critical application like disaster monitoring, defensive programming with careful input validation and proper output encoding is essential. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 11 / April 1 \- April 4, 2025 |

| API Integration and External Connectivity |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed RESTful API endpoints for external integration<br>• Created comprehensive API documentation<br>• Implemented authentication for API access<br>• Built webhook functionality for event notifications<br>• Added CORS configuration for cross-origin requests<br>• Implemented rate limiting to prevent abuse<br>• Created example client code and integration guides<br>• Built API versioning for backward compatibility<br>• Developed response standards for consistent integration<br><br>**API Authentication Implementation:**<br>```typescript<br>// API Key authentication middleware
const apiKeyAuth = async (req: Request, res: Response, next: NextFunction) => {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey) {
    return res.status(401).json({
      error: 'API key is required',
      code: 'MISSING_API_KEY'
    });
  }
  
  try {
    // Verify API key
    const apiKeyDoc = await storage.getApiKeyByToken(apiKey as string);
    
    if (!apiKeyDoc) {
      return res.status(401).json({
        error: 'Invalid API key',
        code: 'INVALID_API_KEY'
      });
    }
    
    // Check if API key is expired
    if (apiKeyDoc.expiresAt && new Date(apiKeyDoc.expiresAt) < new Date()) {
      return res.status(401).json({
        error: 'API key has expired',
        code: 'EXPIRED_API_KEY'
      });
    }
    
    // Attach API key info to request for usage tracking
    req.apiKey = apiKeyDoc;
    
    // Track API usage
    await storage.trackApiUsage({
      apiKeyId: apiKeyDoc.id,
      endpoint: req.path,
      method: req.method,
      timestamp: new Date()
    });
    
    next();
  } catch (error) {
    next(error);
  }
};
<br>```<br><br>**API Documentation:**<br>![API Documentation](https://i.imgur.com/K3dVb4T.png)<br><br>**Webhook Implementation:**<br>```typescript<br>// Webhook registration endpoint
app.post('/api/webhooks', authenticate, async (req: Request, res: Response) => {
  try {
    const { url, events, description } = req.body;
    
    // Validate webhook data
    if (!url || !Array.isArray(events) || events.length === 0) {
      return res.status(400).json({
        error: 'Invalid webhook configuration',
        details: 'URL and at least one event type are required'
      });
    }
    
    // Validate URL format
    try {
      new URL(url);
    } catch (error) {
      return res.status(400).json({
        error: 'Invalid URL format',
        details: 'Please provide a valid URL'
      });
    }
    
    // Validate event types
    const validEvents = ['new_sentiment_post', 'new_disaster_event', 'model_update'];
    const invalidEvents = events.filter(event => !validEvents.includes(event));
    
    if (invalidEvents.length > 0) {
      return res.status(400).json({
        error: 'Invalid event types',
        details: `The following event types are invalid: ${invalidEvents.join(', ')}`,
        validEvents
      });
    }
    
    // Create webhook
    const webhook = await storage.createWebhook({
      url,
      events,
      description,
      userId: req.user!.id,
      secretKey: crypto.randomBytes(32).toString('hex') // Generate secret for signature
    });
    
    res.status(201).json({
      id: webhook.id,
      url: webhook.url,
      events: webhook.events,
      description: webhook.description,
      secretKey: webhook.secretKey,
      createdAt: webhook.createdAt
    });
  } catch (error) {
    handleError(error, res);
  }
});

// Function to trigger webhooks for events
async function triggerWebhooks(eventType: string, payload: any) {
  try {
    // Get all webhooks subscribed to this event
    const webhooks = await storage.getWebhooksByEvent(eventType);
    
    for (const webhook of webhooks) {
      // Prepare the payload
      const webhookPayload = {
        event: eventType,
        timestamp: new Date().toISOString(),
        data: payload
      };
      
      // Create signature for webhook verification
      const signature = crypto
        .createHmac('sha256', webhook.secretKey)
        .update(JSON.stringify(webhookPayload))
        .digest('hex');
      
      // Send the webhook in the background
      fetch(webhook.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Webhook-Signature': signature,
          'X-Event-Type': eventType
        },
        body: JSON.stringify(webhookPayload)
      }).catch(error => {
        log(`Error sending webhook to ${webhook.url}: ${error.message}`);
        // Track failed webhook delivery for retry
        storage.trackWebhookFailure(webhook.id, error.message);
      });
    }
  } catch (error) {
    log(`Error triggering webhooks: ${error}`);
  }
}
<br>```<br><br>**API Rate Limiting:**<br>```typescript<br>// Rate limiting configuration
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: async (request, response) => {
    // Different limits based on authentication
    if (request.apiKey) {
      // API key has its own limit stored in the database
      return request.apiKey.rateLimit || 300; // Default to 300 requests per 15 minutes
    } else if (request.user) {
      // Authenticated users get higher limit
      return 150; // 150 requests per 15 minutes
    } else {
      // Unauthenticated requests get lowest limit
      return 30; // 30 requests per 15 minutes
    }
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
  keyGenerator: (request) => {
    // Use API key or user ID or IP address as the rate limit key
    if (request.apiKey) {
      return `api_${request.apiKey.id}`;
    } else if (request.user) {
      return `user_${request.user.id}`;
    } else {
      return request.ip;
    }
  },
  handler: (request, response, next, options) => {
    response.status(429).json({
      error: 'Too many requests',
      code: 'RATE_LIMIT_EXCEEDED',
      retryAfter: Math.ceil(options.windowMs / 1000)
    });
  }
});

// Apply rate limiting to API routes
app.use('/api/v1', apiLimiter);
app.use('/api/v2', apiLimiter);
<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **RESTful API Design**: Applied for consistent interface patterns<br>• **OpenAPI/Swagger**: Used for interactive API documentation<br>• **JWT & API Key Authentication**: Implemented for flexible access<br>• **Rate Limiting**: Applied with dynamic limits by user type<br>• **CORS Configuration**: Implemented for secure cross-origin access<br>• **Webhook Pattern**: Used for event notification delivery<br>• **HMAC Signatures**: Added for webhook verification<br>• **API Versioning**: Implemented for backward compatibility<br>• **Standardized Response Format**: Created for consistent integration |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Designing an API that was both flexible enough for various integration needs while remaining secure and easy to use.<br>• **Solution**: Created a tiered API design with both simple endpoints for basic integration and advanced endpoints for complex scenarios, all with comprehensive documentation.<br><br>• **Challenge**: Webhook deliveries were unreliable, with some failing silently due to network issues or timeouts.<br>• **Solution**: Implemented a retry mechanism with exponential backoff and a delivery logging system to track and recover from failures.<br><br>• **Challenge**: Initial rate limiting was too restrictive for legitimate high-volume users but too permissive for potential abuse.<br>• **Solution**: Implemented dynamic rate limiting based on authentication level and user-specific limits that could be adjusted as needed.<br><br>• **Lesson Learned**: External API design requires more careful planning than internal APIs, as changes can break integrations for external users.<br><br>• **Lesson Learned**: For webhook systems, implementing verification through signatures is essential to prevent security issues with webhook deliveries. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 12 / April 7 \- April 11, 2025 |

| System Testing and Bug Fixing |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Conducted end-to-end testing of all major application flows<br>• Created and executed test cases for critical features<br>• Identified and resolved bugs across the application<br>• Performed stress testing to validate system stability<br>• Tested edge cases for all critical functionality<br>• Implemented comprehensive logging for debugging<br>• Conducted cross-browser and cross-device testing<br>• Created automated tests for critical components<br>• Fixed UI inconsistencies across different devices<br><br>**Test Implementation:**<br>```typescript<br>// Testing sentiment analysis API
describe('Sentiment Analysis API', () => {
  // Test positive sentiment detection
  it('should correctly analyze positive sentiment in disaster context', async () => {
    const response = await request(app)
      .post('/api/analyze-text')
      .send({
        text: 'The relief efforts after the typhoon have been remarkably effective. Volunteers are working tirelessly to help affected communities.'
      })
      .set('Accept', 'application/json');
      
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('sentiment', 'positive');
    expect(response.body).toHaveProperty('confidence');
    expect(response.body.confidence).toBeGreaterThan(0.7);
    expect(response.body).toHaveProperty('disasterType', 'typhoon');
  });
  
  // Test negative sentiment detection
  it('should correctly analyze negative sentiment in disaster context', async () => {
    const response = await request(app)
      .post('/api/analyze-text')
      .send({
        text: 'The flood has devastated our community. Many families have lost their homes and possessions.'
      })
      .set('Accept', 'application/json');
      
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('sentiment', 'negative');
    expect(response.body).toHaveProperty('confidence');
    expect(response.body.confidence).toBeGreaterThan(0.7);
    expect(response.body).toHaveProperty('disasterType', 'flood');
  });
  
  // Test location extraction
  it('should correctly extract Philippine locations from text', async () => {
    const response = await request(app)
      .post('/api/analyze-text')
      .send({
        text: 'The earthquake in Bohol caused significant damage to historical structures, especially churches.'
      })
      .set('Accept', 'application/json');
      
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('location');
    expect(response.body.location).toContain('Bohol');
  });
  
  // Test invalid input handling
  it('should handle empty text input', async () => {
    const response = await request(app)
      .post('/api/analyze-text')
      .send({ text: '' })
      .set('Accept', 'application/json');
      
    expect(response.status).toBe(400);
    expect(response.body).toHaveProperty('error');
  });
  
  // Test input with mixed sentiments
  it('should correctly handle text with mixed sentiments', async () => {
    const response = await request(app)
      .post('/api/analyze-text')
      .send({
        text: 'While the typhoon caused significant damage, the community response has been inspiring with neighbors helping each other recover.'
      })
      .set('Accept', 'application/json');
      
    expect(response.status).toBe(200);
    // Should detect the overall sentiment correctly based on context
    expect(response.body).toHaveProperty('sentiment');
    // Should have lower confidence due to mixed sentiment
    expect(response.body.confidence).toBeLessThan(0.9);
  });
});
<br>```<br><br>**Comprehensive Logging Implementation:**<br>```typescript<br>// Enhanced logging system for debugging
import winston from 'winston';
import { format } from 'winston';

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  debug: 4,
};

// Define level based on environment
const level = () => {
  const env = process.env.NODE_ENV || 'development';
  return env === 'development' ? 'debug' : 'info';
};

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  debug: 'blue',
};

// Add colors to winston
winston.addColors(colors);

// Define format
const logFormat = format.combine(
  format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
  format.printf(
    (info) => `${info.timestamp} ${info.level}: ${info.message}${info.data ? ' ' + JSON.stringify(info.data) : ''}`
  )
);

// Create file transports
const transports = [
  new winston.transports.Console({
    format: format.combine(format.colorize({ all: true }), logFormat),
  }),
  new winston.transports.File({
    filename: 'logs/error.log',
    level: 'error',
  }),
  new winston.transports.File({ filename: 'logs/combined.log' }),
];

// Create the logger
export const logger = winston.createLogger({
  level: level(),
  levels,
  format: logFormat,
  transports,
});

// Middleware to log HTTP requests
export const httpLogger = (req: Request, res: Response, next: NextFunction) => {
  // Skip logging for static assets
  if (req.path.startsWith('/assets') || req.path.startsWith('/favicon')) {
    return next();
  }
  
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.http(
      `${req.method} ${req.path} ${res.statusCode} ${duration}ms`,
      {
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        ip: req.ip,
        userAgent: req.get('user-agent') || 'unknown',
      }
    );
    
    // Log slow requests separately
    if (duration > 1000) {
      logger.warn(`Slow request: ${req.method} ${req.path} took ${duration}ms`, {
        method: req.method,
        path: req.path,
        duration,
        body: req.method !== 'GET' ? req.body : undefined,
        query: req.query,
      });
    }
  });
  
  next();
};
<br>```<br><br>**Bug Tracking System:**<br>![Bug Tracking](https://i.imgur.com/JfkQa2L.png)<br><br>**Cross-browser Testing Results:**<br>![Cross-browser Testing](https://i.imgur.com/76hXdAG.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Jest**: Used for unit and integration testing<br>• **Supertest**: Implemented for API endpoint testing<br>• **React Testing Library**: Applied for component testing<br>• **Cypress**: Used for end-to-end testing of user flows<br>• **Winston**: Implemented for comprehensive logging<br>• **Browser Stack**: Used for cross-browser compatibility testing<br>• **Lighthouse**: Applied for performance and accessibility audit<br>• **Artillery**: Used for load and stress testing<br>• **Error Boundary Components**: Implemented for graceful error handling |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Certain bugs only appeared in specific browser/device combinations, making them difficult to reproduce and fix.<br>• **Solution**: Implemented a comprehensive cross-browser testing strategy with BrowserStack and created a standardized test matrix covering all major combinations.<br><br>• **Challenge**: Load testing revealed that the system would degrade significantly under high concurrent user loads (>100 simultaneous users).<br>• **Solution**: Implemented performance optimizations including query caching, connection pooling, and server-side rendering of critical components.<br><br>• **Challenge**: Error reporting was inconsistent, making it difficult to identify the root cause of issues in production.<br>• **Solution**: Implemented a comprehensive logging system with contextual information and proper categorization of error types.<br><br>• **Lesson Learned**: Testing should cover not just functionality but also performance, security, and compatibility across different environments.<br><br>• **Lesson Learned**: For disaster monitoring systems, resilience testing is particularly important - the system must remain operational under adverse conditions including high load and partial service failures. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 \- April 19, 2025 |

| Final Documentation and Deployment Preparation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Created comprehensive user documentation with guides<br>• Developed technical documentation covering system architecture<br>• Prepared deployment configurations for production environment<br>• Created database migration scripts for smooth deployment<br>• Implemented asset bundling and optimization for production<br>• Conducted final security review with necessary adjustments<br>• Prepared maintenance documentation and runbooks<br>• Created custom error pages and fallback states<br>• Finalized a roadmap for future development<br><br>**User Documentation:**<br>![User Documentation](https://i.imgur.com/pNIJjKG.png)<br><br>**System Architecture Documentation:**<br>![System Architecture](https://i.imgur.com/bE8DVLK.png)<br><br>**Deployment Configuration:**<br>```typescript<br>// Production configuration
const productionConfig = {
  // Server settings
  server: {
    port: process.env.PORT || 3000,
    host: '0.0.0.0', // Bind to all interfaces
    cors: {
      origin: [
        'https://disaster-resilience.ph', 
        'https://www.disaster-resilience.ph'
      ],
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token', 'X-API-Key'],
      credentials: true
    }
  },
  
  // Database settings
  database: {
    connectionString: process.env.DATABASE_URL,
    ssl: {
      rejectUnauthorized: false // Required for some hosting providers
    },
    pool: {
      min: 2,
      max: 20,
      idle: 10000
    }
  },
  
  // Python service settings
  pythonService: {
    binary: '/app/venv/bin/python3',
    scriptPath: '/app/server/python/process.py',
    maxProcesses: 5, // Maximum concurrent Python processes
    timeout: 300000, // 5 minutes timeout for long-running processes
  },
  
  // Security settings
  security: {
    jwtSecret: process.env.JWT_SECRET,
    tokenExpiry: '15m', // 15 minutes for access tokens
    refreshTokenExpiry: '7d', // 7 days for refresh tokens
    passwordResetExpiry: '1h', // 1 hour for password reset tokens
    bcryptRounds: 12, // Work factor for password hashing
    csrfSecret: process.env.CSRF_SECRET,
    rateLimits: {
      login: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 10 // 10 login attempts per 15 minutes
      },
      api: {
        windowMs: 60 * 1000, // 1 minute
        max: 60 // 60 requests per minute
      }
    }
  },
  
  // File storage settings
  storage: {
    uploadDir: '/app/uploads',
    maxFileSize: 50 * 1024 * 1024, // 50MB max file size
    allowedTypes: ['text/csv', 'application/vnd.ms-excel', 'application/csv']
  },
  
  // Redis cache settings (if used)
  redis: {
    url: process.env.REDIS_URL,
    ttl: {
      default: 3600, // 1 hour
      query: 600, // 10 minutes for DB query cache
      page: 300 // 5 minutes for rendered page cache
    }
  },
  
  // Logging settings
  logging: {
    level: 'info',
    format: 'json',
    transports: [
      { type: 'console' },
      { type: 'file', filename: '/app/logs/app.log' }
    ]
  }
};
<br>```<br><br>**Database Migration Script:**<br>```typescript<br>// Database migration script for deployment
import { drizzle } from 'drizzle-orm/neon-serverless';
import { migrate } from 'drizzle-orm/neon-serverless/migrator';
import { Pool, neonConfig } from '@neondatabase/serverless';
import ws from 'ws';
import * as schema from '../shared/schema';

// Configure WebSocket for serverless environments
neonConfig.webSocketConstructor = ws;

// Migration function
async function runMigrations() {
  console.log('Starting database migrations...');
  
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL environment variable is required');
  }
  
  try {
    // Create connection pool
    const pool = new Pool({ connectionString: process.env.DATABASE_URL });
    const db = drizzle(pool, { schema });
    
    // Run migrations from the migrations folder
    await migrate(db, { migrationsFolder: './migrations' });
    
    console.log('Migrations completed successfully');
    
    // Verify database connection and schema
    const tables = await db.select({
      tableName: sql`tablename`.mapWith(String)
    }).from(sql`information_schema.tables`)
      .where(sql`table_schema = 'public'`);
    
    console.log(`Verified ${tables.length} tables in database:`);
    tables.forEach(t => console.log(` - ${t.tableName}`));
    
    // Close connection
    await pool.end();
  } catch (error) {
    console.error('Migration error:', error);
    process.exit(1);
  }
}

// Run migrations if this file is executed directly
if (require.main === module) {
  runMigrations()
    .then(() => process.exit(0))
    .catch(err => {
      console.error('Unhandled error during migration:', err);
      process.exit(1);
    });
}
<br>```<br><br>**Production Build Configuration:**<br>```typescript<br>// Vite production build configuration
export default defineConfig({
  plugins: [
    react(),
    cartographer(),
    runtimeErrorModal(),
    shadcnThemeJson({ preset: 'vibrant', primaryColor: '#0ea5e9' })
  ],
  build: {
    outDir: 'dist/client',
    emptyOutDir: true,
    sourcemap: false, // Disable source maps in production
    minify: 'terser', // Use Terser for better minification
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'wouter'],
          ui: ['@/components/ui'],
          charts: ['chart.js', 'recharts'],
          maps: ['leaflet']
        }
      }
    },
    target: 'es2018', // Target older browsers for compatibility
    reportCompressedSize: false, // Speed up build
  },
  resolve: {
    alias: {
      '@': '/client/src',
      '@assets': '/client/src/assets',
      '@shared': '/shared'
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:3000',
        ws: true
      }
    }
  }
});
<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Markdown Documentation**: Created with comprehensive examples<br>• **System Architecture Diagrams**: Generated with draw.io<br>• **Drizzle Migrations**: Used for database schema version control<br>• **Environment Variables**: Applied for configuration management<br>• **Vite Build Optimization**: Implemented for production assets<br>• **Code Splitting**: Used for optimized bundle loading<br>• **Docker Containerization**: Prepared for deployment portability<br>• **Nginx Configuration**: Created for production web serving<br>• **Continuous Integration**: Set up with GitHub Actions |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Creating documentation that served both technical and non-technical users effectively.<br>• **Solution**: Implemented a tiered documentation approach with separate user guides, administrator guides, and developer documentation, each with appropriate detail levels.<br><br>• **Challenge**: Ensuring that the deployment process would be smooth across different environments.<br>• **Solution**: Created comprehensive deployment scripts with environment-specific configurations and thorough testing in staging environments.<br><br>• **Challenge**: Database migration needed to preserve existing data while updating the schema.<br>• **Solution**: Developed and tested incremental migration scripts that could handle schema changes without data loss.<br><br>• **Lesson Learned**: Documentation should be treated as a first-class deliverable, not an afterthought, as it significantly impacts usability and maintainability.<br><br>• **Lesson Learned**: Deployment considerations should be addressed throughout the development process, not just at the end, to avoid last-minute complications. |