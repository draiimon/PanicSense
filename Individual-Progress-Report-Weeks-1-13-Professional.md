Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 \- January 24, 2025 |

| Project Setup and Database Design |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Designed initial database schema for disaster monitoring platform using PostgreSQL and Drizzle ORM<br>• Implemented 8 core database tables: users, sessions, sentimentPosts, disasterEvents, analyzedFiles, sentimentFeedback, trainingExamples, and uploadSessions<br>• Set up project structure with TypeScript configuration and folder organization<br>• Created initial Express server with basic routing configuration<br>• Implemented authentication schema with secure password storage<br>• Configured development environment with necessary dependencies<br>• Established GitHub repository with proper branching strategy<br>• Setup database schema with validation rules using Zod<br><br>**Code Screenshot: Database Schema Implementation**<br>![Schema Implementation](https://i.imgur.com/FdH7Jw1.png)<br><br>**Database ER Diagram:**<br>![Database ER Diagram](https://i.imgur.com/K7gLd2R.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **PostgreSQL**: Implemented relational database with proper indexing<br>• **Drizzle ORM**: Used for type-safe database interactions<br>• **TypeScript**: Implemented for type safety across the codebase<br>• **Zod**: Utilized for schema validation and type generation<br>• **Git Flow**: Implemented for version control with feature branches<br>• **Entity-Relationship Modeling**: Used for database design<br>• **VS Code**: Primary development environment with ESLint and Prettier<br>• **Express.js**: Server framework for RESTful API development<br>• **JWT**: Implemented for secure authentication tokens |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Designing an efficient schema that could handle complex disaster monitoring data was difficult. Initial attempts had redundant fields and inefficient relationships.<br>• **Solution**: Restructured schema after research on similar systems and implemented proper normalization while maintaining query performance.<br>• **Challenge**: Setting up TypeScript with proper paths and configurations caused initial integration issues.<br>• **Solution**: Created comprehensive tsconfig.json with proper path aliases and strict type checking.<br>• **Lesson Learned**: Planning database structure thoroughly before implementation saves significant refactoring time later.<br>• **Lesson Learned**: Investing time in proper development environment setup pays dividends throughout the project lifecycle. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 2 / January 27 \- January 31, 2025 |

| API Development and Authentication System |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented core API endpoints for user management, data retrieval, and analysis<br>• Created authentication system with signup, login, and session management<br>• Built storage abstraction layer for database operations<br>• Implemented the following API endpoints:<br>  - POST /api/auth/signup<br>  - POST /api/auth/login<br>  - GET /api/auth/me<br>  - GET /api/sentiment-posts<br>  - GET /api/disaster-events<br>  - GET /api/analyzed-files<br>• Developed secure password hashing using bcrypt<br>• Created JWT token generation and validation<br>• Implemented proper error handling middleware<br><br>**Authentication System Implementation:**<br>![Authentication System](https://i.imgur.com/wP3dsEL.png)<br><br>**API Endpoint Implementation:**<br>```typescript<br>app.post('/api/auth/signup', async (req: Request, res: Response) => {<br>  try {<br>    const userData = insertUserSchema.parse(req.body);<br>    const newUser = await storage.createUser(userData);<br>    const token = await storage.createSession(newUser.id);<br>    res.status(201).json({ user: {...newUser, password: undefined }, token });<br>  } catch (error) {<br>    handleError(error, res);<br>  }<br>});<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Express.js**: Used for RESTful API development<br>• **bcryptjs**: Implemented for secure password hashing<br>• **JSON Web Tokens**: Used for stateless authentication<br>• **Zod**: Implemented for request validation<br>• **Repository Pattern**: Applied for data access abstraction<br>• **Postman**: Used for API testing and documentation<br>• **Error Handling Middleware**: Created for consistent error responses<br>• **Status Code Standards**: Followed HTTP standards for response codes<br>• **Insomnia**: Used for API endpoint testing |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial implementation of authentication had security vulnerabilities with token storage and validation.<br>• **Solution**: Researched OWASP guidelines and implemented proper token management with expirations and refresh mechanisms.<br>• **Challenge**: Error handling was inconsistent across endpoints, leading to unpredictable client responses.<br>• **Solution**: Created centralized error handling middleware with standardized error formats.<br>• **Lesson Learned**: Security considerations should be integrated from the beginning, not added later.<br>• **Lesson Learned**: Standardized API response formats save significant time in frontend development and debugging. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 \- February 7, 2025 |

| Python Integration and Sentiment Analysis Pipeline |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed Python integration service for sentiment analysis using Node.js child processes<br>• Created sentiment analysis pipeline for text processing<br>• Implemented extraction algorithms for disaster types and locations from text<br>• Built CSV processing functionality for batch analysis<br>• Implemented caching system for repeated sentiment analysis requests<br>• Created upload session tracking for progress monitoring<br>• Developed process management for cancellation and resource cleanup<br>• Added logging system for Python process output<br><br>**Python Service Integration:**<br>![Python Service](https://i.imgur.com/Jn4CWzO.png)<br><br>**Sentiment Analysis Results:**<br>![Sentiment Analysis](https://i.imgur.com/H3fUbK7.png)<br><br>**Process Flow Diagram:**<br>![Process Flow](https://i.imgur.com/pTZg9A5.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Node.js Child Processes**: Used for Python integration<br>• **Natural Language Processing**: Implemented for sentiment analysis<br>• **NLTK (Python)**: Utilized for text processing and analysis<br>• **Memory Management**: Implemented for handling large datasets<br>• **Temp File Management**: Used for secure data passing between processes<br>• **Process Signal Handling**: Implemented for graceful process termination<br>• **Stream Processing**: Used for efficient CSV handling<br>• **Caching Strategies**: Implemented for performance optimization<br>• **Error Recovery Mechanisms**: Built for process failures |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial Python integration caused memory leaks due to improper process management.<br>• **Solution**: Implemented proper process cleanup and monitoring, with resource limits and timeout handling.<br>• **Challenge**: Large CSV files caused system crashes during processing.<br>• **Solution**: Developed streaming approach with chunked processing and progress tracking.<br>• **Challenge**: Communication between Node and Python was unreliable with binary data.<br>• **Solution**: Implemented a structured protocol using temporary files for large data exchange.<br>• **Lesson Learned**: Cross-language integration requires careful consideration of resource management and error handling.<br>• **Lesson Learned**: Large file processing needs to be designed with memory constraints in mind from the beginning. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 4 / February 10 \- February 14, 2025 |

| Frontend Components and Dashboard Development |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed main application layout with navigation using shadcn components<br>• Created responsive dashboard with key statistics and visualizations<br>• Implemented authentication forms with validation<br>• Built file upload component with drag-and-drop functionality<br>• Created data display components for sentiment posts and disaster events<br>• Implemented TanStack Query for data fetching and state management<br>• Developed toast notification system for user feedback<br>• Created loading states and error handling for API interactions<br><br>**Dashboard Implementation:**<br>![Dashboard](https://i.imgur.com/5nJZH3K.png)<br><br>**File Upload Component:**<br>![File Upload](https://i.imgur.com/YfL4dMo.png)<br><br>**Form Implementation:**<br>```typescript<br>const form = useForm<z.infer<typeof formSchema>>({<br>  resolver: zodResolver(formSchema),<br>  defaultValues: {<br>    username: "",<br>    password: "",<br>  },<br>});<br><br>const onSubmit = async (values: z.infer<typeof formSchema>) => {<br>  try {<br>    const result = await loginUser(values);<br>    toast({ title: "Login successful" });<br>    // Handle successful login<br>  } catch (error) {<br>    toast({ title: "Login failed", variant: "destructive" });<br>  }<br>};<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **React**: Used for component development<br>• **TypeScript**: Implemented for type safety<br>• **Tailwind CSS**: Used for styling with utility classes<br>• **shadcn/ui**: Utilized for accessible component library<br>• **TanStack React Query**: Implemented for data fetching and caching<br>• **React Hook Form**: Used for form state management<br>• **Zod**: Utilized for form validation<br>• **Framer Motion**: Implemented for animations<br>• **Responsive Design**: Applied with mobile-first approach<br>• **React Context**: Used for global state management |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial dashboard design was cluttered and difficult to navigate on mobile devices.<br>• **Solution**: Redesigned with responsive breakpoints and collapsible sections for mobile viewing.<br>• **Challenge**: Form validation errors were not clearly communicated to users.<br>• **Solution**: Implemented inline validation with clear error messages and visual indicators.<br>• **Challenge**: Data fetching strategy caused multiple redundant API calls.<br>• **Solution**: Implemented proper caching and request deduplication with TanStack Query.<br>• **Lesson Learned**: Mobile-first design prevents significant rework for responsive interfaces.<br>• **Lesson Learned**: Investing in proper form validation and error handling greatly improves user experience. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 \- February 21, 2025 |

| Data Visualization and Geographic Analysis |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Integrated Leaflet for interactive mapping of disaster data<br>• Implemented clustering for efficient rendering of multiple data points<br>• Created heatmap visualization for sentiment concentration<br>• Built timeline charts for sentiment trend analysis<br>• Developed filtering system for geographic and temporal data<br>• Implemented color coding for different sentiment categories<br>• Created marker popups with detailed information<br>• Enhanced location extraction from text in backend<br>• Developed responsive design for visualizations<br><br>**Geographic Analysis Implementation:**<br>![Geographic Analysis](https://i.imgur.com/XeVd2q3.png)<br><br>**Timeline Visualization:**<br>![Timeline Visualization](https://i.imgur.com/R6mNfWC.png)<br><br>**Map Integration Code:**<br>```typescript<br>const MapComponent = ({ data }: { data: SentimentPost[] }) => {<br>  const mapRef = useRef<L.Map | null>(null);<br><br>  useEffect(() => {<br>    if (!mapRef.current) {<br>      mapRef.current = L.map('map').setView([12.8797, 121.7740], 6);<br>      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(mapRef.current);<br>    }<br>    <br>    // Add markers for data points<br>    const markers = L.markerClusterGroup();<br>    data.forEach(post => {<br>      if (post.location) {<br>        const coords = getCoordinates(post.location);<br>        if (coords) {<br>          const marker = L.marker(coords)<br>            .bindPopup(`<b>${post.sentiment}</b><br>${post.text}`);<br>          markers.addLayer(marker);<br>        }<br>      }<br>    });<br>    <br>    mapRef.current.addLayer(markers);<br>    <br>    return () => {<br>      if (mapRef.current) {<br>        mapRef.current.remove();<br>      }<br>    };<br>  }, [data]);<br><br>  return <div id="map" style={{ height: '500px', width: '100%' }} />;<br>};<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Leaflet**: Used for interactive mapping<br>• **Chart.js**: Implemented for timeline visualizations<br>• **GeoJSON**: Utilized for geographic data formatting<br>• **Marker Clustering**: Applied for efficient map rendering<br>• **Color Theory**: Used for sentiment visualization<br>• **Heat Maps**: Implemented for density visualization<br>• **Responsive Canvas Rendering**: Applied for different screen sizes<br>• **Time Series Analysis**: Used for trend visualization<br>• **Spatial Data Processing**: Implemented for geographic analysis |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial map implementation was extremely slow with large datasets (>1000 points).<br>• **Solution**: Implemented marker clustering and data aggregation for improved performance.<br>• **Challenge**: Many text entries had ambiguous location references that couldn't be accurately mapped.<br>• **Solution**: Enhanced location extraction algorithm with context-aware analysis and validation against known locations.<br>• **Challenge**: Visualizations looked distorted on different screen sizes.<br>• **Solution**: Implemented responsive canvas rendering with proper resizing handlers.<br>• **Lesson Learned**: Geographic visualization requires careful performance consideration with large datasets.<br>• **Lesson Learned**: Natural language processing for location extraction benefits greatly from domain-specific knowledge and validation. |

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

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented WebSocket integration for real-time updates<br>• Developed upload progress modal with live status tracking<br>• Created cross-tab synchronization using localStorage and storage events<br>• Built real-time sentiment analysis component for immediate text processing<br>• Implemented notification system for system events and updates<br>• Created reconnection logic for network disruptions<br>• Built progress tracking with percentage completion<br>• Developed broadcast mechanism for server-side events<br><br>**WebSocket Implementation:**<br>![WebSocket Implementation](https://i.imgur.com/nK2xMwB.png)<br><br>**Cross-tab Synchronization:**<br>```typescript<br>// Server-side broadcast function<br>function broadcastUpdate(data: any) {<br>  wss.clients.forEach((client) => {<br>    if (client.readyState === WebSocket.OPEN) {<br>      client.send(JSON.stringify(data));<br>    }<br>  });<br>}<br><br>// Client-side synchronization<br>useEffect(() => {<br>  const handleStorageChange = (e: StorageEvent) => {<br>    if (e.key === 'uploadProgress' && e.newValue) {<br>      const progress = JSON.parse(e.newValue);<br>      setLocalProgress(progress);<br>    }<br>  };<br><br>  window.addEventListener('storage', handleStorageChange);<br>  <br>  return () => {<br>    window.removeEventListener('storage', handleStorageChange);<br>  };<br>}, []);<br>```<br><br>**Upload Progress Modal:**<br>![Upload Progress](https://i.imgur.com/fXuYtTu.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **WebSockets**: Used for real-time communication<br>• **localStorage Events**: Implemented for cross-tab synchronization<br>• **React Context API**: Used for global state management<br>• **Event-driven Architecture**: Applied for real-time updates<br>• **Reconnection Strategies**: Implemented for network resilience<br>• **Progress Tracking Algorithms**: Used for accurate progress reporting<br>• **Optimistic UI Updates**: Applied for responsive user experience<br>• **Modal Management**: Implemented for persistent progress visibility<br>• **State Machine Pattern**: Used for upload status management |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: WebSocket connections were unstable and sometimes failed to reconnect properly.<br>• **Solution**: Implemented exponential backoff strategy for reconnection with proper error handling.<br>• **Challenge**: State synchronization across tabs had race conditions with simultaneous updates.<br>• **Solution**: Implemented a primary tab coordination system with proper locking mechanisms.<br>• **Challenge**: Upload progress reporting was inconsistent with large files.<br>• **Solution**: Redesigned progress calculation with chunked processing and server-side verification.<br>• **Lesson Learned**: Real-time features require robust error handling and reconnection logic.<br>• **Lesson Learned**: Cross-tab synchronization is complex and needs careful consideration of edge cases and race conditions. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 \- March 7, 2025 |

| AI Model Training and Feedback System |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed feedback mechanism for correcting AI predictions<br>• Implemented trainModelWithFeedback functionality for model improvement<br>• Created system to detect similar texts for consistent analysis<br>• Built confidence scoring to indicate prediction reliability<br>• Implemented visualization for confidence levels<br>• Developed training example management system<br>• Created API endpoints for feedback submission and training<br>• Built UI components for feedback submission<br>• Implemented tracking for model improvement metrics<br><br>**Feedback System Implementation:**<br>![Feedback System](https://i.imgur.com/9dVpWoB.png)<br><br>**Model Training Function:**<br>```typescript<br>public async trainModelWithFeedback(feedback: SentimentFeedback): Promise<boolean> {<br>  try {<br>    // Create temp file for feedback data<br>    const tempFilePath = path.join(this.tempDir, `feedback_${nanoid()}.json`);<br>    await fs.promises.mkdir(this.tempDir, { recursive: true });<br>    await fs.promises.writeFile(tempFilePath, JSON.stringify(feedback));<br>    <br>    // Spawn Python process for training<br>    const pythonProcess = spawn(this.pythonBinary, [<br>      this.scriptPath,<br>      '--train',<br>      '--feedback-file',<br>      tempFilePath<br>    ]);<br>    <br>    // Process output and log<br>    let result = false;<br>    pythonProcess.stdout.on('data', (data) => {<br>      const output = data.toString().trim();<br>      log(`Python training: ${output}`, 'python');<br>      pythonConsoleMessages.push({<br>        message: output,<br>        timestamp: new Date()<br>      });<br>      <br>      if (output.includes('Training successful')) {<br>        result = true;<br>      }<br>    });<br>    <br>    // Wait for process to complete<br>    await new Promise<void>((resolve) => {<br>      pythonProcess.on('close', () => {<br>        resolve();<br>      });<br>    });<br>    <br>    // Clean up temp file<br>    await fs.promises.unlink(tempFilePath);<br>    <br>    // Clear cache for this text<br>    this.clearCacheForText(feedback.originalText);<br>    <br>    return result;<br>  } catch (error) {<br>    log(`Error training model: ${error}`, 'python');<br>    return false;<br>  }<br>}<br>```<br><br>**Confidence Visualization:**<br>![Confidence Visualization](https://i.imgur.com/d7YcRu5.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Machine Learning Techniques**: Implemented for sentiment analysis<br>• **Transfer Learning**: Applied for model adaptation<br>• **Feedback Loop System**: Developed for model improvement<br>• **Text Similarity Algorithms**: Used for consistent analysis<br>• **Confidence Scoring**: Implemented for prediction reliability<br>• **Data Validation**: Applied for training data integrity<br>• **Model Versioning**: Used for tracking improvements<br>• **Incremental Learning**: Implemented for continuous improvement<br>• **Neural Network Fine-tuning**: Applied for domain adaptation |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial training approach required too many examples to show improvement.<br>• **Solution**: Implemented domain adaptation techniques to make better use of limited feedback data.<br>• **Challenge**: Text similarity detection had many false positives with standard algorithms.<br>• **Solution**: Developed domain-specific similarity metrics with contextual awareness.<br>• **Challenge**: Confidence scoring was not accurately reflecting actual prediction reliability.<br>• **Solution**: Calibrated confidence scores with historical accuracy data and implemented better uncertainty estimation.<br>• **Lesson Learned**: AI systems need careful design to learn effectively from limited user feedback.<br>• **Lesson Learned**: Domain-specific knowledge significantly improves NLP task performance compared to general techniques. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 \- March 14, 2025 |

| Evaluation and Data Export Features |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented evaluation metrics calculation (accuracy, precision, recall, F1)<br>• Created visualization components for model performance metrics<br>• Built CSV export functionality with proper formatting<br>• Developed filtering options for customized data exports<br>• Created RawData component for viewing complete dataset<br>• Implemented pagination and search for large datasets<br>• Added sorting functionality for data tables<br>• Built download mechanism for exported data<br>• Created metrics storage in database for tracking improvement<br><br>**Evaluation Metrics Dashboard:**<br>![Evaluation Dashboard](https://i.imgur.com/zMhIjkW.png)<br><br>**Raw Data Explorer:**<br>![Raw Data Explorer](https://i.imgur.com/UeK1DWJ.png)<br><br>**CSV Export Implementation:**<br>```typescript<br>app.get('/api/export-csv', async (req: Request, res: Response) => {<br>  try {<br>    // Parse query parameters for filtering<br>    const { startDate, endDate, sentiment, source } = req.query;<br>    <br>    // Get filtered data<br>    const posts = await storage.getSentimentPostsFiltered({<br>      startDate: startDate ? new Date(startDate as string) : undefined,<br>      endDate: endDate ? new Date(endDate as string) : undefined,<br>      sentiment: sentiment as string,<br>      source: source as string<br>    });<br>    <br>    // Generate CSV headers<br>    const csvHeader = 'text,timestamp,source,language,sentiment,confidence,location,disasterType';<br>    <br>    // Generate CSV content with streaming to handle large datasets<br>    res.setHeader('Content-Type', 'text/csv');<br>    res.setHeader('Content-Disposition', 'attachment; filename="sentiment-data.csv"');<br>    <br>    // Write header<br>    res.write(csvHeader + '\\n');<br>    <br>    // Stream rows to avoid memory issues with large datasets<br>    for (const post of posts) {<br>      const row = [<br>        post.text.replace(/,/g, ' '),<br>        post.timestamp.toISOString(),<br>        post.source || '',<br>        post.language || '',<br>        post.sentiment,<br>        post.confidence.toString(),<br>        post.location || '',<br>        post.disasterType || ''<br>      ].join(',');<br>      <br>      res.write(row + '\\n');<br>    }<br>    <br>    res.end();<br>  } catch (error) {<br>    log(`Error exporting CSV: ${error}`);<br>    res.status(500).json({ error: 'Failed to export data' });<br>  }<br>});<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Statistical Analysis**: Implemented for evaluation metrics<br>• **Streaming CSV Generation**: Used for efficient data export<br>• **Data Pagination**: Applied for large dataset handling<br>• **Chart.js**: Utilized for metrics visualization<br>• **Data Filtering Algorithms**: Implemented for customized exports<br>• **Search Optimization**: Applied for efficient data retrieval<br>• **Table Sorting Algorithms**: Used for data exploration<br>• **Content Disposition Headers**: Implemented for file downloads<br>• **MIME Type Handling**: Used for proper file downloading |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial export implementation loaded all data into memory, causing out-of-memory errors with large datasets.<br>• **Solution**: Implemented streaming export with chunked processing to handle arbitrarily large datasets.<br>• **Challenge**: Evaluation metrics were not reflecting real-world performance accurately.<br>• **Solution**: Enhanced metrics calculation with stratified sampling and confidence weighting.<br>• **Challenge**: Raw data explorer was extremely slow with large datasets.<br>• **Solution**: Implemented virtual scrolling and optimized rendering with pagination.<br>• **Lesson Learned**: Data export features need to be designed with scalability in mind from the beginning.<br>• **Lesson Learned**: Evaluation metrics need careful design to accurately reflect system performance in the intended use cases. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 9 / March 17 \- March 21, 2025 |

| Performance Optimization and Error Handling |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Implemented database query optimization for faster retrieval<br>• Created client-side caching for reduced server load<br>• Developed comprehensive error handling system<br>• Improved memory management for large dataset processing<br>• Implemented batch processing for resource-intensive operations<br>• Created usage tracking system to monitor resource utilization<br>• Added database indexing for common queries<br>• Implemented connection pooling for database efficiency<br>• Created performance monitoring dashboard<br><br>**Performance Monitoring:**<br>![Performance Monitoring](https://i.imgur.com/2wbLhRD.png)<br><br>**Optimized Query Implementation:**<br>```typescript<br>// Before optimization - N+1 query problem<br>async function getPostsWithUsers() {<br>  const posts = await db.select().from(sentimentPosts);<br>  <br>  // This causes N additional queries, one for each post<br>  for (let i = 0; i < posts.length; i++) {<br>    if (posts[i].processedBy) {<br>      posts[i].user = await db.select()<br>        .from(users)<br>        .where(eq(users.id, posts[i].processedBy))<br>        .limit(1)<br>        .then(rows => rows[0]);<br>    }<br>  }<br>  <br>  return posts;<br>}<br><br>// After optimization - Single join query<br>async function getPostsWithUsers() {<br>  return db.select({<br>    post: sentimentPosts,<br>    user: {<br>      id: users.id,<br>      username: users.username,<br>      fullName: users.fullName<br>    }<br>  })<br>  .from(sentimentPosts)<br>  .leftJoin(users, eq(sentimentPosts.processedBy, users.id));<br>}<br>```<br><br>**Error Handling System:**<br>![Error Handling](https://i.imgur.com/qk5iQG5.png) |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Database Indexing**: Applied for query optimization<br>• **Query Execution Planning**: Used for SQL optimization<br>• **Connection Pooling**: Implemented for efficient database connections<br>• **React Query Caching**: Applied for client-side data caching<br>• **Memory Profiling**: Used for identifying memory leaks<br>• **Batch Processing**: Implemented for large data operations<br>• **Error Boundary Pattern**: Applied for component-level error handling<br>• **Centralized Error Logging**: Used for error tracking<br>• **Performance Metrics Collection**: Implemented for monitoring |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial database queries were inefficient with N+1 query patterns.<br>• **Solution**: Refactored to use proper joins and eager loading patterns for related data.<br>• **Challenge**: Memory usage grew unbounded with certain operations, eventually causing crashes.<br>• **Solution**: Implemented proper memory management with streaming operations and resource limits.<br>• **Challenge**: Error handling was inconsistent across the application.<br>• **Solution**: Created centralized error handling with proper logging and user feedback.<br>• **Lesson Learned**: Performance optimization should focus on measurable bottlenecks rather than premature optimization.<br>• **Lesson Learned**: A well-designed error handling system significantly improves both user experience and debugging capabilities. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 10 / March 24 \- March 28, 2025 |

| Security Enhancements and User Profile Management |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Conducted security audit and implemented fixes<br>• Enhanced authentication with improved token management<br>• Implemented comprehensive input validation across all endpoints<br>• Created security measures against common web vulnerabilities (XSS, CSRF, SQL injection)<br>• Developed user profile management with customization options<br>• Implemented profile image upload functionality<br>• Added role-based access control for protected operations<br>• Created audit logging for sensitive actions<br>• Implemented password reset functionality<br><br>**Security Implementation:**<br>![Security Implementation](https://i.imgur.com/fQ7Vw3Y.png)<br><br>**User Profile Management:**<br>![User Profile](https://i.imgur.com/8wTkRpL.png)<br><br>**Security Measures Implementation:**<br>```typescript<br>// XSS Prevention middleware<br>app.use((req, res, next) => {<br>  res.setHeader('X-XSS-Protection', '1; mode=block');<br>  res.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' data: https:; script-src 'self'; style-src 'self' 'unsafe-inline';");<br>  res.setHeader('X-Frame-Options', 'SAMEORIGIN');<br>  res.setHeader('X-Content-Type-Options', 'nosniff');<br>  next();<br>});<br><br>// CSRF Protection<br>const csrfProtection = (req: Request, res: Response, next: NextFunction) => {<br>  // Skip for GET requests<br>  if (req.method === 'GET') {<br>    return next();<br>  }<br>  <br>  const csrfToken = req.headers['x-csrf-token'];<br>  const sessionToken = req.headers['authorization']?.split(' ')[1];<br>  <br>  if (!csrfToken || !sessionToken) {<br>    return res.status(403).json({ error: 'CSRF token missing' });<br>  }<br>  <br>  // Verify CSRF token matches a hash of the session token<br>  const expectedToken = crypto<br>    .createHash('sha256')<br>    .update(sessionToken + process.env.CSRF_SECRET)<br>    .digest('hex');<br>    <br>  if (csrfToken !== expectedToken) {<br>    return res.status(403).json({ error: 'Invalid CSRF token' });<br>  }<br>  <br>  next();<br>};<br><br>// Apply CSRF protection to all state-changing endpoints<br>app.post('*', csrfProtection);<br>app.put('*', csrfProtection);<br>app.patch('*', csrfProtection);<br>app.delete('*', csrfProtection);<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **OWASP Security Guidelines**: Followed for web security best practices<br>• **Content Security Policy (CSP)**: Implemented to prevent XSS attacks<br>• **CSRF Protection**: Applied with token validation<br>• **Parameterized Queries**: Used to prevent SQL injection<br>• **Input Sanitization**: Implemented for user inputs<br>• **Role-based Access Control (RBAC)**: Developed for permission management<br>• **Secure File Upload**: Implemented with proper validation<br>• **Audit Logging**: Created for security monitoring<br>• **Password Policy Enforcement**: Applied for account security |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Initial security audit revealed several vulnerabilities in the application.<br>• **Solution**: Systematically addressed each vulnerability following OWASP guidelines and best practices.<br>• **Challenge**: Implementing security measures without negatively impacting user experience.<br>• **Solution**: Designed security measures to operate transparently when possible and with clear guidance when user action was required.<br>• **Challenge**: File upload functionality introduced potential security risks.<br>• **Solution**: Implemented strict validation, secure storage, and proper access controls for uploaded files.<br>• **Lesson Learned**: Security is a continuous process that requires regular auditing and updates.<br>• **Lesson Learned**: The most effective security measures balance protection with usability. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 11 / April 1 \- April 4, 2025 |

| API Integration and External System Connectivity |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Developed RESTful API endpoints for external integration<br>• Implemented API documentation using OpenAPI<br>• Created authentication mechanisms for API access<br>• Built webhook functionality for event notifications<br>• Implemented CORS configuration for secure cross-origin requests<br>• Created example client code for API integration<br>• Added rate limiting to prevent API abuse<br>• Developed API versioning for backward compatibility<br>• Created comprehensive error responses for API consumers<br><br>**API Documentation:**<br>![API Documentation](https://i.imgur.com/K3dVb4T.png)<br><br>**Webhook Implementation:**<br>![Webhook Implementation](https://i.imgur.com/wP3XdYL.png)<br><br>**API Rate Limiting Implementation:**<br>```typescript<br>// Rate limiting middleware<br>const apiRateLimiter = rateLimit({<br>  windowMs: 15 * 60 * 1000, // 15 minutes<br>  max: 100, // limit each IP to 100 requests per windowMs<br>  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers<br>  legacyHeaders: false, // Disable the `X-RateLimit-*` headers<br>  message: {<br>    error: 'Too many requests, please try again later.',<br>    retryAfter: 15 * 60, // seconds until retry is available<br>  },<br>  skip: (req) => {<br>    // Skip rate limiting for internal requests<br>    return req.ip === '127.0.0.1' || req.ip === '::1';<br>  },<br>});<br><br>// Apply rate limiting to API routes<br>app.use('/api/v1', apiRateLimiter);<br>app.use('/api/v2', apiRateLimiter);<br><br>// API versioning<br>const v1Router = express.Router();<br>const v2Router = express.Router();<br><br>// Register API version routers<br>app.use('/api/v1', v1Router);<br>app.use('/api/v2', v2Router);<br><br>// V1 API endpoints<br>v1Router.get('/sentiment-posts', async (req, res) => { /* implementation */ });<br><br>// V2 API endpoints with enhanced features<br>v2Router.get('/sentiment-posts', async (req, res) => { /* implementation */ });<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **RESTful API Design**: Applied for consistent interfaces<br>• **OpenAPI/Swagger**: Used for API documentation<br>• **JWT Authentication**: Implemented for API security<br>• **API Key Management**: Developed for external access<br>• **Rate Limiting**: Applied to prevent abuse<br>• **CORS Configuration**: Implemented for cross-origin security<br>• **Webhook Pattern**: Used for event notifications<br>• **API Versioning**: Applied for compatibility<br>• **Request Validation**: Implemented for data integrity |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Designing API endpoints that balanced flexibility with usability.<br>• **Solution**: Created clear endpoint patterns with consistent parameter naming and documented thoroughly.<br>• **Challenge**: Implementing effective rate limiting without blocking legitimate high-volume users.<br>• **Solution**: Developed tiered rate limiting with authentication-based limits rather than just IP-based limits.<br>• **Challenge**: Ensuring backward compatibility with API changes.<br>• **Solution**: Implemented proper API versioning with clear deprecation policies.<br>• **Lesson Learned**: API design is a critical aspect that significantly impacts integration ease and maintenance.<br>• **Lesson Learned**: Well-documented APIs with clear examples greatly reduce support requirements for external integrations. |

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

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Conducted end-to-end testing of all major application flows<br>• Identified and resolved bugs across the application<br>• Performed stress testing for stability under load<br>• Tested edge cases for critical functionalities<br>• Improved error recovery for unexpected situations<br>• Implemented comprehensive logging for debugging<br>• Conducted cross-browser testing for compatibility<br>• Fixed UI inconsistencies across different devices<br>• Added automated tests for critical components<br><br>**Testing Dashboard:**<br>![Testing Dashboard](https://i.imgur.com/QvWrDv9.png)<br><br>**Bug Tracking System:**<br>![Bug Tracking](https://i.imgur.com/JfkQa2L.png)<br><br>**Test Implementation:**<br>```typescript<br>// Example test for sentiment analysis API<br>describe('Sentiment Analysis API', () => {<br>  it('should correctly analyze positive sentiment', async () => {<br>    const response = await request(app)<br>      .post('/api/analyze-text')<br>      .send({ text: 'I am feeling happy about the community response after the earthquake. The volunteers are doing amazing work.' })<br>      .set('Accept', 'application/json');<br>      <br>    expect(response.status).toBe(200);<br>    expect(response.body).toHaveProperty('sentiment', 'positive');<br>    expect(response.body).toHaveProperty('confidence');<br>    expect(response.body.confidence).toBeGreaterThan(0.7);<br>    expect(response.body).toHaveProperty('disasterType', 'earthquake');<br>  });<br>  <br>  it('should correctly analyze negative sentiment', async () => {<br>    const response = await request(app)<br>      .post('/api/analyze-text')<br>      .send({ text: 'The flood has destroyed everything. We are suffering without clean water and food.' })<br>      .set('Accept', 'application/json');<br>      <br>    expect(response.status).toBe(200);<br>    expect(response.body).toHaveProperty('sentiment', 'negative');<br>    expect(response.body).toHaveProperty('confidence');<br>    expect(response.body.confidence).toBeGreaterThan(0.7);<br>    expect(response.body).toHaveProperty('disasterType', 'flood');<br>  });<br>  <br>  it('should handle empty text input', async () => {<br>    const response = await request(app)<br>      .post('/api/analyze-text')<br>      .send({ text: '' })<br>      .set('Accept', 'application/json');<br>      <br>    expect(response.status).toBe(400);<br>    expect(response.body).toHaveProperty('error');<br>  });<br>});<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Jest**: Used for automated testing<br>• **Supertest**: Implemented for API testing<br>• **React Testing Library**: Applied for component testing<br>• **End-to-End Testing**: Used for workflow validation<br>• **Load Testing**: Implemented with artillery.io<br>• **Cross-browser Testing**: Conducted with BrowserStack<br>• **Logging and Monitoring**: Enhanced with Winston<br>• **Error Tracking**: Implemented with detailed stack traces<br>• **Performance Profiling**: Used to identify bottlenecks |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Some bugs only appeared under specific conditions or with certain data patterns.<br>• **Solution**: Created comprehensive test suite with edge cases and data variations.<br>• **Challenge**: Load testing revealed performance degradation under concurrent usage.<br>• **Solution**: Implemented connection pooling, query caching, and optimized critical paths.<br>• **Challenge**: Cross-browser testing showed inconsistent rendering in older browsers.<br>• **Solution**: Added polyfills and fallback behaviors for improved compatibility.<br>• **Lesson Learned**: Thorough testing across different environments is essential for robust applications.<br>• **Lesson Learned**: Many bugs stem from incorrect assumptions about component interactions, highlighting the need for integration testing. |

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

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | • Created comprehensive user documentation with guides<br>• Developed technical documentation covering system architecture<br>• Prepared deployment configurations for production<br>• Implemented final UI polish based on user feedback<br>• Created database migration scripts for deployment<br>• Conducted final security review and adjustments<br>• Optimized asset loading for production performance<br>• Implemented proper error pages and fallback states<br>• Prepared maintenance documentation for future development<br><br>**User Documentation:**<br>![User Documentation](https://i.imgur.com/pNIJjKG.png)<br><br>**System Architecture Documentation:**<br>![System Architecture](https://i.imgur.com/bE8DVLK.png)<br><br>**Deployment Configuration:**<br>```typescript<br>// Production configuration for database connection<br>const getDbConfig = () => {<br>  const connectionString = process.env.DATABASE_URL;<br>  <br>  if (!connectionString) {<br>    throw new Error('DATABASE_URL environment variable is required');<br>  }<br>  <br>  return {<br>    connectionString,<br>    ssl: process.env.NODE_ENV === 'production' ? {<br>      rejectUnauthorized: false<br>    } : false,<br>    max: parseInt(process.env.DB_POOL_SIZE || '10'),<br>    idleTimeoutMillis: 30000,<br>  };<br>};<br><br>// Optimized asset serving for production<br>if (process.env.NODE_ENV === 'production') {<br>  app.use(compression());<br>  app.use(express.static('dist/client', {<br>    maxAge: '1y',<br>    etag: true,<br>    lastModified: true,<br>  }));<br>  <br>  // Ensure all routes not handled by API are sent to client app<br>  app.get('*', (req, res, next) => {<br>    if (req.path.startsWith('/api')) {<br>      return next();<br>    }<br>    res.sendFile(path.join(__dirname, '../client/index.html'));<br>  });<br>}<br>``` |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | • **Markdown Documentation**: Used for comprehensive guides<br>• **System Architecture Diagrams**: Created with draw.io<br>• **Environment-based Configuration**: Implemented for deployment<br>• **Asset Compression**: Applied for optimized loading<br>• **Content Delivery Networks (CDN)**: Used for static assets<br>• **Database Migration Tools**: Created for version control<br>• **Security Headers**: Implemented for production<br>• **Load Balancing Configuration**: Prepared for scaling<br>• **Logging and Monitoring Setup**: Configured for production |
| **Reflection: Problems Encountered and Lessons Learned** | • **Challenge**: Creating documentation that served both technical and non-technical users.<br>• **Solution**: Developed tiered documentation with user guides separate from technical documentation, with appropriate detail levels for each audience.<br>• **Challenge**: Preparing for deployment to environments that might differ from development.<br>• **Solution**: Implemented environment-agnostic configuration with fallbacks and proper error reporting.<br>• **Challenge**: Ensuring proper database migration without data loss.<br>• **Solution**: Created and thoroughly tested migration scripts with backup strategies.<br>• **Lesson Learned**: Documentation is a crucial deliverable that requires planning and effort comparable to code development.<br>• **Lesson Learned**: Deployment planning should start early in the development process to avoid last-minute complications. |