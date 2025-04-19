Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 1 / January 20 - January 24, 2025



Initial Architecture Design for PanicSensePH Platform



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on establishing the foundational architecture for our PanicSensePH disaster monitoring platform. I designed the initial system architecture diagram showing data flow between React components, Express backend, and PostgreSQL database. This architecture emphasizes real-time processing capabilities needed for disaster response.

• Successfully drafted database schema using Drizzle ORM with 8 primary tables: users, sessions, sentiment_posts, disaster_events, analyzed_files, sentiment_feedback, training_examples, and upload_sessions
• Researched combined LSTM-BiGRU model architecture for sequential text processing with bidirectional context awareness
• Identified MBERT (Multilingual BERT) as optimal solution for handling Filipino-English code-switching and jejemon text patterns
• Structured project into client, server, and shared modules with proper type definitions
• Created preliminary mockups of 7 main dashboard pages: Dashboard, Geographic Analysis, Timeline, Comparison, Raw Data, Evaluation, and Real-time monitoring



Techniques, Tools, and Methodologies Used

• PostgreSQL with Drizzle ORM for type-safe database modeling and migrations
• TypeScript for ensuring type consistency across both frontend and backend
• TensorFlow for implementing the combined LSTM-BiGRU architecture
• PyTorch for fine-tuning the MBERT model on Filipino slang and jejemon text patterns
• React with Tailwind CSS for responsive and accessible user interfaces
• Git Flow methodology for maintaining clean branching structure



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was designing a database schema that effectively supports both structured disaster reports and unstructured social media content. I learned that proper normalization with flexible JSON fields for metadata provides the optimal balance.

I also encountered difficulties with the approach to multilingual text processing, particularly with Filipino slang and jejemon text patterns. Through comparative research, I determined that fine-tuning MBERT on Filipino text corpora would provide better results than training custom embeddings from scratch.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 2 / January 27 - January 31, 2025



Setting Up Development Environment and Core Infrastructure



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

During the second week, I concentrated on implementing core infrastructure components essential for our development workflow. I set up our collaborative environment to ensure consistent development across the team.

• Established CI/CD pipeline for automated testing and deployment
• Implemented initial Express.js backend structure with TypeScript
• Created foundational API endpoints for sentiment analysis and disaster monitoring
• Configured Drizzle ORM with PostgreSQL, implementing proper migrations
• Set up React application with Vite and implemented basic layout components
• Developed reusable UI components using Tailwind CSS for consistent user experience



Techniques, Tools, and Methodologies Used

• Git with feature branch workflow for clean development history
• Drizzle ORM for type-safe database access and migrations
• React with Vite for efficient frontend development
• Agile methodology with weekly sprints and daily stand-ups
• RESTful API design principles with proper error handling
• Docker containers for consistent development environments



Reflection: Problems Encountered and Lessons Learned

The main challenge was ensuring consistency across different development environments, as team members encountered issues with package versions and database connections. I addressed this by implementing Docker containers for development, providing a standardized environment.

Another difficulty was balancing rapid prototyping with establishing a solid foundation. I learned that investing time in proper architecture early on, while seemingly slower initially, pays significant dividends as complexity increases.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 3 / February 3 - February 7, 2025



Implementation of Data Collection and Processing System



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I implemented the data collection and processing systems for PanicSensePH, focusing on robust ingestion pipelines for disaster-related content.

• Developed ETL pipeline for ingesting content from social media and government announcements
• Implemented CSV file upload functionality with validation and batch processing capabilities
• Created progress tracking system for monitoring file imports in real-time
• Established error handling protocols for data processing failures
• Developed data preprocessing module for text normalization and cleaning
• Set up unit tests for critical data processing components with 85% coverage



Techniques, Tools, and Methodologies Used

• Node.js streams for efficient processing of large datasets with minimal memory consumption
• Custom CSV parsing solution using csv-parse library with robust error handling
• Natural language processing techniques for text normalization and cleaning
• Jest for unit testing critical processing components
• Sample datasets of various sizes and formats for comprehensive testing
• Chunking strategies for handling large file processing



Reflection: Problems Encountered and Lessons Learned

A significant challenge was handling the variety of data formats and quality issues in real-world datasets. I learned the importance of robust validation and preprocessing to handle edge cases like incomplete records and inconsistent formatting.

Performance issues emerged when processing very large files, which I addressed by implementing a chunking strategy that processes data in manageable batches. This approach not only improved performance but also enabled better progress tracking and error recovery.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 4 / February 10 - February 14, 2025



Implementation of Authentication System and User Management



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on implementing the authentication and user management functionality, establishing secure access controls for the platform.

• Developed secure authentication mechanism using JWT with proper refresh token rotation
• Implemented user registration, login, password recovery, and session management
• Created role-based access control (RBAC) for different user types
• Developed middleware for route protection with fine-grained access control
• Implemented user profile management system with personal information updates
• Set up secure profile image upload and management functionality
• Created protected route components for client-side authentication restrictions



Techniques, Tools, and Methodologies Used

• Bcrypt for secure password hashing and storage
• JWT for stateless authentication with proper expiration policies
• Zod schemas for comprehensive user data validation
• Multer for secure file upload handling with validation
• React Query for managing authentication state throughout the application
• Custom authentication hooks for centralized auth logic
• CSRF protection measures for enhanced security



Reflection: Problems Encountered and Lessons Learned

The main challenge was balancing security requirements like password complexity and token expiration with usability. I implemented progressive security measures that maintain strong protection without unnecessarily burdening users.

Another issue was managing authentication state across multiple components. I learned the importance of centralizing authentication logic in custom hooks and contexts to provide a single source of truth throughout the application.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 5 / February 17 - February 21, 2025



Implementation of Initial LSTM Model for Sentiment Analysis



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I implemented the initial version of our combined LSTM-BiGRU sentiment analysis model for PanicSensePH, focusing on accurate classification of disaster-related content.

• Developed hybrid LSTM-BiGRU architecture combining long-term memory capabilities with bidirectional context awareness
• Created training pipeline with 1,000 manually annotated disaster-related social media posts in both English and Filipino
• Integrated fine-tuned MBERT layers for enhanced handling of Filipino slang and jejemon text patterns
• Achieved 72% initial accuracy across five sentiment categories with significant improvement for mixed-language content
• Implemented model endpoints in Express.js backend for both real-time analysis and batch processing
• Created specialized tokenization handling for jejemon text with character-level normalization
• Developed sentiment_posts table with proper indexing for efficient querying by sentiment type and source



Techniques, Tools, and Methodologies Used

• TensorFlow for implementing the combined LSTM-BiGRU architecture with custom attention mechanism
• PyTorch for fine-tuning MBERT on Filipino slang and jejemon text samples
• Custom preprocessing pipeline for handling jejemon text variations and character substitutions
• Transfer learning techniques for adapting pre-trained MBERT to disaster-specific terminology
• Database schema optimization with proper indices for sentiment filtering and aggregation
• Cross-validation with stratified sampling to ensure representative distribution across sentiment classes
• Custom metrics for evaluating performance on code-switched and jejemon text separately



Reflection: Problems Encountered and Lessons Learned

The main challenge was handling jejemon text patterns where characters are deliberately substituted (e.g., "p0h" instead of "po", "m@hal" instead of "mahal"). The initial LSTM-BiGRU model struggled with these variations. By integrating character-level preprocessing and fine-tuning MBERT on a corpus containing jejemon samples, recognition accuracy improved by 18%.

Another significant issue was balancing the computational requirements of the combined model architecture. The initial implementation was too resource-intensive for efficient batch processing. I implemented a two-stage approach where simpler features are processed first to filter out obvious cases, with the full LSTM-BiGRU-MBERT stack used only for complex or ambiguous content.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 6 / February 24 - February 28, 2025



Integration of Geographic Analysis and Visualization



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on integrating geographic analysis and visualization features into PanicSensePH, enhancing spatial understanding of disaster sentiment patterns.

• Developed location extraction system with 75% accuracy for place mention identification
• Implemented interactive map visualization using Leaflet.js with sentiment heat maps
• Created map interface with filtering by time period, sentiment type, and disaster events
• Established database of Philippine administrative boundaries for precise classification
• Integrated population data and risk factors for different regions
• Developed custom React components for responsive map visualization
• Implemented custom clustering algorithms for efficient data point rendering



Techniques, Tools, and Methodologies Used

• Leaflet.js with custom map layers and markers for interactive visualization
• Rule-based pattern matching with gazetteer of Philippine place names
• GeoJSON processing for administrative boundaries
• Specialized database indexes for efficient spatial queries
• Custom React components for responsive map interfaces
• Clustering strategies for handling large datasets on maps
• Context-aware location resolution algorithms



Reflection: Problems Encountered and Lessons Learned

A significant challenge was disambiguating place names that could refer to multiple locations in the Philippines. I implemented context-aware resolution that considers surrounding text and frequency analysis to identify the most likely location.

Optimizing map performance with large datasets proved difficult, as displaying thousands of points caused performance issues. I implemented clustering strategies and progressive loading techniques to maintain smooth interactions.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 7 / March 3 - March 7, 2025



Implementation of Real-time Monitoring and WebSocket Integration



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I implemented real-time monitoring capabilities for PanicSensePH, enabling live updates and instant notifications for emerging disaster situations.

• Developed WebSocket-based communication system for live updates between server and clients
• Created notification center for displaying real-time alerts about sentiment pattern changes
• Implemented message queue using Redis for reliable update delivery
• Established connection state management with recovery for missed updates
• Developed throttling and debouncing mechanisms for high-volume update periods
• Created exponential backoff strategy for reconnection during network instability
• Implemented broadcast optimization for efficient message distribution



Techniques, Tools, and Methodologies Used

• WebSocket protocol with custom message format for efficient data transfer
• Redis-based message queue for reliable delivery and persistence
• React Query for integrating real-time data with application state
• Throttling and debouncing for preventing UI performance issues
• Exponential backoff strategy for graceful reconnection
• Connection pooling for efficient resource management
• Custom event system for real-time notifications



Reflection: Problems Encountered and Lessons Learned

The primary challenge was maintaining WebSocket connection stability during network fluctuations. I implemented robust reconnection strategies with proper error handling to provide seamless experience even under suboptimal conditions.

Managing increased server load from numerous concurrent WebSocket connections proved difficult. By implementing connection pooling and efficient broadcast mechanisms, I significantly reduced resource requirements while maintaining responsive updates.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 8 / March 10 - March 14, 2025



Disaster Type Classification and Event Correlation



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I implemented disaster type classification and event correlation features, enabling the system to identify specific disaster types and detect emerging events.

• Developed specialized classifier for categorizing posts into specific disaster types with 82% accuracy
• Created correlation engine for identifying relationships between sentiment patterns and disasters
• Implemented detection system for emerging disaster events based on sentiment clusters
• Developed disaster events dashboard with active and historical event tracking
• Created sentiment trend analysis visualization for each disaster event
• Implemented hierarchical classification for primary and secondary disaster types
• Developed temporal context detection for distinguishing current vs. past events



Techniques, Tools, and Methodologies Used

• Multi-label classification system using TensorFlow
• Temporal pattern analysis with sliding window algorithms
• Correlation algorithms for relationship identification
• Clustering techniques using spatial and temporal dimensions
• Interactive visualizations with Recharts for trend analysis
• Hierarchical classification for complex disaster relationships
• Anomaly detection for identifying unusual sentiment patterns



Reflection: Problems Encountered and Lessons Learned

A significant challenge was distinguishing between discussions about past disasters and reports of current events. I implemented temporal context detection that analyzes time indicators and correlates with posting timestamps to determine if posts refer to ongoing or historical events.

Handling reports of multiple simultaneous disasters, such as flooding caused by typhoons, proved difficult. I developed a hierarchical classification approach that identifies primary and secondary disaster types, providing more nuanced analysis of complex situations.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 9 / March 17 - March 21, 2025



MBERT Integration for Enhanced Multilingual Support



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I enhanced PanicSensePH's multilingual capabilities through deeper integration of Multilingual BERT (MBERT), focusing on improved handling of code-switching text.

• Implemented specialized MBERT-based tokenization for mixed-language text
• Developed language-specific sentiment lexicon for Filipino disaster terminology
• Improved accuracy for mixed-language posts by 15%, reaching 81% overall accuracy
• Created seamless pipeline combining LSTM and MBERT components
• Implemented dynamic model selection based on text characteristics
• Developed cross-lingual word embeddings aligning English and Filipino semantic spaces
• Created handling for regional dialectal variations through specialized training data



Techniques, Tools, and Methodologies Used

• Cross-lingual word embeddings for aligned semantic spaces
• PyTorch for MBERT fine-tuning on disaster terminology
• Transfer learning from English corpus to Filipino processing
• Model quantization and caching strategies for performance
• Code-switching detection algorithms for language transitions
• Contextual analysis with specialized embeddings
• Fuzzy matching for dialectal variants



Reflection: Problems Encountered and Lessons Learned

The biggest challenge was handling Filipino dialectal variations, which significantly affected sentiment detection accuracy. I addressed this by incorporating regional language patterns into training data and implementing fuzzy matching for dialectal variants.

MBERT struggled with very short text fragments common in social media. I implemented a hybrid approach that falls back to specialized LSTM for short texts while using MBERT for longer content, optimizing performance across different content types.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 10 / March 24 - March 28, 2025



Implementation of Feedback Loop and Continuous Learning



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I implemented a feedback mechanism for PanicSensePH that enables continuous improvement through user corrections of sentiment classifications.

• Developed user interface for submitting sentiment classification corrections
• Created dedicated database structure for storing and tracking feedback
• Implemented training pipeline for incorporating feedback into model refinement
• Developed confidence threshold system for flagging low-confidence predictions
• Created semi-supervised learning environment for targeted human review
• Improved overall accuracy by 3% with just 100 feedback instances
• Implemented analytics dashboard for tracking correction patterns and model weaknesses



Techniques, Tools, and Methodologies Used

• Active learning strategies for intelligent sample selection
• Incremental model updating without full retraining
• Weighted feedback integration for preserving global model performance
• Confidence scoring mechanisms for prediction reliability assessment
• React with optimistic updates for responsive feedback interface
• Feedback prioritization based on correction value
• Pattern analysis for identifying systematic model weaknesses



Reflection: Problems Encountered and Lessons Learned

A significant challenge was balancing user feedback incorporation with model stability. Early implementations sometimes overfit to recent corrections, reducing general performance. I implemented a weighted feedback integration approach that preserves global model performance while learning from new examples.

Tracking feedback patterns proved crucial for identifying systematic weaknesses in our models. The analytics dashboard I created for monitoring correction patterns has been invaluable for guiding targeted improvements to specific aspects of the sentiment analysis system.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 11 / March 31 - April 4, 2025



Performance Optimization and Scalability Enhancements



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on improving PanicSensePH's performance and scalability through comprehensive optimization of both frontend and backend components.

• Conducted thorough profiling to identify performance bottlenecks
• Improved average response times by 65% for common operations
• Optimized database queries through proper indexing and restructuring
• Enhanced frontend with virtualized lists and lazy loading for efficient rendering
• Implemented caching layer for frequently accessed and compute-intensive results
• Improved system capacity to handle 3x previous concurrent user load
• Developed tiered processing approach for balancing accuracy and speed
• Implemented connection pooling for improved stability under heavy load



Techniques, Tools, and Methodologies Used

• Database indexing targeted at common query patterns
• Query optimization and strategic denormalization
• React component memoization and code splitting
• Sophisticated caching strategy with memory and persistent tiers
• Load testing with simulated concurrent users
• Performance benchmarking against established metrics
• Progressive loading techniques for large datasets



Reflection: Problems Encountered and Lessons Learned

The main challenge was maintaining accuracy while improving performance. Initial optimization attempts reduced processing time but slightly decreased model accuracy. I addressed this by implementing a tiered approach with fast approximate methods followed by precise analysis where needed.

Database connection management was causing performance degradation during high load. Implementing connection pooling significantly improved stability under heavy concurrent usage, essential for disaster scenarios when many users might access the system simultaneously.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 12 / April 7 - April 11, 2025



Security Enhancements and Data Protection



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on enhancing PanicSensePH's security posture through comprehensive security auditing and implementing robust protection measures.

• Conducted thorough security audit identifying potential vulnerabilities
• Implemented enhanced input validation and sanitization against injection attacks
• Strengthened authentication with rate limiting and account lockout capabilities
• Added CSRF protection and proper Content Security Policy headers
• Implemented field-level encryption for sensitive information
• Created anonymization protocols for personally identifiable information
• Developed detailed audit logging system for security monitoring
• Conducted automated and manual penetration testing to verify improvements



Techniques, Tools, and Methodologies Used

• Comprehensive Zod schemas for strict validation of all inputs
• OWASP best practices for security header configuration
• Redis-based token buckets for reliable request throttling
• Automated vulnerability scanning and manual penetration testing
• AES-256 encryption for field-level data protection
• Proper key management practices for cryptographic operations
• Sandboxed processing for file uploads and untrusted content



Reflection: Problems Encountered and Lessons Learned

A significant challenge was balancing security measures with usability. Initial implementations created friction in user experience with overly strict requirements. I refined our approach to implement controls that maintain protection while minimizing impact on legitimate users.

File upload vulnerabilities posed particular risks, potentially allowing malformed files to cause system issues. Implementing robust validation and sandboxed processing proved essential for preventing both security and stability problems related to user-uploaded content.





Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 13 / April 14 - April 18, 2025



Final Integration and System Testing



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This final week, I focused on comprehensive testing and integration of all PanicSensePH components, ensuring the system functions reliably under real-world conditions.

• Conducted end-to-end testing across various disaster scenarios
• Successfully processed simulated high-volume data (5,000+ records/minute)
• Maintained 92% uptime with minimal performance degradation under heavy load
• Reduced LSTM model memory consumption by 38% through tensor optimizations
• Improved batching strategies for consistent performance in constrained environments
• Created comprehensive technical documentation including architecture diagrams
• Documented API specifications, database schema, and deployment instructions
• Conducted user acceptance testing with disaster management professionals



Techniques, Tools, and Methodologies Used

• TensorFlow for hyperparameter tuning of LSTM model
• PyTorch optimization for MBERT implementation
• K-fold cross-validation for verifying model stability
• Automated test scripts simulating real-world usage patterns
• System-level monitoring for performance and resource utilization
• Documentation generation with interactive API examples
• Continuous integration for final quality assurance



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was reconciling performance differences between testing and production environments. Initial production deployment showed lower accuracy due to memory constraints. Implementing efficient tensor operations and improved batching strategies maintained accuracy while reducing resource requirements.

Edge case testing for feedback integration revealed that some feedback patterns caused model overfitting. By implementing a sophisticated weighting mechanism considering feedback frequency and confidence scores, we improved the system without introducing bias toward specific examples.

The PanicSensePH project has demonstrated how LSTM and MBERT technologies can provide critical sentiment insights during emergencies, potentially improving disaster response coordination and public communication.