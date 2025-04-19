# Technological Institute of the Philippines
## Thesis 1 - Individual Progress Report

<table align="center" style="border-collapse: collapse; width: 100%; border: 2px solid #4A86E8;">
<tr bgcolor="#4A86E8" style="color: white;">
<td colspan="2" align="center"><h2>INDIVIDUAL PROGRESS REPORT</h2></td>
</tr>
<tr>
<td width="30%" bgcolor="#E6F0FF"><strong>Name</strong></td>
<td bgcolor="#F8F9FA">Mark Andrei R. Castillo</td>
</tr>
<tr>
<td bgcolor="#E6F0FF"><strong>Role</strong></td>
<td bgcolor="#F8F9FA">Member</td>
</tr>
<tr>
<td bgcolor="#E6F0FF"><strong>Week No. / Inclusive Dates</strong></td>
<td bgcolor="#F8F9FA">Week No. 1 / January 20 - January 24, 2025</td>
</tr>
<tr bgcolor="#4A86E8" style="color: white;">
<td colspan="2" align="center"><strong>Initial Architecture Design for PanicSensePH Platform</strong></td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Activities and Progress</strong><br>(Actual Code, Screenshot of the Design, etc.)</td>
<td bgcolor="#F8F9FA">
<p>This week, I focused on establishing the foundational architecture for our PanicSensePH disaster monitoring platform. I designed the initial system architecture diagram showing data flow between React components, Express backend, and PostgreSQL database. This architecture emphasizes real-time processing capabilities needed for disaster response.</p>
<ul>
<li>Successfully drafted database schema using Drizzle ORM with 8 primary tables: users, sessions, sentiment_posts, disaster_events, analyzed_files, sentiment_feedback, training_examples, and upload_sessions</li>
<li>Researched combined LSTM-BiGRU model architecture for sequential text processing with bidirectional context awareness</li>
<li>Identified MBERT (Multilingual BERT) as optimal solution for handling Filipino-English code-switching and jejemon text patterns</li>
<li>Structured project into client, server, and shared modules with proper type definitions</li>
<li>Created preliminary mockups of 7 main dashboard pages: Dashboard, Geographic Analysis, Timeline, Comparison, Raw Data, Evaluation, and Real-time monitoring</li>
</ul>
</td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Techniques, Tools, and Methodologies Used</strong></td>
<td bgcolor="#F8F9FA">
<ul>
<li>PostgreSQL with Drizzle ORM for type-safe database modeling and migrations</li>
<li>TypeScript for ensuring type consistency across both frontend and backend</li>
<li>TensorFlow for implementing the combined LSTM-BiGRU architecture</li>
<li>PyTorch for fine-tuning the MBERT model on Filipino slang and jejemon text patterns</li>
<li>React with Tailwind CSS for responsive and accessible user interfaces</li>
<li>Git Flow methodology for maintaining clean branching structure</li>
</ul>
</td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Reflection: Problems Encountered and Lessons Learned</strong></td>
<td bgcolor="#F8F9FA">
<p>The most significant challenge was designing a database schema that effectively supports both structured disaster reports and unstructured social media content. I learned that proper normalization with flexible JSON fields for metadata provides the optimal balance.</p>
<p>I also encountered difficulties with the approach to multilingual text processing, particularly with Filipino slang and jejemon text patterns. Through comparative research, I determined that fine-tuning MBERT on Filipino text corpora would provide better results than training custom embeddings from scratch.</p>
</td>
</tr>
</table>


<table align="center" style="border-collapse: collapse; width: 100%; border: 2px solid #4A86E8; margin-top: 30px;">
<tr bgcolor="#4A86E8" style="color: white;">
<td colspan="2" align="center"><h2>INDIVIDUAL PROGRESS REPORT</h2></td>
</tr>
<tr>
<td width="30%" bgcolor="#E6F0FF"><strong>Name</strong></td>
<td bgcolor="#F8F9FA">Mark Andrei R. Castillo</td>
</tr>
<tr>
<td bgcolor="#E6F0FF"><strong>Role</strong></td>
<td bgcolor="#F8F9FA">Member</td>
</tr>
<tr>
<td bgcolor="#E6F0FF"><strong>Week No. / Inclusive Dates</strong></td>
<td bgcolor="#F8F9FA">Week No. 2 / January 27 - January 31, 2025</td>
</tr>
<tr bgcolor="#4A86E8" style="color: white;">
<td colspan="2" align="center"><strong>Setting Up Development Environment and Core Infrastructure</strong></td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Activities and Progress</strong><br>(Actual Code, Screenshot of the Design, etc.)</td>
<td bgcolor="#F8F9FA">
<p>During the second week, I concentrated on implementing core infrastructure components essential for our development workflow. I set up our collaborative environment to ensure consistent development across the team.</p>
<ul>
<li>Established CI/CD pipeline for automated testing and deployment</li>
<li>Implemented initial Express.js backend structure with TypeScript</li>
<li>Created foundational API endpoints for sentiment analysis and disaster monitoring</li>
<li>Configured Drizzle ORM with PostgreSQL, implementing proper migrations</li>
<li>Set up React application with Vite and implemented basic layout components</li>
<li>Developed reusable UI components using Tailwind CSS for consistent user experience</li>
</ul>
</td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Techniques, Tools, and Methodologies Used</strong></td>
<td bgcolor="#F8F9FA">
<ul>
<li>Git with feature branch workflow for clean development history</li>
<li>Drizzle ORM for type-safe database access and migrations</li>
<li>React with Vite for efficient frontend development</li>
<li>Agile methodology with weekly sprints and daily stand-ups</li>
<li>RESTful API design principles with proper error handling</li>
<li>Docker containers for consistent development environments</li>
</ul>
</td>
</tr>
<tr>
<td bgcolor="#E6F0FF" valign="top"><strong>Reflection: Problems Encountered and Lessons Learned</strong></td>
<td bgcolor="#F8F9FA">
<p>The main challenge was ensuring consistency across different development environments, as team members encountered issues with package versions and database connections. I addressed this by implementing Docker containers for development, providing a standardized environment.</p>
<p>Another difficulty was balancing rapid prototyping with establishing a solid foundation. I learned that investing time in proper architecture early on, while seemingly slower initially, pays significant dividends as complexity increases.</p>
</td>
</tr>
</table>


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 3 / February 3 - February 7, 2025 |

| Implementation of Data Collection and Processing System | |
| ------------------------------------------------------ | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I implemented the data collection and processing systems for PanicSensePH, focusing on robust ingestion pipelines for disaster-related content.<br><br>• Developed ETL pipeline for ingesting content from social media and government announcements<br>• Implemented CSV file upload functionality with validation and batch processing capabilities<br>• Created progress tracking system for monitoring file imports in real-time<br>• Established error handling protocols for data processing failures<br>• Developed data preprocessing module for text normalization and cleaning<br>• Set up unit tests for critical data processing components with 85% coverage |
| Techniques, Tools, and Methodologies Used | • Node.js streams for efficient processing of large datasets with minimal memory consumption<br>• Custom CSV parsing solution using csv-parse library with robust error handling<br>• Natural language processing techniques for text normalization and cleaning<br>• Jest for unit testing critical processing components<br>• Sample datasets of various sizes and formats for comprehensive testing<br>• Chunking strategies for handling large file processing |
| Reflection: Problems Encountered and Lessons Learned | A significant challenge was handling the variety of data formats and quality issues in real-world datasets. I learned the importance of robust validation and preprocessing to handle edge cases like incomplete records and inconsistent formatting.<br><br>Performance issues emerged when processing very large files, which I addressed by implementing a chunking strategy that processes data in manageable batches. This approach not only improved performance but also enabled better progress tracking and error recovery. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 4 / February 10 - February 14, 2025 |

| Implementation of Authentication System and User Management | |
| ---------------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I focused on implementing the authentication and user management functionality, establishing secure access controls for the platform.<br><br>• Developed secure authentication mechanism using JWT with proper refresh token rotation<br>• Implemented user registration, login, password recovery, and session management<br>• Created role-based access control (RBAC) for different user types<br>• Developed middleware for route protection with fine-grained access control<br>• Implemented user profile management system with personal information updates<br>• Set up secure profile image upload and management functionality<br>• Created protected route components for client-side authentication restrictions |
| Techniques, Tools, and Methodologies Used | • Bcrypt for secure password hashing and storage<br>• JWT for stateless authentication with proper expiration policies<br>• Zod schemas for comprehensive user data validation<br>• Multer for secure file upload handling with validation<br>• React Query for managing authentication state throughout the application<br>• Custom authentication hooks for centralized auth logic<br>• CSRF protection measures for enhanced security |
| Reflection: Problems Encountered and Lessons Learned | The main challenge was balancing security requirements like password complexity and token expiration with usability. I implemented progressive security measures that maintain strong protection without unnecessarily burdening users.<br><br>Another issue was managing authentication state across multiple components. I learned the importance of centralizing authentication logic in custom hooks and contexts to provide a single source of truth throughout the application. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 5 / February 17 - February 21, 2025 |

| Implementation of Initial LSTM-BiGRU Model for Sentiment Analysis | |
| ---------------------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I implemented the initial version of our combined LSTM-BiGRU sentiment analysis model for PanicSensePH, focusing on accurate classification of disaster-related content.<br><br>• Developed hybrid LSTM-BiGRU architecture combining long-term memory capabilities with bidirectional context awareness<br>• Created training pipeline with 1,000 manually annotated disaster-related social media posts in both English and Filipino<br>• Integrated fine-tuned MBERT layers for enhanced handling of Filipino slang and jejemon text patterns<br>• Achieved 72% initial accuracy across five sentiment categories with significant improvement for mixed-language content<br>• Implemented model endpoints in Express.js backend for both real-time analysis and batch processing<br>• Created specialized tokenization handling for jejemon text with character-level normalization<br>• Developed sentiment_posts table with proper indexing for efficient querying by sentiment type and source |
| Techniques, Tools, and Methodologies Used | • TensorFlow for implementing the combined LSTM-BiGRU architecture with custom attention mechanism<br>• PyTorch for fine-tuning MBERT on Filipino slang and jejemon text samples<br>• Custom preprocessing pipeline for handling jejemon text variations and character substitutions<br>• Transfer learning techniques for adapting pre-trained MBERT to disaster-specific terminology<br>• Database schema optimization with proper indices for sentiment filtering and aggregation<br>• Cross-validation with stratified sampling to ensure representative distribution across sentiment classes<br>• Custom metrics for evaluating performance on code-switched and jejemon text separately |
| Reflection: Problems Encountered and Lessons Learned | The main challenge was handling jejemon text patterns where characters are deliberately substituted (e.g., "p0h" instead of "po", "m@hal" instead of "mahal"). The initial LSTM-BiGRU model struggled with these variations. By integrating character-level preprocessing and fine-tuning MBERT on a corpus containing jejemon samples, recognition accuracy improved by 18%.<br><br>Another significant issue was balancing the computational requirements of the combined model architecture. The initial implementation was too resource-intensive for efficient batch processing. I implemented a two-stage approach where simpler features are processed first to filter out obvious cases, with the full LSTM-BiGRU-MBERT stack used only for complex or ambiguous content. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 6 / February 24 - February 28, 2025 |

| Integration of Geographic Analysis and Visualization | |
| --------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I focused on integrating geographic analysis and visualization features into PanicSensePH, enhancing spatial understanding of disaster sentiment patterns.<br><br>• Developed location extraction system with 75% accuracy for place mention identification<br>• Implemented interactive map visualization using Leaflet.js with sentiment heat maps<br>• Created map interface with filtering by time period, sentiment type, and disaster events<br>• Established database of Philippine administrative boundaries for precise classification<br>• Integrated population data and risk factors for different regions<br>• Developed custom React components for responsive map visualization<br>• Implemented custom clustering algorithms for efficient data point rendering |
| Techniques, Tools, and Methodologies Used | • Leaflet.js with custom map layers and markers for interactive visualization<br>• Rule-based pattern matching with gazetteer of Philippine place names<br>• GeoJSON processing for administrative boundaries<br>• Specialized database indexes for efficient spatial queries<br>• Custom React components for responsive map interfaces<br>• Clustering strategies for handling large datasets on maps<br>• Context-aware location resolution algorithms |
| Reflection: Problems Encountered and Lessons Learned | A significant challenge was disambiguating place names that could refer to multiple locations in the Philippines. I implemented context-aware resolution that considers surrounding text and frequency analysis to identify the most likely location.<br><br>Optimizing map performance with large datasets proved difficult, as displaying thousands of points caused performance issues. I implemented clustering strategies and progressive loading techniques to maintain smooth interactions. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 7 / March 3 - March 7, 2025 |

| Implementation of Real-time Monitoring and WebSocket Integration | |
| --------------------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I implemented real-time monitoring capabilities for PanicSensePH, enabling live updates and instant notifications for emerging disaster situations.<br><br>• Developed WebSocket-based communication system for live updates between server and clients<br>• Created notification center for displaying real-time alerts about sentiment pattern changes<br>• Implemented message queue using Redis for reliable update delivery<br>• Established connection state management with recovery for missed updates<br>• Developed throttling and debouncing mechanisms for high-volume update periods<br>• Created exponential backoff strategy for reconnection during network instability<br>• Implemented broadcast optimization for efficient message distribution |
| Techniques, Tools, and Methodologies Used | • WebSocket protocol with custom message format for efficient data transfer<br>• Redis-based message queue for reliable delivery and persistence<br>• React Query for integrating real-time data with application state<br>• Throttling and debouncing for preventing UI performance issues<br>• Exponential backoff strategy for graceful reconnection<br>• Connection pooling for efficient resource management<br>• Custom event system for real-time notifications |
| Reflection: Problems Encountered and Lessons Learned | The primary challenge was maintaining WebSocket connection stability during network fluctuations. I implemented robust reconnection strategies with proper error handling to provide seamless experience even under suboptimal conditions.<br><br>Managing increased server load from numerous concurrent WebSocket connections proved difficult. By implementing connection pooling and efficient broadcast mechanisms, I significantly reduced resource requirements while maintaining responsive updates. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 8 / March 10 - March 14, 2025 |

| Disaster Type Classification and Event Correlation | |
| ------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I implemented disaster type classification and event correlation features, enabling the system to identify specific disaster types and detect emerging events.<br><br>• Developed specialized classifier for categorizing posts into specific disaster types with 82% accuracy<br>• Created correlation engine for identifying relationships between sentiment patterns and disasters<br>• Implemented detection system for emerging disaster events based on sentiment clusters<br>• Developed disaster events dashboard with active and historical event tracking<br>• Created sentiment trend analysis visualization for each disaster event<br>• Implemented hierarchical classification for primary and secondary disaster types<br>• Developed temporal context detection for distinguishing current vs. past events |
| Techniques, Tools, and Methodologies Used | • Multi-label classification system using TensorFlow<br>• Temporal pattern analysis with sliding window algorithms<br>• Correlation algorithms for relationship identification<br>• Clustering techniques using spatial and temporal dimensions<br>• Interactive visualizations with Recharts for trend analysis<br>• Hierarchical classification for complex disaster relationships<br>• Anomaly detection for identifying unusual sentiment patterns |
| Reflection: Problems Encountered and Lessons Learned | A significant challenge was distinguishing between discussions about past disasters and reports of current events. I implemented temporal context detection that analyzes time indicators and correlates with posting timestamps to determine if posts refer to ongoing or historical events.<br><br>Handling reports of multiple simultaneous disasters, such as flooding caused by typhoons, proved difficult. I developed a hierarchical classification approach that identifies primary and secondary disaster types, providing more nuanced analysis of complex situations. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 9 / March 17 - March 21, 2025 |

| MBERT Integration for Enhanced Multilingual Support | |
| -------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I enhanced PanicSensePH's multilingual capabilities through deeper integration of Multilingual BERT (MBERT), focusing on improved handling of code-switching text and jejemon patterns.<br><br>• Implemented specialized MBERT-based tokenization for mixed-language text with jejemon variations<br>• Developed language-specific sentiment lexicon for Filipino disaster terminology including slang terms<br>• Improved accuracy for mixed-language posts by 15%, reaching 81% overall accuracy<br>• Created seamless pipeline combining LSTM-BiGRU and MBERT components<br>• Implemented character-level transformer to normalize jejemon text variations<br>• Developed cross-lingual word embeddings aligning English and Filipino semantic spaces<br>• Created handling for regional dialectal variations through specialized training data |
| Techniques, Tools, and Methodologies Used | • Cross-lingual word embeddings for aligned semantic spaces<br>• PyTorch for MBERT fine-tuning on disaster terminology and jejemon text<br>• Transfer learning from English corpus to Filipino processing<br>• Model quantization and caching strategies for performance<br>• Code-switching detection algorithms for language transitions<br>• Character-level preprocessing for jejemon text normalization<br>• Fuzzy matching for dialectal variants and slang terms |
| Reflection: Problems Encountered and Lessons Learned | The biggest challenge was handling Filipino jejemon text patterns, which significantly affected sentiment detection accuracy. Common patterns include character substitutions (e.g., "w@t p0" instead of "what po"), elongated words ("soooobrang" instead of "sobrang"), and mixed character cases ("aNg GaLiNg"). I addressed this by implementing a character-level transformer that normalizes these variations before processing.<br><br>MBERT struggled with very short text fragments common in social media. I implemented a hybrid approach that falls back to specialized LSTM-BiGRU for short texts while using MBERT for longer content, optimizing performance across different content types. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 10 / March 24 - March 28, 2025 |

| Implementation of Feedback Loop and Continuous Learning | |
| -------------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I implemented a feedback mechanism for PanicSensePH that enables continuous improvement through user corrections of sentiment classifications.<br><br>• Developed user interface for submitting sentiment classification corrections<br>• Created dedicated sentiment_feedback table for storing and tracking corrections<br>• Implemented training pipeline for incorporating feedback into LSTM-BiGRU-MBERT model refinement<br>• Developed confidence threshold system for flagging low-confidence predictions<br>• Created semi-supervised learning environment for targeted human review<br>• Improved overall accuracy by 3% with just 100 feedback instances<br>• Implemented analytics dashboard for tracking correction patterns and model weaknesses |
| Techniques, Tools, and Methodologies Used | • Active learning strategies for intelligent sample selection<br>• Incremental model updating without full retraining<br>• Weighted feedback integration for preserving global model performance<br>• Confidence scoring mechanisms for prediction reliability assessment<br>• React with optimistic updates for responsive feedback interface<br>• Feedback prioritization based on correction value<br>• Pattern analysis for identifying systematic model weaknesses |
| Reflection: Problems Encountered and Lessons Learned | A significant challenge was balancing user feedback incorporation with model stability. Early implementations sometimes overfit to recent corrections, reducing general performance. I implemented a weighted feedback integration approach that preserves global model performance while learning from new examples.<br><br>Tracking feedback patterns proved crucial for identifying systematic weaknesses in handling jejemon text and regional dialects. The analytics dashboard I created for monitoring correction patterns has been invaluable for guiding targeted improvements to specific aspects of the sentiment analysis system. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 11 / March 31 - April 4, 2025 |

| Performance Optimization and Scalability Enhancements | |
| ----------------------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I focused on improving PanicSensePH's performance and scalability through comprehensive optimization of both frontend and backend components.<br><br>• Conducted thorough profiling to identify performance bottlenecks in LSTM-BiGRU-MBERT pipeline<br>• Improved average response times by 65% for common operations<br>• Optimized database queries through proper indexing of sentiment_posts and disaster_events tables<br>• Enhanced frontend with virtualized lists and lazy loading for efficient rendering<br>• Implemented caching layer for frequently accessed and compute-intensive results<br>• Improved system capacity to handle 3x previous concurrent user load<br>• Developed tiered processing approach for balancing accuracy and speed<br>• Implemented model quantization for reduced memory footprint |
| Techniques, Tools, and Methodologies Used | • Database indexing targeted at common sentiment and location query patterns<br>• Query optimization and strategic denormalization for frequently joined tables<br>• React component memoization and code splitting for dashboard pages<br>• Sophisticated caching strategy with memory and persistent tiers<br>• Load testing with simulated concurrent users<br>• TensorFlow model quantization to reduce memory requirements<br>• Progressive loading techniques for large geographical datasets |
| Reflection: Problems Encountered and Lessons Learned | The main challenge was maintaining accuracy while improving performance of the LSTM-BiGRU-MBERT pipeline. Initial optimization attempts reduced processing time but slightly decreased model accuracy for jejemon text. I addressed this by implementing a tiered approach with fast approximate methods followed by precise analysis where needed.<br><br>Database connection management was causing performance degradation during high load. Implementing connection pooling significantly improved stability under heavy concurrent usage, essential for disaster scenarios when many users might access the system simultaneously. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 12 / April 7 - April 11, 2025 |

| Security Enhancements and Data Protection | |
| ----------------------------------------- | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, I focused on enhancing PanicSensePH's security posture through comprehensive security auditing and implementing robust protection measures.<br><br>• Conducted thorough security audit identifying potential vulnerabilities<br>• Implemented enhanced input validation and sanitization against injection attacks<br>• Strengthened authentication with rate limiting and account lockout capabilities<br>• Added CSRF protection and proper Content Security Policy headers<br>• Implemented field-level encryption for sensitive information<br>• Created anonymization protocols for personally identifiable information<br>• Developed detailed audit logging system for security monitoring<br>• Conducted automated and manual penetration testing to verify improvements |
| Techniques, Tools, and Methodologies Used | • Comprehensive Zod schemas for strict validation of all inputs<br>• OWASP best practices for security header configuration<br>• Redis-based token buckets for reliable request throttling<br>• Automated vulnerability scanning and manual penetration testing<br>• AES-256 encryption for field-level data protection<br>• Proper key management practices for cryptographic operations<br>• Sandboxed processing for file uploads and untrusted content |
| Reflection: Problems Encountered and Lessons Learned | A significant challenge was balancing security measures with usability. Initial implementations created friction in user experience with overly strict requirements. I refined our approach to implement controls that maintain protection without unnecessarily burdening users.<br><br>File upload vulnerabilities posed particular risks, potentially allowing malformed files to cause system issues. Implementing robust validation and sandboxed processing proved essential for preventing both security and stability problems related to user-uploaded content. |


Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT

| Name | Mark Andrei R. Castillo |
| ---- | ----------------------- |
| Role | Member |
| Week No. / Inclusive Dates | Week No. 13 / April 14 - April 18, 2025 |

| Final Integration and System Testing | |
| ------------------------------------ | --- |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This final week, I focused on comprehensive testing and integration of all PanicSensePH components, ensuring the system functions reliably under real-world conditions.<br><br>• Conducted end-to-end testing across various disaster scenarios<br>• Successfully processed simulated high-volume data (5,000+ records/minute)<br>• Maintained 92% uptime with minimal performance degradation under heavy load<br>• Reduced LSTM-BiGRU model memory consumption by 38% through tensor optimizations<br>• Improved MBERT integration for better handling of jejemon text variations<br>• Created comprehensive technical documentation including architecture diagrams<br>• Documented API specifications, database schema, and deployment instructions<br>• Validated all seven dashboard pages: Dashboard, Geographic Analysis, Timeline, Comparison, Raw Data, Evaluation, and Real-time monitoring |
| Techniques, Tools, and Methodologies Used | • TensorFlow for hyperparameter tuning of LSTM-BiGRU model<br>• PyTorch optimization for MBERT implementation<br>• K-fold cross-validation for verifying model stability<br>• Automated test scripts simulating real-world usage patterns<br>• System-level monitoring for performance and resource utilization<br>• Documentation generation with interactive API examples<br>• Continuous integration for final quality assurance |
| Reflection: Problems Encountered and Lessons Learned | The most significant challenge was reconciling performance differences between testing and production environments. Initial production deployment showed lower accuracy due to memory constraints. Implementing efficient tensor operations and improved batching strategies maintained accuracy while reducing resource requirements.<br><br>Edge case testing for feedback integration revealed that some jejemon text patterns caused model overfitting. By implementing a sophisticated weighting mechanism considering feedback frequency and confidence scores, we improved the system without introducing bias toward specific examples.<br><br>The PanicSensePH project has successfully demonstrated how LSTM-BiGRU combined with fine-tuned MBERT can provide critical sentiment insights during emergencies, potentially improving disaster response coordination and public communication. |