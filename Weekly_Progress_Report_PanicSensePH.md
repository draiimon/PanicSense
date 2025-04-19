# Technological University of the Philippines

## WEEKLY PROGRESS REPORT - PANICSENSE PH

### Project Progress Summary
**Week 1: January 20 - January 24, 2025**

#### System Architecture Design and Requirements Analysis

##### Activities and Progress
This week marked the official start of the PanicSensePH project development. Our team focused on establishing the foundational architecture and defining technical requirements. We successfully designed the initial system architecture diagram, which illustrates the data flow between our front-end React components, back-end Express services, and PostgreSQL database integration.

We conducted a comprehensive stakeholder needs analysis through interviews with three disaster management officials from NDRRMC. Their input helped us identify critical features for the minimum viable product (MVP), particularly regarding real-time data processing needs.

The team also implemented the initial project setup with the Node.js/TypeScript environment and established the PostgreSQL database connectivity, setting up all essential tables as defined in our schema.

##### Techniques, Tools, and Methodologies Used
- **PostgreSQL Database Design** - Implemented normalized schemas for disaster-related data
- **TypeScript & Drizzle ORM** - Established type-safe database models with migration capabilities
- **Express.js Backend** - Set up RESTful API endpoints for data exchange
- **React/Vite Frontend** - Configured development environment with hot module replacement
- **Git Flow Methodology** - Implemented feature branch workflow for parallel development

##### Reflection: Problems Encountered and Lessons Learned
The biggest challenge was selecting the appropriate schema design that would accommodate both structured and unstructured data from social media sources. We learned that planning for future expansion from the beginning is crucial, particularly regarding the sentiment analysis models we'll implement in the coming weeks.

Balancing real-time processing capabilities with database performance was another considerable challenge. We resolved this by implementing an efficient batch processing strategy that will be crucial for handling large CSV imports.

---

### Project Progress Summary
**Week 2: January 27 - January 31, 2025**

#### Sentiment Analysis Model Development

##### Activities and Progress
This week, our focus shifted to developing the core sentiment analysis capabilities. We successfully implemented the initial version of our custom Long Short-Term Memory (LSTM) neural network model for disaster-related text classification. The architecture includes bidirectional layers that capture contextual information from both directions of the text sequence.

We began training the model on our preliminary dataset of 1,000 disaster-related social media posts, which we manually annotated with sentiment categories (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral). Initial validation tests showed promising results with approximately 68% accuracy.

Additionally, we developed the file processing system for CSV uploads, which will be a critical component for bulk analysis of historical data. This included implementation of a progress tracking mechanism to provide real-time feedback during processing.

##### Techniques, Tools, and Methodologies Used
- **Custom LSTM Neural Network** - Developed a bidirectional LSTM specialized for disaster text sequences
- **MBERT (Multilingual BERT)** - Implemented for handling Filipino text and code-switching
- **Word Embeddings** - Created 300-dimensional embeddings that capture disaster terminology semantics
- **TensorFlow and PyTorch** - Used TensorFlow for LSTM and PyTorch for MBERT implementation
- **Ensemble Learning** - Combined predictions from LSTM and MBERT for improved accuracy
- **Batch Processing** - Implemented efficient batch processing to handle large datasets

##### Reflection: Problems Encountered and Lessons Learned
The primary challenge was handling multilingual text, as many disaster reports in the Philippines contain code-switching between English and Filipino. We addressed this by integrating MBERT for improved language detection and processing.

We also encountered performance bottlenecks during batch processing of large files. This led us to implement a chunking strategy that processes data in manageable segments, providing regular progress updates to the user interface.

---

### Project Progress Summary
**Week 3: February 3 - February 7, 2025**

#### Location Detection and Geographic Analysis

##### Activities and Progress
This week focused on enhancing our system with location detection capabilities. We implemented a rule-based system that extracts location mentions from text data, which was then integrated with our existing sentiment analysis pipeline.

We created a comprehensive database of Philippine locations, including major cities, municipalities, and regions, which our system uses to identify disaster-affected areas. This location data is now properly linked to sentiment analysis results, allowing for geographic visualization of sentiment patterns.

The user interface was extended with a map visualization component using Leaflet.js, which now displays color-coded sentiment data across different regions. Initial testing shows the system can accurately identify locations in approximately 75% of disaster-related posts.

##### Techniques, Tools, and Methodologies Used
- **Named Entity Recognition** - Implemented custom rules for extracting location mentions
- **Geographic Information System (GIS)** - Integrated with Leaflet.js for interactive maps
- **Location Database** - Created comprehensive database of Philippine locations
- **Pattern Matching Algorithms** - Developed robust patterns for identifying location contexts
- **Spatial Data Visualization** - Implemented heat maps showing sentiment distribution

##### Reflection: Problems Encountered and Lessons Learned
A significant challenge was disambiguating common place names that could refer to multiple locations in the Philippines. We addressed this by implementing context-aware location resolution that considers surrounding text for disambiguation.

We also realized that relying solely on explicit location mentions was insufficient, as many social media posts imply locations without directly naming them. This led us to begin work on a contextual location inference system for future implementation.

---

### Project Progress Summary
**Week 4: February 10 - February 14, 2025**

#### Authentication System and User Management

##### Activities and Progress
This week, we developed and integrated the user authentication and management system. We successfully implemented secure user registration, login functionality, and session management using JWT tokens and bcrypt for password hashing.

We created role-based access controls to distinguish between regular users and administrators, with appropriate permission levels for each. The database schema was extended to support user profiles, including profile image storage and retrieval.

We also implemented the front-end components for user authentication, including login and registration forms with client-side validation. The system now maintains persistent sessions and properly handles authentication-protected routes.

##### Techniques, Tools, and Methodologies Used
- **Secure Authentication** - Implemented JWT-based authentication with proper token management
- **Password Security** - Used bcrypt for secure password hashing and storage
- **Role-Based Access Control** - Developed permission system for different user types
- **React Hook Form** - Implemented form validation with zod schema integration
- **Session Management** - Created secure, persistent sessions with proper timeout handling

##### Reflection: Problems Encountered and Lessons Learned
The main challenge was ensuring proper security practices throughout the authentication flow. We conducted a thorough security review and implemented measures to prevent common vulnerabilities like CSRF attacks and token exposure.

We also discovered the importance of clear error messages for users during the authentication process. After user testing, we improved the error handling to provide more informative feedback, which significantly enhanced the user experience.

---

### Project Progress Summary
**Week 5: February 17 - February 21, 2025**

#### CSV Processing and Batch Analysis

##### Activities and Progress
This week centered on enhancing our batch processing capabilities for historical data analysis. We successfully implemented an advanced CSV processing system that can handle files containing thousands of records efficiently.

The system now provides real-time progress updates during processing, with detailed statistics about completion percentage and estimated time remaining. We also implemented error handling for malformed records, ensuring that individual problematic entries don't halt the entire processing operation.

We developed a user interface for CSV upload and processing management, including the ability to cancel ongoing processes and resume from previous failures. Additionally, we created a results visualization dashboard that displays aggregate statistics and sentiment distribution from processed files.

##### Techniques, Tools, and Methodologies Used
- **Stream Processing** - Implemented efficient file streaming for large dataset handling
- **Parallel Processing** - Utilized worker threads for simultaneous record processing
- **Progress Tracking** - Developed real-time progress monitoring system
- **Error Recovery** - Created mechanisms to handle and recover from processing failures
- **Batch Optimization** - Improved processing efficiency through chunking and memory management

##### Reflection: Problems Encountered and Lessons Learned
The primary challenge was managing memory usage during large file processing. Initial implementations caused memory overflow with files exceeding 10,000 records. We resolved this by implementing a streaming approach that processes data in manageable chunks.

We also encountered issues with varying CSV formats from different data sources. This led us to implement a more flexible parsing system that can adapt to different column arrangements and naming conventions, making the system more robust for real-world use.

---

### Project Progress Summary
**Week 6: February 24 - February 28, 2025**

#### LSTM Model Refinement and Accuracy Improvement

##### Activities and Progress
This week was dedicated to improving our sentiment analysis model's accuracy. We conducted a thorough evaluation of our existing LSTM model and identified key areas for enhancement, particularly in distinguishing between closely related sentiment categories like Fear/Anxiety and Panic.

We collected and annotated an additional 1,500 disaster-related posts to expand our training dataset, with special focus on ambiguous cases. The enhanced training set was used to retrain our LSTM model, resulting in an accuracy improvement from 68% to 74%.

We also implemented a confidence scoring system that provides a reliability metric for each sentiment prediction. This allows users to filter results based on confidence thresholds and helps identify cases that may require human review.

##### Techniques, Tools, and Methodologies Used
- **Model Hyperparameter Tuning** - Optimized LSTM architecture for better performance
- **Data Augmentation** - Expanded training dataset with strategic new examples
- **Cross-Validation** - Implemented k-fold validation for more reliable accuracy measurement
- **Confidence Scoring** - Developed probabilistic confidence metrics for predictions
- **Ensemble Methods** - Combined multiple model outputs for improved results

##### Reflection: Problems Encountered and Lessons Learned
A significant challenge was balancing the model's performance across different sentiment categories. Initial improvements in Panic detection came at the cost of reduced accuracy for Neutral classifications. We addressed this through careful class weighting and targeted data augmentation.

We also found that Filipino expressions of distress often use distinctive patterns that weren't being captured by our initial model. This led us to implement specialized tokenization and feature extraction for Filipino disaster terminology, which significantly improved detection of culturally specific expressions.

---

### Project Progress Summary
**Week 7: March 3 - March 7, 2025**

#### Real-time Monitoring and WebSocket Integration

##### Activities and Progress
This week focused on implementing real-time monitoring capabilities. We successfully integrated WebSocket connections to provide live updates to users as new data is processed or sentiments are detected.

We developed the server-side event broadcasting system that notifies connected clients about newly analyzed posts, processing progress updates, and system status changes. The client-side was enhanced with components that dynamically update to reflect these real-time changes without requiring page refreshes.

Additionally, we implemented a notification system that alerts users to significant sentiment pattern changes, such as sudden increases in panic-related posts for specific locations, which could indicate emerging disaster situations.

##### Techniques, Tools, and Methodologies Used
- **WebSocket Protocol** - Implemented bidirectional communication for real-time updates
- **Event-Driven Architecture** - Developed publisher-subscriber pattern for efficient notifications
- **React Query Integration** - Enhanced data fetching with optimistic updates and cache synchronization
- **Throttling and Debouncing** - Implemented rate limiting to prevent UI performance issues
- **Connection Management** - Created robust handling for connection drops and reconnection

##### Reflection: Problems Encountered and Lessons Learned
The main challenge was managing WebSocket connection stability, particularly during periods of network instability. We implemented a robust reconnection strategy with exponential backoff, which significantly improved the user experience during intermittent connectivity.

We also discovered performance issues when large numbers of updates were sent simultaneously. This led us to implement batching and throttling mechanisms that group updates together, reducing overhead and preventing UI rendering bottlenecks.

---

### Project Progress Summary
**Week 8: March 10 - March 14, 2025**

#### Disaster Type Classification and Event Correlation

##### Activities and Progress
This week, we expanded the system's analytical capabilities by implementing disaster type classification. We developed a specialized classifier that can categorize posts into specific disaster types, including typhoons, earthquakes, floods, fires, and volcanic eruptions.

We created a correlation engine that identifies relationships between sentiment patterns and disaster types across different locations. This enables the system to detect emerging disaster events based on sentiment clusters and provide early warnings.

The user interface was enhanced with a disaster events dashboard that displays active and historical events, along with sentiment trend analysis for each. Initial testing shows the system can accurately detect and classify disaster events with approximately 82% accuracy.

##### Techniques, Tools, and Methodologies Used
- **Multi-label Classification** - Implemented disaster type categorization capabilities
- **Temporal Pattern Analysis** - Developed time-series analysis for trend detection
- **Correlation Algorithms** - Created methods to identify relationships between sentiments and events
- **Clustering Techniques** - Implemented spatial and temporal clustering for event detection
- **Interactive Visualizations** - Enhanced UI with dynamic charts for trend visualization

##### Reflection: Problems Encountered and Lessons Learned
A significant challenge was distinguishing between discussions about past disasters and reports of current events. We addressed this by implementing temporal context detection that analyzes time indicators in the text and correlates them with posting timestamps.

We also encountered difficulties in handling reports of multiple simultaneous disasters, such as flooding caused by typhoons. This led us to implement a hierarchical classification approach that can identify primary and secondary disaster types from a single post.

---

### Project Progress Summary
**Week 9: March 17 - March 21, 2025**

#### MBERT Integration for Enhanced Language Support

##### Activities and Progress
This week's focus was on improving multilingual support through deeper integration of Multilingual BERT (MBERT). We enhanced our language detection and analysis capabilities to better handle code-switching between English and Filipino, which is common in disaster-related social media posts.

We implemented a specialized MBERT-based tokenizer that properly processes mixed-language text, preserving the semantic meaning across language boundaries. This was integrated with our existing LSTM pipeline to create a hybrid model that leverages the strengths of both approaches.

Additionally, we developed a language-specific sentiment lexicon for Filipino disaster terminology, which provides better context for sentiment analysis in culturally specific expressions. Testing shows a 15% improvement in accuracy for mixed-language posts.

##### Techniques, Tools, and Methodologies Used
- **MBERT Fine-tuning** - Specialized the model for disaster-related terminologies
- **Cross-lingual Embeddings** - Implemented aligned word embeddings across languages
- **Code-switching Detection** - Developed algorithms to identify language transitions in text
- **Contextual Analysis** - Enhanced semantic understanding with contextual embeddings
- **Transfer Learning** - Applied knowledge from English corpus to improve Filipino processing

##### Reflection: Problems Encountered and Lessons Learned
The primary challenge was handling Filipino dialectal variations, which can significantly affect sentiment detection. We addressed this by incorporating regional language patterns into our training data and implementing fuzzy matching for dialectal variants.

We also discovered that MBERT sometimes struggled with very short text fragments common in social media. This led us to implement a hybrid approach that falls back to our specialized LSTM for very short texts while using MBERT for longer, more complex content.

---

### Project Progress Summary
**Week 10: March 24 - March 28, 2025**

#### Feedback Loop and Model Adaptation

##### Activities and Progress
This week centered on implementing a feedback mechanism that allows users to correct sentiment classifications and improve the system over time. We developed a user interface for submitting corrections, which are then stored in a dedicated feedback table in the database.

We created a training pipeline that periodically incorporates user feedback to retrain and refine our models. This closed-loop system ensures continuous improvement based on real-world usage patterns. Initial testing shows that incorporating just 100 feedback instances improved overall accuracy by 3%.

Additionally, we implemented a confidence threshold system that automatically flags low-confidence predictions for human review, creating a semi-supervised learning environment that prioritizes human intervention where it's most needed.

##### Techniques, Tools, and Methodologies Used
- **Active Learning** - Implemented strategic sample selection for human annotation
- **Incremental Learning** - Developed continuous model updating without full retraining
- **Feedback Prioritization** - Created system to identify most valuable correction opportunities
- **Confidence Estimation** - Enhanced reliability metrics for prediction trustworthiness
- **Human-in-the-loop Processing** - Designed workflow for efficient human review integration

##### Reflection: Problems Encountered and Lessons Learned
A significant challenge was balancing the incorporation of user feedback with maintaining model stability. Early implementations sometimes overfit to recent corrections, reducing general performance. We resolved this by implementing a weighted feedback integration approach that preserves global model performance.

We also discovered the importance of tracking feedback patterns to identify systematic model weaknesses. This led us to implement an analytics dashboard specifically for monitoring correction patterns, which has been invaluable for guiding targeted improvements.

---

### Project Progress Summary
**Week 11: March 31 - April 4, 2025**

#### Performance Optimization and Scalability Enhancements

##### Activities and Progress
This week was dedicated to improving system performance and scalability. We conducted comprehensive profiling of both front-end and back-end components to identify bottlenecks and optimization opportunities.

We implemented database query optimization, including proper indexing and query restructuring, which reduced average response times by 65% for common operations. The front-end was enhanced with virtualized lists and lazy loading for more efficient rendering of large datasets.

We also implemented a caching layer for frequently accessed and compute-intensive results, particularly for sentiment analysis of common phrases. This reduced redundant processing and improved overall system responsiveness. Load testing confirms the system can now handle 3x the previous concurrent user load.

##### Techniques, Tools, and Methodologies Used
- **Database Indexing** - Implemented strategic indexes for query optimization
- **Query Optimization** - Restructured complex queries for efficiency
- **React Component Memoization** - Reduced unnecessary re-renders
- **Response Caching** - Implemented intelligent caching strategies
- **Lazy Loading** - Applied code-splitting and dynamic imports for faster initial loads
- **Load Balancing** - Designed architecture to distribute processing efficiently

##### Reflection: Problems Encountered and Lessons Learned
The main challenge was maintaining accuracy while improving performance. Initial optimization attempts reduced processing time but slightly decreased model accuracy. We addressed this by implementing a tiered approach that uses fast, approximate methods for initial results followed by more precise analysis where needed.

We also discovered that database connection management was causing performance degradation during high load. This led us to implement a connection pooling strategy that significantly improved stability under heavy concurrent usage.

---

### Project Progress Summary
**Week 12: April 7 - April 11, 2025**

#### Security Enhancements and Data Protection

##### Activities and Progress
This week focused on strengthening the security posture of PanicSensePH. We conducted a comprehensive security audit, identifying and addressing potential vulnerabilities throughout the system.

We implemented enhanced input validation and sanitization to prevent injection attacks, particularly for user-submitted content and file uploads. The authentication system was strengthened with rate limiting, account lockout capabilities, and improved password policies.

Data protection was improved through implementation of field-level encryption for sensitive information and proper anonymization of personally identifiable information in exported datasets. We also created a detailed audit logging system that tracks all significant system actions for security monitoring.

##### Techniques, Tools, and Methodologies Used
- **Input Validation** - Implemented comprehensive validation using zod schemas
- **Rate Limiting** - Applied request throttling to prevent abuse
- **Content Security Policy** - Configured proper CSP headers
- **XSS Prevention** - Implemented context-aware output encoding
- **CSRF Protection** - Added token-based cross-site request forgery prevention
- **Secure Headers** - Configured appropriate security headers

##### Reflection: Problems Encountered and Lessons Learned
A significant challenge was balancing security measures with usability. Some initial implementations created excessive friction in the user experience. We refined our approach to implement security controls that maintain protection while minimizing user impact.

We also discovered vulnerabilities in our file upload processing pipeline, which could potentially allow malformed files to cause system issues. This led us to implement more robust file validation and sandboxed processing to prevent security and stability problems.

---

### Project Progress Summary
**Week 13: April 14 - April 18, 2025**

#### Final Integration and System Testing

##### Activities and Progress
In this final week, we focused on comprehensive testing and integration of all system components. We conducted end-to-end testing across various disaster scenarios, validating the entire workflow from data ingestion to visualization and alerting.

We refined the user interface based on feedback from our test users, improving navigation flow and information presentation. Dashboard components were reorganized to prioritize critical information and streamline common workflows.

Documentation was completed, including technical specifications, API documentation, and user guides. We conducted performance testing under various load conditions, confirming that the system meets all scalability requirements.

##### Techniques, Tools, and Methodologies Used
- **End-to-end Testing** - Validated complete system workflows
- **User Acceptance Testing** - Gathered and incorporated user feedback
- **Performance Benchmarking** - Measured system performance under various conditions
- **Documentation Generation** - Created comprehensive technical and user documentation
- **Regression Testing** - Ensured new features didn't compromise existing functionality
- **API Stability Verification** - Confirmed consistency of all external interfaces

##### Reflection: Problems Encountered and Lessons Learned
The main challenge during this phase was addressing edge cases discovered during comprehensive testing. We identified several rare but significant scenarios where the system behavior wasn't optimal, particularly in handling unusual data patterns and recovery from processing interruptions.

Throughout the project, we learned the importance of early integration testing rather than leaving it to the final phase. Some components that worked well in isolation required modifications when operating together in the complete system. In future projects, we'll implement continuous integration testing from earlier stages to identify these issues sooner.

The PanicSensePH project has successfully achieved its goals of providing advanced disaster monitoring capabilities with intelligent sentiment analysis. The combination of LSTM neural networks and MBERT for multilingual support has proven effective in accurately analyzing disaster-related communications across languages. The system is now ready for deployment and will provide valuable insights for disaster management and community resilience efforts.