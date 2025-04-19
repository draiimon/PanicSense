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

This week, I focused on establishing the foundational architecture for our PanicSensePH disaster monitoring platform. I successfully designed the initial system architecture diagram showing data flow between frontend React components, Express backend services, and PostgreSQL database. This architecture emphasizes the real-time data processing capabilities needed for disaster response.

I conducted research on state-of-the-art sentiment analysis approaches, particularly focusing on LSTM and MBERT applications in disaster management contexts. Based on this research, I drafted preliminary specifications for our sentiment analysis pipeline that will form the core of our platform.

Additionally, I implemented the initial database schema using Drizzle ORM, defining all essential tables for storing sentiment posts, disaster events, and user data. This schema design ensures proper normalization while maintaining compatibility with our planned sentiment analysis components.



Techniques, Tools, and Methodologies Used

I utilized PostgreSQL with Drizzle ORM for database design, implementing normalized schemas for disaster-related data. TypeScript provided type safety throughout the system, particularly for data model definitions. For backend implementation, I used Express.js to establish RESTful API endpoints for data exchange.

For frontend development, I implemented React with Vite for an efficient development environment with hot module replacement. The development workflow was structured using Git Flow methodology to facilitate parallel development across team members.



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was designing a schema that could accommodate both structured data from official disaster reports and unstructured content from social media sources. I learned that planning for future expansion from the beginning is crucial, particularly regarding the sentiment analysis models we'll implement.

I also encountered difficulties determining the optimal approach for multilingual support, as our system needs to handle both English and Filipino text, often with code-switching. Through research, I identified MBERT as a promising solution for this challenge, which will be implemented in upcoming sprints.





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

This week, I focused on setting up our development environment and implementing the core infrastructure components. I established our Git repository with proper branching strategy and configured the CI/CD pipeline for automated testing and deployment. This ensures that all team members can collaborate effectively while maintaining code quality.

I implemented the initial backend structure using Express.js with TypeScript, creating the foundational API endpoints for sentiment analysis and disaster monitoring. For database connectivity, I configured Drizzle ORM to work with our PostgreSQL instance, implementing proper migrations and type safety throughout the data layer.

On the frontend side, I set up the React application with Vite, implementing the basic layout components and navigation structure. I also created reusable UI components using Tailwind CSS that will be used throughout the application, ensuring a consistent user experience.



Techniques, Tools, and Methodologies Used

For version control and collaboration, I utilized Git with a feature branch workflow to maintain clean development history. Drizzle ORM provided type-safe database access with migration capabilities for PostgreSQL. React with Vite offered an efficient frontend development environment with hot module replacement.

The development process followed an Agile methodology with weekly sprints and daily stand-ups to ensure everyone stays aligned. For API development, I implemented RESTful principles with proper error handling and response formatting.



Reflection: Problems Encountered and Lessons Learned

The main challenge was ensuring consistency across different development environments, as some team members encountered issues with package versions and database connections. I addressed this by implementing Docker containers for development, providing a standardized environment for all team members.

Another difficulty was balancing between rapid prototyping and establishing a solid foundation. I learned that investing time in proper architecture and infrastructure early on, while seemingly slower initially, pays significant dividends as the project grows in complexity.





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

This week, I focused on implementing the data collection and processing system for PanicSensePH. I developed a robust ETL (Extract, Transform, Load) pipeline that can ingest disaster-related content from various sources, including social media platforms and official government announcements. The system efficiently processes incoming data, extracts relevant information, and stores it in our PostgreSQL database.

I also implemented the file upload functionality that allows users to import CSV files containing historical disaster data. This feature includes validation, parsing, and batch processing capabilities to handle large datasets efficiently. The backend now provides detailed progress tracking and error handling during the import process.

Additionally, I created the foundation for our data preprocessing module, which normalizes text data, removes irrelevant content, and prepares it for sentiment analysis. This preprocessing step is crucial for ensuring accurate results from our LSTM and MBERT models.



Techniques, Tools, and Methodologies Used

For the ETL pipeline, I utilized Node.js streams to efficiently process large datasets with minimal memory consumption. For CSV parsing and validation, I implemented a custom solution using the csv-parse library with robust error handling. The data preprocessing module leverages natural language processing techniques for text normalization and cleaning.

I also employed unit testing with Jest to ensure the reliability of these critical components, achieving over 85% test coverage for the data processing modules. For manual testing, I created sample datasets of various sizes and formats to verify system behavior under different conditions.



Reflection: Problems Encountered and Lessons Learned

A significant challenge was handling the variety of data formats and quality issues present in real-world datasets. I learned the importance of robust validation and preprocessing to handle edge cases like incomplete records, special characters, and inconsistent formatting.

Performance issues emerged when processing very large files, which I addressed by implementing a chunking strategy that processes data in manageable batches. This approach not only improved performance but also enabled better progress tracking and error recovery, enhancing the overall user experience.





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

This week, I focused on implementing the authentication system and user management functionality for PanicSensePH. I developed a secure authentication mechanism using JSON Web Tokens (JWT) with proper refresh token rotation for maintaining secure sessions. The system now supports user registration, login, password recovery, and session management.

I implemented role-based access control (RBAC) to differentiate between regular users, analysts, and administrators, ensuring that each user type has appropriate permissions for their responsibilities. This included creating middleware for route protection and implementing fine-grained access control throughout the API.

Additionally, I developed the user profile management system, including functionality for updating personal information, changing passwords, and managing notification preferences. The system now also supports profile image upload and management using secure file handling practices.



Techniques, Tools, and Methodologies Used

For secure authentication, I utilized bcrypt for password hashing and JWT for stateless authentication. User data validation was implemented using Zod schemas to ensure proper data integrity. For file uploads, I used Multer with appropriate validation and sanitization to prevent security issues.

The frontend authentication flow was implemented using React Query and custom hooks to manage authentication state throughout the application. I also created protected route components to handle authentication-based navigation restrictions on the client side.



Reflection: Problems Encountered and Lessons Learned

The main challenge was implementing a secure yet user-friendly authentication system. Balancing security requirements like password complexity, token expiration, and CSRF protection with usability proved difficult. I addressed this by implementing progressive security measures that don't unnecessarily burden users while maintaining strong protection against common attacks.

Another issue was handling authentication state across multiple components and pages. I learned the importance of centralizing authentication logic in custom hooks and contexts to prevent inconsistencies and provide a single source of truth for authentication status throughout the application.





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

This week, I focused on implementing the initial version of our LSTM-based sentiment analysis model for PanicSensePH. I developed a bidirectional LSTM architecture specifically designed for processing disaster-related text sequences. The model analyzes contextual information from both directions in text, significantly improving detection of sentiment nuances in emergency communications.

I created a training pipeline using a preliminary dataset of 1,000 disaster-related social media posts that I manually annotated with our five sentiment categories (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral). Initial validation showed promising results with approximately 68% accuracy, with particularly strong performance in detecting Panic and Resilience categories.

Additionally, I integrated this model with our backend API, creating endpoints to process both individual text inputs and batch analysis for CSV uploads. This integration now enables our platform to provide real-time sentiment analysis for incoming disaster-related content.



Techniques, Tools, and Methodologies Used

I implemented the bidirectional LSTM neural network using TensorFlow, configuring the architecture with 300-dimensional word embeddings to capture disaster terminology semantics effectively. For handling Filipino text, I began integrating MBERT components using PyTorch, creating a hybrid approach that leverages both frameworks' strengths.

I employed ensemble learning techniques to combine predictions from both models, implementing a weighted voting mechanism that significantly improved accuracy for multilingual text. For efficiently processing large datasets, I developed a batch processing system that handles data in manageable chunks.



Reflection: Problems Encountered and Lessons Learned

The main challenge was achieving acceptable accuracy for multilingual text, as our dataset contains both English and Filipino content with frequent code-switching. The initial LSTM implementation performed well on English text but struggled with Filipino expressions. By integrating MBERT for language detection and specialized tokenization, I was able to improve handling of multilingual content.

I also discovered performance bottlenecks during batch processing of large files, which I addressed by implementing a chunking strategy. This approach not only improved processing efficiency but also enabled progress tracking for better user experience.





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

This week, I focused on integrating geographic analysis and visualization features into PanicSensePH. I developed a location extraction system that identifies place mentions in text data, mapping them to standardized geographic coordinates. This system now accurately detects locations mentioned in disaster-related content with approximately 75% accuracy.

I implemented an interactive map visualization using Leaflet.js that displays sentiment data across different regions. The map interface includes heat maps showing concentration of different sentiment categories, allowing users to identify areas with high levels of panic or distress during disaster events. Users can now filter by time period, sentiment type, and specific disaster events.

Additionally, I created a database of Philippine administrative boundaries for provinces, cities, and municipalities, enabling precise geographic classification of disaster-related content. This geographic database includes population data and risk factors for different regions, providing valuable context for sentiment analysis.



Techniques, Tools, and Methodologies Used

For geographic visualization, I utilized Leaflet.js with custom map layers and markers to create an interactive interface. The location extraction system combines rule-based pattern matching with a gazetteer of Philippine place names to identify and normalize location mentions in text.

I implemented GeoJSON processing for handling administrative boundaries and created specialized database indexes for efficient spatial queries. For the user interface, I developed custom React components for the map visualization with responsive design for different device sizes.



Reflection: Problems Encountered and Lessons Learned

A significant challenge was disambiguating place names that could refer to multiple locations in the Philippines. I addressed this by implementing context-aware resolution that considers surrounding text and frequency analysis to identify the most likely location based on the context.

Another issue was optimizing the performance of map rendering with large datasets. Initially, displaying thousands of data points caused significant performance degradation. I implemented clustering strategies and progressive loading techniques to maintain smooth interactions even with large volumes of data.





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

This week, I focused on implementing real-time monitoring capabilities for PanicSensePH. I developed a WebSocket-based communication system that enables live updates between the server and clients. This system now broadcasts notifications about newly analyzed posts, processing progress updates, and system status changes to connected users.

I created a notification center on the frontend that displays real-time alerts about significant sentiment pattern changes, such as sudden increases in panic-related posts for specific locations. These alerts can indicate emerging disaster situations, enabling earlier response from authorities.

Additionally, I implemented a message queue system using Redis to ensure reliable delivery of updates even during periods of network instability. The system now maintains connection state and can recover missed updates when clients reconnect, ensuring users always have the latest information.



Techniques, Tools, and Methodologies Used

For real-time communication, I used WebSocket protocol with a custom message format for efficient data transfer. The Redis-based message queue ensures reliable message delivery and provides persistence for critical updates. On the frontend, I implemented React Query for seamless integration of real-time data with the application state.

I also employed throttling and debouncing techniques to prevent UI performance issues during high-volume update periods. For connection management, I implemented an exponential backoff strategy for reconnection attempts to handle network instability gracefully.



Reflection: Problems Encountered and Lessons Learned

The primary challenge was maintaining WebSocket connection stability, particularly during periods of network fluctuation. I learned the importance of implementing robust reconnection strategies with proper error handling to provide a seamless experience for users even under suboptimal network conditions.

Another issue was managing the increased server load from maintaining numerous concurrent WebSocket connections. By implementing connection pooling and efficient broadcast mechanisms, I was able to significantly reduce the resource requirements while maintaining responsive real-time updates.





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

This week, I focused on implementing disaster type classification and event correlation features for PanicSensePH. I developed a specialized classifier that categorizes posts into specific disaster types, including typhoons, earthquakes, floods, fires, and volcanic eruptions, with approximately 82% accuracy.

I created a correlation engine that identifies relationships between sentiment patterns and disaster types across different locations. This system now can detect emerging disaster events based on sentiment clusters and provide early warnings to users. The correlation algorithm considers both spatial and temporal factors to identify related posts that likely refer to the same event.

Additionally, I developed a disaster events dashboard that displays active and historical events, along with sentiment trend analysis for each. This interface allows users to track how public sentiment evolves throughout the course of a disaster, providing valuable insights for response planning and communication strategies.



Techniques, Tools, and Methodologies Used

For disaster classification, I implemented a multi-label classification system using TensorFlow with specialized features for disaster-related terminology. The correlation engine utilizes temporal pattern analysis with sliding window algorithms to detect trends and anomalies in sentiment data over time.

I implemented clustering techniques using both spatial and temporal dimensions to group related content. The visualization components were created using Recharts for interactive time-series displays that show sentiment evolution during disaster events.



Reflection: Problems Encountered and Lessons Learned

A significant challenge was distinguishing between discussions about past disasters and reports of current events. I addressed this by implementing temporal context detection that analyzes time indicators in the text and correlates them with posting timestamps to determine if a post refers to an ongoing or historical event.

I also encountered difficulties in handling reports of multiple simultaneous disasters, such as flooding caused by typhoons. This led me to implement a hierarchical classification approach that can identify primary and secondary disaster types from a single post, providing more nuanced analysis.





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

This week, I focused on enhancing the multilingual capabilities of PanicSensePH through deeper integration of Multilingual BERT (MBERT). I implemented specialized MBERT-based tokenization that properly processes mixed-language text, preserving semantic meaning across language boundaries. This integration significantly improved our system's ability to handle code-switching between English and Filipino, which is common in disaster-related social media posts.

I developed a language-specific sentiment lexicon for Filipino disaster terminology, providing better context for sentiment analysis in culturally specific expressions. Testing showed a 15% improvement in accuracy for mixed-language posts, bringing our overall system performance to 81% accuracy across all language patterns.

Additionally, I created a seamless pipeline that combines our existing LSTM model with the new MBERT components, creating a hybrid approach that leverages the strengths of both architectures. The system now dynamically selects the appropriate model based on text characteristics, optimizing for both accuracy and performance.



Techniques, Tools, and Methodologies Used

I implemented cross-lingual word embeddings that align semantic spaces between English and Filipino, enabling more accurate representation of mixed-language content. For MBERT fine-tuning, I used PyTorch with a specialized training regimen focused on disaster-related terminology in both languages.

I employed transfer learning techniques to apply knowledge from our larger English corpus to improve Filipino processing, addressing the imbalance in available training data. For performance optimization, I implemented model quantization and caching strategies to reduce resource requirements while maintaining accuracy.



Reflection: Problems Encountered and Lessons Learned

The biggest challenge was handling Filipino dialectal variations, which significantly affected sentiment detection accuracy. I addressed this by incorporating regional language patterns into our training data and implementing fuzzy matching for dialectal variants, which improved detection of region-specific expressions.

I also discovered that MBERT sometimes struggled with very short text fragments common in social media. This led me to implement a hybrid approach that falls back to our specialized LSTM for very short texts while using MBERT for longer, more complex content, optimizing performance across different content types.





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

This week, I focused on implementing a feedback mechanism for PanicSensePH that allows users to correct sentiment classifications, creating a continuous learning system. I developed a user interface for submitting corrections, which are stored in a dedicated feedback table in the database for further analysis and model improvement.

I created a training pipeline that incorporates user feedback to refine our models over time. This closed-loop system ensures continuous improvement based on real-world usage patterns. Initial testing shows that incorporating just 100 feedback instances improved overall accuracy by 3%, demonstrating the value of human-in-the-loop learning.

Additionally, I implemented a confidence threshold system that automatically flags low-confidence predictions for human review. This creates a semi-supervised learning environment that prioritizes human intervention where it's most needed, optimizing the use of expert time while maximizing improvements to the system.



Techniques, Tools, and Methodologies Used

I implemented active learning strategies to intelligently select samples for human annotation, focusing on cases where model uncertainty is highest. For incremental model updating, I developed a technique that incorporates new feedback without requiring full retraining, enabling continuous improvement with minimal computational overhead.

I created a prioritization system that identifies the most valuable correction opportunities based on confidence scores and potential impact on overall system performance. The feedback interface was implemented using React with optimistic updates for a responsive user experience.



Reflection: Problems Encountered and Lessons Learned

A significant challenge was balancing the incorporation of user feedback with maintaining model stability. Early implementations sometimes overfit to recent corrections, reducing general performance. I resolved this by implementing a weighted feedback integration approach that preserves global model performance while still learning from new examples.

I also discovered the importance of tracking feedback patterns to identify systematic model weaknesses. This led me to implement an analytics dashboard specifically for monitoring correction patterns, which has been invaluable for guiding targeted improvements to the sentiment analysis system.





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

This week, I focused on improving the performance and scalability of PanicSensePH. I conducted comprehensive profiling of both frontend and backend components to identify bottlenecks and optimization opportunities. Based on the results, I implemented targeted improvements that reduced average response times by 65% for common operations.

I optimized database queries through proper indexing and query restructuring, significantly improving data retrieval speed for large datasets. The frontend was enhanced with virtualized lists and lazy loading for more efficient rendering of large sentiment data collections, providing smooth scrolling even with thousands of records.

Additionally, I implemented a caching layer for frequently accessed and compute-intensive results, particularly for sentiment analysis of common phrases. This reduced redundant processing and improved overall system responsiveness. Load testing confirms the system can now handle 3x the previous concurrent user load without performance degradation.



Techniques, Tools, and Methodologies Used

I utilized database indexing strategies targeted at our most common query patterns, improving performance for sentiment data retrieval by location and time period. Query optimization techniques included denormalization where appropriate and restructuring complex joins for efficiency.

For frontend optimization, I implemented React component memoization and code splitting to reduce unnecessary re-renders and initial load times. I also developed a sophisticated caching strategy using a combination of memory caching for frequent requests and persistent caching for compute-intensive operations.



Reflection: Problems Encountered and Lessons Learned

The main challenge was maintaining accuracy while improving performance. Initial optimization attempts reduced processing time but slightly decreased model accuracy. I addressed this by implementing a tiered approach that uses fast, approximate methods for initial results followed by more precise analysis where needed, preserving both speed and accuracy.

I also discovered that database connection management was causing performance degradation during high load. This led me to implement a connection pooling strategy that significantly improved stability under heavy concurrent usage, an essential improvement for disaster scenarios when many users might access the system simultaneously.





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

This week, I focused on enhancing the security posture of PanicSensePH. I conducted a comprehensive security audit, identifying and addressing potential vulnerabilities throughout the system. The improvements included implementing enhanced input validation and sanitization to prevent injection attacks, particularly for user-submitted content and file uploads.

I strengthened the authentication system with rate limiting, account lockout capabilities, and improved password policies to protect against brute force attacks. I also implemented Cross-Site Request Forgery (CSRF) protection and proper Content Security Policy (CSP) headers to mitigate common web security threats.

Additionally, I improved data protection through implementation of field-level encryption for sensitive information and proper anonymization of personally identifiable information in exported datasets. I created a detailed audit logging system that tracks all significant system actions for security monitoring and compliance purposes.



Techniques, Tools, and Methodologies Used

For input validation, I implemented comprehensive Zod schemas with strict validation rules for all user inputs. Security headers were configured according to OWASP best practices, including proper CSP directives to prevent XSS attacks. Rate limiting was implemented using Redis-based token buckets for reliable request throttling.

I conducted security testing using both automated vulnerability scanners and manual penetration testing techniques to identify potential weak points. For encryption, I implemented AES-256 for field-level protection of sensitive data, with proper key management practices.



Reflection: Problems Encountered and Lessons Learned

A significant challenge was balancing security measures with usability. Some initial implementations created excessive friction in the user experience, such as overly strict password requirements and frequent re-authentication. I refined our approach to implement security controls that maintain protection while minimizing user impact.

I also discovered vulnerabilities in our file upload processing pipeline, which could potentially allow malformed files to cause system issues. This led me to implement more robust file validation and sandboxed processing to prevent security and stability problems, ensuring that malicious uploads cannot compromise the system.





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

This week, I focused on comprehensive testing and final integration of all PanicSensePH components. I conducted end-to-end testing across various disaster scenarios, validating the entire workflow from data ingestion to visualization and alerting. The system successfully processed simulated high-volume data (5,000+ records per minute) while maintaining 92% uptime with minimal performance degradation.

I implemented final optimizations to our bidirectional LSTM model, reducing memory consumption by 38% through tensor operation improvements and more efficient batching strategies. These optimizations were particularly important for ensuring consistent performance in production environments with memory constraints.

Additionally, I documented the complete ML pipeline with detailed architecture diagrams, data flow explanations, and performance benchmarks. This technical documentation includes API specifications, database schema details, and deployment instructions to ensure the system can be properly maintained and extended in the future.



Techniques, Tools, and Methodologies Used

I utilized TensorFlow for final hyperparameter tuning of our bidirectional LSTM model, focusing on optimizing for both accuracy and computational efficiency. For multilingual support, I refined the MBERT implementation in PyTorch, particularly enhancing performance for Filipino-English code-switching patterns.

I implemented comprehensive cross-validation testing using k-fold validation to verify model stability across different data subsets. For system-level testing, I developed automated test scripts that simulate real-world usage patterns and verify correct behavior across all components.



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was reconciling performance differences between our testing and production environments. Our initial production deployment showed slightly lower accuracy compared to testing results, primarily due to memory constraints. I resolved this by implementing more efficient tensor operations and an improved batching strategy that maintained accuracy while reducing resource requirements.

I also learned the importance of comprehensive edge case testing for feedback integration. Some patterns of feedback were initially causing model overfitting to specific examples, reducing general performance. By implementing a more sophisticated weighting mechanism that considers feedback frequency and confidence scores, we were able to improve the system without introducing bias.

Through this project, I've gained valuable experience in combining robust machine learning techniques with efficient implementation strategies for real-world disaster response applications. The PanicSensePH platform demonstrates how sentiment analysis can provide critical insights during emergencies, potentially improving response coordination and public communication during disasters.