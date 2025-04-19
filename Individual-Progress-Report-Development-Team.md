# Technological Institute of the Philippines  
Thesis 2

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 \- January 24, 2025 |

| Initial Project Setup and Architecture Design |
| :---: |

| Activities and Progress | During this first week, our team established the foundation for the disaster monitoring platform. We set up the core project structure using Next.js/React with TypeScript for the frontend and integrated PostgreSQL with Drizzle ORM for the backend. The initial focus was on designing a robust database schema that could accommodate disaster monitoring data while ensuring performance optimization. We created the basic tables including users, sentiment posts, disaster events, and analyzed files. Setting up the authentication system was a priority to ensure secure access to the platform. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We configured Drizzle ORM with PostgreSQL to handle database interactions efficiently. For frontend setup, we implemented shadcn components with Tailwind CSS to ensure a responsive and modern UI. We set up the workflow system to manage the application's running state and established proper project structure for maintainability. |
| **Reflection: Problems Encountered and Lessons Learned** | One challenge was determining the right database schema structure to accommodate real-time disaster monitoring data while ensuring quick retrieval. We initially underestimated the complexity of sentiment analysis data storage, but resolved it by creating a more comprehensive schema with proper relationships. This taught us the importance of thorough planning before implementation, especially for complex data systems that need to scale. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 2 / January 27 \- January 31, 2025 |

| Data Processing Pipeline Development |
| :---: |

| Activities and Progress | This week focused on building the core data processing pipeline for sentiment analysis. We implemented the Python service integration to analyze social media and text data for disaster-related content. The team developed the uploadSession mechanism to handle large file uploads and processing, ensuring persistence across page refreshes. We also created the initial API endpoints for sentiment analysis and file processing, implementing proper error handling and progress tracking to ensure reliability. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We integrated Python with Node.js using child processes for efficient data processing. For handling large datasets, we implemented a chunking mechanism with proper memory management. We used TanStack Query for frontend data fetching with optimistic updates to provide a responsive user experience during long-running operations. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was managing memory usage during large CSV processing. Initial implementations caused memory overflow with large files. We resolved this by implementing a streaming approach with batch processing. This experience highlighted the importance of considering resource constraints when designing data processing systems, and the value of incremental processing for large datasets. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 \- February 7, 2025 |

| AI Model Integration and Training System |
| :---: |

| Activities and Progress | Week 3 saw significant progress in AI integration. We implemented the sentiment analysis model with neural network capabilities for behavioral computing. The team developed a training and feedback system allowing users to correct AI predictions and use these corrections to improve the model's accuracy over time. We created the trainingExamples and sentimentFeedback database tables to store this learning data. Additionally, we built the confidence scoring system to indicate prediction reliability and implemented the initial version of disaster type and location extraction from text. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We utilized natural language processing techniques for sentiment analysis and entity extraction. For the AI training system, we implemented a feedback loop mechanism with proper data validation. We used a caching system to improve performance for repeated analysis requests and implemented proper error handling for AI processing failures. |
| **Reflection: Problems Encountered and Lessons Learned** | A major challenge was balancing AI model accuracy with processing speed. Initial implementations were accurate but too slow for real-time analysis. We optimized by implementing a multi-tier approach: quick analysis for real-time data and deeper analysis for batch processing. This taught us about the trade-offs between speed and accuracy in AI applications and reinforced the importance of designing systems that can adapt to different usage patterns. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 4 / February 10 \- February 14, 2025 |

| Geospatial Analysis and Visualization |
| :---: |

| Activities and Progress | This week focused on implementing geospatial features for the platform. We integrated Leaflet for interactive mapping and built the geographic analysis component to visualize disaster sentiment data by location. We enhanced the location extraction capabilities to identify geographic mentions in text more accurately and implemented clustering for data points to improve map visualization with large datasets. We also created the map filtering system to allow users to focus on specific regions or disaster types. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We implemented Leaflet with React integration for interactive maps and used geographic clustering algorithms for efficient data visualization. For location extraction from text, we implemented both rule-based and machine learning approaches for better accuracy. We used responsive design techniques to ensure maps worked well on different screen sizes. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was accurately extracting and standardizing location data from free text. Many references were ambiguous or used colloquial names. We improved this by implementing a multi-stage approach with validation against known locations. This experience taught us the importance of data cleaning and normalization in geographic information systems, and how combining multiple approaches can lead to better results than relying on a single method. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 \- February 21, 2025 |

| Timeline and Temporal Analysis Features |
| :---: |

| Activities and Progress | Week 5 focused on implementing temporal analysis capabilities. We developed the timeline component to visualize sentiment trends over time and created charting systems to represent emotion changes during disasters. We built filterable time series visualizations to allow users to focus on specific periods and implemented comparison features for multiple disaster events. We also enhanced the database queries to optimize for time-based data retrieval and implemented proper date formatting and timezone handling throughout the application. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We utilized Chart.js for time series visualization with date-fns for proper date handling. For efficient temporal data querying, we implemented database indexes and optimized query patterns. We used responsive design techniques to ensure charts remained usable on different screen sizes and implemented proper data aggregation for viewing different time scales. |
| **Reflection: Problems Encountered and Lessons Learned** | One challenge was handling data aggregation across different time scales (hourly, daily, weekly). Initial implementations were either too granular, causing performance issues, or too aggregated, losing important details. We solved this by implementing adaptive aggregation based on the selected timeframe. This taught us about the importance of dynamic data processing based on user context and reinforced that visualization is not just about displaying data but presenting it in the most meaningful way for the current analysis. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 6 / February 24 \- February 28, 2025 |

| Real-time Analysis and Monitoring System |
| :---: |

| Activities and Progress | This week was devoted to implementing real-time features. We built the WebSocket integration for live updates and notifications and developed the real-time sentiment analysis component for immediate text processing. We implemented the cross-tab synchronization mechanism to ensure consistent state across multiple browser windows and created live progress indicators for long-running operations. We also built the real-time error recovery system to handle connection disruptions gracefully. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used WebSockets for real-time server-client communication and implemented a custom state synchronization protocol for cross-tab consistency. For real-time UI updates, we used React's state management with optimistic updates. We implemented proper error handling and reconnection logic for network disruptions. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was maintaining consistent state across multiple tabs while avoiding race conditions. Initial implementations caused data inconsistencies when multiple tabs made updates simultaneously. We resolved this by implementing a primary tab coordination system with proper locking mechanisms. This experience highlighted the complexity of distributed state management in web applications and taught us the importance of designing for concurrency from the beginning. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 \- March 7, 2025 |

| Comparative Analysis and Reporting |
| :---: |

| Activities and Progress | Week 7 focused on building comparative analysis capabilities. We implemented the side-by-side comparison features for multiple disaster events and created the evaluation metrics system to assess sentiment analysis accuracy. We built report generation features for exporting analysis results and developed visualization components for comparing sentiment patterns across different events or timeframes. We also implemented the initial version of the dashboard with key performance indicators and summary statistics. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used advanced charting libraries for comparative visualization and implemented statistical analysis methods for evaluation metrics. For report generation, we created a structured data export system with CSV formatting. We used React's component composition patterns to create reusable comparison widgets. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was designing meaningful comparisons between datasets with different characteristics (size, timeframe, context). Initial implementations made direct comparisons that were potentially misleading. We improved this by incorporating normalization and context-aware comparisons. This taught us the importance of statistical rigor in data analysis applications and how presentation can significantly impact interpretation. Providing proper context and normalization is critical for meaningful comparative analysis. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 \- March 14, 2025 |

| User Interface Refinement and Dashboard Development |
| :---: |

| Activities and Progress | This week was dedicated to refining the user interface and building a comprehensive dashboard. We implemented the main dashboard with summarized analytics and key statistics and created customizable dashboard widgets for user-specific monitoring needs. We refined the navigation system for intuitive application flow and improved form designs for better user experience during data input. We also implemented responsive layouts to ensure the application worked well on different devices and screen sizes. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used shadcn components with Tailwind CSS for consistent UI elements and implemented responsive design patterns for multi-device support. For the dashboard, we used modular component architecture with dynamic loading. We conducted internal usability testing to identify and resolve navigation issues. |
| **Reflection: Problems Encountered and Lessons Learned** | A challenge was balancing information density with clarity in the dashboard design. Initial designs were either too cluttered with data or too simplified to be useful. We improved this by implementing progressive disclosure patterns and user-customizable layouts. This experience reinforced that effective data visualization is about finding the right balance between comprehensive information and cognitive load, and that user preferences for information presentation can vary significantly. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 9 / March 17 \- March 21, 2025 |

| Authentication and User Management |
| :---: |

| Activities and Progress | Week 9 focused on enhancing the authentication and user management system. We implemented role-based access control with different permission levels and created the user profile management features with customization options. We built secure session handling with proper token management and implemented password reset and account recovery functionality. We also created user activity logging for audit purposes and implemented profile image upload and management. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used bcrypt for secure password handling and implemented JWT-based session management. For file uploads, we used multer with proper validation and secure storage. We implemented a comprehensive permission system using role-based checks throughout the application. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was designing a flexible permission system that could accommodate different user roles without making the codebase overly complex. Initial implementations were either too rigid or had too many special cases. We improved this by implementing a capability-based approach rather than just role checks. This experience taught us about the balance between security and usability in authentication systems, and how thoughtful permission design can make applications both secure and maintainable. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 10 / March 24 \- March 28, 2025 |

| Data Export and Integration Capabilities |
| :---: |

| Activities and Progress | This week was dedicated to implementing data export and integration features. We built the CSV export functionality for analysis results and created the API endpoints for external system integration. We implemented data filtering and selection for customized exports and developed the export scheduling system for automated reporting. We also created documentation for the API to facilitate integration with other systems. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We implemented streaming CSV generation for efficient large data exports and created RESTful API patterns for consistent external access. For scheduled exports, we developed a task scheduling system with proper error handling. We used OpenAPI specifications for clear API documentation. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was handling large data exports without causing server memory issues. Initial implementations loaded all data into memory before creating the export file, causing out-of-memory errors with large datasets. We resolved this by implementing streaming exports that processed data in chunks. This reinforced the importance of considering resource constraints in data-intensive operations and how streaming approaches can solve scaling problems that batch processing cannot. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 11 / March 31 \- April 4, 2025 |

| Performance Optimization and Scaling |
| :---: |

| Activities and Progress | Week 11 focused on performance optimization throughout the application. We implemented database query optimization for faster data retrieval and created client-side caching strategies to reduce server load. We built the usage tracking system to monitor system resource utilization and implemented batch processing for resource-intensive operations. We also conducted comprehensive performance testing to identify and resolve bottlenecks. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used database indexing and query optimization techniques for improved performance and implemented React Query's caching capabilities for efficient data management. For performance testing, we created automated tests with metrics collection. We used profiling tools to identify CPU and memory bottlenecks. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was optimizing the sentiment analysis process, which was initially too slow for large datasets. We improved this by implementing parallel processing and smarter caching strategies. This experience highlighted that performance optimization is often about identifying the specific bottlenecks rather than general improvements, and that measuring actual performance with real workloads is essential for effective optimization. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 12 / April 7 \- April 11, 2025 |

| Security Enhancements and Error Handling |
| :---: |

| Activities and Progress | This week was dedicated to security improvements and robust error handling. We implemented comprehensive input validation across all endpoints and created the advanced error handling system with useful feedback for users. We built security measures against common web vulnerabilities (XSS, CSRF, SQL injection) and implemented rate limiting to prevent abuse. We also conducted security testing and fixed identified vulnerabilities. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used Zod schemas for comprehensive input validation and implemented error boundary components for graceful failure handling. For security, we used proper escaping and parameterized queries throughout. We conducted automated security scanning with manual verification of results. |
| **Reflection: Problems Encountered and Lessons Learned** | A challenge was balancing comprehensive error handling with meaningful user feedback, as too detailed errors could expose system details. We solved this by implementing a layered approach: detailed logging for developers and user-friendly messages for end users. This experience reinforced that security is a system-wide concern that needs to be addressed at multiple levels, and that good error handling is as much about user experience as it is about system stability. |

---

## INDIVIDUAL PROGRESS REPORT

| Name | Development Team |
| :---: | :---: |
| **Role** | Full-Stack Development |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 \- April 19, 2025 |

| Final Testing and Documentation |
| :---: |

| Activities and Progress | In the final week, we focused on comprehensive testing and documentation. We conducted end-to-end testing of all major application flows and finalized user documentation with guides and tutorials. We created technical documentation for system architecture and APIs and prepared the final deployment configuration. We also implemented final UI polish based on user feedback and conducted a systematic review of all features to ensure completeness. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used systematic testing methodologies covering unit, integration, and end-to-end tests. For documentation, we created structured documentation with clear examples and visual aids. We implemented a final code review process with focus on quality and maintainability. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was ensuring that all features worked together cohesively after individual development. We found some integration issues that weren't apparent during feature development. We resolved these by implementing more comprehensive integration testing. This final phase taught us the importance of early integration testing alongside feature development, and how documentation forces a comprehensive review of the system, often uncovering overlooked details or inconsistencies. |
