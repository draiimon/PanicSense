Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 \- January 24, 2025 |

| Project Setup and Database Design |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | During the first week of development, we focused on setting up the core project structure and designing the database schema. We created a comprehensive PostgreSQL database schema with tables for users, sentiment analysis posts, disaster events, analyzed files, and profile images. We implemented authentication tables with proper security considerations and set up the Drizzle ORM for database interactions. Additionally, we configured the project with TypeScript, established the frontend using Next.js/React, and set up the development workflow for the team. We also implemented the basic Express server structure to handle API requests and configured the environment for development. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We utilized PostgreSQL for the database with Drizzle ORM for database interactions. For the schema design, we employed a systematic approach, defining relationships between tables and implementing proper validation using Zod schemas. We set up TypeScript configurations for type safety throughout the project and implemented the project structure following modern best practices for fullstack JavaScript applications. |
| **Reflection: Problems Encountered and Lessons Learned** | One challenge we faced was designing an efficient database schema that could handle the complex requirements of disaster monitoring data. We had to carefully consider relationships between entities and how to structure sentiment analysis data for optimal retrieval. Another challenge was setting up the TypeScript configuration to work seamlessly with both frontend and backend code. Through these challenges, we learned the importance of thorough planning before implementation, especially for complex data systems. We also recognized the value of establishing strong foundational patterns early in the project to ensure consistency throughout development. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 2 / January 27 \- January 31, 2025 |

| API Development and Authentication System |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | In our second week, we developed the core API endpoints for the platform. We implemented user authentication with signup, login, and session management using secure practices. We created routes for managing sentiment posts, disaster events, and analyzed files. We implemented proper error handling and validation for all endpoints using Zod schemas to ensure data integrity. We also established the storage interface in server/storage.ts to abstract database operations and ensure consistency across the application. The API endpoints were tested extensively to ensure reliability and correct operation. We also started work on the file upload system for CSV processing, implementing the initial version of the upload functionality with progress tracking. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used Express.js for API development and implemented authentication with bcrypt for password hashing and JWT tokens for session management. We utilized Zod for input validation throughout the API and implemented standard RESTful patterns for endpoint design. We also used multer for handling file uploads with proper validation and storage. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was implementing secure authentication with proper session management. We initially had issues with token validation and session persistence. Another difficulty was designing the storage interface to be flexible enough for future development while remaining simple to use. Through these challenges, we learned the importance of thorough testing for authentication systems and the value of well-designed abstractions for database operations. We also realized the importance of implementing proper error handling from the beginning to make debugging easier during development. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 \- February 7, 2025 |

| Python Integration and Sentiment Analysis Pipeline |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | This week, we focused on implementing the Python service for sentiment analysis. We developed the PythonService class to manage communication between Node.js and Python processes, implementing proper process management and error handling. We created the sentiment analysis pipeline that can process text and CSV data, extract sentiment, disaster types, and locations from text. We implemented a caching system for confidence scores to improve performance and built the ProcessCSV function to handle bulk data processing. We also developed the uploadSession mechanism to track progress of long-running operations and ensure data persistence. Additionally, we started work on the training system by implementing the trainingExamples and sentimentFeedback tables and creating the initial API endpoints for sentiment feedback. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used Node.js child processes for Python integration and implemented proper inter-process communication with error handling. For sentiment analysis, we utilized natural language processing techniques and implemented a structured pipeline for text processing. We also used file system operations for temporary file management during processing and implemented caching strategies for improved performance. |
| **Reflection: Problems Encountered and Lessons Learned** | A major challenge was managing the communication between Node.js and Python processes reliably. We encountered issues with process termination and error handling that required careful implementation. Another challenge was efficiently processing large CSV files without causing memory issues. We learned the importance of proper resource management when dealing with cross-language integration and the value of implementing progress tracking for long-running operations. We also recognized the need for careful error handling when dealing with external processes and file system operations. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 4 / February 10 \- February 14, 2025 |

| Frontend Components and Dashboard Development |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | During week 4, we focused on developing the frontend components and the main dashboard. We implemented the main application layout with navigation and responsive design using shadcn components and Tailwind CSS. We created the dashboard page with key statistics and visualizations of sentiment data. We developed components for displaying sentiment posts, disaster events, and analyzed files. We implemented the upload interface with progress tracking and cancellation capability. We also created forms for user authentication, sentiment analysis, and disaster event creation with proper validation using react-hook-form and zod. We set up the TanStack Query configuration for efficient data fetching and state management. Additionally, we started work on the real-time features by implementing the initial WebSocket connection for live updates. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used React with TypeScript for component development and shadcn UI components with Tailwind CSS for styling. We implemented TanStack Query for data fetching and state management with proper caching strategies. For forms, we used react-hook-form with zod validation and implemented responsive design patterns for multi-device support. We also used WebSocket for real-time updates and notifications. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was designing a responsive UI that worked well on different devices while presenting complex data effectively. We also faced difficulties in managing form state with validation for complex forms. Through these challenges, we learned the importance of component reusability and proper state management in React applications. We also recognized the value of implementing consistent UI patterns throughout the application to improve usability. The implementation of real-time features revealed the complexity of maintaining consistent state across multiple communication channels. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 \- February 21, 2025 |

| Data Visualization and Geographic Analysis |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | In week 5, we focused on implementing data visualization and geographic analysis features. We integrated Leaflet for interactive mapping and developed the GeographicAnalysis component to visualize sentiment data by location. We implemented clustering for map data points to improve visualization with large datasets. We created chart components for sentiment trend analysis using Chart.js, with proper time series handling. We developed the Timeline component to visualize sentiment changes over time with filtering capabilities. We also implemented the Comparison component for side-by-side analysis of different disaster events. Additionally, we enhanced the Python service to improve location extraction from text, using both rule-based and machine learning approaches for better accuracy. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used Leaflet with React integration for interactive maps and implemented geographic clustering algorithms for efficient data visualization. For chart creation, we utilized Chart.js with proper date handling using date-fns. We implemented advanced CSS techniques for creating responsive and interactive visualizations and used React's context API for sharing visualization state across components. |
| **Reflection: Problems Encountered and Lessons Learned** | A major challenge was efficiently rendering large datasets on maps without performance issues. We initially faced slowdowns with large numbers of markers, which we resolved by implementing clustering. Another challenge was creating time-series visualizations that worked well with irregular data points. Through these challenges, we learned the importance of data processing before visualization and the value of implementing proper aggregation techniques. We also recognized the need for careful performance optimization when dealing with large datasets in interactive visualizations. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 6 / February 24 \- February 28, 2025 |

| Real-time Features and Cross-tab Synchronization |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | During the sixth week, we focused on implementing real-time features and cross-tab synchronization. We developed the complete WebSocket integration for live updates, implementing proper connection management and error handling. We created the real-time sentiment analysis component for immediate text processing with instant feedback. We implemented the upload progress modal that stays visible across page navigations, providing real-time status updates. We built the cross-tab synchronization mechanism using browser storage and WebSocket events to ensure consistent state across multiple tabs. We also developed the notification system for alerts about new data and system events. Additionally, we improved error recovery for network disruptions and implemented proper reconnection logic. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used WebSocket for real-time communication with reconnection handling and implemented custom protocols for different types of real-time events. We utilized browser storage (localStorage) combined with storage events for cross-tab communication and implemented optimistic UI updates for improved user experience during real-time operations. We also created custom React hooks for real-time state management. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was maintaining consistent state across multiple tabs while handling concurrent updates. We faced race conditions when multiple tabs tried to update the same data simultaneously. Another difficulty was ensuring that real-time features gracefully degraded when connections were unstable. Through these challenges, we learned the importance of designing for distributed state management from the beginning and the value of implementing proper locking mechanisms. We also recognized the complexity of real-time systems and the need for comprehensive error handling to provide a good user experience despite network issues. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 \- March 7, 2025 |

| AI Model Training and Feedback System |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Week 7 was dedicated to enhancing the AI capabilities and implementing the feedback system. We developed the complete feedback mechanism for correcting AI predictions, with UI components for submitting corrections. We implemented the trainModelWithFeedback function to incorporate user feedback into the AI model. We created the analyzeSimilarityForFeedback function to detect similar texts and apply consistent sentiment analysis. We built the training example management system with API endpoints for creating and updating examples. We also enhanced the sentiment analysis algorithm to use the training data for improved accuracy. Additionally, we implemented confidence scoring with proper visualization to indicate prediction reliability. We also started work on evaluation metrics to assess the AI model's performance over time. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We utilized machine learning techniques for sentiment analysis and similarity detection. We implemented a feedback loop system with data validation and storage. For confidence visualization, we created custom UI components with color coding and tooltips. We also used statistical methods for calculating evaluation metrics such as accuracy, precision, and recall. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was creating an effective training system that could improve the AI model without requiring massive amounts of feedback data. We also faced difficulties in accurately calculating similarity between texts to ensure consistent analysis. Through these challenges, we learned about the importance of designing AI systems that can learn incrementally from limited feedback and the value of combining rule-based and machine learning approaches. We also recognized the importance of transparent AI systems that communicate confidence levels to users and allow for corrections. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 \- March 14, 2025 |

| Evaluation and Data Export Features |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | During the eighth week, we focused on implementing evaluation and data export features. We developed the Evaluation component for assessing sentiment analysis accuracy with visualization of metrics. We implemented CSV export functionality for analyzed data, with proper formatting and header information. We created filtering options for customized data exports based on date ranges, sentiment types, or disaster categories. We built the metrics calculation system to evaluate model performance (accuracy, precision, recall, F1 score). We also implemented the RawData component for viewing and filtering the complete dataset with pagination and search functionality. Additionally, we enhanced the database queries to optimize for exporting large datasets efficiently. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We implemented streaming CSV generation for efficient large data exports and created custom React components for displaying evaluation metrics. For metrics calculation, we used statistical methods implemented in TypeScript. We also developed custom data filtering and aggregation functions for flexible data selection. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was implementing efficient data export for large datasets without causing memory issues. We initially loaded all data into memory, which caused problems with large datasets. We resolved this by implementing streaming exports. Another challenge was designing meaningful evaluation metrics that accurately reflected the AI model's performance. Through these challenges, we learned about the importance of efficient data processing patterns for large datasets and the value of well-designed metrics for assessing AI performance. We also recognized the importance of providing flexible data export options to support different user needs. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 9 / March 17 \- March 21, 2025 |

| Performance Optimization and Error Handling |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Week 9 was dedicated to performance optimization and improving error handling throughout the application. We implemented database query optimization with proper indexing and query restructuring for faster data retrieval. We developed client-side caching strategies using TanStack Query's capabilities to reduce server load. We created the comprehensive error handling system with user-friendly error messages and recovery options. We improved memory management in the Python service for processing large datasets. We implemented batch processing for resource-intensive operations to prevent timeouts. We also created the usage tracking system to monitor resource utilization and prevent abuse. Additionally, we conducted performance testing to identify and resolve bottlenecks in both frontend and backend code. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used database indexing and query optimization techniques for improved performance. We implemented React Query's caching capabilities with proper invalidation strategies. For error handling, we created a layered approach with different levels of detail for users and logs. We also used profiling tools to identify performance bottlenecks and optimize code. |
| **Reflection: Problems Encountered and Lessons Learned** | A major challenge was balancing performance optimization with code readability and maintainability. We initially created complex optimizations that were difficult to understand and maintain. Another challenge was implementing error handling that provided useful information for debugging while still being user-friendly. Through these challenges, we learned the importance of targeted performance optimization based on actual bottlenecks rather than premature optimization. We also recognized the value of a well-designed error handling system that provides appropriate information to different audiences (users vs. developers). |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 10 / March 24 \- March 28, 2025 |

| Security Enhancements and User Profile Management |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | The tenth week focused on security enhancements and implementing user profile management. We conducted a security audit and implemented fixes for identified vulnerabilities. We enhanced the authentication system with improved token management and session validation. We implemented comprehensive input validation across all endpoints using Zod schemas. We created security measures against common web vulnerabilities (XSS, CSRF, SQL injection). We developed the user profile management features with customization options. We implemented profile image upload functionality with proper validation and secure storage. We also added role-based access control for protected operations and created audit logging for sensitive actions. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used secure coding practices throughout the application and implemented thorough input validation using Zod. For file uploads, we used multer with proper validation and secure storage. We implemented security best practices for preventing common web vulnerabilities and created a comprehensive permission system using role-based checks. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was implementing security measures without negatively impacting user experience or performance. We also faced difficulties in designing a flexible permission system that could accommodate different user roles. Through these challenges, we learned about the balance between security and usability, and how thoughtful security design can protect the application without creating excessive friction for users. We also recognized the importance of considering security from the beginning of the project rather than adding it as an afterthought. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 11 / April 1 \- April 4, 2025 |

| API Integration and External System Connectivity |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | In week 11, we focused on API integration and preparing the system for external connectivity. We developed RESTful API endpoints for external system integration with proper documentation. We implemented API rate limiting and throttling to prevent abuse. We created authentication mechanisms for API access using API keys and JWT tokens. We built comprehensive API documentation using OpenAPI specifications. We implemented CORS configuration for secure cross-origin requests. We also developed webhook functionality for notification of external systems about important events. Additionally, we created example client code for API integration and conducted thorough testing of all API endpoints to ensure reliability and correct operation. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We implemented RESTful API patterns for consistent external access and created comprehensive API documentation with examples. We used rate limiting middleware to prevent API abuse and implemented secure authentication for API access. We also developed automated tests for API endpoints to ensure reliability. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was designing API endpoints that were both flexible enough to support various use cases and simple enough to be easily understood. We also faced difficulties in implementing effective rate limiting that prevented abuse without blocking legitimate high-volume users. Through these challenges, we learned the importance of thoughtful API design with consistent patterns and the value of comprehensive documentation for external integrations. We also recognized the need for proper security measures specifically designed for API access. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 12 / April 7 \- April 11, 2025 |

| System Testing and Bug Fixing |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | Week 12 was dedicated to comprehensive system testing and bug fixing. We conducted end-to-end testing of all major application flows and identified and resolved bugs across the application. We performed stress testing to ensure stability under heavy load and tested edge cases for all critical functionalities. We improved error recovery for unexpected situations and implemented comprehensive logging for better debugging. We also conducted cross-browser testing to ensure compatibility. We fixed UI inconsistencies and improved responsive behavior on different devices. Additionally, we implemented automated tests for critical components and functions, and created a system for reporting and tracking bugs. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We used systematic testing methodologies covering unit, integration, and end-to-end tests. For stress testing, we created scripts to simulate heavy usage patterns. We implemented improved logging for debugging and error tracking. We also conducted manual testing across different browsers and devices. |
| **Reflection: Problems Encountered and Lessons Learned** | A significant challenge was identifying and fixing subtle bugs that only appeared under specific conditions or with certain data patterns. We also faced difficulties in reproducing and resolving issues that only occurred under heavy load. Through these challenges, we learned the importance of systematic testing that covers both common and edge cases, and the value of comprehensive logging for debugging complex issues. We also recognized that many bugs stem from incorrect assumptions about how components interact, reinforcing the need for integration testing alongside unit testing. |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Development Team |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 \- April 19, 2025 |

| Final Documentation and Deployment Preparation |
| :---: |

| Activities and Progress (Actual Code, Screenshot of the Design, etc.) | The final week focused on documentation and preparing the system for deployment. We created comprehensive user documentation with guides and tutorials. We developed technical documentation covering system architecture, API endpoints, and database schema. We prepared deployment configurations for production environments. We implemented final UI polish based on user feedback. We created database migration scripts for production deployment. We conducted a final security review and made necessary adjustments. We also optimized asset loading for production performance and implemented proper error pages and fallback states. Additionally, we prepared maintenance documentation and created a roadmap for future development. |
| :---: | :---- |
| **Techniques, Tools, and Methodologies Used** | We created structured documentation with clear examples and visual aids. For deployment preparation, we implemented environment-specific configurations and optimized build processes. We conducted a final code review with focus on quality and maintainability. We also created database migration tools for smooth deployment. |
| **Reflection: Problems Encountered and Lessons Learned** | A key challenge was creating documentation that was comprehensive enough for technical users while still being accessible to non-technical users. We also faced difficulties in preparing for deployment to environments that might differ from our development setup. Through these challenges, we learned the importance of tailoring documentation to different audiences and the value of environment-agnostic configuration. We also recognized that documentation forces a comprehensive review of the system, often uncovering overlooked details or inconsistencies that can be addressed before final release. |