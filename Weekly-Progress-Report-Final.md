Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 \- January 24, 2025 |

| Project Setup |
| :---: |

| Activities and Progress | • Set up PostgreSQL database with 8 core tables for disaster data<br>• Created database schema with proper relations for user management<br>• Implemented authentication with bcrypt password hashing<br>• Set up TypeScript configuration with strict type checking<br>• Created shared type definitions for frontend and backend consistency |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • PostgreSQL/Drizzle ORM for database management<br>• TypeScript for type-safe development<br>• Zod schema validation for data integrity<br>• Git Flow for version control<br>• VS Code with ESLint/Prettier for code quality |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Initial database schema was too rigid for complex sentiment data<br>• Solution: Redesigned schema with flexible fields and proper relations<br>• Lesson: Thorough planning of data structure prevents major refactoring later |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 2 / January 27 \- January 31, 2025 |

| API Development |
| :---: |

| Activities and Progress | • Built core API endpoints for auth and data retrieval<br>• Created storage interface for database operations<br>• Implemented JWT authentication with proper security<br>• Added request validation for all endpoints<br>• Created error handling middleware for consistent responses |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Express.js for REST API development<br>• bcrypt for secure password management<br>• JWT for authentication tokens<br>• Repository pattern for database access<br>• Postman for API testing |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Session management had scaling issues with concurrent users<br>• Solution: Implemented stateless JWT authentication with token refresh<br>• Lesson: Standardized API responses greatly simplify frontend development |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 3 / February 3 \- February 7, 2025 |

| Sentiment Analysis Integration |
| :---: |

| Activities and Progress | • Integrated Python service for disaster sentiment analysis<br>• Built text processing pipeline for content extraction<br>• Implemented location and disaster type identification<br>• Created CSV processing for batch data analysis<br>• Set up caching system for improved performance |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Node.js child processes for Python integration<br>• NLTK for natural language processing<br>• Streaming for efficient file handling<br>• Inter-process communication patterns<br>• Memory optimization techniques |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Processing large CSV files caused memory overflows<br>• Solution: Implemented chunked streaming approach with batch processing<br>• Lesson: Cross-language integration requires careful resource management |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 4 / February 10 \- February 14, 2025 |

| Frontend Implementation |
| :---: |

| Activities and Progress | • Developed responsive layout with modern component design<br>• Created dashboard with key disaster monitoring stats<br>• Built auth forms with validation and error handling<br>• Implemented file upload component with progress tracking<br>• Added data visualization components for sentiment display |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • React with TypeScript for component development<br>• TanStack Query for efficient data fetching<br>• React Hook Form for form state management<br>• Tailwind CSS for responsive styling<br>• Shadcn UI for accessible components |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Dashboard was cluttered with too much information<br>• Solution: Redesigned with card-based layout and progressive disclosure<br>• Lesson: User interfaces should prioritize clarity over comprehensiveness |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 \- February 21, 2025 |

| Geographic Analysis |
| :---: |

| Activities and Progress | • Integrated Leaflet for interactive Philippine disaster mapping<br>• Created heatmap visualization for sentiment concentration<br>• Built custom location extraction for Philippine places<br>• Implemented filtering by disaster type and region<br>• Added responsive map containers for mobile views |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Leaflet with React integration<br>• Custom gazetteer for Philippine locations<br>• GeoJSON for geographic data representation<br>• Marker clustering for performance<br>• Administrative boundary overlays |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Standard NER failed to identify Philippine locations<br>• Solution: Created custom location database with 1,700+ places<br>• Lesson: Country-specific optimizations are essential for local applications |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 6 / February 24 \- February 28, 2025 |

| Real-time Features |
| :---: |

| Activities and Progress | • Implemented WebSocket for live disaster updates<br>• Built upload progress modal with real-time status<br>• Created cross-tab synchronization for consistent state<br>• Added immediate sentiment analysis feedback<br>• Implemented reconnection logic for network resilience |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • WebSocket Protocol for bidirectional communication<br>• localStorage Events for cross-tab messaging<br>• Event-driven architecture for real-time updates<br>• Optimistic UI updates for responsiveness<br>• Reconnection strategies with exponential backoff |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: WebSocket connections dropped on mobile networks<br>• Solution: Added robust reconnection with fallback polling<br>• Lesson: Real-time features need comprehensive error handling |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 7 / March 3 \- March 7, 2025 |

| AI Feedback System |
| :---: |

| Activities and Progress | • Developed system for correcting AI sentiment predictions<br>• Built model training pipeline incorporating user feedback<br>• Implemented confidence scoring with visual indicators<br>• Added text similarity detection for consistent analysis<br>• Created UI components for feedback submission |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Transfer learning for model adaptation<br>• Text similarity algorithms for comparison<br>• Confidence calibration techniques<br>• Bayesian methods for uncertainty estimation<br>• TanStack Mutation for optimistic updates |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Model required too many examples to improve<br>• Solution: Implemented category-based learning with fewer examples<br>• Lesson: Domain-specific training greatly improves AI performance |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 8 / March 10 \- March 14, 2025 |

| Data Export Features |
| :---: |

| Activities and Progress | • Created model evaluation metrics dashboard<br>• Built CSV export functionality for disaster data<br>• Implemented custom filtering for targeted exports<br>• Added data explorer with search and pagination<br>• Created visualization for model performance metrics |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Confusion matrix analysis for performance metrics<br>• Streaming CSV generation for efficiency<br>• Virtual scrolling for large dataset display<br>• Content disposition headers for file downloads<br>• Chart.js for metric visualization |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Large exports caused memory exhaustion<br>• Solution: Implemented streaming with chunked processing<br>• Lesson: Data export features must consider scalability from the start |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 9 / March 17 \- March 21, 2025 |

| Performance Optimization |
| :---: |

| Activities and Progress | • Optimized database queries with proper join techniques<br>• Implemented database indexing for frequent queries<br>• Added client-side caching for reduced server load<br>• Created monitoring dashboard for performance metrics<br>• Improved memory management for data processing |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Query execution planning and optimization<br>• Database indexing for critical fields<br>• Connection pooling for efficiency<br>• React Query caching strategies<br>• Memory profiling tools |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Application slowed significantly with 10,000+ records<br>• Solution: Added proper indexes and optimized N+1 queries<br>• Lesson: Performance optimization should target measured bottlenecks |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 10 / March 24 \- March 28, 2025 |

| Security Enhancements |
| :---: |

| Activities and Progress | • Implemented security headers for protection<br>• Added CSRF protection for state-changing operations<br>• Created role-based access control system<br>• Enhanced authentication with token refresh<br>• Implemented input validation and sanitization |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • OWASP security guidelines implementation<br>• Content Security Policy for XSS prevention<br>• CSRF token validation<br>• Input sanitization techniques<br>• Rate limiting for abuse prevention |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: XSS vulnerabilities in user content display<br>• Solution: Implemented strict CSP and proper output encoding<br>• Lesson: Security must be implemented across all application layers |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 11 / April 1 \- April 4, 2025 |

| External API Integration |
| :---: |

| Activities and Progress | • Built RESTful API endpoints for external integration<br>• Created comprehensive API documentation<br>• Implemented webhook system for event notifications<br>• Added proper CORS configuration for cross-origin access<br>• Built rate limiting with tiered access levels |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • RESTful API design principles<br>• OpenAPI/Swagger for documentation<br>• HMAC signatures for webhook security<br>• Dynamic rate limiting by authentication level<br>• API versioning for compatibility |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Webhook deliveries failed without notification<br>• Solution: Implemented retry mechanism with delivery logging<br>• Lesson: External integrations need robust failure handling |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 12 / April 7 \- April 11, 2025 |

| Testing and Quality Assurance |
| :---: |

| Activities and Progress | • Conducted comprehensive end-to-end testing<br>• Created automated tests for critical functions<br>• Implemented enhanced logging for debugging<br>• Performed cross-browser compatibility testing<br>• Fixed UI inconsistencies across devices |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Jest for unit and integration testing<br>• Cypress for end-to-end testing<br>• Winston for structured logging<br>• BrowserStack for compatibility testing<br>• Lighthouse for performance auditing |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Inconsistent rendering across browsers<br>• Solution: Added browser-specific styles and polyfills<br>• Lesson: Testing must cover functionality, performance, and compatibility |

---

Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 \- April 19, 2025 |

| Documentation and Deployment |
| :---: |

| Activities and Progress | • Created comprehensive user documentation<br>• Prepared deployment configuration for production<br>• Built database migration scripts for smooth deployment<br>• Optimized asset loading for production performance<br>• Conducted final security review before release |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Markdown with diagrams for documentation<br>• Drizzle migrations for database versioning<br>• Environment-based configuration management<br>• Asset bundling and compression<br>• Docker containerization for deployment |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Documentation needed for different user types<br>• Solution: Created tiered documentation by technical level<br>• Lesson: Deployment planning should start earlier in development |