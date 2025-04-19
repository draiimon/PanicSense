Technological Institute of the Philippines  
Thesis 2

**INDIVIDUAL PROGRESS REPORT**

| Name | Mark Andrei R. Castillo |
| :---: | :---: |
| **Role** | Lead Developer |
| **Inclusive Dates** | January 20 - April 19, 2025 |

## Week No. 1 / January 20 - January 24, 2025: Project Setup and Database Design

| **Activities and Progress** | • Set up PostgreSQL database with 8 core tables<br>• Created database schema with proper relations<br>• Set up TypeScript and project structure<br>• Implemented authentication schema<br>• Created shared types for FE/BE consistency |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • PostgreSQL, Drizzle ORM<br>• TypeScript, Node.js, Express<br>• Zod for schema validation<br>• Git Flow for version control<br>• VS Code with ESLint and Prettier |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Database design for complex sentiment data<br>• Solution: Restructured schema with flexible fields<br>• Lesson: Proper planning prevents future refactoring |

## Week No. 2 / January 27 - January 31, 2025: API Development and Authentication

| **Activities and Progress** | • Developed core API endpoints (auth, data retrieval)<br>• Built auth system with signup/login/sessions<br>• Created storage interface for DB operations<br>• Implemented error handling middleware<br>• Added validation for API requests |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Express.js for API routing<br>• bcrypt for password hashing<br>• JWT for authentication tokens<br>• Repository pattern for data access<br>• Postman for API testing |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Session management scaling issues<br>• Solution: Implemented stateless JWT with Redis<br>• Lesson: Standardized API responses simplify frontend work |

## Week No. 3 / February 3 - February 7, 2025: Sentiment Analysis Integration

| **Activities and Progress** | • Built Python service for sentiment analysis<br>• Created text processing pipeline<br>• Implemented location/disaster type extraction<br>• Added CSV processing for batch analysis<br>• Developed caching for performance |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Node.js child processes<br>• NLTK for NLP processing<br>• Stream processing for CSV files<br>• Memory management techniques<br>• Inter-process communication |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Memory leaks with large CSV files<br>• Solution: Implemented streaming with chunked processing<br>• Lesson: Resource management critical for cross-language integration |

## Week No. 4 / February 10 - February 14, 2025: Frontend Implementation

| **Activities and Progress** | • Created responsive main layout and navigation<br>• Built dashboard with key statistics<br>• Implemented auth forms with validation<br>• Developed file upload component<br>• Added data display components |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • React with TypeScript<br>• TanStack Query for data fetching<br>• React Hook Form for forms<br>• Tailwind CSS and shadcn/ui<br>• Mobile-first responsive design |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Information overload in dashboard<br>• Solution: Redesigned with clearer hierarchy<br>• Lesson: User interface should prioritize clarity |

## Week No. 5 / February 17 - February 21, 2025: Geographic Analysis

| **Activities and Progress** | • Integrated Leaflet for interactive mapping<br>• Created heatmap for sentiment concentration<br>• Enhanced location extraction for Philippine places<br>• Built responsive map containers<br>• Added filtering by disaster type and date |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Leaflet for interactive maps<br>• GeoJSON for geographic data<br>• Custom gazetteer for Philippine locations<br>• Marker clustering for performance<br>• Administrative boundary overlays |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Poor NER for Philippine locations<br>• Solution: Created custom gazetteer with 1,700+ locations<br>• Lesson: Country-specific optimizations improve results |

## Week No. 6 / February 24 - February 28, 2025: Real-time Features

| **Activities and Progress** | • Implemented WebSocket for live updates<br>• Created upload progress modal<br>• Built cross-tab synchronization<br>• Added real-time sentiment analysis<br>• Implemented reconnection for network issues |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • WebSocket Protocol<br>• localStorage Events for cross-tab sync<br>• Event-driven architecture<br>• Reconnection strategies<br>• Optimistic UI updates |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Unreliable WebSocket connections<br>• Solution: Added reconnection with exponential backoff<br>• Lesson: Real-time features need robust error handling |

## Week No. 7 / March 3 - March 7, 2025: Feedback System

| **Activities and Progress** | • Created system for correcting AI predictions<br>• Built training pipeline for model improvement<br>• Added confidence scoring visualization<br>• Implemented text similarity detection<br>• Created UI for feedback submission |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Transfer Learning for model adaptation<br>• Text similarity algorithms<br>• Confidence calibration<br>• React Hook Form for feedback UI<br>• TanStack Mutation for submission |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Model needed too many examples to improve<br>• Solution: Implemented category-based learning<br>• Lesson: Domain-specific knowledge improves performance |

## Week No. 8 / March 10 - March 14, 2025: Data Export

| **Activities and Progress** | • Implemented evaluation metrics calculation<br>• Created visualization for model performance<br>• Built CSV export functionality<br>• Added filtering for customized exports<br>• Created data explorer with pagination |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Confusion Matrix Analysis<br>• Chart.js for visualization<br>• Streaming CSV generation<br>• Virtual scrolling for large datasets<br>• Content disposition headers |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Memory issues with large exports<br>• Solution: Implemented streaming with chunked processing<br>• Lesson: Stream processing needed throughout the stack |

## Week No. 9 / March 17 - March 21, 2025: Performance Optimization

| **Activities and Progress** | • Optimized database queries with proper joins<br>• Implemented client-side caching<br>• Added database indexes for common queries<br>• Created performance monitoring dashboard<br>• Improved memory management |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Database indexing strategies<br>• Query execution planning<br>• Connection pooling<br>• Memory profiling<br>• React Query caching |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Performance degradation with large datasets<br>• Solution: Added indexes and optimized queries<br>• Lesson: Performance optimization should be data-driven |

## Week No. 10 / March 24 - March 28, 2025: Security Enhancements

| **Activities and Progress** | • Implemented security headers<br>• Added CSRF protection<br>• Created role-based access control<br>• Enhanced authentication with token management<br>• Improved input validation |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • OWASP security guidelines<br>• Content Security Policy<br>• Rate limiting<br>• bcrypt for password hashing<br>• JWT with short expiry |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: XSS in user-generated content<br>• Solution: Strict CSP and output encoding<br>• Lesson: Security needs multiple layers to be effective |

## Week No. 11 / April 1 - April 4, 2025: API Integration

| **Activities and Progress** | • Created RESTful API endpoints for integration<br>• Implemented API authentication<br>• Added webhook functionality<br>• Created CORS configuration<br>• Built rate limiting system |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • RESTful API design patterns<br>• OpenAPI/Swagger documentation<br>• HMAC signatures for webhooks<br>• Dynamic rate limiting<br>• API versioning |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Unreliable webhook deliveries<br>• Solution: Added retry mechanism with logging<br>• Lesson: External APIs need careful backward compatibility |

## Week No. 12 / April 7 - April 11, 2025: Testing and Bug Fixing

| **Activities and Progress** | • Conducted end-to-end testing<br>• Created test cases for critical features<br>• Implemented enhanced logging system<br>• Performed cross-browser testing<br>• Fixed UI inconsistencies |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Jest for unit testing<br>• Supertest for API testing<br>• Winston for logging<br>• BrowserStack for compatibility<br>• Cypress for E2E testing |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Browser-specific rendering issues<br>• Solution: Added polyfills and fallbacks<br>• Lesson: Testing needs to cover functionality and environments |

## Week No. 13 / April 14 - April 19, 2025: Documentation and Deployment

| **Activities and Progress** | • Created comprehensive user documentation<br>• Prepared deployment configuration<br>• Built database migration scripts<br>• Optimized assets for production<br>• Conducted final security review |
| :---: | :--- |
| **Techniques, Tools, and Methodologies Used** | • Markdown documentation<br>• Drizzle migrations<br>• Environment-based configuration<br>• Asset compression and bundling<br>• Docker containerization |
| **Reflection: Problems Encountered and Lessons Learned** | • Challenge: Documentation for different user types<br>• Solution: Created tiered documentation approach<br>• Lesson: Deployment planning should start early |