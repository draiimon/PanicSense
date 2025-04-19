# PanicSensePH: Disaster Monitoring Platform
## Project Reports Compilation
*January 20, 2025 - April 18, 2025*

---

# SECTION I: WEEKLY PROGRESS SUMMARIES

## Technological University of the Philippines

### WEEKLY PROGRESS REPORT - PANICSENSE PH

#### Project Progress Summary
**Week 1: January 20 - January 24, 2025**

##### System Architecture Design and Requirements Analysis

###### Activities and Progress
This week marked the official start of the PanicSensePH project development. Our team focused on establishing the foundational architecture and defining technical requirements. We successfully designed the initial system architecture diagram, which illustrates the data flow between our front-end React components, back-end Express services, and PostgreSQL database integration.

We conducted a comprehensive stakeholder needs analysis through interviews with three disaster management officials from NDRRMC. Their input helped us identify critical features for the minimum viable product (MVP), particularly regarding real-time data processing needs.

The team also implemented the initial project setup with the Node.js/TypeScript environment and established the PostgreSQL database connectivity, setting up all essential tables as defined in our schema.

###### Techniques, Tools, and Methodologies Used
- **PostgreSQL Database Design** - Implemented normalized schemas for disaster-related data
- **TypeScript & Drizzle ORM** - Established type-safe database models with migration capabilities
- **Express.js Backend** - Set up RESTful API endpoints for data exchange
- **React/Vite Frontend** - Configured development environment with hot module replacement
- **Git Flow Methodology** - Implemented feature branch workflow for parallel development

###### Reflection: Problems Encountered and Lessons Learned
The biggest challenge was selecting the appropriate schema design that would accommodate both structured and unstructured data from social media sources. We learned that planning for future expansion from the beginning is crucial, particularly regarding the sentiment analysis models we'll implement in the coming weeks.

Balancing real-time processing capabilities with database performance was another considerable challenge. We resolved this by implementing an efficient batch processing strategy that will be crucial for handling large CSV imports.

---

#### Project Progress Summary
**Week 2: January 27 - January 31, 2025**

##### Sentiment Analysis Model Development

###### Activities and Progress
This week, our focus shifted to developing the core sentiment analysis capabilities. We successfully implemented the initial version of our custom Long Short-Term Memory (LSTM) neural network model for disaster-related text classification. The architecture includes bidirectional layers that capture contextual information from both directions of the text sequence.

We began training the model on our preliminary dataset of 1,000 disaster-related social media posts, which we manually annotated with sentiment categories (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral). Initial validation tests showed promising results with approximately 68% accuracy.

Additionally, we developed the file processing system for CSV uploads, which will be a critical component for bulk analysis of historical data. This included implementation of a progress tracking mechanism to provide real-time feedback during processing.

###### Techniques, Tools, and Methodologies Used
- **Custom LSTM Neural Network** - Developed a bidirectional LSTM specialized for disaster text sequences
- **MBERT (Multilingual BERT)** - Implemented for handling Filipino text and code-switching
- **Word Embeddings** - Created 300-dimensional embeddings that capture disaster terminology semantics
- **TensorFlow and PyTorch** - Used TensorFlow for LSTM and PyTorch for MBERT implementation
- **Ensemble Learning** - Combined predictions from LSTM and MBERT for improved accuracy
- **Batch Processing** - Implemented efficient batch processing to handle large datasets

###### Reflection: Problems Encountered and Lessons Learned
The primary challenge was handling multilingual text, as many disaster reports in the Philippines contain code-switching between English and Filipino. We addressed this by integrating MBERT for improved language detection and processing.

We also encountered performance bottlenecks during batch processing of large files. This led us to implement a chunking strategy that processes data in manageable segments, providing regular progress updates to the user interface.

---

*[Content continues with Weeks 3-13 from the weekly progress report]*

---

# SECTION II: INDIVIDUAL PROGRESS REPORT

## Technological University of the Philippines

### INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Joseph L. Santos |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 - April 18, 2025 |

#### Final Integration and System Testing

##### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on comprehensive end-to-end testing of the PanicSensePH platform, particularly the sentiment analysis engine. I conducted stress testing with simulated high-volume data ingestion (5,000+ records per minute) to validate system stability under peak loads. Our LSTM-MBERT hybrid model maintained 92% uptime during these tests, with only minimal performance degradation.

I also completed the final integration of the feedback loop system that allows corrected sentiment classifications to be incorporated back into the training pipeline. This included resolution of several edge cases where corrections weren't being properly weighted in the retraining process. Testing revealed a 4.7% improvement in accuracy after incorporating 200 real-world feedback samples.

Additionally, I documented all API endpoints with detailed Swagger specifications and created comprehensive technical documentation for future maintenance and extension of the sentiment analysis subsystem. The documentation includes architecture diagrams, data flow explanations, and performance benchmarks for different components.

##### Techniques, Tools, and Methodologies Used

- **Bidirectional LSTM Neural Networks** - Finalized hyperparameter tuning for the context-sensitive language model
- **MBERT Language Detection** - Enhanced handling of Filipino-English code-switching patterns common in disaster communication
- **Word Embeddings** - Implemented 300-dimensional semantic vector space capturing disaster terminology
- **TensorFlow and PyTorch** - Used TensorFlow for LSTM and PyTorch for MBERT implementation
- **Ensemble Learning** - Combined predictions from both models through weighted voting mechanism
- **Cross-validation** - Implemented k-fold cross-validation to verify model stability across different data subsets

##### Reflection: Problems Encountered and Lessons Learned

The most significant challenge this week was reconciling performance differences between testing and production environments. Our model showed slightly lower accuracy in the production configuration due to memory constraints. I resolved this by implementing a more efficient batching strategy and optimizing tensor operations, bringing production performance in line with testing results.

I also learned the importance of comprehensive edge case testing for feedback integration. Some patterns of feedback were initially causing overfitting to specific examples, reducing general performance. By implementing a more sophisticated weighting mechanism that considers feedback frequency and confidence scores, we were able to improve the system without introducing bias toward frequently occurring patterns.

Moving forward, I recommend implementing an automated regression testing pipeline that continuously validates model performance against a growing test set. This would help catch potential degradation as the system evolves over time.