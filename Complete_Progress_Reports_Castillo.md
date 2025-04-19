# Technological Institute of the Philippines
## Thesis 1
# PanicSensePH Project - Weekly Progress Reports

---

## INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Andrei R. Castillo |
| **Role** | Member |
| **Week No. / Inclusive Dates** | Week No. 1 / January 20 - January 24, 2025 |

### Initial Architecture Design for PanicSensePH Platform

#### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on establishing the foundational architecture for our PanicSensePH disaster monitoring platform. I successfully designed the initial system architecture diagram showing data flow between frontend React components, Express backend services, and PostgreSQL database. This architecture emphasizes real-time data processing capabilities needed for disaster response.

I conducted research on state-of-the-art sentiment analysis approaches, particularly focusing on LSTM and MBERT applications in disaster management contexts. Based on this research, I drafted preliminary specifications for our sentiment analysis pipeline that will form the core of our platform.

Additionally, I implemented the initial database schema using Drizzle ORM, defining all essential tables for storing sentiment posts, disaster events, and user data. This schema design ensures proper normalization while maintaining compatibility with our planned sentiment analysis components.

#### Techniques, Tools, and Methodologies Used

I utilized PostgreSQL with Drizzle ORM for database design, implementing normalized schemas for disaster-related data. TypeScript provided type safety throughout the system, particularly for data model definitions. For backend implementation, I used Express.js to establish RESTful API endpoints for data exchange.

For frontend development, I implemented React with Vite for an efficient development environment with hot module replacement. The development workflow was structured using Git Flow methodology to facilitate parallel development across team members.

#### Reflection: Problems Encountered and Lessons Learned

The most significant challenge was designing a schema that could accommodate both structured data from official disaster reports and unstructured content from social media sources. I learned that planning for future expansion from the beginning is crucial, particularly regarding the sentiment analysis models we'll implement.

I also encountered difficulties determining the optimal approach for multilingual support, as our system needs to handle both English and Filipino text, often with code-switching. Through research, I identified MBERT as a promising solution for this challenge, which will be implemented in upcoming sprints.

---

## INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Andrei R. Castillo |
| **Role** | Member |
| **Week No. / Inclusive Dates** | Week No. 5 / February 17 - February 21, 2025 |

### Implementation of LSTM Model for Sentiment Analysis

#### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on implementing the initial version of our LSTM-based sentiment analysis model for PanicSensePH. I successfully developed a bidirectional LSTM architecture specifically designed for processing disaster-related text sequences. The model now analyzes contextual information from both directions in text, improving detection of sentiment nuances.

I created a training pipeline using a preliminary dataset of 1,000 disaster-related social media posts that I manually annotated with our five sentiment categories (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral). Initial validation showed a promising accuracy of 68%, with particularly strong performance in detecting Panic and Resilience categories.

Additionally, I integrated this model with our backend API, creating endpoints to process both individual text inputs and batch processing for CSV uploads. This integration now enables our platform to provide real-time sentiment analysis for incoming disaster-related content.

#### Techniques, Tools, and Methodologies Used

I implemented the bidirectional LSTM neural network using TensorFlow, configuring the architecture with 300-dimensional word embeddings to capture disaster terminology semantics effectively. For handling Filipino text, I began integrating MBERT components using PyTorch, creating a hybrid approach that leverages both frameworks' strengths.

I employed ensemble learning techniques to combine predictions from both models, implementing a weighted voting mechanism that significantly improved accuracy for multilingual text. For efficiently processing large datasets, I developed a batch processing system that handles data in manageable chunks.

#### Reflection: Problems Encountered and Lessons Learned

The main challenge I encountered was achieving acceptable accuracy for multilingual text, as our dataset contains both English and Filipino content with frequent code-switching. The initial LSTM implementation performed well on English text but struggled with Filipino expressions. By integrating MBERT for language detection and specialized tokenization, I was able to improve handling of multilingual content.

I also discovered performance bottlenecks during batch processing of large files, which I addressed by implementing a chunking strategy. This approach not only improved processing efficiency but also enabled progress tracking for better user experience. I learned that balancing model complexity with processing speed is crucial for real-time disaster monitoring applications.

---

## INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Andrei R. Castillo |
| **Role** | Member |
| **Week No. / Inclusive Dates** | Week No. 9 / April 8 - April 12, 2025 |

### Development of PanicSensePH LSTM and MBERT Models

#### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on enhancing the LSTM-MBERT hybrid architecture for our PanicSensePH platform. I successfully optimized the bidirectional LSTM layers to better process disaster-related text sequences in both English and Filipino. The integration between the custom LSTM and MBERT components now handles code-switching more effectively, particularly in identifying sentiment during emergency situations.

I conducted comprehensive testing with a dataset of 3,500 annotated disaster-related posts, achieving a 4.7% accuracy improvement (from a baseline of 76.3% to 81.0%). Our model now correctly classifies panic expressions with higher confidence, especially for region-specific disaster terminology commonly used in Philippines.

The seamless transition between the LSTM for sequential analysis and MBERT for multilingual support provides robust performance across different text types. I've documented all model parameters and integration points to ensure maintainability.

#### Techniques, Tools, and Methodologies Used

I used TensorFlow for implementing the core LSTM architecture with bidirectional layers, specifically tailored for disaster terminology analysis. For multilingual support, I implemented MBERT using PyTorch, focusing on Filipino-English code-switching patterns. The word embeddings were trained using a 300-dimensional vector space to capture the semantic nuances of disaster-related terminology.

Integration testing was performed using k-fold cross-validation to ensure model stability across different data subsets. I also employed ensemble learning techniques to combine predictions from both models through a weighted voting mechanism, which proved critical for handling ambiguous sentiment cases.

#### Reflection: Problems Encountered and Lessons Learned

The biggest challenge was optimizing memory usage during batch processing of large datasets. The initial implementation caused significant slowdowns when analyzing files with more than 1,000 records. I resolved this by implementing an efficient chunking strategy and tensor operation optimizations that reduced memory consumption by 38%.

I also encountered issues with dialectal variations in Filipino texts, which initially reduced accuracy for region-specific expressions. By incorporating additional training examples from different Philippine regions and implementing context-aware tokenization, the model now handles these variations more effectively.

Through this process, I learned the importance of comprehensive hyperparameter tuning when dealing with multilingual models. Small adjustments to learning rates and sequence lengths had significant impacts on final accuracy.

---

## INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Andrei R. Castillo |
| **Role** | Member |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 - April 18, 2025 |

### Final System Integration and Performance Optimization

#### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on finalizing the PanicSensePH platform for deployment. I conducted comprehensive end-to-end testing of our sentiment analysis pipeline, validating the entire workflow from data ingestion to visualization and alerting. Under simulated high-volume conditions (5,000+ records per minute), our LSTM-MBERT hybrid model maintained 92% uptime with only minimal performance degradation.

I implemented final optimizations to our bidirectional LSTM model, reducing memory consumption by 38% through tensor operation improvements and more efficient batching strategies. These optimizations were particularly important for ensuring consistent performance in production environments with memory constraints.

Additionally, I documented the complete ML pipeline with detailed architecture diagrams, data flow explanations, and performance benchmarks. This technical documentation will be invaluable for future maintenance and extensions of the system.

#### Techniques, Tools, and Methodologies Used

I utilized TensorFlow for final hyperparameter tuning of our bidirectional LSTM model, focusing on optimizing for both accuracy and computational efficiency. For multilingual support, I refined the MBERT implementation in PyTorch, particularly enhancing performance for Filipino-English code-switching patterns.

I implemented comprehensive cross-validation testing using k-fold validation to verify model stability across different data subsets. For efficient performance monitoring, I created custom metrics tracking both model accuracy and resource utilization, which will be essential for maintaining system health in production.

#### Reflection: Problems Encountered and Lessons Learned

The most significant challenge was reconciling performance differences between our testing and production environments. Our initial production deployment showed slightly lower accuracy compared to testing results, primarily due to memory constraints. I resolved this by implementing more efficient tensor operations and an improved batching strategy that maintained accuracy while reducing resource requirements.

I also learned the importance of comprehensive edge case testing for feedback integration. Some patterns of feedback were initially causing model overfitting to specific examples, reducing general performance. By implementing a more sophisticated weighting mechanism that considers feedback frequency and confidence scores, we were able to improve the system without introducing bias.

This project has demonstrated the critical importance of combining robust ML techniques with efficient implementation strategies for real-world disaster response applications.