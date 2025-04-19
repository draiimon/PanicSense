Technological University of the Philippines

# INDIVIDUAL PROGRESS REPORT

| | |
|:---|:---|
| **Name** | Mark Joseph L. Santos |
| **Role** | Lead Developer |
| **Week No. / Inclusive Dates** | Week No. 13 / April 14 - April 18, 2025 |

## Final Integration and System Testing

### Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on comprehensive end-to-end testing of the PanicSensePH platform, particularly the sentiment analysis engine. I conducted stress testing with simulated high-volume data ingestion (5,000+ records per minute) to validate system stability under peak loads. Our LSTM-MBERT hybrid model maintained 92% uptime during these tests, with only minimal performance degradation.

I also completed the final integration of the feedback loop system that allows corrected sentiment classifications to be incorporated back into the training pipeline. This included resolution of several edge cases where corrections weren't being properly weighted in the retraining process. Testing revealed a 4.7% improvement in accuracy after incorporating 200 real-world feedback samples.

Additionally, I documented all API endpoints with detailed Swagger specifications and created comprehensive technical documentation for future maintenance and extension of the sentiment analysis subsystem. The documentation includes architecture diagrams, data flow explanations, and performance benchmarks for different components.

### Techniques, Tools, and Methodologies Used

- **Bidirectional LSTM Neural Networks** - Finalized hyperparameter tuning for the context-sensitive language model
- **MBERT Language Detection** - Enhanced handling of Filipino-English code-switching patterns common in disaster communication
- **Word Embeddings** - Implemented 300-dimensional semantic vector space capturing disaster terminology
- **TensorFlow and PyTorch** - Used TensorFlow for LSTM and PyTorch for MBERT implementation
- **Ensemble Learning** - Combined predictions from both models through weighted voting mechanism
- **Cross-validation** - Implemented k-fold cross-validation to verify model stability across different data subsets

### Reflection: Problems Encountered and Lessons Learned

The most significant challenge this week was reconciling performance differences between testing and production environments. Our model showed slightly lower accuracy in the production configuration due to memory constraints. I resolved this by implementing a more efficient batching strategy and optimizing tensor operations, bringing production performance in line with testing results.

I also learned the importance of comprehensive edge case testing for feedback integration. Some patterns of feedback were initially causing overfitting to specific examples, reducing general performance. By implementing a more sophisticated weighting mechanism that considers feedback frequency and confidence scores, we were able to improve the system without introducing bias toward frequently occurring patterns.

Moving forward, I recommend implementing an automated regression testing pipeline that continuously validates model performance against a growing test set. This would help catch potential degradation as the system evolves over time.