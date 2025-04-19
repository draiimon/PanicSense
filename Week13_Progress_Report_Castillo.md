Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 13 / April 14 - April 18, 2025



Final System Integration and Performance Optimization



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on finalizing the PanicSensePH platform for deployment. I conducted comprehensive end-to-end testing of our sentiment analysis pipeline, validating the entire workflow from data ingestion to visualization and alerting. Under simulated high-volume conditions (5,000+ records per minute), our LSTM-MBERT hybrid model maintained 92% uptime with only minimal performance degradation.

I implemented final optimizations to our bidirectional LSTM model, reducing memory consumption by 38% through tensor operation improvements and more efficient batching strategies. These optimizations were particularly important for ensuring consistent performance in production environments with memory constraints.

Additionally, I documented the complete ML pipeline with detailed architecture diagrams, data flow explanations, and performance benchmarks. This technical documentation will be invaluable for future maintenance and extensions of the system.



Techniques, Tools, and Methodologies Used

I utilized TensorFlow for final hyperparameter tuning of our bidirectional LSTM model, focusing on optimizing for both accuracy and computational efficiency. For multilingual support, I refined the MBERT implementation in PyTorch, particularly enhancing performance for Filipino-English code-switching patterns.

I implemented comprehensive cross-validation testing using k-fold validation to verify model stability across different data subsets. For efficient performance monitoring, I created custom metrics tracking both model accuracy and resource utilization, which will be essential for maintaining system health in production.



Reflection: Problems Encountered and Lessons Learned

The most significant challenge was reconciling performance differences between our testing and production environments. Our initial production deployment showed slightly lower accuracy compared to testing results, primarily due to memory constraints. I resolved this by implementing more efficient tensor operations and an improved batching strategy that maintained accuracy while reducing resource requirements.

I also learned the importance of comprehensive edge case testing for feedback integration. Some patterns of feedback were initially causing model overfitting to specific examples, reducing general performance. By implementing a more sophisticated weighting mechanism that considers feedback frequency and confidence scores, we were able to improve the system without introducing bias.

This project has demonstrated the critical importance of combining robust ML techniques with efficient implementation strategies for real-world disaster response applications.