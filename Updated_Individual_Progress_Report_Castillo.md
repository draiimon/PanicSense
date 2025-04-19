Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 9 / April 8 - April 12, 2025



Development of PanicSensePH LSTM and MBERT Models



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on enhancing the LSTM-MBERT hybrid architecture for our PanicSensePH platform. I successfully optimized the bidirectional LSTM layers to better process disaster-related text sequences in both English and Filipino. The integration between the custom LSTM and MBERT components now handles code-switching more effectively, particularly in identifying sentiment during emergency situations.

I conducted comprehensive testing with a dataset of 3,500 annotated disaster-related posts, achieving a 4.7% accuracy improvement (from a baseline of 76.3% to 81.0%). Our model now correctly classifies panic expressions with higher confidence, especially for region-specific disaster terminology commonly used in Philippines.

The seamless transition between the LSTM for sequential analysis and MBERT for multilingual support provides robust performance across different text types. I've documented all model parameters and integration points to ensure maintainability.



Techniques, Tools, and Methodologies Used

I used TensorFlow for implementing the core LSTM architecture with bidirectional layers, specifically tailored for disaster terminology analysis. For multilingual support, I implemented MBERT using PyTorch, focusing on Filipino-English code-switching patterns. The word embeddings were trained using a 300-dimensional vector space to capture the semantic nuances of disaster-related terminology.

Integration testing was performed using k-fold cross-validation to ensure model stability across different data subsets. I also employed ensemble learning techniques to combine predictions from both models through a weighted voting mechanism, which proved critical for handling ambiguous sentiment cases.



Reflection: Problems Encountered and Lessons Learned

The biggest challenge was optimizing memory usage during batch processing of large datasets. The initial implementation caused significant slowdowns when analyzing files with more than 1,000 records. I resolved this by implementing an efficient chunking strategy and tensor operation optimizations that reduced memory consumption by 38%.

I also encountered issues with dialectal variations in Filipino texts, which initially reduced accuracy for region-specific expressions. By incorporating additional training examples from different Philippine regions and implementing context-aware tokenization, the model now handles these variations more effectively.

Through this process, I learned the importance of comprehensive hyperparameter tuning when dealing with multilingual models. Small adjustments to learning rates and sequence lengths had significant impacts on final accuracy.