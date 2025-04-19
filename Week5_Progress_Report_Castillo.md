Technological Institute of the Philippines
Thesis 1

INDIVIDUAL PROGRESS REPORT



Name

Mark Andrei R. Castillo

Role

Member

Week No. / Inclusive Dates

Week No. 5 / February 17 - February 21, 2025



Implementation of LSTM Model for Sentiment Analysis



Activities and Progress (Actual Code, Screenshot of the Design, etc.)

This week, I focused on implementing the initial version of our LSTM-based sentiment analysis model for PanicSensePH. I successfully developed a bidirectional LSTM architecture specifically designed for processing disaster-related text sequences. The model now analyzes contextual information from both directions in text, improving detection of sentiment nuances.

I created a training pipeline using a preliminary dataset of 1,000 disaster-related social media posts that I manually annotated with our five sentiment categories (Panic, Fear/Anxiety, Disbelief, Resilience, Neutral). Initial validation showed a promising accuracy of 68%, with particularly strong performance in detecting Panic and Resilience categories.

Additionally, I integrated this model with our backend API, creating endpoints to process both individual text inputs and batch processing for CSV uploads. This integration now enables our platform to provide real-time sentiment analysis for incoming disaster-related content.



Techniques, Tools, and Methodologies Used

I implemented the bidirectional LSTM neural network using TensorFlow, configuring the architecture with 300-dimensional word embeddings to capture disaster terminology semantics effectively. For handling Filipino text, I began integrating MBERT components using PyTorch, creating a hybrid approach that leverages both frameworks' strengths.

I employed ensemble learning techniques to combine predictions from both models, implementing a weighted voting mechanism that significantly improved accuracy for multilingual text. For efficiently processing large datasets, I developed a batch processing system that handles data in manageable chunks.



Reflection: Problems Encountered and Lessons Learned

The main challenge I encountered was achieving acceptable accuracy for multilingual text, as our dataset contains both English and Filipino content with frequent code-switching. The initial LSTM implementation performed well on English text but struggled with Filipino expressions. By integrating MBERT for language detection and specialized tokenization, I was able to improve handling of multilingual content.

I also discovered performance bottlenecks during batch processing of large files, which I addressed by implementing a chunking strategy. This approach not only improved processing efficiency but also enabled progress tracking for better user experience. I learned that balancing model complexity with processing speed is crucial for real-time disaster monitoring applications.