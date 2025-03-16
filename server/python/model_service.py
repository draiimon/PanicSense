import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import logging

class SentimentClassifier(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', num_classes=5):
        """
        Initialize the model architecture combining mBERT, LSTM, and BiGRU

        Parameters:
        - model_name: The pre-trained BERT model to use
        - num_classes: Number of sentiment classes (Panic, Fear/Anxiety, etc.)
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)

        # LSTM layer after BERT
        self.lstm = nn.LSTM(
            input_size=768,  # BERT's output dimension
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )

        # BiGRU layer after LSTM
        self.bigru = nn.GRU(
            input_size=512,  # 256*2 due to bidirectional LSTM
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        # Final classification layer
        self.fc = nn.Linear(256, num_classes)  # 128*2 due to bidirectional GRU

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the network

        Flow:
        1. BERT processes the input text
        2. LSTM processes BERT's output sequence
        3. BiGRU processes LSTM's output
        4. Final classification
        """
        # Get BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]

        # Apply LSTM
        lstm_out, _ = self.lstm(bert_output)

        # Apply BiGRU
        gru_out, _ = self.bigru(lstm_out)

        # Get the final hidden state
        pooled_output = gru_out[:, -1]

        # Apply dropout and classification layer
        return self.fc(self.dropout(pooled_output))

class DisasterSentimentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with the model and tokenizer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = SentimentClassifier()
        self.model.to(self.device)

        # Define sentiment labels
        self.sentiment_labels = [
            'Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral'
        ]

        logging.info(f"Initialized DisasterSentimentAnalyzer on {self.device}")

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text

        Process:
        1. Tokenize the input text
        2. Pass through the model
        3. Get prediction and confidence
        4. Generate explanation
        """
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(predictions, dim=1)

        sentiment = self.sentiment_labels[predicted.item()]
        confidence = confidence.item()

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "explanation": self._generate_explanation(sentiment, text),
            "language": self._detect_language(text)
        }

    def _generate_explanation(self, sentiment, text):
        """Generate human-readable explanation for the sentiment"""
        explanations = {
            'Panic': 'Shows immediate distress or urgent need for help',
            'Fear/Anxiety': 'Expresses worry or concern about the situation',
            'Disbelief': 'Shows shock or surprise about events',
            'Resilience': 'Demonstrates community support and determination',
            'Neutral': 'Provides factual information without strong emotion'
        }
        return explanations.get(sentiment, 'Sentiment analysis completed')

    def _detect_language(self, text):
        """Detect the language of the text"""
        # For now returning English, but you can implement proper language detection
        return 'en'

# Create a global instance
analyzer = DisasterSentimentAnalyzer()