"""
Hybrid Sentiment Analysis Trainer for PanicSense
This script combines keyword-based approach with ML-based sentiment analysis 
to create a hybrid model with real performance metrics.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import joblib
import re
import pickle

# Add the parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from emoji_utils
from python.emoji_utils import preprocess_text, preserve_exclamations, clean_text_preserve_indicators

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SENTIMENT_CATEGORIES = ["panic", "fear", "disbelief", "resilience", "neutral"]

# Keyword dictionaries for each sentiment category
KEYWORD_DICT = {
    "panic": [
        "emergency", "help", "tulong", "naiipit", "evacuate", "rescue", "trap", 
        "mabilis", "tubig", "baha", "malakas", "ulan", "lindol", "tumataas",
        "😱", "😨", "😰", "nakakatakot", "takot", "naiipit", "nasira", "kawawa",
        "pagtulong", "hindi makahinga", "nasalanta", "bumabaha", "nagkalat", 
        "nawalan", "nasalanta", "nasiraan", "nasira", "sinira", "nag-collapse",
        "gumuho", "rumaragasa", "pumapatay", "nadadamay", "nadadale", "nakakatakot",
        "buhawi", "pag-alboroto", "rescue"
    ],
    "fear": [
        "worry", "scared", "afraid", "fear", "dread", "horror", "terror",
        "nervous", "anxious", "kabado", "kinakabahan", "nag-aalala", "natatakot",
        "matatakot", "delikado", "mapanganib", "dangerous", "banta", "nangangamba",
        "maaaring", "posible", "hindi ligtas", "hindi safe", "magbibigay ng pinsala",
        "magiging sanhi ng", "makakapinsala", "maaring makapinsala", "nanganganib",
        "nararanasan", "tumatagal", "papalakas", "lumalala", "nagbabanta", "nakaamba"
    ],
    "disbelief": [
        "what", "how", "why", "unbelievable", "shocking", "di makapaniwala", "bakit",
        "paano", "anong nangyari", "hindi maintindihan", "hindi kapani-paniwala",
        "hindi expected", "hindi inaasahan", "bakit ganito", "bakit nangyari",
        "hindi dapat", "nakakagulat", "kakaiba", "hindi aakalain", "kagulat-gulat",
        "nakakagulat", "iba sa inaasahan", "malayo sa inaasahan", "sobrang bilis",
        "sobrang lakas", "sobrang dami", "sobrang laki", "napakarami", "napakalaki", 
        "napakagrabe", "biglaan", "bigla", "walang warning", "walang abiso"
    ],
    "resilience": [
        "strong", "together", "rebuild", "recover", "help", "support", "community",
        "volunteers", "donation", "contribute", "fundraise", "survive", "overcome",
        "endure", "withstand", "adapt", "cope", "manage", "tulungan", "sama-sama",
        "tulong", "bangon", "malalagpasan", "malalampasan", "babangon", "makakayanan",
        "makakabangon", "sama-sama", "tulong-tulong", "magkaisa", "nagkakaisa", 
        "nagtutulong", "magkapit-bisig", "makababangon", "makalalampas", "relief",
        "donation", "support", "pagtulong", "volunteer", "sumasaludo", "hope", "pag-asa"
    ],
    "neutral": [
        "report", "update", "announce", "inform", "status", "condition", "situation",
        "news", "development", "advisory", "bulletin", "notification", "alert",
        "announcement", "briefing", "ulat", "balita", "updates", "abiso", "impormasyon",
        "kalagayan", "sitwasyon", "kaganapan", "nangyari", "nagaganap", "naganap",
        "nilikha", "magkakaroon", "nagkaroon", "may", "merong", "magpapasimula", 
        "nagsimula", "maguumpisa", "uumpisa", "currently", "ngayon", "kasalukuyan"
    ]
}

# Utils for text preprocessing
def tokenize_text(text):
    """Simple tokenization function for text"""
    # Convert text to lowercase
    text = text.lower()
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Split text into tokens
    tokens = text.split()
    return tokens

class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracts keyword-based features from text
    This is the backbone of our hybrid approach - combining rule-based keyword
    matching with ML-based text classification
    """
    def __init__(self, keyword_dict=KEYWORD_DICT):
        self.keyword_dict = keyword_dict
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Transform text data to keyword-based features
        For each sentiment category, count how many keywords match in the text
        """
        features = np.zeros((len(X), len(self.keyword_dict)))
        
        for i, text in enumerate(X):
            # Preprocess text
            processed_text = text.lower()
            tokens = tokenize_text(processed_text)
            
            # Calculate keyword matches for each sentiment category
            for j, (category, keywords) in enumerate(self.keyword_dict.items()):
                # Check both single word and phrase matches
                single_word_matches = sum(1 for keyword in keywords if keyword.lower() in tokens)
                
                # Check for phrase matches (keywords with multiple words)
                phrase_matches = sum(1 for keyword in keywords if ' ' in keyword and keyword.lower() in processed_text)
                
                # Add special handling for emojis and exclamations
                emoji_matches = sum(1 for keyword in keywords if len(keyword) == 1 and keyword in text)
                exclamation_count = text.count('!') if category in ['panic', 'fear', 'disbelief'] else 0
                
                # Count total matches with proper weighting
                total_matches = (
                    single_word_matches + 
                    phrase_matches * 2 +  # Phrases are stronger indicators
                    emoji_matches * 3 +   # Emojis are very strong indicators
                    min(exclamation_count, 3)  # Cap exclamation impact
                )
                
                features[i, j] = total_matches
                
        return features
    
    def get_feature_names_out(self):
        """Return feature names for the transformer"""
        return np.array([f"keyword_{category}" for category in self.keyword_dict.keys()])

class HybridSentimentAnalysisTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_path = os.path.join(MODEL_DIR, 'hybrid_sentiment_model.pkl')
        self.vectorizer_path = os.path.join(MODEL_DIR, 'hybrid_vectorizer.pkl')
        
        # Load model if it exists
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model if it exists"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                print("Loaded pre-trained model and vectorizer")
                return True
            else:
                print("No pre-trained model found")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self):
        """Save the trained model and vectorizer"""
        if self.model and self.vectorizer:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print(f"Model and vectorizer saved to {MODEL_DIR}")
    
    def prepare_data(self, data):
        """Prepare data for training"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a list of dictionaries or a pandas DataFrame")
        
        # Ensure required columns exist
        required_cols = ['text', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain the following columns: {required_cols}")
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(clean_text_preserve_indicators)
        
        # Check if sentiment values are valid
        valid_sentiments = set(SENTIMENT_CATEGORIES)
        invalid_sentiments = set(df['sentiment'].unique()) - valid_sentiments
        if invalid_sentiments:
            print(f"Warning: Found invalid sentiment values: {invalid_sentiments}")
            # Filter out rows with invalid sentiments
            df = df[df['sentiment'].isin(valid_sentiments)]
            
        return df
    
    def train(self, data, test_size=0.2, random_state=42):
        """Train the hybrid sentiment analysis model and return evaluation metrics"""
        df = self.prepare_data(data)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['sentiment'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['sentiment'] if len(df) > len(SENTIMENT_CATEGORIES) * 2 else None
        )
        
        # Create text vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Create keyword feature extractor
        keyword_extractor = KeywordFeatureExtractor(KEYWORD_DICT)
        
        # Create a pipeline with a FeatureUnion that combines both TF-IDF and keyword features
        self.vectorizer = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', Pipeline([
                    ('vectorizer', tfidf_vectorizer)
                ])),
                ('keywords', keyword_extractor)
            ]))
        ])
        
        # Transform text to combined feature vectors (TF-IDF + keywords)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train the model using LogisticRegression
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs',
            random_state=random_state
        )
        self.model.fit(X_train_vec, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=SENTIMENT_CATEGORIES)
        
        # Save the model
        self.save_model()
        
        # Return evaluation metrics with hybrid model indication
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1Score': float(f1),
            'confusionMatrix': cm.tolist(),
            'labels': SENTIMENT_CATEGORIES,
            'trainingDate': datetime.datetime.now().isoformat(),
            'testSize': test_size,
            'sampleCount': len(df),
            'modelType': 'hybrid',  # Indicate this is a hybrid model
            'keywordFeatures': True
        }
        
        return metrics
    
    def predict(self, text, include_confidence=False):
        """Predict sentiment for a given text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        # Preprocess text
        processed_text = clean_text_preserve_indicators(text)
        
        # Vectorize text
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict sentiment
        sentiment = self.model.predict(text_vec)[0]
        
        if include_confidence:
            # Get prediction probabilities
            proba = self.model.predict_proba(text_vec)[0]
            # Get the confidence for the predicted class
            confidence = float(proba[self.model.classes_ == sentiment][0])
            return sentiment, confidence
        
        return sentiment
    
    def batch_predict(self, texts):
        """Predict sentiment for a batch of texts"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        # Preprocess texts
        processed_texts = [clean_text_preserve_indicators(text) for text in texts]
        
        # Vectorize texts
        texts_vec = self.vectorizer.transform(processed_texts)
        
        # Predict sentiments
        sentiments = self.model.predict(texts_vec)
        
        # Get prediction probabilities
        probas = self.model.predict_proba(texts_vec)
        
        results = []
        for i, (text, sentiment, proba) in enumerate(zip(texts, sentiments, probas)):
            # Get the confidence for the predicted class
            confidence = float(proba[self.model.classes_ == sentiment][0])
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return results
    
    def evaluate_csv(self, file_path, text_column='text', label_column='sentiment'):
        """
        Evaluate a CSV file containing text and labels
        Returns evaluation metrics
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            
            # If label column exists, use it for evaluation
            if label_column in df.columns:
                # Prepare data for evaluation
                data = []
                for _, row in df.iterrows():
                    data.append({
                        'text': row[text_column],
                        'sentiment': row[label_column] if row[label_column] in SENTIMENT_CATEGORIES else 'neutral'
                    })
                
                # Evaluate model on data
                metrics = self.train(data)
                return metrics, data
            
            # If label column doesn't exist, just predict
            else:
                # Predict sentiment for each row
                predictions = self.batch_predict(df[text_column].tolist())
                return None, predictions
                
        except Exception as e:
            print(f"Error evaluating CSV file: {e}")
            raise

def main():
    """Main function for testing"""
    trainer = HybridSentimentAnalysisTrainer()
    
    # Sample data for testing
    sample_data = [
        {"text": "Oh my god! It's flooding everywhere! We're trapped!", "sentiment": "panic"},
        {"text": "I hope everyone stays safe during this typhoon. 🙏", "sentiment": "resilience"},
        {"text": "This earthquake is scary. I'm worried about aftershocks.", "sentiment": "fear"},
        {"text": "Did that really just happen? I can't believe it.", "sentiment": "disbelief"},
        {"text": "PAGASA reports the typhoon will make landfall at 2pm today.", "sentiment": "neutral"},
        {"text": "We need to evacuate now! The flood is rising quickly!", "sentiment": "panic"},
        {"text": "Together, we can overcome this disaster. Stay strong!", "sentiment": "resilience"},
        {"text": "I'm afraid for my family's safety during this storm.", "sentiment": "fear"},
        {"text": "Is this real? I've never seen flooding this bad before.", "sentiment": "disbelief"},
        {"text": "The governor announced relief operations will begin tomorrow.", "sentiment": "neutral"}
    ]
    
    # Train and evaluate
    metrics = trainer.train(sample_data)
    print(json.dumps(metrics, indent=2))
    
    # Test prediction
    text = "Help! The earthquake destroyed our house!"
    sentiment, confidence = trainer.predict(text, include_confidence=True)
    print(f"Text: '{text}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.2f})")
    
    # Test how keywords affect prediction
    text_with_keywords = "Emergency! Help! The earthquake is causing panic everywhere!"
    sentiment, confidence = trainer.predict(text_with_keywords, include_confidence=True)
    print(f"Text with keywords: '{text_with_keywords}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()