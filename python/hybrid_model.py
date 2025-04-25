#!/usr/bin/env python3

"""
NeonDB-Powered Sentiment Analysis Model for PanicSense
Advanced cloud-based sentiment analysis system using NeonDB
Specifically designed for batch processing of CSV files
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
import re
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import emoji utils for text preprocessing
try:
    from emoji_utils import clean_text_preserve_indicators, preprocess_text
except ImportError:
    # Try with full path
    try:
        from python.emoji_utils import clean_text_preserve_indicators, preprocess_text
    except ImportError:
        logger.warning("Could not import emoji_utils module. Using simplified text processing.")
        def clean_text_preserve_indicators(text):
            return text
        def preprocess_text(text):
            return text

class NeonDBPanicSenseModel:
    """
    NeonDB-powered sentiment analysis model that uses cloud database for storage and analysis
    """
    def __init__(self, api_key=None, database_url=None):
        self.api_key = api_key or os.environ.get('NEON_API_KEY', '')
        self.database_url = database_url or os.environ.get('DATABASE_URL', '')
        
        # Sentiment label mappings
        self.sentiment_labels = ['Panic', 'Fear/Anxiety', 'Disbelief', 'Resilience', 'Neutral']
        
        # Check database connection
        self._check_database_connection()
        
    def _check_database_connection(self):
        """Check if the database connection is working"""
        try:
            # Import psycopg2 for database connection
            import psycopg2
            
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            cur.execute("SELECT current_database();")
            db_name = cur.fetchone()[0]
            
            logger.info(f"Successfully connected to NeonDB database: {db_name}")
            
            # Close connections
            cur.close()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to NeonDB: {e}")
            return False
    
    def predict_sentiment(self, text, max_length=128):
        """
        Predict sentiment for a single text input using rule-based approach
        
        Args:
            text (str): Input text to analyze
            max_length (int): Maximum text length to consider
            
        Returns:
            dict: Prediction results with sentiment, confidence, and explanation
        """
        # Preprocess text
        preprocessed_text = preprocess_text(clean_text_preserve_indicators(text))
        
        # Truncate text if needed
        if len(preprocessed_text) > max_length:
            preprocessed_text = preprocessed_text[:max_length]
        
        # Simple rule-based sentiment analysis
        # This is a placeholder for the removed neural network
        panic_indicators = ['help', 'emergency', 'trapped', 'danger', 'tulong', 'sunog', 'evacuate']
        fear_indicators = ['scared', 'afraid', 'worried', 'fear', 'anxious', 'takot', 'natatakot']
        disbelief_indicators = ['unbelievable', 'cannot believe', 'hindi kapani-paniwala', 'bakit']
        resilience_indicators = ['strong', 'together', 'overcome', 'hope', 'pray', 'malalagpasan']
        
        # Count occurrences of each indicator type in lowercase text
        text_lower = preprocessed_text.lower()
        
        panic_count = sum(1 for word in panic_indicators if word in text_lower)
        fear_count = sum(1 for word in fear_indicators if word in text_lower)
        disbelief_count = sum(1 for word in disbelief_indicators if word in text_lower)
        resilience_count = sum(1 for word in resilience_indicators if word in text_lower)
        
        # Determine sentiment based on highest count
        counts = {
            'Panic': panic_count,
            'Fear/Anxiety': fear_count,
            'Disbelief': disbelief_count,
            'Resilience': resilience_count,
            'Neutral': 0  # Default value
        }
        
        # If no indicators found, set to Neutral
        if sum(counts.values()) == 0:
            sentiment = 'Neutral'
            confidence_score = 0.7
        else:
            # Get sentiment with highest count
            sentiment = max(counts, key=counts.get)
            
            # Calculate confidence based on relative frequency
            total_indicators = sum(counts.values())
            sentiment_count = counts[sentiment]
            confidence_score = min(0.95, 0.5 + (sentiment_count / total_indicators) * 0.5)
        
        # Generate explanation
        explanation = self._generate_explanation(preprocessed_text, sentiment, confidence_score)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence_score,
            "explanation": explanation
        }
        
    def _generate_explanation(self, text, sentiment, confidence):
        """Generate human-readable explanation for the sentiment prediction"""
        # Define explanation templates based on sentiment and confidence
        high_confidence = confidence > 0.85
        medium_confidence = 0.65 <= confidence <= 0.85
        low_confidence = confidence < 0.65
        
        explanations = {
            "Panic": {
                "high": "Strong indicators of panic detected, such as urgent language, emotional distress, and calls for help.",
                "medium": "Moderate signs of panic present, including some urgent language and expressions of distress.",
                "low": "Some panic indicators detected, but mixed with other sentiments or ambiguous language."
            },
            "Fear/Anxiety": {
                "high": "Clear expressions of fear and anxiety, including worry about safety and uncertain outcomes.",
                "medium": "Moderate indicators of fear or anxiety, showing concern without extreme emotional distress.",
                "low": "Some anxiety indicators present, but possibly mixed with other sentiments."
            },
            "Disbelief": {
                "high": "Strong expressions of disbelief, shock, or sarcasm in response to the situation.",
                "medium": "Moderate indicators of disbelief or questioning of the circumstances.",
                "low": "Some elements of disbelief or surprise, but possibly mixed with other reactions."
            },
            "Resilience": {
                "high": "Strong resilience indicators, including positive outlook, support for others, or calls for unity.",
                "medium": "Moderate resilience signals showing some coping or adaptation strategies.",
                "low": "Some resilience elements detected, but possibly mixed with other emotions."
            },
            "Neutral": {
                "high": "Highly factual and objective content with minimal emotional indicators.",
                "medium": "Mostly factual content with some subtle emotional undertones.",
                "low": "Somewhat neutral content but contains some emotional elements."
            }
        }
        
        # Select explanation based on confidence level
        if high_confidence:
            explanation = explanations[sentiment]["high"]
        elif medium_confidence:
            explanation = explanations[sentiment]["medium"]
        else:
            explanation = explanations[sentiment]["low"]
            
        # Add text-specific details
        # Extract potential keywords or phrases that influenced the decision
        key_phrases = self._extract_key_phrases(text, sentiment)
        if key_phrases:
            explanation += f" Key phrases like '{key_phrases}' contributed to this assessment."
            
        return explanation
    
    def _extract_key_phrases(self, text, sentiment):
        """Extract key phrases that likely influenced the sentiment prediction"""
        # Define sentiment-specific keywords
        sentiment_keywords = {
            "Panic": ["help", "rescue", "emergency", "trapped", "danger", "tulong", "tulungan", "naiipit"],
            "Fear/Anxiety": ["scared", "afraid", "worried", "fear", "anxious", "takot", "natatakot", "kabado"],
            "Disbelief": ["unbelievable", "cannot believe", "what", "seriously", "hindi kapani-paniwala", "talaga"],
            "Resilience": ["strong", "together", "overcome", "hope", "pray", "kapit", "kakayanin", "tulong"],
            "Neutral": ["report", "announce", "information", "update", "advisory", "balita", "ulat"]
        }
        
        # Check for keywords in text
        keywords = sentiment_keywords.get(sentiment, [])
        matched_phrases = []
        
        for keyword in keywords:
            # Find whole word matches (with word boundaries)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text.lower()):
                # Get the context around the keyword (up to 5 words)
                for match in re.finditer(pattern, text.lower()):
                    start, end = match.span()
                    # Get words before and after
                    words = text.split()
                    keyword_index = -1
                    for i, word in enumerate(words):
                        if keyword.lower() in word.lower():
                            keyword_index = i
                            break
                    
                    if keyword_index >= 0:
                        start_idx = max(0, keyword_index - 2)
                        end_idx = min(len(words), keyword_index + 3)
                        context = " ".join(words[start_idx:end_idx])
                        matched_phrases.append(context)
        
        # Return up to 2 unique phrases
        unique_phrases = list(set(matched_phrases))
        if len(unique_phrases) > 2:
            return ", ".join(unique_phrases[:2])
        elif unique_phrases:
            return ", ".join(unique_phrases)
        else:
            return ""


class HybridModelProcessor:
    """
    Processor class for batch processing of CSV files using the hybrid model
    """
    def __init__(self, model_path=None, device="cpu"):
        """
        Initialize the processor
        
        Args:
            model_path (str): Path to the pretrained model weights (if None, load from default or train)
            device (str): Device to run model on ('cpu' or 'cuda')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        
        # Initialize model
        self.model = HybridPanicSenseModel()
        
        # Load model weights if available
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model weights found. Using initialized model.")
        
        # Move model to device
        self.model.to(self.device)
        
        # Rule-based system for sanity checking and fallbacks
        self.setup_rule_based_system()
        
    def setup_rule_based_system(self):
        """Set up the rule-based system for fallbacks and validation"""
        # Define keyword lists for each sentiment category
        self.sentiment_keywords = {
            'Panic': [
                'help', 'rescue', 'emergency', 'trapped', 'danger', 'evacuate now', 'sos',
                'tulong', 'saklolo', 'tulungan', 'naiipit', 'delikado', 'HELP', 'TULONG',
                'mamamatay na', 'hindi makalabas', 'hindi makahinga'
            ],
            'Fear/Anxiety': [
                'scared', 'afraid', 'worried', 'fear', 'anxiety', 'nervous', 'terrified',
                'takot', 'natatakot', 'kabado', 'kinakabahan', 'nangangamba', 'hindi mapakali',
                'may balita ba', 'safe ba', 'scary', 'frightening'
            ],
            'Disbelief': [
                'unbelievable', 'cannot believe', 'what?', 'seriously?', 'really?', 'omg',
                'hindi kapani-paniwala', 'talaga?', 'totoo ba?', 'ganun?', 'grabe naman',
                'hindi makapaniwala', 'jusko', 'nakakagulat'
            ],
            'Resilience': [
                'strong', 'together', 'overcome', 'hope', 'pray', 'support', 'help each other',
                'kapit', 'kakayanin', 'magtulung-tulungan', 'magtulungan', 'dasal', 'kaya natin',
                'stay safe', 'kaya natin ito', 'matibay tayo', 'magkaisa'
            ],
            'Neutral': [
                'report', 'announce', 'information', 'update', 'advisory', 'notice', 'bulletin',
                'balita', 'ulat', 'abiso', 'impormasyon', 'anunsyo', 'according to',
                'ayon sa', 'magnitude', 'intensity', 'location'
            ]
        }
        
        # Weights for different signals in rule-based analysis
        self.rule_weights = {
            'keywords': 0.5,    # Weight for keyword matches
            'punctuation': 0.2, # Weight for punctuation analysis (!!!, caps)
            'length': 0.1,      # Weight for text length
            'emojis': 0.2,      # Weight for emoji presence
        }
        
        # Emotion-indicative punctuation and patterns
        self.punctuation_patterns = {
            'Panic': [r'!{2,}', r'[A-Z]{3,}', r'\?{2,}!+'],
            'Fear/Anxiety': [r'\?{2,}', r'\.{3,}'],
            'Disbelief': [r'\?!+', r'\?{3,}'],
            'Resilience': [r'!', r'💪', r'🙏'],
            'Neutral': [r'\.', r':']
        }
        
    def process_csv(self, csv_path, text_column='text', max_length=128, batch_size=32):
        """
        Process a CSV file using the hybrid model
        
        Args:
            csv_path (str): Path to CSV file
            text_column (str): Column name containing text to analyze
            max_length (int): Maximum sequence length for tokenization
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of dictionaries with analysis results
        """
        try:
            logger.info(f"Reading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            if text_column not in df.columns:
                logger.error(f"Text column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
                return []
            
            # Prepare inputs for batch processing
            total_rows = len(df)
            results = []
            
            # Process in batches
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
                batch_df = df.iloc[i:min(i+batch_size, total_rows)]
                batch_texts = batch_df[text_column].tolist()
                batch_results = self.process_batch(batch_texts, max_length)
                
                # Add results and report progress
                results.extend(batch_results)
                
                # Report progress for integration with Node.js
                progress_percentage = min(100, int((i + batch_size) / total_rows * 100))
                progress_data = {
                    "processed": min(i + batch_size, total_rows),
                    "total": total_rows,
                    "stage": "Hybrid Model Analysis"
                }
                print(f"PROGRESS:{json.dumps(progress_data)}::END_PROGRESS", flush=True)
            
            # Add additional fields from the original CSV
            final_results = []
            for i, result in enumerate(results):
                row_data = df.iloc[i].to_dict()
                # Remove text column to avoid duplication
                if text_column in row_data:
                    row_data.pop(text_column)
                    
                # Create final result with all metadata
                final_result = {
                    "text": df.iloc[i][text_column],
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"]
                }
                
                # Add other columns from original CSV
                for key, value in row_data.items():
                    if key not in final_result and pd.notna(value):
                        final_result[key] = value
                
                final_results.append(final_result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return []
    
    def process_batch(self, texts, max_length=128):
        """
        Process a batch of texts using the model
        
        Args:
            texts (list): List of text strings to analyze
            max_length (int): Maximum sequence length for tokenization
            
        Returns:
            list: List of dictionaries with analysis results
        """
        self.model.eval()
        results = []
        
        # Clean and preprocess texts
        preprocessed_texts = [preprocess_text(clean_text_preserve_indicators(text)) for text in texts]
        
        # Tokenize batch
        encoded_inputs = self.tokenizer(
            preprocessed_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded_inputs['input_ids'],
                attention_mask=encoded_inputs['attention_mask']
            )
            probs = F.softmax(outputs, dim=1)
            confidences, predicted_classes = torch.max(probs, dim=1)
        
        # Process results
        for i, (text, pred_class, confidence) in enumerate(zip(texts, predicted_classes, confidences)):
            sentiment = self.model.sentiment_labels[pred_class.item()]
            confidence_score = confidence.item()
            
            # Apply rule-based validation and potentially override
            rule_based_result = self.apply_rule_based_analysis(text)
            
            # If model confidence is low and rule-based system has a strong opinion, use rule-based result
            if confidence_score < 0.65 and rule_based_result["confidence"] > 0.75:
                sentiment = rule_based_result["sentiment"]
                confidence_score = rule_based_result["confidence"]
                explanation = f"The model initially detected '{self.model.sentiment_labels[pred_class.item()]}' with lower confidence, but rule-based analysis strongly indicated '{sentiment}'. " + rule_based_result["explanation"]
            else:
                # Generate explanation based on the model's prediction
                explanation = f"The model detected '{sentiment}' with {confidence_score:.2f} confidence. "
                explanation += self.model._generate_explanation(preprocessed_texts[i], sentiment, confidence_score)
            
            results.append({
                "sentiment": sentiment,
                "confidence": confidence_score,
                "explanation": explanation
            })
        
        return results
    
    def apply_rule_based_analysis(self, text):
        """
        Apply rule-based analysis as a fallback or validation mechanism
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Rule-based analysis result with sentiment and confidence
        """
        # Initialize scores for each sentiment category
        scores = {sentiment: 0.0 for sentiment in self.sentiment_keywords.keys()}
        
        # Clean text for analysis
        clean_text = text.lower()
        
        # 1. Check for keywords (50% weight)
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', clean_text):
                    scores[sentiment] += self.rule_weights['keywords'] / len(keywords)
        
        # 2. Check punctuation patterns (20% weight)
        for sentiment, patterns in self.punctuation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    scores[sentiment] += self.rule_weights['punctuation'] / len(patterns)
        
        # 3. Text length analysis (10% weight)
        text_length = len(text)
        if text_length < 15:  # Very short texts often indicate panic or strong emotions
            scores['Panic'] += self.rule_weights['length'] * 0.5
            scores['Disbelief'] += self.rule_weights['length'] * 0.3
        elif 15 <= text_length <= 50:  # Medium texts could be fear/anxiety or disbelief
            scores['Fear/Anxiety'] += self.rule_weights['length'] * 0.4
            scores['Disbelief'] += self.rule_weights['length'] * 0.4
        else:  # Longer texts often indicate neutral reporting or resilience
            scores['Neutral'] += self.rule_weights['length'] * 0.5
            scores['Resilience'] += self.rule_weights['length'] * 0.3
        
        # 4. Emoji analysis (20% weight)
        emoji_patterns = {
            'Panic': [r'😱', r'😨', r'🆘', r'⚠️', r'🔥'],
            'Fear/Anxiety': [r'😟', r'😰', r'😧', r'😦', r'😮'],
            'Disbelief': [r'😮', r'😕', r'🤔', r'😑', r'🙄'],
            'Resilience': [r'💪', r'🙏', r'❤️', r'🤝', r'✊'],
            'Neutral': [r'🔍', r'📋', r'📢', r'📰', r'📊']
        }
        
        for sentiment, patterns in emoji_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    scores[sentiment] += self.rule_weights['emojis'] / len(patterns)
        
        # Get the sentiment with the highest score
        max_sentiment = max(scores.items(), key=lambda x: x[1])
        sentiment = max_sentiment[0]
        confidence = min(0.9, max(0.5, max_sentiment[1]))  # Scale confidence between 0.5 and 0.9
        
        # Generate explanation
        explanation = "Rule-based analysis detected "
        
        # Add key indicators to explanation
        indicators = []
        if any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', clean_text) 
               for keyword in self.sentiment_keywords.get(sentiment, [])):
            indicators.append("key emotional words")
        
        if any(re.search(pattern, text) for pattern in self.punctuation_patterns.get(sentiment, [])):
            indicators.append("punctuation patterns")
        
        if any(re.search(pattern, text) for pattern in emoji_patterns.get(sentiment, [])):
            indicators.append("emotional emojis")
            
        if indicators:
            explanation += f"{', '.join(indicators)} indicating {sentiment.lower()}."
        else:
            explanation += f"a pattern consistent with {sentiment.lower()}."
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "explanation": explanation
        }
    
    def train_model(self, train_texts, train_labels, validation_split=0.2, epochs=5, batch_size=16, learning_rate=2e-5):
        """
        Train the hybrid model on labeled data
        
        Args:
            train_texts (list): List of text strings for training
            train_labels (list): List of sentiment labels corresponding to texts
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for optimization
            
        Returns:
            dict: Training metrics
        """
        logger.info("Starting model training...")
        
        # Convert sentiment labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(self.model.sentiment_labels)}
        train_label_indices = [label_to_idx.get(label, 0) for label in train_labels]
        
        # Split into train and validation
        val_size = int(len(train_texts) * validation_split)
        train_size = len(train_texts) - val_size
        
        # Shuffle and split
        indices = np.random.permutation(len(train_texts))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Set up data
        train_x = [train_texts[i] for i in train_indices]
        train_y = [train_label_indices[i] for i in train_indices]
        val_x = [train_texts[i] for i in val_indices]
        val_y = [train_label_indices[i] for i in val_indices]
        
        # Set model to training mode
        self.model.train()
        
        # Optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            # Process in batches
            for i in range(0, len(train_x), batch_size):
                batch_texts = train_x[i:i+batch_size]
                batch_labels = torch.tensor(train_y[i:i+batch_size], device=self.device)
                
                # Tokenize
                encoded_inputs = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=encoded_inputs['input_ids'],
                    attention_mask=encoded_inputs['attention_mask']
                )
                
                # Calculate loss and backpropagate
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / (len(train_x) // batch_size + 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(val_x), batch_size):
                    batch_texts = val_x[i:i+batch_size]
                    batch_labels = torch.tensor(val_y[i:i+batch_size], device=self.device)
                    
                    # Tokenize
                    encoded_inputs = self.tokenizer(
                        batch_texts,
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=encoded_inputs['input_ids'],
                        attention_mask=encoded_inputs['attention_mask']
                    )
                    
                    # Calculate loss
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            avg_val_loss = val_loss / (len(val_x) // batch_size + 1)
            val_losses.append(avg_val_loss)
            val_accuracy = correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save model
                os.makedirs("models", exist_ok=True)
                torch.save(self.model.state_dict(), "models/hybrid_model_best.pt")
                logger.info("Saved best model checkpoint")
        
        # Load best model for inference
        self.model.load_state_dict(torch.load("models/hybrid_model_best.pt", map_location=self.device))
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_val_accuracy": val_accuracy,
            "final_val_loss": avg_val_loss
        }
    
    def save_model(self, path="models/hybrid_model.pt"):
        """Save model weights to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path="models/hybrid_model.pt"):
        """Load model weights from file"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
            return True
        else:
            logger.error(f"Model file not found at {path}")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Neural Network Model for Disaster Sentiment Analysis")
    parser.add_argument("--csv", type=str, help="CSV file to process")
    parser.add_argument("--output", type=str, default="output.json", help="Output JSON file")
    parser.add_argument("--text_column", type=str, default="text", help="Column name with text data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model weights")
    
    args = parser.parse_args()
    
    if args.csv:
        # Initialize processor
        processor = HybridModelProcessor(model_path=args.model_path, device=args.device)
        
        # Process CSV
        results = processor.process_csv(args.csv, text_column=args.text_column, batch_size=args.batch_size)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {args.output}")
    else:
        logger.error("No CSV file specified. Use --csv to specify input file.")