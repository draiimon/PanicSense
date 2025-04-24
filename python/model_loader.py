"""
Model Loader for PanicSense Hybrid Model
Handles loading of PyTorch models from .pth files
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available for model loading")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not available - model loading will be simulated")

class BiGRULSTMModel(nn.Module):
    """
    Hybrid Bi-GRU & LSTM model for sentiment analysis
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim * 2 if bidirectional else hidden_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0 if n_layers < 2 else dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        gru_output, _ = self.gru(embedded)
        # gru_output = [batch size, sent len, hid dim * num directions]
        
        lstm_output, (hidden, cell) = self.lstm(gru_output)
        # lstm_output = [batch size, sent len, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        
        if self.gru.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
            
        # hidden = [batch size, hid dim * num directions]
        
        output = self.fc(self.dropout(hidden))
        # output = [batch size, out dim]
        
        return output

class ModelLoader:
    """
    Handles loading of pre-trained PyTorch models for sentiment analysis
    Supports .pth model files
    """
    
    def __init__(self, model_dir='models/sentiment'):
        """
        Initialize the model loader
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.models = {}
        self.vocab = None
        self.vocab_path = os.path.join(model_dir, 'vocab.json')
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize default configuration
        self.config = {
            'vocab_size': 25000,
            'embedding_dim': 300,
            'hidden_dim': 256,
            'output_dim': 5,  # 5 sentiment categories
            'n_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'pad_idx': 0
        }
        
        logger.info(f"Model loader initialized with directory: {model_dir}")
        self.load_available_models()
    
    def load_available_models(self):
        """Scan the model directory and load available models"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
        
        # Look for model files
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.endswith('.pth') or f.endswith('.pt')]
        
        if not model_files:
            logger.info(f"No model files found in {self.model_dir}")
            return
        
        # Try to load each model
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            try:
                self.load_model(model_path, model_name)
                logger.info(f"Successfully loaded model {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
    
    def load_model(self, model_path, model_name=None):
        """
        Load a model from a .pth file
        
        Args:
            model_path: Path to the model file
            model_name: Name to assign to the model (defaults to filename without extension)
        
        Returns:
            The loaded model
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, simulating model load")
            self.models[model_name or os.path.basename(model_path)] = "SIMULATED_MODEL"
            return None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Use filename as model name if not provided
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Check if config file exists alongside the model
        config_path = os.path.splitext(model_path)[0] + '.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded model config from {config_path}")
        
        # Create model with config
        model = BiGRULSTMModel(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim'],
            n_layers=self.config['n_layers'],
            bidirectional=self.config['bidirectional'],
            dropout=self.config['dropout']
        )
        
        # Load model state dict
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()  # Set model to evaluation mode
            
            # Store the loaded model
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_vocab(self, vocab_path=None):
        """
        Load vocabulary from a JSON file
        
        Args:
            vocab_path: Path to vocabulary file (defaults to vocab.json in model_dir)
        
        Returns:
            Dictionary mapping tokens to indices
        """
        if vocab_path is None:
            vocab_path = self.vocab_path
            
        if not os.path.exists(vocab_path):
            logger.warning(f"Vocabulary file not found: {vocab_path}")
            return None
            
        try:
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            self.vocab = vocab
            logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
            return vocab
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            return None
    
    def get_model(self, model_name=None):
        """
        Get a loaded model by name
        
        Args:
            model_name: Name of the model to retrieve
                       If None, returns the first loaded model
        
        Returns:
            The loaded model or None if not found
        """
        if not self.models:
            logger.warning("No models have been loaded")
            return None
            
        if model_name is None:
            # Return first model
            return next(iter(self.models.values()))
            
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return None
            
        return self.models[model_name]
    
    def predict(self, text, model_name=None):
        """
        Make a prediction with a loaded model
        
        Args:
            text: Text to analyze
            model_name: Name of the model to use (uses first model if None)
        
        Returns:
            List of prediction results
        """
        if not TORCH_AVAILABLE:
            # Simulate prediction
            import random
            sentiment_categories = ['Panic', 'Fear/Anxiety', 'Resilience', 'Neutral', 'Disbelief']
            return {
                'sentiment': random.choice(sentiment_categories),
                'confidence': random.uniform(0.6, 0.95)
            }
        
        model = self.get_model(model_name)
        if model is None:
            logger.error("No model available for prediction")
            return None
            
        # Here we would implement the actual tokenization and prediction
        # For now this is a placeholder for the full implementation
        # This would require the vocabulary and tokenizer
        
        # Simulating results since we need tokenizer and preprocessing
        import random
        sentiment_categories = ['Panic', 'Fear/Anxiety', 'Resilience', 'Neutral', 'Disbelief']
        return {
            'sentiment': random.choice(sentiment_categories),
            'confidence': random.uniform(0.6, 0.95)
        }


# Function to get available models
def get_available_models(model_dir='models/sentiment'):
    """
    Get a list of available model files
    
    Args:
        model_dir: Directory to search for model files
    
    Returns:
        List of model file paths
    """
    if not os.path.exists(model_dir):
        return []
        
    return [
        os.path.join(model_dir, f) 
        for f in os.listdir(model_dir) 
        if f.endswith('.pth') or f.endswith('.pt')
    ]


# Helper function to check if PyTorch and models are available
def check_model_availability():
    """
    Check if PyTorch is available and if models exist
    
    Returns:
        Dict with availability info
    """
    models = get_available_models()
    return {
        'torch_available': TORCH_AVAILABLE,
        'models_available': len(models) > 0,
        'model_count': len(models),
        'model_paths': models
    }