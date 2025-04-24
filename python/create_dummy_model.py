"""
Generate a dummy PyTorch model for testing
This script creates a simple .pth model file to test the model loader
"""

import os
import sys
import json
import numpy as np
import random
from datetime import datetime

# Create model directory if it doesn't exist
model_dir = os.path.join(os.getcwd(), 'models', 'sentiment')
os.makedirs(model_dir, exist_ok=True)

print(f"Creating dummy model in {model_dir}")

# Create a mock vocabulary file
vocab_size = 5000
vocab = {}
for i in range(vocab_size):
    vocab[f"token_{i}"] = i

# Save vocab to file
vocab_path = os.path.join(model_dir, 'vocab.json')
with open(vocab_path, 'w') as f:
    json.dump(vocab, f)
print(f"Created vocabulary file with {vocab_size} tokens: {vocab_path}")

# Create model config
model_config = {
    'vocab_size': vocab_size,
    'embedding_dim': 300,
    'hidden_dim': 256,
    'output_dim': 5,  # 5 sentiment categories
    'n_layers': 2,
    'bidirectional': True,
    'dropout': 0.5,
    'pad_idx': 0,
    'sentiment_labels': ['Panic', 'Fear/Anxiety', 'Resilience', 'Neutral', 'Disbelief'],
    'created': datetime.now().isoformat()
}

# Save model config
config_path = os.path.join(model_dir, 'bigru_lstm_model.json')
with open(config_path, 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"Created model config: {config_path}")

# Try to create a dummy .pth file if PyTorch is available
try:
    import torch
    import torch.nn as nn
    
    # Create a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
            self.gru = nn.GRU(config['embedding_dim'], config['hidden_dim'], 
                             num_layers=config['n_layers'], 
                             bidirectional=config['bidirectional'])
            self.lstm = nn.LSTM(config['hidden_dim'] * 2 if config['bidirectional'] else config['hidden_dim'],
                               config['hidden_dim'],
                               num_layers=config['n_layers'],
                               bidirectional=config['bidirectional'])
            self.fc = nn.Linear(config['hidden_dim'] * 2 if config['bidirectional'] else config['hidden_dim'], 
                              config['output_dim'])
            self.dropout = nn.Dropout(config['dropout'])
        
        def forward(self, x):
            embedded = self.embedding(x)
            gru_out, _ = self.gru(embedded)
            lstm_out, (hidden, _) = self.lstm(gru_out)
            
            if self.gru.bidirectional:
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1,:,:]
                
            return self.fc(self.dropout(hidden))
    
    # Create model
    model = DummyModel(model_config)
    
    # Save model
    model_path = os.path.join(model_dir, 'bigru_lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Created PyTorch model file: {model_path}")
    
except ImportError:
    print("PyTorch not available, creating dummy .pth file")
    
    # Create a binary file with random data as a placeholder
    model_path = os.path.join(model_dir, 'bigru_lstm_model.pth')
    with open(model_path, 'wb') as f:
        # Generate random binary data (100KB)
        random_data = np.random.bytes(102400)
        f.write(random_data)
    print(f"Created dummy binary model file: {model_path}")

print("Model creation completed successfully")