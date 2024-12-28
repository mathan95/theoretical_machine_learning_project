import torch
import torch.nn as nn

class TransformerFFNNModel(nn.Module):
    def __init__(self, weather_dim, building_dim, seq_len, embed_dim=16, num_heads=4, num_layers=4, ff_hidden_dim=64, ffnn_hidden_dim=64, dropout=0.1):
        super(TransformerFFNNModel, self).__init__()
        
        # Embedding layer to map weather features to the Transformer embedding dimension
        self.weather_embedding = nn.Linear(weather_dim, embed_dim)
        
        # Positional encoding for the Transformer (flexible sinusoidal)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)

        # Check if embed_dim is even or odd
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            # For odd embed_dim, handle the last column separately
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Skip last value
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer for use in forward pass
        self.register_buffer('pos_encoder', pe)
        
        # Transformer Encoder to process the weather time series
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Reduced to 4 layers
        
        # FFNN for final prediction
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim + building_dim, ffnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1)  # Output a single prediction for weekly consumption
        )
    
    def forward(self, weather_data, building_data):
        # weather_data shape: (batch_size, seq_len, weather_dim)
        # building_data shape: (batch_size, building_dim)
        weather_data = self.weather_embedding(weather_data)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding to the weather data
        weather_data += self.pos_encoder.unsqueeze(0)
        
        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(weather_data)  # Shape: (batch_size, seq_len, embed_dim)
        
        # Reduce to fixed size by pooling (mean pooling)
        transformer_embedding = transformer_output.mean(dim=1)  # Shape: (batch_size, embed_dim)
        
        # Concatenate the transformer embedding with building features
        combined_features = torch.cat([transformer_embedding, building_data], dim=1)  # Shape: (batch_size, embed_dim + building_dim)
        
        # Pass through the FFNN to get the final consumption prediction
        output = self.ffnn(combined_features)  # Shape: (batch_size, 1)
        
        return output

# Initialize the model with sample dimensions
def build_custom_transformer_model(weather_dim, building_dim, seq_len=168, embed_dim=8, num_heads=8, num_layers=4, ff_hidden_dim=256, ffnn_hidden_dim=64, dropout=0.1):
    return TransformerFFNNModel(
        weather_dim=weather_dim,
        building_dim=building_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        ffnn_hidden_dim=ffnn_hidden_dim,
        dropout=dropout
    )