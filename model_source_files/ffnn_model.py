import torch
import torch.nn as nn

class FFNNModel(nn.Module):
    def __init__(self, weather_dim, building_dim, hidden_dim=64, dropout=0.1):
        super(FFNNModel, self).__init__()
        
        # Fully connected layers to process concatenated weather and building features
        self.ffnn = nn.Sequential(
            nn.Linear(weather_dim + building_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(hidden_dim/ 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim/ 2), 1)  # Output a single prediction for weekly consumption
        )
    
    def forward(self, weather_embedding, building_features):
        # Concatenate weather embeddings and building features along the feature dimension
        combined_features = torch.cat([weather_embedding, building_features], dim=1)  # Shape: (batch_size, weather_dim + building_dim)
        
        # Pass through the feedforward network
        output = self.ffnn(combined_features)  # Shape: (batch_size, 1)
        
        return output

# Initialize the model with specified dimensions
def build_ffnn_model(weather_dim, building_dim, hidden_dim=128, dropout=0.2456):
    return FFNNModel(
        weather_dim=weather_dim,
        building_dim=building_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )