import torch
import torch.nn as nn

class LSTMFFNNModel(nn.Module):
    def __init__(self, weather_dim, building_dim, lstm_hidden_dim, ffnn_hidden_dim, dropout=0.1):
        super(LSTMFFNNModel, self).__init__()
        
        # LSTM to process weather time series
        self.lstm = nn.LSTM(input_size=weather_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        
        # FFNN for final prediction
        self.ffnn = nn.Sequential(
            nn.Linear(lstm_hidden_dim + building_dim, ffnn_hidden_dim),
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
    
    def forward(self, weather_data, building_features):
        # weather_data shape: (batch_size, seq_len, weather_dim)
        # building_features shape: (batch_size, building_dim)

        # Pass through LSTM
        lstm_output, _ = self.lstm(weather_data)  # Shape: (batch_size, seq_len, lstm_hidden_dim)
        
        # Reduce LSTM output to a fixed size by taking the mean across timesteps
        lstm_embedding = lstm_output.mean(dim=1)  # Shape: (batch_size, lstm_hidden_dim)
        
        # Concatenate the LSTM embedding with building features
        combined_features = torch.cat([lstm_embedding, building_features], dim=1)  # Shape: (batch_size, lstm_hidden_dim + building_dim)
        
        # Pass through the FFNN to get the final consumption prediction
        output = self.ffnn(combined_features)  # Shape: (batch_size, 1)
        
        return output

# Define function to build LSTM model
def build_lstm_ffnn_model(weather_dim, building_dim, lstm_hidden_dim=8, ffnn_hidden_dim=256, dropout=0.1):
    return LSTMFFNNModel(
        weather_dim=weather_dim,
        building_dim=building_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        ffnn_hidden_dim=ffnn_hidden_dim,
        dropout=dropout
    )