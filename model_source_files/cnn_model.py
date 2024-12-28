import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        
        # If dimensions change, apply a 1x1 conv to match dimensions for residual addition
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetEncoder1D(nn.Module):
    def __init__(self, input_dim, embedding_dim, cnn_filters):
        super(ResNetEncoder1D, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_dim, cnn_filters),
            nn.MaxPool1d(2),  # Reduces sequence length by 2
            ResidualBlock1D(cnn_filters, 2 * cnn_filters),
            nn.MaxPool1d(2),  # Further reduces sequence length by 2
            ResidualBlock1D(2 * cnn_filters, embedding_dim),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to reduce sequence length to 1
        )

    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # Transpose to (batch_size, input_dim, seq_len)
        encoded = self.encoder(x)  # Output: (batch_size, embedding_dim, 1)
        return encoded.squeeze(-1)  # Squeeze to (batch_size, embedding_dim)

class CNNFFNNModelWithResNet(nn.Module):
    def __init__(self, weather_dim, building_dim, embedding_dim, cnn_filters, ffnn_hidden_dim, dropout=0.1):
        super(CNNFFNNModelWithResNet, self).__init__()
        
        # ResNet-based encoder for weather data
        self.resnet_encoder = ResNetEncoder1D(input_dim=weather_dim, embedding_dim=embedding_dim, cnn_filters=cnn_filters)
        
        # FFNN for final prediction
        self.ffnn = nn.Sequential(
            nn.Linear(embedding_dim + building_dim, ffnn_hidden_dim),
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
        
        # Pass weather data through ResNet encoder
        resnet_embedding = self.resnet_encoder(weather_data)  # Shape: (batch_size, embedding_dim)
        
        # Concatenate the ResNet embedding with building features
        combined_features = torch.cat([resnet_embedding, building_features], dim=1)  # Shape: (batch_size, embedding_dim + building_dim)
        
        # Pass through the FFNN to get the final consumption prediction
        output = self.ffnn(combined_features)  # Shape: (batch_size, 1)
        
        return output

# Define function to build the model
def build_cnn_ffnn_model_with_resnet(weather_dim, building_dim, embedding_dim=16, cnn_filters=64, ffnn_hidden_dim=256, dropout=0.1):
    return CNNFFNNModelWithResNet(
        weather_dim=weather_dim,
        building_dim=building_dim,
        embedding_dim=embedding_dim,
        cnn_filters=cnn_filters,
        ffnn_hidden_dim=ffnn_hidden_dim,
        dropout=dropout
    )