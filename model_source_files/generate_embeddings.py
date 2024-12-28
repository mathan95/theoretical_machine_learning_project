import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from weather_encoder import fit_minmax_scaler



# Load your trained scaler
def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please ensure it was saved during training.")
    return joblib.load(scaler_path)

# Load your trained model
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetAutoencoder1D(nn.Module):
    def __init__(self, input_dim, embedding_dim=16, seq_len=168):
        super(ResNetAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_dim, 64),
            nn.MaxPool1d(2),
            ResidualBlock1D(64, 128),
            nn.MaxPool1d(2),
            ResidualBlock1D(128, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        return encoded

# Set device and load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 6  # Number of weather features used
embedding_dim = 32  # Embedding dimension as used during training
model = ResNetAutoencoder1D(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
model.load_state_dict(torch.load("model/autoencoder_model_2.pth"))
model.eval()

# Load and preprocess the selected weather file
def load_and_preprocess_weather(file_path, scaler, weather_features):
    weather_df = pd.read_csv(file_path, skiprows=8, header=None)
    weather_df.columns = ["Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags",
                          "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", 
                          "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation", 
                          "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity from Sky", 
                          "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation", 
                          "Global Horizontal Illuminance", "Direct Normal Illuminance", "Diffuse Horizontal Illuminance", 
                          "Zenith Luminance", "Wind Direction", "Wind Speed", "Total Sky Cover", 
                          "Opaque Sky Cover", "Visibility", "Ceiling Height", "Present Weather Observation", 
                          "Present Weather Codes", "Precipitable Water", "Aerosol Optical Depth (AOD)", 
                          "Snow Depth", "Days Since Last Snowfall", "Albedo", "Liquid Precipitation Depth", 
                          "Liquid Precipitation Quantity"]
    weather_df = weather_df[weather_features]
    weather_df = pd.DataFrame(scaler.transform(weather_df), columns=weather_features)
    return weather_df

# Generate embeddings for the entire 52 weeks and save them
def generate_embeddings_for_weeks(weather_file_path, scaler, model, embedding_dim):
    weather_data = load_and_preprocess_weather(weather_file_path, scaler, weather_features)
    weekly_embeddings = []
    
    for i in range(52):
        weekly_segment = weather_data.iloc[i * 168: (i + 1) * 168].values  # 168 hours per week
        weekly_segment = torch.tensor(weekly_segment, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 168, num_features)
        
        with torch.no_grad():
            embedding = model(weekly_segment).squeeze(0).cpu().numpy()  # Shape: (embedding_dim, seq_len)
            weekly_embeddings.append(np.mean(embedding, axis=1))  # Average pooling over the sequence dimension
    
    # Save embeddings to a file
    weekly_embeddings = np.array(weekly_embeddings)  # Shape: (52, embedding_dim)
    output_df = pd.DataFrame(weekly_embeddings, columns=[f"embedding_{i+1}" for i in range(embedding_dim)])
    output_df.to_csv("../embeddings/CAN_AB_Rocky_09_23_embeddings.csv", index=False)
    print("Embeddings saved to output directory")

# Define weather features to select from EPW data
weather_features = ["Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", "Wind Speed", "Total Sky Cover", "Wind Direction"]

# Example usage
weather_file_path = "../weather_files/CAN_AB_Rocky_2009_2023.epw"
scaler_path = "model/scaler.pkl"  # Path to the saved scaler
scaler = load_scaler(scaler_path)  # Load the saved scaler

generate_embeddings_for_weeks(weather_file_path, scaler, model, embedding_dim)