import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from torch.utils.data import random_split


# Define a function to fit the MinMaxScaler across all files
def fit_minmax_scaler(file_paths, weather_features):
    scaler = MinMaxScaler()
    combined_data = []

    for file_path in file_paths:
        # Load EPW data
        weather_df = pd.read_csv(file_path, skiprows=8, header=None)
        weather_df.columns = [
            "Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags", 
            "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", 
            "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation", 
            "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity from Sky", 
            "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation", 
            "Global Horizontal Illuminance", "Direct Normal Illuminance", "Diffuse Horizontal Illuminance", 
            "Zenith Luminance", "Wind Direction", "Wind Speed", "Total Sky Cover", 
            "Opaque Sky Cover", "Visibility", "Ceiling Height", "Present Weather Observation", 
            "Present Weather Codes", "Precipitable Water", "Aerosol Optical Depth (AOD)", 
            "Snow Depth", "Days Since Last Snowfall", "Albedo", "Liquid Precipitation Depth", 
            "Liquid Precipitation Quantity"
        ]
        
        # Filter only the selected weather features
        weather_df = weather_df[weather_features]
        combined_data.append(weather_df)
    
    # Concatenate all data for fitting
    combined_data = pd.concat(combined_data)
    
    # Fit the scaler on the combined data
    scaler.fit(combined_data)

    return scaler

# Define a function to parse and scale the data using the fitted scaler
def parse_multiple_epw(file_paths, weather_features, scaler):
    weekly_data_all = []

    for file_path in file_paths:
        # Load EPW data
        weather_df = pd.read_csv(file_path, skiprows=8, header=None)
        weather_df.columns = [
            "Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags", 
            "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", 
            "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation", 
            "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity from Sky", 
            "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation", 
            "Global Horizontal Illuminance", "Direct Normal Illuminance", "Diffuse Horizontal Illuminance", 
            "Zenith Luminance", "Wind Direction", "Wind Speed", "Total Sky Cover", 
            "Opaque Sky Cover", "Visibility", "Ceiling Height", "Present Weather Observation", 
            "Present Weather Codes", "Precipitable Water", "Aerosol Optical Depth (AOD)", 
            "Snow Depth", "Days Since Last Snowfall", "Albedo", "Liquid Precipitation Depth", 
            "Liquid Precipitation Quantity"
        ]
        
        # Filter only the selected weather features
        weather_df = weather_df[weather_features]
        
        # Scale the data
        weather_df = pd.DataFrame(scaler.transform(weather_df), columns=weather_features)
        
        # Reshape data to create 52 weeks, each with 168 hourly records
        for i in range(52):
            weekly_segment = weather_df.iloc[i * 168: (i + 1) * 168].values
            weekly_data_all.append(weekly_segment)
    
    return np.array(weekly_data_all)  # Shape: (num_weeks, 168, num_features)

# Custom Dataset for concatenated weekly weather data from multiple files
class WeeklyWeatherDataset(Dataset):
    def __init__(self, weather_data):
        self.weather_data = weather_data

    def __len__(self):
        return len(self.weather_data)

    def __getitem__(self, idx):
        return torch.tensor(self.weather_data[idx], dtype=torch.float32)

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

class ResNetAutoencoder1D(nn.Module):
    def __init__(self, input_dim, embedding_dim=16, cnn_filters=64):
        super(ResNetAutoencoder1D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_dim, cnn_filters),
            nn.MaxPool1d(2),  # Reduces sequence length by 2
            ResidualBlock1D(cnn_filters, 2*cnn_filters),
            nn.MaxPool1d(2),  # Further reduces sequence length by 2
            ResidualBlock1D(2*cnn_filters, embedding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 2*cnn_filters, kernel_size=4, stride=2, padding=1),  # Upsample by 2
            nn.ReLU(),
            nn.ConvTranspose1d(2*cnn_filters, cnn_filters, kernel_size=4, stride=2, padding=1),  # Upsample by 2
            nn.ReLU(),
            nn.ConvTranspose1d(cnn_filters, input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()  # Use Sigmoid if input data is normalized between 0 and 1
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add a batch dimension if missing

        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)  
        decoded = self.decoder(encoded)  
        decoded = decoded.permute(0, 2, 1)
        return encoded, decoded

# Function to plot and save the reconstruction after each epoch
def plot_and_save_reconstruction(model, dataset, epoch, index=0):
    model.eval()
    original_data = dataset[index].unsqueeze(0).to(device)  # Add batch dimension and move to device
    _, reconstructed_data = model(original_data)
    
    original_data = original_data.squeeze(0).cpu().detach().numpy()
    reconstructed_data = reconstructed_data.squeeze(0).cpu().detach().numpy()
    
    num_features = original_data.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 2 * num_features))
    for i in range(num_features):
        axes[i].plot(original_data[:, i], label='Original', color='b')
        axes[i].plot(reconstructed_data[:, i], label='Reconstructed', color='r', linestyle='--')
        axes[i].set_title(weather_features[i])
        axes[i].legend(loc='upper right')
    plt.tight_layout()
    
    os.makedirs("reconstructions2", exist_ok=True)
    plt.savefig(f"reconstructions2/reconstruction_epoch_{epoch+1}.png")
    plt.close()

def train_autoencoder(weather_file_paths, val_file_paths, num_epochs=50, batch_size=8, learning_rate=0.0008311009890997349, embedding_dim=32):
    # Fit the scaler and prepare the datasets
    scaler = fit_minmax_scaler(weather_file_paths + val_file_paths, weather_features)
    
    # Parse training data
    weekly_weather_data_train = parse_multiple_epw(weather_file_paths, weather_features, scaler)
    input_dim = weekly_weather_data_train.shape[2]
    
    # Parse validation data
    weekly_weather_data_val = parse_multiple_epw(val_file_paths, weather_features, scaler)

    # Save the scaler
    os.makedirs("model", exist_ok=True)
    scaler_path = "model/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Create the Datasets
    train_dataset = WeeklyWeatherDataset(weekly_weather_data_train)
    val_dataset = WeeklyWeatherDataset(weekly_weather_data_val)
    
    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = ResNetAutoencoder1D(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to the GPU if available
            optimizer.zero_grad()
            
            # Forward pass
            _, decoded = model(batch)
            loss = criterion(decoded, batch)  # Compute the reconstruction loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, decoded = model(batch)
                loss = criterion(decoded, batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Plot and save the reconstruction for one sample after each epoch
        plot_and_save_reconstruction(model, train_dataset, epoch, index=0)
    
    # Save the trained model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/autoencoder_model_2.pth")
    print("Model saved to model/autoencoder_model.pth")


if __name__ == "__main__":
    # Define weather features to select from EPW data
    weather_features = ["Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", "Wind Speed", "Total Sky Cover", "Wind Direction"]

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training and validation file paths
    train_file_paths = [
        "../weather_files/CAN_ON_London_2009_2023.epw",
        "../weather_files/CAN_ON_London_2004_2018.epw",
        "../weather_files/CAN_AB_Rocky_2009_2023.epw",
        "../weather_files/CAN_ON_Toronto_2009_2023.epw",
        "../weather_files/CAN_BC_Summerland_2009_2023.epw"
    ]
    val_file_paths = [
        "../weather_files/CAN_ON_London_2004_2018.epw"
    ]
    
    # Train the autoencoder
    autoencoder_model = train_autoencoder(train_file_paths, val_file_paths)