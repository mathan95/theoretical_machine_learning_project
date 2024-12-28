import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import DataLoader
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from weather_encoder import ResNetAutoencoder1D, WeeklyWeatherDataset, fit_minmax_scaler, parse_multiple_epw

# Define results file path
results_file = "resnet_autoencoder_tuning.csv"

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Weather features to select from EPW data
weather_features = ["Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", "Wind Speed", "Total Sky Cover", "Wind Direction"]

# Weather file paths
weather_file_paths = [
    "../weather_files/CAN_ON_London_2009_2023.epw",
    "../weather_files/CAN_ON_London_2004_2018.epw",
    "../weather_files/CAN_ON_Toronto_2009_2023.epw",
    "../weather_files/CAN_BC_Summerland_2009_2023.epw",
    "../weather_files/CAN_AB_Rocky_2009_2023.epw",
]


# # Weather file paths
# weather_file_paths = [
#     "/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2009_2023.epw",
#     "/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2004_2018.epw",
#     "/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_Toronto_2009_2023.epw",
#     "/home/pmanmat/scratch/tldo/weatherfiles/CAN_BC_Summerland_2009_2023.epw"
# ]


# Objective function for Optuna
def objective(trial):
    print(f"Starting trial {trial.number} with parameters...")

    # Hyperparameter suggestions
    embedding_dim = trial.suggest_int("embedding_dim", 8, 128, step=8)
    batch_size = trial.suggest_int("batch_size", 8, 64, step=8)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    cnn_filters = trial.suggest_int("cnn_filters", 16, 128, step=16)
    num_epochs = 10  # Set lower epochs for faster tuning

    # Fit the scaler
    scaler = fit_minmax_scaler(weather_file_paths, weather_features)
    weekly_weather_data = parse_multiple_epw(weather_file_paths, weather_features, scaler)
    input_dim = weekly_weather_data.shape[2]

    # Create the dataset and dataloader
    dataset = WeeklyWeatherDataset(weekly_weather_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = ResNetAutoencoder1D(
        input_dim=input_dim,
        embedding_dim=embedding_dim
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Trial {trial.number} - Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss

    avg_loss = total_loss / num_epochs

    # Save trial results
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([trial.number, embedding_dim, batch_size, learning_rate, avg_loss])

    return avg_loss

# Run Optuna study
if __name__ == "__main__":
    # Initialize CSV if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["trial", "embedding_dim", "batch_size", "learning_rate", "avg_loss"])

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)

    # Output best trial
    print("Best trial:")
    print(study.best_trial)