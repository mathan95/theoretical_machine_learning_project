import optuna
import torch
import csv
import os
from transformer_model import build_custom_transformer_model  # Adjust based on actual module import path
from data_loader import create_data_loaders
import torch.optim as optim

# Define results file path
results_file = "/home/pmanmat/scratch/tldo/data/hp_tuning_transformer_out.csv"

# results_file = "hp_tuning_out.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Script Begins")

# Initialize CSV file for recording results if it doesn't exist
if not os.path.exists(results_file):
    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["trial", "embed_dim", "num_heads", "num_layers", "ff_hidden_dim", "ffnn_hidden_dim", "dropout", "learning_rate", "val_loss"])
        print("CSV Created")

param_ranges = {
    'wall_insulation': (0.02, 0.1),
    'roof_insulation': (0.02, 0.1),
    'window_u_factor': (1.2, 2.0),
    'window_shgc': (0.3, 0.7),
    'transmittance': (0.5, 1.0),
    'wall_thickness': (0.1, 0.5),
    'roof_thickness': (0.1, 0.5),
    'north_axis': (0, 360),
    'wall_solar_absorptance': (0.5, 0.9),
    'roof_solar_absorptance': (0.5, 0.9),
    'equipment_gain': (5, 15),
    'window_scale_factor': (0.5, 1.0),  # Change from area to scale factor
    'hvac_setpoint_heating': (18, 22),
    'hvac_setpoint_cooling': (24, 28)
}

def objective(trial):
    print(f"Starting trial {trial.number} with parameters...")

    # Define hyperparameters to tune
    embed_dim = trial.suggest_int("embed_dim", 8, 128, step=8)
    num_heads = int(embed_dim/8)
    num_layers = trial.suggest_int("num_layers", 2, 16)
    ff_hidden_dim = trial.suggest_int("ff_hidden_dim", 32, 128)
    ffnn_hidden_dim = trial.suggest_int("ffnn_hidden_dim", 32, 256)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    
    print(f"Starting trial {trial.number} with parameters: "
          f"embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}, ff_hidden_dim={ff_hidden_dim}, "
          f"ffnn_hidden_dim={ffnn_hidden_dim}, dropout={dropout}, learning_rate={learning_rate}")

    # Skip the trial if `embed_dim` is not divisible by `num_heads`
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")

    # Load data
    train_loader, val_loader, _ = create_data_loaders(
        train_csv_file="/home/pmanmat/scratch/tldo/data/London_ON_09_23_results.csv",  # Adjusted path
        train_weather_file="/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2009_2023.epw",  # Adjusted path
        test_csv_file="/home/pmanmat/scratch/tldo/data/London_ON_04_18_results.csv",  # Adjusted path
        test_weather_file="/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2004_2018.epw",  # Adjusted path
        param_ranges=param_ranges,
        batch_size=16
    )

    #     # Load data
    # train_loader, val_loader, _ = create_data_loaders(
    #     train_csv_file="output/energy_simulation_results_1.csv",  # Adjusted path
    #     train_weather_file="weather_files/CAN_ON_London_2009_2023.epw",  # Adjusted path
    #     test_csv_file="output/energy_simulation_results_2.csv",  # Adjusted path
    #     test_weather_file="weather_files/CAN_ON_London_2004_2018.epw",  # Adjusted path
    #     param_ranges=param_ranges,
    #     batch_size=16
    # )

    # Initialize model
    model = build_custom_transformer_model(
        weather_dim=6,  # Adjust based on actual input features
        building_dim=train_loader.dataset.building_features.shape[1],
        seq_len=168,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_hidden_dim=ff_hidden_dim,
        ffnn_hidden_dim=ffnn_hidden_dim,
        dropout=dropout
    ).to(device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop (simplified)
    for epoch in range(5):  # Fewer epochs for quick tuning
        model.train()
        for weather_data, building_features, consumption in train_loader:
            optimizer.zero_grad()
            predictions = model(weather_data.to(device), building_features.to(device))
            loss = criterion(predictions, consumption.to(device))
            loss.backward()
            optimizer.step()

    # Validation
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for weather_data, building_features, consumption in val_loader:
            predictions = model(weather_data.to(device), building_features.to(device))
            loss = criterion(predictions, consumption.to(device))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)


    # Save trial results
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([trial.number, embed_dim, num_heads, num_layers, ff_hidden_dim, ffnn_hidden_dim, dropout, learning_rate, avg_val_loss])

    return avg_val_loss

# Run optuna study
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
print("Best trial:", study.best_trial)