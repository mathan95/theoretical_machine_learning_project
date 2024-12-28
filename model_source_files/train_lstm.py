import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lstm_model import build_lstm_ffnn_model  # Import the LSTM model here
from data_loader import create_data_loaders  # Assuming dataset loader script

# Hyperparameters
num_epochs = 30
learning_rate = 0.0001
batch_size = 4

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

# Load data
train_loader, val_loader, test_loader = create_data_loaders(
    train_csv_file="../output_files/London_ON_09_23_results.csv",
    train_weather_file="../weather_files/CAN_ON_London_2009_2023.epw",
    test_csv_file="../output_files/London_ON_04_18_results.csv",
    test_weather_file="../weather_files/CAN_ON_London_2004_2018.epw",
    param_ranges=param_ranges,
    batch_size=batch_size,
    num_weeks=52, 
    train_split=1, 
    val_split=0.4, 
    test_split=0.6
)

# Model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weather_dim = 6  # Adjust this to the actual number of weather features
building_dim = train_loader.dataset.building_features.shape[1]  # Automatically determined from dataset
lstm_hidden_dim = 64
ffnn_hidden_dim = 64
dropout = 0.1

model = build_lstm_ffnn_model(
    weather_dim=weather_dim,
    building_dim=building_dim,
    lstm_hidden_dim=lstm_hidden_dim,
    ffnn_hidden_dim=ffnn_hidden_dim,
    dropout=dropout
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# R² and MAPE functions
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

def mape(y_true, y_pred):
    return (torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100).item()

def smape(targets, predictions):
    return (2 * torch.abs(targets - predictions) / (torch.abs(targets) + torch.abs(predictions) + 1e-8) * 100).mean()


# Initialize variables to track the best model
best_val_smape = float('inf')
# best_model_path = '/home/pmanmat/scratch/tldo/model/model.pth'
best_model_path = 'model/lstm_model.pth'

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for weather_data, building_features, consumption in train_loader:
        weather_data = weather_data.to(device)
        building_features = building_features.to(device)
        consumption = consumption.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(weather_data, building_features)
        loss = criterion(predictions, consumption)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    all_val_consumption = []
    all_val_predictions = []

    with torch.no_grad():
        for weather_data, building_features, consumption in val_loader:
            weather_data = weather_data.to(device)
            building_features = building_features.to(device)
            consumption = consumption.to(device)
            
            # Forward pass
            predictions = model(weather_data, building_features)
            loss = criterion(predictions, consumption)
            val_loss += loss.item()

            # Store all predictions and ground truths for validation
            all_val_consumption.append(consumption.cpu())
            all_val_predictions.append(predictions.cpu())

    # Concatenate all predictions and actual values for validation
    all_val_consumption = torch.cat(all_val_consumption, dim=0)
    all_val_predictions = torch.cat(all_val_predictions, dim=0).squeeze()

    # Calculate R² and MAPE on the entire validation set
    val_r2 = r2_score(all_val_consumption, all_val_predictions)
    val_smape = smape(all_val_consumption, all_val_predictions)

    # Save the model if it has the best validation loss so far
    val_loss_avg = val_loss / len(val_loader)
    if val_smape < best_val_smape:
        best_val_smape = val_smape
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model at epoch {epoch+1} with val loss: {val_smape:.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss_avg:.4f}, R2: {val_r2:.4f}, SMAPE: {val_smape:.2f}%")

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_loss = 0.0
all_test_consumption = []
all_test_predictions = []

with torch.no_grad():
    for weather_data, building_features, consumption in test_loader:
        weather_data = weather_data.to(device)
        building_features = building_features.to(device)
        consumption = consumption.to(device)

        predictions = model(weather_data, building_features)
        loss = criterion(predictions, consumption)
        test_loss += loss.item()

        # Store all predictions and ground truths for test
        all_test_consumption.append(consumption.cpu())
        all_test_predictions.append(predictions.cpu())

# Concatenate all predictions and actual values for test
all_test_consumption = torch.cat(all_test_consumption, dim=0)
all_test_predictions = torch.cat(all_test_predictions, dim=0).squeeze()

# Calculate R² and MAPE on the entire test set
test_r2 = r2_score(all_test_consumption, all_test_predictions)
test_smape = smape(all_test_consumption, all_test_predictions)

print(f"Test Loss: {test_loss/len(test_loader):.4f}, R2: {test_r2:.4f}, MAPE: {test_smape:.2f}%")