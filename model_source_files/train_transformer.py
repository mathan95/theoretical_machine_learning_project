import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_model import build_custom_transformer_model  # Assuming your model script is named `model.py`
from data_loader import create_data_loaders  # Assuming the dataset loader script is named `data_loader_script.py`

# Hyperparameters
num_epochs = 100
learning_rate = 0.0013797303433641557
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
    val_split=0.95, 
    test_split=0.1
)


# # Load data
# train_loader, val_loader, test_loader = create_data_loaders(
#     train_csv_file="/home/pmanmat/scratch/tldo/data/energy_simulation_results_1.csv",
#     train_weather_file="/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2009_2023.epw",
#     test_csv_file="/home/pmanmat/scratch/tldo/data/energy_simulation_results_2.csv",
#     test_weather_file="/home/pmanmat/scratch/tldo/weatherfiles/CAN_ON_London_2004_2018.epw",
#     param_ranges=param_ranges,
#     batch_size=batch_size,
#     num_weeks=52, 
#     train_split=1, 
#     val_split=0.2, 
#     test_split=0.2
# )

# Model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weather_dim = 6  # Adjust this to the actual number of weather features
building_dim = train_loader.dataset.building_features.shape[1]  # Automatically determined from dataset

model = build_custom_transformer_model(
    weather_dim=weather_dim,
    building_dim=building_dim,
    seq_len=168,
    embed_dim=16,
    num_heads=1,
    num_layers=16,
    ff_hidden_dim=64,
    ffnn_hidden_dim=256,
    dropout=0.16
).to(device)

# model = build_custom_transformer_model(
#     weather_dim=weather_dim,
#     building_dim=building_dim,
#     seq_len=168,
#     embed_dim=8,
#     num_heads=4,
#     num_layers=1,
#     ff_hidden_dim=32,
#     ffnn_hidden_dim=64,
#     dropout=0.44
# ).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

def mape(y_true, y_pred):
    return (torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100).item()

def smape(targets, predictions):
    return (2 * torch.abs(targets - predictions) / (torch.abs(targets) + torch.abs(predictions) + 1e-8) * 100).mean()

def combined_mse_mape_weighted_loss(predictions, targets, mape_weight=0.8):
    mse_loss = nn.MSELoss()(predictions, targets)
    mape_loss = (torch.abs((targets - predictions) / (targets + 1e-8))).mean()
    return (1 - mape_weight) * mse_loss + mape_weight * mape_loss


# Initialize variables to track the best model
best_val_smape = float('inf')
# best_model_path = '/home/pmanmat/scratch/tldo/model/model.pth'
best_model_path = 'model/transformer_model_2.pth'

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for weather_data, building_features, consumption in train_loader:
        # Move data to device
        weather_data = weather_data.to(device)
        building_features = building_features.to(device)
        consumption = consumption.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(weather_data, building_features)
        loss = combined_mse_mape_weighted_loss(predictions, consumption)
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
            # Move data to device
            weather_data = weather_data.to(device)
            building_features = building_features.to(device)
            consumption = consumption.to(device)
            
            # Forward pass
            predictions = model(weather_data, building_features)
            loss = criterion(predictions, consumption)
            val_loss += loss.item()

            # Store all predictions and ground truths
            all_val_consumption.append(consumption.cpu())
            all_val_predictions.append(predictions.cpu())

    # Concatenate all predictions and actual values
    all_val_consumption = torch.cat(all_val_consumption, dim=0)
    all_val_predictions = torch.cat(all_val_predictions, dim=0).squeeze()

    # # Convert to lists for easier handling
    # val_consumption_list = all_val_consumption.view(-1).tolist()
    # val_predictions_list = all_val_predictions.view(-1).tolist()

    # # Select 20 random indices to sample
    # num_samples = 20
    # random_indices = random.sample(range(len(val_consumption_list)), num_samples)

    # # Print selected consumption and predictions
    # print("\nRandomly selected consumption values and predictions:")
    # for idx in random_indices:
    #     actual = val_consumption_list[idx]
    #     prediction = val_predictions_list[idx]
    #     print(f"Actual: {actual:.3f}, Prediction: {prediction:.3f}")

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

        # Forward pass
        predictions = model(weather_data, building_features)
        loss = criterion(predictions, consumption)
        test_loss += loss.item()
        
        # Store all predictions and ground truths
        all_test_consumption.append(consumption.cpu())
        all_test_predictions.append(predictions.cpu())

# Concatenate all predictions and actual values
all_test_consumption = torch.cat(all_test_consumption, dim=0)
all_test_predictions = torch.cat(all_test_predictions, dim=0).squeeze()

# Calculate R² and MAPE on the entire test set
test_r2 = r2_score(all_test_consumption, all_test_predictions)
test_mape = mape(all_test_consumption, all_test_predictions)

print(f"Test Loss: {test_loss/len(test_loader):.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.2f}%")