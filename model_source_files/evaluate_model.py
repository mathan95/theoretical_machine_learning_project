import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from transformer_model import build_custom_transformer_model
from data_loader import create_data_loaders


# Hyperparameters
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
    'window_scale_factor': (0.5, 1.0),
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
    val_split=0, 
    test_split=1
)


# Model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weather_dim = 6  # Adjust this to the actual number of weather features
building_dim = train_loader.dataset.building_features.shape[1]

model = build_custom_transformer_model(
    weather_dim=weather_dim,
    building_dim=building_dim,
    seq_len=168,
    embed_dim=16, #8,
    num_heads=1,
    num_layers=9,
    ff_hidden_dim=64, #110,
    ffnn_hidden_dim=256, #109,
    dropout=0.16
).to(device)

# Define metrics
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

criterion = combined_mse_mape_weighted_loss  # Use custom combined loss function

# Load the best model and evaluate on the test set
# Load the best model and evaluate on the test set
best_model_path = 'model/transformer_model_2.pth'

# Load the model for CPU use
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
model.eval()

test_loss = 0.0
all_test_consumption = []
all_test_predictions = []

with torch.no_grad():
    for weather_data, building_features, consumption in test_loader:
        #print(weather_data)
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
        #break

# Concatenate all predictions and actual values
all_test_consumption = torch.cat(all_test_consumption, dim=0)
all_test_predictions = torch.cat(all_test_predictions, dim=0).squeeze() 
#print(all_test_consumption, all_test_predictions)
# Calculate R² and MAPE on the entire test set
test_r2 = r2_score(all_test_consumption, all_test_predictions)
test_smape = smape(all_test_consumption, all_test_predictions)

print(f"Test Loss: {test_loss/len(test_loader):.4f}, R2: {test_r2:.4f}, MAPE: {test_smape:.2f}%")

# Convert tensors to lists for easier DataFrame creation
consumption_list = all_test_consumption.cpu().numpy().tolist()
predictions_list = all_test_predictions.cpu().numpy().tolist()

# Create a DataFrame with two columns: 'Actual' and 'Prediction'
df = pd.DataFrame({
    'Actual': consumption_list,
    'Prediction': predictions_list
})

# Save the DataFrame to a CSV file
df.to_csv("test_london_w2_transformer_2.csv", index=False)
print("Results saved to test_results.csv")