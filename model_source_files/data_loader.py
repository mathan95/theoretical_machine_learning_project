import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler  # or StandardScaler

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def parse_epw(file_path, scaler=None):
    # Load the EPW file, skipping the first 8 rows and without headers
    weather_df = pd.read_csv(file_path, skiprows=8, header=None)

    # Assign the 35 column names as provided
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
    
    # Select relevant columns for weather features
    weather_features = weather_df[[
        "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", 
        "Wind Speed", "Total Sky Cover", "Wind Direction"
    ]].values

    # Apply scaling if a scaler is provided
    if scaler:
        weather_features = scaler.transform(weather_features)

    # Reshape data to create 52 weeks with 168 hourly records each
    weekly_weather = np.array([weather_features[i * 168:(i + 1) * 168] for i in range(52)])
    
    return weekly_weather  # Shape: (52, 168, num_features)

# Define a function to scale each feature based on param_ranges
def scale_building_features(features, param_ranges):
    scaled_features = []
    for i, (param, (low, high)) in enumerate(param_ranges.items()):
        # Scale the feature based on its min and max range
        scaled_feature = (features[:, i] - low) / (high - low)
        scaled_features.append(scaled_feature)
    return np.column_stack(scaled_features)

# Custom Dataset
class BuildingWeatherDataset(Dataset):
    def __init__(self, csv_file, weather_data, num_weeks=52, week_range=None, param_ranges=None):
        # Load building parameters and weekly consumption data from CSV
        self.data = pd.read_csv(csv_file)
        raw_building_features = self.data.drop(columns=[f'Week {i+1}' for i in range(num_weeks)]).values

        # Scale building features using the defined ranges
        self.building_features = scale_building_features(raw_building_features, param_ranges)

        # Initialize scaler and normalize consumption data
        #self.scaler = MinMaxScaler()
        self.weekly_consumption = (self.data[[f'Week {i+1}' for i in range(num_weeks)]].values / 2000).round(6)
        #self.scaler.fit_transform(
           # self.data[[f'Week {i+1}' for i in range(num_weeks)]].values
        #)

        self.weather_data = weather_data  # Shape: (52, 168, num_weather_features)
        self.num_samples = self.building_features.shape[0]
        self.num_weeks = num_weeks
        self.week_range = week_range if week_range is not None else range(num_weeks)

    def __len__(self):
        # Length is the number of samples times the selected weeks in the range
        return self.num_samples * len(self.week_range)

    def __getitem__(self, idx):
        # Determine the sample and week based on the index and week range
        sample_idx = idx // len(self.week_range)
        week_relative_idx = idx % len(self.week_range)  # Relative position within the week range
        week_idx = min(self.week_range) + week_relative_idx  # Offset the week index by the start of week_range
        #print(sample_idx, week_idx)

        # Get building features and weekly consumption for this specific week
        building_features = torch.tensor(self.building_features[sample_idx], dtype=torch.float32)
        consumption = torch.tensor(self.weekly_consumption[sample_idx, week_idx], dtype=torch.float32)
        
        # Get corresponding weekly weather data
        weekly_weather = torch.tensor(self.weather_data[week_idx], dtype=torch.float32)  # Shape: (168, num_weather_features)

        return weekly_weather, building_features, consumption


def create_data_loaders(train_csv_file, train_weather_file, test_csv_file, test_weather_file, param_ranges, batch_size=16, num_weeks=52, train_split=1, val_split=0.9, test_split=0.1):
    # Adjust train_split if set to 1 for a full-year train split
    adjusted_train_split = 0 if train_split == 1 else train_split
    num_train_weeks = int(num_weeks * train_split)
    num_train_weeks_csv_2 = int(num_weeks * adjusted_train_split)

    num_val_weeks = int(num_weeks * val_split)
    num_test_weeks = min(num_weeks - num_train_weeks_csv_2 - num_val_weeks, int(num_weeks * test_split))

    # Define the week ranges for each set
    train_week_range = range(num_train_weeks)
    val_week_range = range(num_train_weeks_csv_2, num_train_weeks_csv_2 + num_val_weeks)
    test_week_range = range(num_train_weeks_csv_2 + num_val_weeks, num_train_weeks_csv_2 + num_val_weeks + num_test_weeks)

    # Step 1: Fit the scaler on training weather data
    train_weather_data_raw = parse_epw(train_weather_file)  # Get raw data for fitting
    weather_features = train_weather_data_raw.reshape(-1, train_weather_data_raw.shape[-1])  # Flatten weekly weather for fitting

    scaler = MinMaxScaler()
    scaler.fit(weather_features)


    # Print scaling information
    print("Feature Minimums (data_min_):", scaler.data_min_)
    print("Feature Maximums (data_max_):", scaler.data_max_)
    print("Feature Ranges (data_range_):", scaler.data_range_)
    print("Scaler Minimum (min_):", scaler.min_)
    print("Scaler Scale (scale_):", scaler.scale_)

    print(train_week_range, test_week_range, val_week_range)

    # Step 2: Scale the data using the same scaler
    train_weather_data = parse_epw(train_weather_file, scaler=scaler)
    test_weather_data = parse_epw(test_weather_file, scaler=scaler)

    # Create datasets for each set
    train_dataset = BuildingWeatherDataset(train_csv_file, train_weather_data, num_weeks=num_weeks, week_range=train_week_range, param_ranges=param_ranges)
    val_dataset = BuildingWeatherDataset(test_csv_file, test_weather_data, num_weeks=num_weeks, week_range=val_week_range, param_ranges=param_ranges)
    test_dataset = BuildingWeatherDataset(test_csv_file, test_weather_data, num_weeks=num_weeks, week_range=test_week_range, param_ranges=param_ranges)

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# # Set up train and test datasets based on week range
# train_dataset = BuildingWeatherDataset(
#     csv_file='energy_simulation_results.csv', 
#     weather_data=weather_data, 
#     week_range=range(35)  # First 35 weeks for training
# )

# test_dataset = BuildingWeatherDataset(
#     csv_file='energy_simulation_results.csv', 
#     weather_data=weather_data, 
#     week_range=range(35, 52)  # Last 17 weeks for testing
# )


# # Define DataLoaders for train and test datasets
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Example usage: iterate over the dataset
# for weekly_weather, building_features, consumption in train_loader:
#     print("Weather data shape:", weekly_weather.shape)          # (batch_size, 168, num_weather_features)
#     print("Building features shape:", building_features.shape)   # (batch_size, num_building_features)
#     print("Consumption shape:", consumption.shape)               # (batch_size,)
#     break


# for weekly_weather, building_features, consumption in test_loader:
#     print("Weather data shape:", weekly_weather.shape)          # (batch_size, 168, num_weather_features)
#     print("Building features shape:", building_features.shape)   # (batch_size, num_building_features)
#     print("Consumption shape:", consumption.shape)               # (batch_size,)
#     break