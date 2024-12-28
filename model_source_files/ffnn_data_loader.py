import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Load embeddings from file
def load_embeddings(embedding_file):
    embeddings_df = pd.read_csv(embedding_file)
    return embeddings_df.values  # Shape: (52, embedding_dim)

# Scale building features based on defined ranges
def scale_building_features(features, param_ranges):
    scaled_features = []
    for i, (param, (low, high)) in enumerate(param_ranges.items()):
        scaled_feature = (features[:, i] - low) / (high - low)
        scaled_features.append(scaled_feature)
    return np.column_stack(scaled_features)

# Custom Dataset to include embeddings, building features, and consumption
class BuildingWeatherDataset(Dataset):
    def __init__(self, csv_file, embedding_file, num_weeks=52, week_range=None, param_ranges=None):
        # Load building parameters and weekly consumption data from CSV
        self.data = pd.read_csv(csv_file)
        raw_building_features = self.data.drop(columns=[f'Week {i+1}' for i in range(num_weeks)]).values

        # Scale building features using the defined ranges
        self.building_features = scale_building_features(raw_building_features, param_ranges)

        # Load embeddings from the specified file
        self.weather_embeddings = load_embeddings(embedding_file)  # Shape: (52, embedding_dim)

        # Normalize consumption data
        self.weekly_consumption = (self.data[[f'Week {i+1}' for i in range(num_weeks)]].values / 2000).round(6)

        self.num_samples = self.building_features.shape[0]
        self.num_weeks = num_weeks
        self.week_range = week_range if week_range is not None else range(num_weeks)

    def __len__(self):
        # Length is the number of samples times the selected weeks in the range
        return self.num_samples * len(self.week_range)

    def __getitem__(self, idx):
        # Determine the sample and week based on the index and week range
        sample_idx = idx // len(self.week_range)
        week_relative_idx = idx % len(self.week_range)
        week_idx = min(self.week_range) + week_relative_idx
        # print(idx, week_idx, week_relative_idx, self.week_range)
        # Get building features, weekly consumption, and weather embeddings for this specific week
        building_features = torch.tensor(self.building_features[sample_idx], dtype=torch.float32)
        consumption = torch.tensor(self.weekly_consumption[sample_idx, week_idx], dtype=torch.float32)
        weather_embedding = torch.tensor(self.weather_embeddings[week_idx], dtype=torch.float32)  # Shape: (embedding_dim,)

        return weather_embedding, building_features, consumption

# Create data loaders with the modified dataset class
def create_data_loaders(train_csv_file, train_embedding_file, test_csv_file, test_embedding_file, param_ranges, batch_size=16, num_weeks=52, train_split=1, val_split=0.2, test_split=0.2):
    adjusted_train_split = 0 if train_split == 1 else train_split
    num_train_weeks = int(num_weeks * train_split)
    num_train_weeks_csv_2 = int(num_weeks * adjusted_train_split)

    num_val_weeks = int(num_weeks * val_split)
    num_test_weeks = min(num_weeks - num_train_weeks_csv_2 - num_val_weeks, int(num_weeks * test_split))

    # Define the week ranges for each set
    train_week_range = range(num_train_weeks)
    val_week_range = range(num_train_weeks_csv_2, num_train_weeks_csv_2 + num_val_weeks)
    test_week_range = range(num_train_weeks_csv_2 + num_val_weeks, num_train_weeks_csv_2 + num_val_weeks + num_test_weeks)

    # Create datasets for each set
    train_dataset = BuildingWeatherDataset(train_csv_file, train_embedding_file, num_weeks=num_weeks, week_range=train_week_range, param_ranges=param_ranges)
    val_dataset = BuildingWeatherDataset(test_csv_file, test_embedding_file, num_weeks=num_weeks, week_range=val_week_range, param_ranges=param_ranges)
    test_dataset = BuildingWeatherDataset(test_csv_file, test_embedding_file, num_weeks=num_weeks, week_range=test_week_range, param_ranges=param_ranges)

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader