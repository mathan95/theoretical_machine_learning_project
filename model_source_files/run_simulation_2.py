import os
import pandas as pd
import numpy as np
import shutil
from eppy import modeleditor
from eppy.modeleditor import IDF
from scipy.stats import qmc  # For Latin Hypercube Sampling
import csv

# Set the path to the IDD file
iddfile = '/Applications/EnergyPlus-24-1-0/PreProcess/IDFVersionUpdater/V24-1-0-Energy+.idd'
IDF.setiddname(iddfile)

# Paths for EnergyPlus executable and files
idf_file_path = '../MediumOffice_Chicago.idf'
weather_file_path = '../weather_files/CAN_AB_Rocky_2009_2023.epw'
energyplus_executable = 'energyplus'

# Initialize the IDF class
idf = IDF(idf_file_path, weather_file_path)

# Define parameter ranges for LHS sampling
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

# Number of LHS samples
n_samples = 50
sampler = qmc.LatinHypercube(d=len(param_ranges))
sampled_params = sampler.random(n_samples)
params_scaled = {}

# Scale the sampled parameters to the specified range
for i, (param, (low, high)) in enumerate(param_ranges.items()):
    params_scaled[param] = sampled_params[:, i] * (high - low) + low


# Function to fix the "24:00:00" issue by converting it to "00:00:00" and incrementing the date
def fix_24_hour_time_format(df, time_column):
    df[time_column] = df[time_column].str.replace('24:00:00', '00:00:00')
    df[time_column] = pd.to_datetime(df[time_column], format='%Y %m/%d %H:%M:%S', errors='coerce')
    df[time_column] = df[time_column].fillna(method='ffill') + pd.to_timedelta(df[time_column].isna().astype(int), unit='D')
    return df


# Function to adjust window vertices based on a scale factor
def scale_window_area(idf, window_scale_factor):
    # Retrieve all FenestrationSurface:Detailed objects
    # Example: Scaling only Y-coordinate to adjust width
    for window_surface in idf.idfobjects['FENESTRATIONSURFACE:DETAILED']:
        # Get the original height based on the Z-coordinate difference between the top and bottom vertices
        original_height = window_surface.Vertex_1_Zcoordinate - window_surface.Vertex_2_Zcoordinate

        # Calculate the new height based on the scale factor
        new_height = original_height * window_scale_factor

        # Adjust the top vertices to achieve the new height while keeping the bottom vertices unchanged
        window_surface.Vertex_1_Zcoordinate = round(window_surface.Vertex_2_Zcoordinate + new_height, 3)
        window_surface.Vertex_4_Zcoordinate = round(window_surface.Vertex_3_Zcoordinate + new_height, 3)

# Function to modify parameters in the IDF file
def modify_idf_params(idf, params, output_folder, modified_idf_folder):
    # Wall and roof insulation conductivity
    idf.getobject('MATERIAL', 'Steel Frame NonRes Wall Insulation').Conductivity = params['wall_insulation']
    idf.getobject('MATERIAL', 'IEAD NonRes Roof Insulation').Conductivity = params['roof_insulation']
    
    # Wall and roof thickness
    idf.getobject('MATERIAL', 'Steel Frame NonRes Wall Insulation').Thickness = params['wall_thickness']
    idf.getobject('MATERIAL', 'IEAD NonRes Roof Insulation').Thickness = params['roof_thickness']
    
    # Solar absorptance for walls and roof
    idf.getobject('MATERIAL', 'Steel Frame NonRes Wall Insulation').Solar_Absorptance = params['wall_solar_absorptance']
    idf.getobject('MATERIAL', 'IEAD NonRes Roof Insulation').Solar_Absorptance = params['roof_solar_absorptance']
    
    # Window properties
    idf.getobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', 'NonRes Fixed Assembly Window').UFactor = params['window_u_factor']
    idf.getobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', 'NonRes Fixed Assembly Window').Solar_Heat_Gain_Coefficient = params['window_shgc']
    idf.getobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', 'NonRes Fixed Assembly Window').Visible_Transmittance = params['transmittance']
    
    # Scale window area using the scale factor
    scale_window_area(idf, params['window_scale_factor'])
    
    # Set north axis orientation
    idf.getobject('BUILDING', 'Ref Bldg Medium Office New2004_v1.3_5.0').North_Axis = params['north_axis']

    # Inspect available fields to find the correct field name
    # sample_equip = idf.getobject('ELECTRICEQUIPMENT', 'Core_ZN_MiscPlug_Equip')
    
    # Adjust equipment gains
    for zone in ['Core_bottom', 'Core_top', 'Core_mid', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_3', 'Perimeter_top_ZN_4', 
                 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_4', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_2', 
                 'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_4']:
        electric_equip = idf.getobject('ELECTRICEQUIPMENT', f'{zone}_PlugMisc_Equip')
        # Set the value for "Watts per Zone Floor Area {W/m2}", which is likely the 6th field in the IDF object.
        electric_equip["Watts_per_Floor_Area"] = params['equipment_gain']
    
    # # Modify HVAC heating and cooling setpoints
    # heating_setpoint = idf.getobject('SCHEDULE:COMPACT', 'HTGSETP_SCH')
    # cooling_setpoint = idf.getobject('SCHEDULE:COMPACT', 'CLGSETP_SCH')

    # print(heating_setpoint)
    # print(cooling_setpoint)
    
    # # Assign schedule fields with explicit values
    # heating_setpoint['Field_3'] = "Until: 06:00, " + str(round(params['hvac_setpoint_heating'], 1))
    # heating_setpoint['Field_5'] = "Until: 22:00, " + str(round(params['hvac_setpoint_heating'] + 2, 1))
    # heating_setpoint['Field_7'] = "Until: 24:00, " + str(round(params['hvac_setpoint_heating'] + 2, 1))

    # cooling_setpoint['Field_3'] = "Until: 06:00, " + str(round(params['hvac_setpoint_cooling'], 1))
    # cooling_setpoint['Field_5'] = "Until: 22:00, " + str(round(params['hvac_setpoint_cooling'] - 2,1))
    # cooling_setpoint['Field_7'] = "Until: 24:00, " + str(round(params['hvac_setpoint_cooling'] - 2, 1))


    # print(heating_setpoint)
    # print(cooling_setpoint)
    
    # Save the modified IDF file
    modified_idf_path = os.path.join(modified_idf_folder, f'modified_medium_office_{params["wall_insulation"]:.3f}.idf')
    idf.save(modified_idf_path)
    return modified_idf_path

# Function to run the simulation and aggregate output
def run_simulation(modified_idf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.system(f'{energyplus_executable} -w {weather_file_path} -r {modified_idf_path} -d {output_folder}')
    output_csv = os.path.join(output_folder, 'eplusout.csv')
    
    if os.path.exists(output_csv):
        results = pd.read_csv(output_csv)
        filtered_results = results[['Date/Time', 'Electricity:Facility [J](Hourly)', 'NaturalGas:Facility [J](Hourly)']].copy()
        filtered_results['Date/Time'] = '2019 ' + filtered_results['Date/Time']
        filtered_results = fix_24_hour_time_format(filtered_results, 'Date/Time')
        filtered_results['Electricity_kWh'] = filtered_results['Electricity:Facility [J](Hourly)'] / 3.6e6
        filtered_results['NaturalGas_kWh'] = filtered_results['NaturalGas:Facility [J](Hourly)'] / 3.6e6
        filtered_results['Total_Energy_kWh'] = filtered_results['Electricity_kWh'] + filtered_results['NaturalGas_kWh']
        filtered_results['Week_Number'] = filtered_results['Date/Time'].dt.isocalendar().week
        weekly_consumption = filtered_results.groupby('Week_Number')['Total_Energy_kWh'].sum()
        return weekly_consumption.values[:52]
    else:
        print("No output file found.")
        return None

# Prepare output folder and CSV file
output_file = 'energy_simulation_results.csv'
output_folder = 'energyplus_output'
modified_idf_folder = 'modified_idf'

# Function to clear a folder if it exists
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Delete all contents
    os.makedirs(folder_path, exist_ok=True)  # Recreate the empty folder

# Clear or create folders
clear_folder(output_folder)
clear_folder(modified_idf_folder)

# Create or overwrite the CSV file with the header only once
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Define and write the header
    header = list(params_scaled.keys()) + [f'Week {i+1}' for i in range(52)]
    writer.writerow(header)

# Run simulations and append each result directly to the CSV file
for i in range(n_samples):
    # Get the parameters for this sample
    params = {k: v[i] for k, v in params_scaled.items()}
    # Modify IDF and run simulation
    modified_idf_path = modify_idf_params(idf, params, output_folder, modified_idf_folder)
    weekly_consumption = run_simulation(modified_idf_path, output_folder)
    
    # Write results to CSV if simulation is successful
    if weekly_consumption is not None:
        row = list(params.values()) + list(weekly_consumption)
        # Append row to the CSV
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

print(f'Simulation results saved to {output_file}')