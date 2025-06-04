# utils.py

# === Standard Libraries ===
import os
import shutil
import subprocess
import csv
import random
import gc
import datetime
from datetime import datetime, timedelta
from os import walk
import re
import ast

# === IPython for Notebook Display ===
from IPython.display import display, HTML

# === Numerical and Data Handling ===
import numpy as np
import pandas as pd
import xarray as xr
import joblib

# === Plotting & Visualization ===
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
from matplotlib.table import Table
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.graph_objs as go
import dash
from dash import dcc, html
import dash
from dash import dcc, html
import plotly.figure_factory as ff


# === Scientific Computing and Stats ===
from scipy.stats import skew as sc_skew, pearsonr, linregress
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch, butter, filtfilt, correlate, detrend
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import leastsq, minimize

# === Machine Learning ===
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# === Deep Learning (TensorFlow & Keras) ===
import tensorflow as tf
import keras
import keras_tuner as kt  # Hyperparameter tuning
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, LSTM, Bidirectional, RepeatVector, 
    TimeDistributed, Dense, Attention, LayerNormalization
)
from keras.saving import register_keras_serializable

# === Utility & Miscellaneous ===
import yaml
from tqdm import tqdm
from PIL import Image

# === General Functions ===

# feature_errors: shape (num_sequences, num_features)
def plot_feature_error_correlation(feature_errors, features, save_path=None, figsize=(8,6)):
    """
    Plots a correlation heatmap of feature-specific reconstruction errors.

    Args:
        feature_errors (np.ndarray): Array of shape (num_sequences, num_features)
        features (list): List of feature names (length = num_features)
        save_path (str, optional): If provided, saves the figure to this path.
        figsize (tuple, optional): Figure size.
    """
    feature_errors_df = pd.DataFrame(feature_errors, columns=features)
    error_corr = feature_errors_df.corr()

    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    ax = sns.heatmap(
        error_corr, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmin=-1, vmax=1, 
        linewidths=0.5, 
        square=True, 
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation of Feature-specific Reconstruction Errors")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return error_corr

def scan_tuner_versions(base_dir):
    print(f"Scanning directory: {base_dir}")
    
    # Look for tuner_results_{version} folders directly inside base_dir
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path):
            match = re.match(r'tuner_results_(\w+)', entry)
            if match:
                version = match.group(1)
                
                # Check for LSTM_CNN and Attention inside this folder
                lstm_path = os.path.join(entry_path, 'LSTM_CNN')
                attention_path = os.path.join(entry_path, 'Attention')
                custom_path = os.path.join(entry_path, 'custom')

                has_lstm = os.path.isdir(lstm_path)
                has_attention = os.path.isdir(attention_path)
                has_custom = os.path.isdir(custom_path)

                print(f"Found version '{version}' in '{entry_path}':")
                print(f"    LSTM_CNN: {'✅' if has_lstm else '❌'}")
                print(f"    Attention: {'✅' if has_attention else '❌'}")
                print(f"    Custom: {'✅' if has_custom else '❌'}")

def load_raw_data(nc_path):
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    df = ds.to_dataframe()
    ds.close()
    df = df.iloc[df.index.get_level_values(1) == 0].droplevel(level=1)
    df.index = pd.to_datetime(df.index)
    return df

def compute_stats(df):
    metrics = {}
    # Correlations
    for var in ['ROLL', 'PITCH', 'HEADING','NORTH_SPEED','EAST_SPEED','VERTICAL_SPEED','ROLL','PITCH','HEADING','EASTWARD_WIND','NORTHWARD_WIND','VERTICAL_WIND','ALTITUDE']:
        metrics[f'mean_{var}'] = df[var].mean()

    return metrics

def extract_flight_number(filename):
    match = re.search(r'flight_(\d+)', filename)
    return int(match.group(1)) if match else None
    
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)  # Use safe_load for secure parsing
        return data
    return None

def write_yaml_subsegments(flight_id, subsegments, output_folder):
    yaml_data = {
        'flight_id': flight_id,
        'subsegments': []
    }
    
    for idx, subseg in enumerate(subsegments):
        yaml_data['subsegments'].append({
            'id': idx + 1,
            'name': subseg['name'],
            'start': subseg['start'].isoformat(),
            'end': subseg['end'].isoformat(),
            'score': subseg['score']
        })
    
    yaml_file_path = os.path.join(output_folder, f"flight_{flight_id}_subsegments.yaml")
    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_data, file)
    print(f"Subsegments YAML written to {yaml_file_path}")

def recover_lists_from_remaining(remaining_indices):
    # Sort indices to ensure they are in chronological order

    # Initialize list to store recovered lists
    recovered_lists = []

    # Iterate through remaining indices to recover individual lists
    current_list = []
    for idx in remaining_indices:
        if not current_list or (idx - current_list[-1]) == timedelta(seconds=1):
            current_list.append(idx)
        else:
            if current_list:
                recovered_lists.append(current_list)
                current_list = [idx]

    # Append the last recovered list if not already appended
    if current_list:
        recovered_lists.append(current_list)

    return recovered_lists
    
    
def moving_average(data, window_size):
    """
    Calculate the moving average of a 1D array using a defined window size.
    This function supports input data that may include pint.Quantity objects.

    Parameters:
    data (array-like or pint.Quantity): The input data for which to calculate the moving average.
    window_size (int): The number of data points to include in the moving average window.

    Returns:
    np.ndarray or pint.Quantity: An array of the same length as `data` containing the moving average values.
    """
    # Check if the data is a pint.Quantity object
    if isinstance(data, Quantity):
        # Strip the units, perform the moving average, then reapply the units
        units = data.units
        data = data.magnitude  # Strip units to work with numpy
        
        # Calculate the moving average
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Pad the result to match the original data length
        smoothed_data = np.pad(smoothed_data, (window_size // 2, window_size - 1 - window_size // 2), mode='constant', constant_values=np.nan)
        
        # Reapply the units to the smoothed data
        smoothed_data_with_units = smoothed_data * units
        
        return smoothed_data_with_units
    
    # If data is not a pint.Quantity, perform the moving average directly
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    smoothed_data = np.pad(smoothed_data, (window_size // 2, window_size - 1 - window_size // 2), mode='constant', constant_values=np.nan)
    
    return smoothed_data

def interpolate_to_regular_points(altitude, data, num_points, value_range):
    """
    Interpolates the data to regular intervals within a specified range.

    Parameters:
    altitude (array-like): The altitude values.
    data (array-like): The data to be interpolated.
    num_points (int): The number of points to interpolate.
    value_range (tuple): The range (min, max) of the values to be interpolated.

    Returns:
    np.ndarray: The interpolated values.
    """
    # Define regular points for interpolation
    regular_points = np.linspace(value_range[0], value_range[1], num_points)
    
    # Interpolate data to regular points
    interpolation_function = interp1d(altitude, data, bounds_error=False, fill_value='extrapolate')
    interpolated_data = interpolation_function(regular_points)
    
    return regular_points, interpolated_data

# Function to apply low-pass Butterworth filter
def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Detect 180 turns
def find_180_turns(df, heading_col='HEADING', window_seconds=300, tolerance_deg=20):
    """
    Simple 180° turn detection:
    Compare heading now vs heading after a time window.
    """
    df = df.copy()

    turns = []

    # Ensure time is sorted
    df = df.sort_index()

    timestamps = df.index

    for idx, t0 in enumerate(timestamps):
        t_future = t0 + pd.Timedelta(seconds=window_seconds)
        
        # Find the closest future point
        future_idx = df.index.searchsorted(t_future)
        if future_idx >= len(df):
            continue  # No future data point available
        
        heading_now = df.iloc[idx][heading_col]
        heading_future = df.iloc[future_idx][heading_col]
        
        if pd.isna(heading_now) or pd.isna(heading_future):
            continue  # Skip missing data
        # Actually dont care about 180 turns, any turn is fine?
        # Compute smallest angular difference
        diff = np.abs((heading_future - heading_now + 180) % 360 - 180)
        
        if np.abs(diff - 180) <= tolerance_deg:
            turns.append((t0, df.index[future_idx]))
    
    return turns
    
def merge_turns(turns, max_gap_seconds=60):
    """
    Merge turn detections that are close together into single turns.

    Parameters:
    - turns: list of (start_time, end_time) tuples.
    - max_gap_seconds: maximum allowed gap between detections in the same group.

    Returns:
    - merged_turns: list of merged (start_time, end_time) tuples.
    """
    if not turns:
        return []

    # Sort turns by start time
    turns = sorted(turns)
    merged_turns = []
    current_start, current_end = turns[0]
    for start, end in turns[1:]:
        gap = (start - current_end).total_seconds()

        if gap <= max_gap_seconds:
            # Extend current turn
            current_end = min(current_end, end)
            current_start = max(current_start, start)
        else:
            # Save previous turn and start new one
            merged_turns.append((current_start, current_end))
            current_start, current_end = start, end

    # Don't forget the last one
    merged_turns.append((current_start, current_end))

    return merged_turns
    
def plot_heading_with_turns(df, merged_turns,flight, heading_col='HEADING',campaign='Unknown'):
    """
    Plot heading vs time, and shade detected turns.
    
    Parameters:
    - df: original dataframe with heading data.
    - merged_turns: list of (start_time, end_time) tuples.
    - heading_col: name of the heading column.
    """
    os.makedirs(f'../../../Plots/Turns_analysis/{campaign}',exist_ok=True)
    plt.figure(figsize=(15, 5))

    # Plot the heading
    plt.plot(df.index, df[heading_col], label='Heading', color='blue')
    
    # Shade detected turns
    for start, end in merged_turns:
        plt.axvspan(start, end, color='red', alpha=0.3)

    plt.title(f'Aircraft Heading with Detected 180° Turns_{flight}_{campaign}')
    plt.xlabel('Time')
    plt.ylabel('Heading (degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../../../Plots/Turns_analysis/{campaign}/{flight}_{campaign}_turn_analysis.png')
    plt.close()
    
def wind_to_body_frame(row):
    heading_rad = np.deg2rad(row['HEADING'])  # scalar
    wind_vector = np.array([row['U_WIND'], row['V_WIND']])  # shape (2,)
    
    # Rotation matrix for 2D rotation by -heading
    cos_h = np.cos(-heading_rad)
    sin_h = np.sin(-heading_rad)
    R = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])  # shape (2,2)
    
    wind_body = R @ wind_vector  # shape (2,)
    return pd.Series(wind_body, index=['U_L', 'V_T'])

    
def plot_wind_around_turns(df, merged_turns, 
                           time_window_minutes=5, 
                           u_col='U_WIND', v_col='V_WIND', w_col='VERTICAL_WIND',
                           heading_col='HEADING', alt_col='ALT_INS',
                           flight=None,campaign=None, summary_records=None):
    """
    Plot wind components and heading around each turn if altitude stable.
    Also fills a summary_records list with turn info.
    """
    os.makedirs(f'../../../Plots/Turns_analysis/{campaign}/{flight}',exist_ok=True)
    time_margin = pd.Timedelta(minutes=time_window_minutes)
    for i, (start, end) in enumerate(merged_turns):
        # Define before and after masks
        before_mask = (df.index >= (start - time_margin)) & (df.index < start)
        after_mask = (df.index > end) & (df.index <= (end + time_margin))
        alt_before = df.loc[before_mask, alt_col].mean()
        alt_after = df.loc[after_mask, alt_col].mean()
        start_heading = df.loc[df.index[start, 'HEADING']]
        end_heading = df.loc[df.index[end, 'HEADING']]
        heading_series = df.loc[start:end, 'HEADING']
        heading_diff = heading_series.diff().dropna()
        # Altitude check
        if before_mask.sum() == 0 or after_mask.sum() == 0:
            continue  # Skip if not enough data

        heading_diff = (heading_diff + 180) % 360 - 180
        alt_diff = np.abs(alt_before - alt_after)

        if alt_diff >= 50:
            continue  # Skip if big altitude change before and after turn
        # Should ignore high diff value (359 to 0 jump)
        
        elif heading_diff.mean() < 0:
            direction = "right turn"
        else:
            direction = "left turn"
        # --- Plotting ---
        heading_diff = np.abs(start_heading - end_heading)
        fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=True)

        time_mask = (df.index >= (start - time_margin)) & (df.index <= (end + time_margin))
        df_plot = df.loc[time_mask]
        if (np.abs(df.loc[before_mask, heading_col].mean() - df.loc[after_mask, heading_col].mean()) - 180) > 20 :
            continue
            
        for ax, col, label, color in zip(
            axs[:3],
            [u_col, v_col, w_col],
            ['U wind', 'V wind', 'W wind'],
            ['tab:blue', 'tab:green', 'tab:red']
        ):
            ax.plot(df_plot.index, df_plot[col], label=label, color=color)
            ax.axvspan(start, end, color='orange', alpha=0.3)
            ax.legend()
            ax.grid(True)

        # Plot heading
        axs[3].plot(df_plot.index, df_plot[heading_col], label='Heading', color='purple')
        axs[3].axvspan(start, end, color='orange', alpha=0.3)
        axs[3].legend()
        axs[3].grid(True)
        
        axs[4].plot(df_plot.index, df_plot[alt_col], label='Altitude', color='purple')
        axs[4].axvspan(start, end, color='orange', alpha=0.3)
        axs[4].legend()
        axs[4].grid(True)

        axs[0].set_title(f"{campaign} : Flight {flight} - Turn {i+1} (Start: {start.time()}, End: {end.time()})")
        axs[-1].set_xlabel('Time')
        plt.savefig(f'../../../Plots/Turns_analysis/{campaign}/{flight}/Turn_{i+1}.png')
        plt.tight_layout()
        plt.close()

        # --- Record info for summary ---
        if summary_records is not None:
            summary_records.append({
                'campaign': campaign,
                'flight': flight,
                'turn_start': start,
                'turn_end': end,
                'direction':direction,
                'turn_duration_sec': (end - start).total_seconds(),
                'U_wind_before': df.loc[before_mask, u_col].mean(),
                'V_wind_before': df.loc[before_mask, v_col].mean(),
                'W_wind_before': df.loc[before_mask, w_col].mean(),
                'U_wind_after': df.loc[after_mask, u_col].mean(),
                'V_wind_after': df.loc[after_mask, v_col].mean(),
                'W_wind_after': df.loc[after_mask, w_col].mean(),
                'heading_diff': heading_diff,
                'altitude_before': alt_before,
                'altitude_after': alt_after,
                'altitude_diff': alt_diff
            })

# Function to handle NaNs by interpolation
def handle_nans(data):
    # Convert to pandas Series for easy interpolation
    series = pd.Series(data)
    # Interpolate missing values
    series_interpolated = series.interpolate(method='linear').bfill().ffill()
    return series_interpolated.values

# === Calibration tools ===
# Function to find the optimal lag using correlation
def find_optimal_lag(fast_signal, slow_signal, max_lag):
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        slow_signal_shifted = np.roll(slow_signal, lag)
        corr = np.corrcoef(fast_signal, slow_signal_shifted)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag
    
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (  ss_res / ss_tot)

    
# Function to perform calibration
def calibrate_sensors(slow_signal, fast_signal, sample_rate,max_lag=100):
    # Handle NaNs in signals
    slow_signal = handle_nans(slow_signal)
    fast_signal = handle_nans(fast_signal)
    # Find the optimal lag
    optimal_lag = find_optimal_lag(fast_signal, slow_signal, max_lag)
    slow_signal_shifted = np.roll(slow_signal, optimal_lag)
    # Filter signals
    cutoff_freq = 1/6  # Cutoff frequency in Hz
    slow_filtered = low_pass_filter(slow_signal_shifted, cutoff_freq, sample_rate)
    fast_filtered = low_pass_filter(fast_signal, cutoff_freq, sample_rate)
    
    # Reshape for sklearn
    y = slow_filtered
    X = fast_filtered.reshape(-1, 1)

    # linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # compute the calibrated fast signal
    fast_predicted = fast_signal * model.coef_[0] + model.intercept_
    r2 = compute_r2(slow_filtered,fast_filtered * model.coef_[0] + model.intercept_)
    return model.coef_[0], model.intercept_, fast_predicted,r2, optimal_lag, slow_filtered, fast_filtered

# === Wind computation ===

def W_wind_computation_calib(df,A,B,C,D):
    # expecting df_25
    # all angles in radian
    psi = df['HEADING']*np.pi/180
    theta = df['PITCH']*np.pi/180
    
        
    phi = (df['ROLL'])*np.pi/180 
    beta = (A * df['DPJ_RAD']/df['DP1']) + B
    alpha = (C * df['DPI_RAD']/df['DP1']) + D

    # Airflow speed
    Ua = df['TAS']
    # INS ground speeds
    Up = df['EAST_SPEED']
    Vp = df['NORTH_SPEED']
    Wp = df['VERTICAL_SPEED']
    # Length between INS and radome in the x-axis
    L = 8.05
    D = np.sqrt(1+np.tan(alpha)*np.tan(alpha)+np.tan(beta)*np.tan(beta))
    # time delta of the time series
    dt = 0.25
    # Computing second order gradient
    thetap = np.gradient(theta)/dt
    psip = np.gradient(psi)/dt
    # wind computation

    W = (-Ua * (D**-1)*(np.sin(theta)-np.tan(beta)*np.cos(theta)*np.sin(phi)-np.tan(alpha)*np.cos(theta)*np.cos(phi))
        + Wp + L * thetap *np.cos(theta))
    

    return(W)
    
def wind_computation_calib(df,A = 2.07 ,B = 13.63 ,C = 21.00 ,D = 0.00):
    # expecting df_25
    # Default values are from SAFIRE calibrations
    # all angles in radian
    psi = df['HEADING']*np.pi/180
    theta = df['PITCH']*np.pi/180
    
        
    phi = (df['ROLL'])*np.pi/180 
    beta = (A * df['DPJ_RAD']/df['DP1']) + B
    alpha = (C * df['DPI_RAD']/df['DP1']) + D
    # Airflow speed
    Ua = df['TAS']
    # INS ground speeds
    Up = df['EAST_SPEED']
    Vp = df['NORTH_SPEED']
    Wp = df['VERTICAL_SPEED']
    # Length between INS and radome in the x-axis for the ATR-42 in MAESTRO configuration
    L = 8.05
    D = np.sqrt(1+np.tan(alpha)*np.tan(alpha)+np.tan(beta)*np.tan(beta))
    # time delta of the time series
    dt = 0.25
    # Computing second order gradient
    thetap = np.gradient(theta)/dt
    psip = np.gradient(psi)/dt
    # wind computation
    U = (-Ua * (D**-1)*(np.sin(psi)*np.cos(theta)+np.tan(beta)*(np.cos(psi)*np.cos(phi)+np.sin(psi)
        *np.sin(theta)*np.sin(phi))+np.tan(alpha)*(np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi))
        )+Up - L*(thetap*np.sin(theta)*np.sin(psi)-psip*np.cos(psi)*np.sin(theta)))
    V = (-Ua * (D**-1)*(np.cos(psi)*np.cos(theta)-np.tan(beta)*(np.sin(psi)*np.cos(phi)-np.cos(psi)
        *np.sin(theta)*np.sin(phi))+np.tan(alpha)*(np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi))
        )+Vp - L*(psip*np.sin(psi)*np.cos(theta)+thetap*np.cos(psi)*np.sin(theta)))
    W = (-Ua * (D**-1)*(np.sin(theta)-np.tan(beta)*np.cos(theta)*np.sin(phi)-np.tan(alpha)*np.cos(theta)*np.cos(phi))
        + Wp + L * thetap *np.cos(theta))
    
    
    return([U,V,W])

def wind_computation_bias(df,aoa_bias=0,aos_bias=0,tas_cor=1):
    # expecting df_25
    # all angles in radian
    psi = df['HEADING']*np.pi/180
    theta = df['PITCH']*np.pi/180
    
    phi = (df['ROLL'])*np.pi/180 
    beta = df['AOS_RAD'] + (aos_bias*np.pi/180)
    alpha = df['AOA_RAD'] + (aoa_bias*np.pi/180)

    # Airflow speed
    Ua = df['TAS']*tas_cor
    # INS ground speeds
    Up = df['EAST_SPEED']
    Vp = df['NORTH_SPEED']
    Wp = df['VERTICAL_SPEED']
    # Length between INS and radome in the x-axis for the ATR-42 in MAESTRO configuration
    L = 8.05
    D = np.sqrt(1+np.tan(alpha)*np.tan(alpha)+np.tan(beta)*np.tan(beta))
    # time delta of the time series
    dt = 1/25
    # Computing second order gradient
    thetap = np.gradient(theta)/dt
    psip = np.gradient(psi)/dt
    # wind computation
    U = (-Ua * (D**-1)*(np.sin(psi)*np.cos(theta)+np.tan(beta)*(np.cos(psi)*np.cos(phi)+np.sin(psi)
        *np.sin(theta)*np.sin(phi))+np.tan(alpha)*(np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi))
        )+Up - L*(thetap*np.sin(theta)*np.sin(psi)-psip*np.cos(psi)*np.sin(theta)))
    V = (-Ua * (D**-1)*(np.cos(psi)*np.cos(theta)-np.tan(beta)*(np.sin(psi)*np.cos(phi)-np.cos(psi)
        *np.sin(theta)*np.sin(phi))+np.tan(alpha)*(np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi))
        )+Vp - L*(psip*np.sin(psi)*np.cos(theta)+thetap*np.cos(psi)*np.sin(theta)))
    W = (-Ua * (D**-1)*(np.sin(theta)-np.tan(beta)*np.cos(theta)*np.sin(phi)-np.tan(alpha)*np.cos(theta)*np.cos(phi))
        + Wp + L * thetap *np.cos(theta))
    
    return([U,V,W])

# === Conversion for humidity ===

def rh_to_mr(temp_c, rh, pressure_hpa):
    """
    Convert relative humidity to mixing ratio.
    
    Parameters:
    temp_c (float or array-like): Temperature in degrees Celsius.
    rh (float or array-like): Relative Humidity in percentage.
    pressure_hpa (float or array-like): Atmospheric pressure in hPa.
    
    Returns:
    float or array-like: Mixing ratio in g/kg.
    """
    # Calculate saturation vapor pressure (Tetens formula)
    e_s = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    
    # Calculate actual vapor pressure
    e = (rh / 100.0) * e_s
    
    # Calculate mixing ratio
    mr = (0.622 * e) / (pressure_hpa - e) * 1000  # Convert to g/kg
    
    return mr

def ah_to_mr(temp_c,ah,pressure_hpa):
    """
    Convert absolute humidity to mixing ratio.
    
    Parameters:
    temp_c (float or array-like): Temperature in degrees Celsius.
    rh (float or array-like): Relative Humidity in percentage.
    pressure_hpa (float or array-like): Atmospheric pressure in hPa.
    
    Returns:
    float or array-like: Mixing ratio in g/kg.
    """
    mr=1000*(ah*287.05*1e-3*(temp_c+273.15))/(pressure_hpa*100-(ah*461.5*1e-3*(temp_c+273.15)))
    
    return mr

# === Heterogeneity and segmentation ===

def weight_function_hom(segment):
    # segment is expected to be df_25 type
    # apply coarse variational decomposition, and evaluate the homogeneity
    score = 0
    par = ['U-geo','V-geo','W','T_c','MR']
    for p in par:
        Pert = ( segment[p+'_pert'])
        Cumsum = np.cumsum((Pert*Pert))
        Cumsum_per = 100 * (Cumsum / np.sum((Pert*Pert)))
        one_to_one_line = np.linspace(0, 100, len(Cumsum_per))
        rmse = np.sqrt(mean_squared_error(one_to_one_line, Cumsum_per))
        score = score + rmse
    
    return score

def find_best_segment_simple(df, segment_length,b,a):
    best_segment = None
    best_score = np.inf
    best_start_idx = 0
    
    for start_idx in (range(len(df) - segment_length + 1)):
        segment = df.iloc[start_idx:start_idx + segment_length].copy()
        score = weight_function_hom(segment)
        
        if score < best_score:
            best_score = score
            best_segment = segment
            best_start_idx = start_idx
    
    return best_segment, best_start_idx
    
# === Computing Perturbations and all turbulent scalars ===

def perturbations_and_moments(df_sub):
    ds = pd.Series(dtype='float64')
    df_subw = df_sub.copy()

    # Virtual temperature adjustment
    df_subw['T'] = df_subw['T'] * ((1000 / df_subw['STATIC_PRESSURE1']) ** 0.286)

    for p in ['V_c', 'U_c', 'W', 'T', 'MR']:
        Smoothed = filtfilt(b, a, df_subw[p].ffill().bfill())
        df_subw[p + '_pert'] = (df_subw[p] - Smoothed).dropna()
        df_subw[p + '_detrend'] = detrend(df_subw[p])

    df_subw = df_subw.loc[df_subw.index[0] + pd.Timedelta(seconds=30):df_subw.index[-1] - pd.Timedelta(seconds=30)]

    # General characteristics
    ds['date'] = df_subw.index[0].strftime("%Y-%m-%d")
    ds['time_start'] = df_subw.index[0].strftime("%H-%M-%S")
    ds['time_end'] = df_subw.index[-1].strftime("%H-%M-%S")
    ds['lat_start'] = df_subw['LATITUDE'].iloc[0]
    ds['lat_end'] = df_subw['LATITUDE'].iloc[-1]
    ds['lon_start'] = df_subw['LONGITUDE'].iloc[0]
    ds['lon_end'] = df_subw['LONGITUDE'].iloc[-1]

    # Means
    ds['alt'] = df_subw['ALTITUDE'].mean()
    ds['lat'] = df_subw['LATITUDE'].mean()
    ds['lon'] = df_subw['LONGITUDE'].mean()
    ds['MEAN_THDG'] = df_subw['HEADING'].mean()
    ds['MEAN_TAS'] = df_subw['TAS'].mean()
    ds['MEAN_GS'] = df_subw['GS'].mean()
    ds['MEAN_PS'] = df_subw['STATIC_PRESSURE1'].mean()
    ds['MEAN_TS'] = df_subw['STATIC_TEMPERATURE1'].mean()
    ds['MEAN_WD'] = np.arctan2(df_subw['V-geo'].mean(), df_subw['U-geo'].mean())
    ds['UWE_mean'] = df_subw['U-geo'].mean()
    ds['VSN_mean'] = df_subw['V-geo'].mean()
    ds['WS_mean'] = df_subw['U_c'].mean()
    ds['W_mean'] = df_subw['W'].mean()
    ds['THETA_mean'] = df_subw['T'].mean()
    ds['MR_mean'] = df_subw['MR'].mean()

    def compute_moments(prefix, suffix):
        ds[f'VAR_{prefix}{suffix}'] = (df_subw[f'{prefix}{suffix}'] ** 2).mean()
        ds[f'M3_{prefix}{suffix}'] = (df_subw[f'{prefix}{suffix}'] ** 3).mean()

    for var in ['U_c', 'V_c', 'W', 'T', 'MR']:
        compute_moments(var, '_pert')
        compute_moments(var, '_detrend')

    for var in ['U_c_pert', 'V_c_pert', 'W_pert', 'T_pert', 'MR_pert']:
        ds[f'SKEW_{var.split("_")[0]}'] = sc_skew(df_subw[var])

    def covar(v1, v2, suffix=''):
        ds[f'COVAR_{v1[:1]}{v2[:1]}{suffix}'] = (df_subw[v1] * df_subw[v2]).mean()

    covars = [('U_c_pert', 'V_c_pert'), ('U_c_pert', 'W_pert'), ('U_c_pert', 'T_pert'), ('U_c_pert', 'MR_pert'),
              ('V_c_pert', 'W_pert'), ('V_c_pert', 'T_pert'), ('V_c_pert', 'MR_pert'),
              ('W_pert', 'T_pert'), ('W_pert', 'MR_pert'), ('T_pert', 'MR_pert')]
    
    for v1, v2 in covars:
        covar(v1, v2)

    for v1, v2 in [(v[0].replace('_pert', '_detrend'), v[1].replace('_pert', '_detrend')) for v in covars]:
        covar(v1, v2, '_DET')

    def integral_length(pp, name, slope_key=None):
        acorr = sm.tsa.acf(pp, qstat=True, adjusted=True, nlags=len(pp))
        lags_sec = [(pp.index[i] - pp.index[0]).total_seconds() for i in range(len(pp))]
        lags_dist = [lag * df_subw['TAS'].mean() for lag in lags_sec]
        first_zero = next((i for i, v in enumerate(acorr[0]) if v <= 0), len(pp) - 1)
        ds[f'L_{name}'] = np.trapz(acorr[0][:first_zero+1], lags_dist[:first_zero+1])

        if slope_key:
            freqs, psd = welch(pp, fs=25, nperseg=256)
            wavenum = [df_subw['TAS'].mean() * f for f in freqs]
            IL_idx = np.argmin(np.abs(np.array(wavenum) - ds[f'L_{name}']))
            log_k = np.log10(wavenum[IL_idx:])
            log_psd = np.log10(psd[IL_idx:])
            if not np.isnan(log_psd).all():
                interp = interp1d(log_k, log_psd, fill_value="extrapolate")
                new_log_k = np.linspace(log_k.min(), log_k.max(), 15)
                new_log_psd = interp(new_log_k)
                slope, *_ = linregress(new_log_k, new_log_psd)
                ds[slope_key] = slope

    for name, col, slope_key in [('U', 'U_c_pert', 'Slope_U'), ('V', 'V_c_pert', 'Slope_V'), ('W', 'W_pert', 'Slope_W'),
                                 ('T', 'T_pert', None), ('MR', 'MR_pert', None)]:
        integral_length(df_subw[col], name, slope_key)

    def cospectral_length(v1p, v2p, key):
        ldat = len(v1p)
        corr = correlate(v1p, v2p, mode='full', method='auto')
        norm_corr = corr / (ldat * np.std(v1p) * np.std(v2p))
        lags = df_subw['TAS'].mean() * np.arange(-(ldat - 1), ldat) / 25
        ncorr = norm_corr[ldat - 1:]
        if np.any(ncorr < 0):
            zero_cross_idx = np.where(ncorr < 0)[0][0]
            ds[key] = df_subw['TAS'].mean() * np.abs(np.trapz(ncorr[:zero_cross_idx])) / 25
        else:
            ds[key] = np.nan

    cospectral_length(df_subw['U_c_pert'], df_subw['V_c_pert'], 'L_UV')
    cospectral_length(df_subw['U_c_pert'], df_subw['W_pert'], 'L_UW')

    return ds

# === Netcdf makers ===

# For time series

def df_to_netcdf(df, location, flight_number, leg_or_seg):
    version = '0'
    
    variable_map = {
        'ALTITUDE': ('ALTITUDE', 'm', 'Altitude from IXSEA : AIRINS', 'altitude'),
        'LONGITUDE': ('LONGITUDE', 'degrees_east', 'Longitude from IXSEA : AIRINS', 'longitude'),
        'LATITUDE': ('LATITUDE', 'degrees_north', 'Latitude from IXSEA : AIRINS', 'latitude'),
        'THDG': ('HEADING', 'degrees', 'True Heading Angle : AIRINS', 'heading'),
        'MR': ('MR', 'g.kg-1', 'Water vapor mixing ratio', 'MR'),
        'T': ('T', 'K', 'Potential air Temperature', 'air_temperature'),
        'UWE': ('U-geo', 'm.s-1', 'West-East Wind, corrected', 'eastward_wind'),
        'VSN': ('V-geo', 'm.s-1', 'South-North Wind, corrected', 'northward_wind'),
        'U_c': ('U_c', 'm.s-1', 'Streamwise horizontal wind', 'streamwise_hwind'),
        'V_c': ('V_c', 'm.s-1', 'Transverse horizontal wind', 'transverse_hwind'),
        'W': ('W', 'm.s-1', 'Vertical Wind, corrected', 'upward_wind'),
        'MR_fluc': ('MR_pert', 'g.kg-1', 'Water vapor mixing ratio perturbation', 'MR_pert'),
        'T_fluc': ('T_pert', 'K', 'Potential Temperature perturbation', 'air_temperature_pert'),
        'U_c_fluc': ('U_c_pert', 'm.s-1', 'Streamwise horizontal wind perturbation', 'streamwise_hwind_pert'),
        'V_c_fluc': ('V_c_pert', 'm.s-1', 'Transverse horizontal Wind perturbation', 'transverse_hwind_pert'),
        'W_c_fluc': ('W_pert', 'm.s-1', 'Vertical Wind perturbation', 'vertical_wind_pert'),
    }

    ds = xr.Dataset(coords={'time': df.index})

    for var_name, (df_key, unit, long_name, std_name) in variable_map.items():
        ds[var_name] = (['time'], df[df_key].values)
        ds[var_name].attrs = {
            'units': unit,
            'long_name': long_name,
            'standard_name': std_name
        }

    ds.attrs = {
        'title': f'Turbulence Data for RF {leg_or_seg}',
        'institution': 'Laboratoire d Aerologie, Toulouse, France',
        'source': 'MAESTROATR-42 field campaign, Island of Sal, Cabo-Verde  (11-08-2024 to 10-09-2024) https://orcestra-campaign.org/maestro.html',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'associated paper in writings',
        'Comment': (
            "High rate turbulence processing. Temperature and Mixing ratios were obtained from fast sensors and calibrated using "
            "slower reference sensors over the segment. Wind data was corrected after evaluation of "
            "bias in attack and sideslip angles. The perturbations were obtained through high pass filtering of the full "
            "time series"
        ),
        'Authors': 'Louis Jaffeux, Marie Lothon',
        'Version': f'v{version}',
    }

    os.makedirs(location, exist_ok=True)
    ds.to_netcdf(
        f'{location}/MAESTRO_ATR_TURBULENCE_RF{flight_number-22}_as2400{flight_number}_{20240810}_Time-series_{leg_or_seg}_v{version}.nc'
    )

def dfmom_to_netcdf(df, flight_number, location, err_cal=False):
    ds = df.to_xarray()
    date_t = df['date'].iloc[0]

    # Global attributes
    ds.attrs = {
        'title': f'Turbulent Moments for MAESTRO Flight RF{flight_number-22} (as2400{flight_number}) {date_t}',
        'institution': 'Laboratoire d Aerologie, Toulouse, France',
        'source': 'MAESTROATR-42 field campaign, Island of Sal, Cabo-Verde  (11-08-2024 to 10-09-2024) https://orcestra-campaign.org/maestro.html',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'associated dataset paper in writings',
        'Comment': (
            "High rate turbulence processing. Segmentation is based on the campaign segmentation found on the operational "
            "MAESTRO website. Temperature and Mixing ratios were calibrated using slow reference sensors over each segment. "
            "Wind data was corrected for biases in attack and sideslip angles. Perturbations were computed via high-pass "
            "filtering, then used for moment computations."
        ),
        'Authors': 'Louis Jaffeux, Marie Lothon'
    }

    # Helper function to assign attributes
    def set_attrs(var_map):
        for var, meta in var_map.items():
            if var in ds:
                ds[var].attrs['units'] = meta.get('units', '')
                if 'long_name' in meta:
                    ds[var].attrs['long_name'] = meta['long_name']
                if 'comments' in meta:
                    ds[var].attrs['comments'] = meta['comments']

    # Define attributes
    set_attrs({
        'Heterogeneity_score': {'comments': (
            "Computed as the distance between actual point-by-point variance increase and the expected linear increase, "
            "summed for U,V,W,T,q. Lower score = higher homogeneity."
        )},
        'R2_MR': {'comments': 'R-squared between WVSS2 slow and FASTWAVE fast sensor'},
        'name': {'units': 'MAESTROflightname_Leg_Segment', 'comments': 'Segment'},
        'time_start': {'units': 'YYYY-MM-DD HH:mm:ss'},
        'time_end': {'units': 'YYYY-MM-DD HH:mm:ss'},
        'lat_start': {'units': 'degrees'},
        'lat_end': {'units': 'degrees'},
        'lon_start': {'units': 'degrees'},
        'lon_end': {'units': 'degrees'},
        'alt': {'units': 'm', 'long_name': 'Mean Altitude above sea level from the ATR-42 INS'},
        'lat': {'units': 'degrees', 'long_name': 'Central Latitude from the ATR-42 INS'},
        'lon': {'units': 'degrees', 'long_name': 'Central Longitude from the ATR-42 INS'},
        'MEAN_THDG': {'units': 'degrees', 'long_name': 'Mean heading of the ATR-42'},
        'MEAN_TAS': {'units': 'm.s-1', 'long_name': 'Mean true air speed'},
        'MEAN_GS': {'units': 'm.s-1', 'long_name': 'Mean ground speed'},
        'MEAN_PS': {'units': 'hPa', 'long_name': 'Mean static pressure'},
        'MEAN_TS': {'units': 'm.s-1', 'long_name': 'Mean static temperature'},
        'MEAN_WD': {'units': 'degrees', 'long_name': 'Mean wind direction'},
        'UWE_mean': {'units': 'm.s-1', 'long_name': 'Mean Eastward wind (bias-corrected)'},
        'VSN_mean': {'units': 'm.s-1', 'long_name': 'Mean Northward wind (bias-corrected)'},
        'WS_mean': {'units': 'm.s-1', 'long_name': 'Mean total horizontal wind (bias-corrected)'},
        'W_mean': {'units': 'm.s-1', 'long_name': 'Mean vertical wind (bias-corrected)'},
        'THETA_mean': {'units': 'Celsius', 'long_name': 'Mean potential temperature'},
        'MR_mean': {'units': 'g.kg-1', 'long_name': 'Mean calibrated mixing ratio'},
    })

    # Variable groups to automate repetitive assignments
    stats = ['VAR', 'M3', 'SKEW', 'COVAR']
    components = {
        'U': 'Streamwise horizontal wind',
        'V': 'Transverse horizontal wind',
        'W': 'Vertical wind',
        'THETA': 'Potential temperature',
        'MR': 'Water vapor mixing ratio',
        'UV': 'streamwise and transverse horizontal winds',
        'UW': 'streamwise and vertical winds',
        'UT': 'streamwise wind and potential temperature',
        'UMR': 'streamwise wind and water vapor mixing ratio',
        'VW': 'transverse and vertical winds',
        'VT': 'transverse wind and potential temperature',
        'VMR': 'transverse wind and water vapor mixing ratio',
        'WT': 'vertical wind and potential temperature',
        'WMR': 'vertical wind and water vapor mixing ratio',
        'TMR': 'potential temperature and water vapor mixing ratio'
    }

    units_dict = {
        'VAR': {'U': 'm2.s-2', 'V': 'm2.s-2', 'W': 'm2.s-2', 'THETA': 'K2', 'MR': 'g2.kg-2²'},
        'M3': {'U': 'm3.s-3', 'V': 'm3.s-3', 'W': 'm3.s-3', 'THETA': 'K3', 'MR': 'g3.kg-3'},
        'SKEW': {key: '' for key in ['U', 'V', 'W', 'THETA', 'MR']},
        'COVAR': {
            'UV': 'm2.s-2', 'UW': 'm2.s-2', 'UT': 'm.K.s-1', 'UMR': 'm.g.s-1.kg-1',
            'VW': 'm2.s-2', 'VT': 'm.K.s-1', 'VMR': 'm.g.s-1.kg-1',
            'WT': '', 'WMR': 'm.g.s-1.kg-1', 'TMR': 'K.g.kg-1'
        }
    }

    for stat in stats:
        for comp, desc in components.items():
            varname = f"{stat}_{comp}"
            if varname in ds:
                set_attrs({varname: {
                    'units': units_dict.get(stat, {}).get(comp, ''),
                    'long_name': f"{desc} {stat.lower() if stat != 'M3' else 'third order moment'}"
                }})
            det_varname = f"{varname}_DET"
            if det_varname in ds:
                set_attrs({det_varname: {
                    'units': units_dict.get(stat, {}).get(comp, ''),
                    'long_name': f"{desc} {stat.lower()} (detrended)"
                }})

    # Error attributes
    if err_cal:
        for err_type in ['ERR_S', 'ERR_R']:
            for comp, desc in components.items():
                varname = f'{err_type}_{comp}'
                if varname in ds:
                    set_attrs({varname: {
                        'units': '',
                        'long_name': f"{'Systematic' if err_type == 'ERR_S' else 'Random'} relative error for {desc} {'variance' if comp in ['U', 'V', 'W', 'THETA', 'MR'] else 'covariance'}"
                    }})

    return ds


# ===== Added Functions for ML =====