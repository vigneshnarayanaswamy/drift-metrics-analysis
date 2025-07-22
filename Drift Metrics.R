----setup, include=FALSE----------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


----include=FALSE-----------------------------------------------------------------------

library(reticulate)
use_python("/Users/vigneshn/PycharmProjects/pythonProject/venv/bin/python")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import skewnorm
import seaborn as sns

##
#Defining the Functions
##
def get_pmf_bin_num(data_series, bins):
    hist, bin_edges = np.histogram(data_series, bins = bins)
    return hist/np.sum(hist), bin_edges
##
def get_pmf_bin_cuts(data_series, bin_cut_points):
    hist, bin_edges = np.histogram(data_series, bins = bin_cut_points)
    return hist/np.sum(hist)
##
def get_hellinger(old, new, bins):
    p = get_pmf_bin_num(old, bins)[0]
    q = get_pmf_bin_cuts(new, get_pmf_bin_num(old, bins)[1])
    hellinger = (1/np.sqrt(2))*np.sqrt(sum( (np.sqrt(p) - np.sqrt(q))**2 ))
    return hellinger
##
def get_psi(old, new, bins):  # modified function signature to accept bins argument
    p = get_pmf_bin_num(old, bins)[0]
    q = get_pmf_bin_cuts(new, get_pmf_bin_num(old, bins)[1])
##
    epsilon = 1e-10
    # Ensure p and q are not zero or very small, add epsilon
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
##
    return sum((p-q)*np.log(p/q))
##
def get_jensen_shannon(old, new, bins):
    p = get_pmf_bin_num(old, bins)[0]
    q = get_pmf_bin_cuts(new, get_pmf_bin_num(old, bins)[1])
##
    return distance.jensenshannon(p, q)**2

##
np.random.seed(42)  # Set seed for reproducability
##
# Create an empty DataFrame for the first set of variables
training_data = pd.DataFrame()
##
# Create 5 variables that sample from normal dist with same parameters
training_data['Stationary Series'] = np.random.normal(0, 1, 1000)
training_data['Mean Shift'] = np.random.normal(0, 1, 1000)
training_data['Standard Deviation Shift'] = np.random.normal(0, 1, 1000)
#training_data['Mean and Standard Deviation Shift'] = np.random.normal(1, 1000)
training_data['Distributional Shift'] =np.random.normal(0,1, 1000)
training_data['Transitioning Shift'] = np.random.normal(0,1, 1000)
training_data['Skewness Shift'] = np.random.normal(0, 1, 1000)
training_data['Noisy Signal'] = np.random.normal(0, 10, 1000)  # mean = 0, std dev = 10
##
##
##
##
# Create an empty DataFrame for the second set of variables
live_data1 = pd.DataFrame()
##
# Create 5 variables that sample from normal dist with same parameters
live_data1['Stationary Series'] = np.random.normal(0, 1, 1000)
live_data1['Mean Shift'] = np.random.normal(0, 1, 1000)
live_data1['Standard Deviation Shift'] = np.random.normal(0, 1, 1000)
#live_data1['Normal_Distribution_mean_std_shift'] = np.random.normal(0,1, 1000)
live_data1['Distributional Shift'] =np.random.normal(0,1, 1000)
live_data1['Transitioning Shift'] = np.random.normal(0,1, 1000)
live_data1['Skewness Shift']  = np.random.normal(0, 1, 1000)
live_data1['Noisy Signal'] = np.random.normal(0, 10, 1000)  # mean = 0, std dev = 10
##
##
##
# Create an empty DataFrame for the second set of variables
live_data2 = pd.DataFrame()
##
# Create 5 variables by sampling from different distributions with set 2 parameters
live_data2['Stationary Series'] = np.random.normal(0, 1, 2000)
live_data2['Mean Shift'] = np.random.normal(3, 1, 2000)
live_data2['Standard Deviation Shift'] = np.random.normal(0, 2, 2000)
#live_data2['Normal_Distribution_mean_std_shift'] = np.random.normal(3, 2, 2000)
live_data2['Distributional Shift'] = np.random.poisson(3, 2000)
live_data2['Skewness Shift']  = skewnorm.rvs(a=5, loc=0, scale=1, size=2000)
live_data2['Noisy Signal'] = np.random.normal(0, 10, 2000)  # mean = 0, std dev = 10
##
# Define the length of the Series and the number of time steps for the transition
series_length = 2000
num_time_steps = 1000
##
# Define the initial and final mean values
initial_mean = 0
final_mean = 3
std_dev = 1
##
# Create an array to store the values
values = []
##
# Generate values that transition smoothly over time
for t in range(num_time_steps + 1):
    alpha = t / num_time_steps  # Interpolation factor
##
    # Interpolate the mean parameter
    mean = (1 - alpha) * initial_mean + alpha * final_mean
##
    # Calculate the number of samples for this time step
    num_samples = round(series_length / (num_time_steps + 1))
##
    # Generate samples with the interpolated mean
    samples = np.random.normal(mean, std_dev, num_samples)
##
    # Append the samples to the values array
    values.extend(samples)
##
live_data2['Transitioning Shift'] =  pd.Series(values[:series_length])
##
live_data = pd.concat([live_data1, live_data2], ignore_index=True)
##

# Define the width and height multipliers
width = 15  # Adjust the width of the entire figure
height_per_subplot = 5  # Adjust the height for each individual subplot
##
fig, axs = plt.subplots(3, 2, figsize=(width, height_per_subplot * 3))  # Adjust the 3 here to match the number of rows
##
# Flatten the axis array for easy indexing
axs = axs.flatten()
##
# Plot each smoothed variable in a separate panel of the grid
for i, col in enumerate(live_data.columns[:-1]):
    ax = axs[i]
    ax.plot(live_data.index, live_data[col], label=col)
    ax.set_xlabel('Observations')
    ax.set_ylabel('Values')
    ax.set_title(f'{col}', fontsize=30)
    ax.legend()
    ax.grid(True)
#plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()

hellinger_results = pd.DataFrame()
PSI_results = pd.DataFrame()
##
window_size = 150  # Size of the window
step_size = 1    # Sliding step size
##
for col in training_data.columns:
    hellinger_results[col] = live_data[col].rolling(window=window_size, min_periods=window_size).apply(lambda x: get_hellinger(x, training_data[col], bins = 10)).dropna()
    PSI_results[col] = live_data[col].rolling(window=window_size, min_periods=window_size).apply(lambda x: get_psi(x, training_data[col], bins = 10)).dropna()

import matplotlib.pyplot as plt
import pandas as pd

# Get the column names for Hellinger and PSI results
hellinger_columns = hellinger_results.columns
PSI_columns = PSI_results.columns
##
# Determine the number of plots to create (all but the last column)
num_plots = len(hellinger_columns) - 1
##
# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # Adjust the figure size if necessary
##
# Flatten the axis array for easy indexing
axs = axs.flatten()
##
# Iterate through each subplot and plot the data (all but the last column)
for i in range(num_plots):
    ax = axs[i]  # Get the current axis
##
    # Plot Hellinger distance for the i-th column of hellinger_results
    ax.plot(hellinger_results.index, hellinger_results[hellinger_columns[i]], label=f'Hellinger - {hellinger_columns[i]}', linestyle='-', color='blue')
##
    # Plot PSI for the i-th column of PSI_results
    ax.plot(PSI_results.index, PSI_results[PSI_columns[i]], label=f'PSI - {PSI_columns[i]}', linestyle='-', color='green')
##
    # Set labels, title, legend, and grid
    ax.set_xlabel('Observations')
    ax.set_ylabel('Values')
    ax.set_title(f'Line Plots of Hellinger and PSI for {hellinger_columns[i]}')
    ax.legend()
    ax.grid(True)
##
    # Set the x-axis limits
    x_start = 900  # Replace with your desired starting point
    x_end = 1400    # Replace with your desired ending point
    ax.set_xlim(x_start, x_end)
##
    # Set the y-axis limits
    y_start = 0  # Replace with your desired starting point
    y_end = 0.3    # Replace with your desired ending point
    ax.set_ylim(y_start, y_end)
##
    # Add a horizontal red dashed line at y = 0.2
    ax.axhline(y=0.2, color='red', linestyle='--')
##
# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
##
# Assume hellinger_results, PSI_results, get_psi, and get_hellinger are already defined
# ... other parts of your code ...
##
# Initialize result DataFrames
PSI_bin_size_analysis = pd.DataFrame()
Hellinger_bin_size_analysis = pd.DataFrame()
##
# Loop through bin sizes and calculate PSI and Hellinger
for bins in [10,20,50]:  # Updated bin values
    window_size = 200  # Fixed window size, adjust as per your requirement
##
    # PSI Calculation
    PSI_bin_size_analysis[f"Noisy Signal {bins} bins"] = \
        live_data["Noisy Signal"].rolling(window=window_size, min_periods=window_size).apply(
            lambda x: get_psi(x, training_data["Noisy Signal"], bins=bins)).dropna()
##
    # Hellinger Distance Calculation
    Hellinger_bin_size_analysis[f"Noisy Signal {bins} bins"] = \
        live_data["Noisy Signal"].rolling(window=window_size, min_periods=window_size).apply(
            lambda x: get_hellinger(x, training_data["Noisy Signal"], bins=bins)).dropna()
##
# Create a figure with 1 row and 2 columns for side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
##
# Customize plot properties
title_fontsize = 20  # Change as needed
axis_label_fontsize = 14  # Change as needed
legend_fontsize = 12  # Change as needed
##
# Plot PSI
for column in PSI_bin_size_analysis.columns:
    # Calculate standard deviation and mean for observations after 1000
    std_psi = PSI_bin_size_analysis.loc[1000:, column].std()
    mean_psi = PSI_bin_size_analysis.loc[1000:, column].mean()  # Calculate mean
    label_psi = f"{column} (std: {std_psi:.2f}, mean: {mean_psi:.2f})"  # Include mean in the label
    axs[0].plot(PSI_bin_size_analysis.index, PSI_bin_size_analysis[column], label=label_psi, alpha=0.5)
axs[0].set_title('PSI Bin Size Analysis', fontsize=title_fontsize)
axs[0].set_xlabel('Observations', fontsize=axis_label_fontsize)  # Updated label
axs[0].set_ylabel('Value', fontsize=axis_label_fontsize)
axs[0].legend(loc='upper right', fontsize=legend_fontsize)
##
# Plot Hellinger
for column in Hellinger_bin_size_analysis.columns:
    # Calculate standard deviation and mean for observations after 1000
    std_hellinger = Hellinger_bin_size_analysis.loc[1000:, column].std()
    mean_hellinger = Hellinger_bin_size_analysis.loc[1000:, column].mean()  # Calculate mean
    label_hellinger = f"{column} (std: {std_hellinger:.2f}, mean: {mean_hellinger:.2f})"  # Include mean in the label
    axs[1].plot(Hellinger_bin_size_analysis.index, Hellinger_bin_size_analysis[column], label=label_hellinger, alpha=0.5)
axs[1].set_title('Hellinger Bin Size Analysis', fontsize=title_fontsize)
axs[1].set_xlabel('Observations', fontsize=axis_label_fontsize)  # Updated label
axs[1].set_ylabel('Value', fontsize=axis_label_fontsize)
axs[1].legend(loc='upper right', fontsize=legend_fontsize)
##
# Adjust spacing and display
plt.tight_layout()
plt.show()
##
