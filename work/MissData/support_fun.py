import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def RMSE(actual, imputed, corrupted_indices, n_skip = 17):
    """
    Root Mean Squared Error for corrupted values only

    Args:
    - actual: numpy array of shape (n_samples, n_features) containing the real data
    - imputed: numpy array of shape (n_samples, n_features) containing the imputed data
    - corrupted_indices: boolean array of shape (n_samples, n_features) containing the indices of the corrupted data
    - n_skip: number of days to skip in the corruption process. Default to 17
    
    Returns:
    - rmse: averaged root mean squared error
    """

    # Compute error between actual and imputed values
    error = actual - imputed

    # Select only the values that were previously corrupted
    corrupted_error = error[corrupted_indices]

    # Reshape back to matrix
    corrupted_error = np.reshape(corrupted_error, (corrupted_indices.shape[0] - n_skip, int(corrupted_error.shape[0] / (corrupted_indices.shape[0] - n_skip))))

    # Compute the daily mean squared errors
    mse = np.mean(np.square(corrupted_error), axis=1)

    # Take the average of the daly root mean squared errors
    rmse = np.mean(np.sqrt(mse))

    return rmse

def corruption(input, corruption_rate, missing_scenario, n_skip = 17):
    """
    Randomly corrupt a fraction of the input data by setting it to NaN.

    Args:
    - input: numpy array of shape (n_samples, n_features) containing the input data
    - corruption_rate: float between 0 and 1 representing the fraction of input data to corrupt
    - missing_scenario: string indicating the type of missing data to create ('continuous' or 'random')
    - n_skip: number of days to skip in the corruption process. Default to 17

    Returns:
    - corrupted_input: numpy array of shape (n_samples, n_features) containing the input data with missing values
    - mask: boolean array of shape (n_samples, n_features) containing the indices of the corrupted data
    """
    print("inp type", input.dtype)
    # Get the shape of the input data
    n_samples, n_features = input.shape

    # Create an array of NaNs of the same shape as the input data
    nan_arr = np.full((n_samples, n_features), np.nan)

    # Create an array to hold the corrupted input data
    corrupted_input = np.copy(input)

    # Create an array to hold the indices of the corrupted input data
    mask = np.zeros((n_samples, n_features), dtype=bool)

    # Set the seed for reproducibility
    random.seed(1)

    # Create a list of indices to skip
    skip_indices = random.sample(range(n_samples), n_skip)

    if missing_scenario == 'continuous':
        # Calculate the number of consecutive features to corrupt
        n_consecutive = int(np.round(corruption_rate * n_features))
        for i in range(n_samples):

            if i in skip_indices:
                # Skip this iteration
                continue

            # Fix the seeds of the random generator for each day
            np.random.seed(i)

            # Randomly select a starting index for the consecutive features
            start_idx = np.random.randint(0, n_features - n_consecutive)

            # Corrupt the consecutive features with NaNs
            corrupted_input[i, start_idx:start_idx+n_consecutive] = nan_arr[i, start_idx:start_idx+n_consecutive]

            # Set the corresponding entries in the mask to True
            mask[i, start_idx:start_idx+n_consecutive] = True

    elif missing_scenario == 'random':
        # Calculate the number of features to corrupt
        n_missing_features = np.int(np.round(n_features * corruption_rate))

        for i in range(n_samples):

            if i in skip_indices:
                # Skip this iteration
                continue

            # Fix the seeds of the random generator for each day
            np.random.seed(i)

            # Randomly select the indices of the features to corrupt
            missing_indices = np.random.choice(n_features, size=n_missing_features, replace=False)

            # Corrupt the selected features with NaNs
            corrupted_input[i, missing_indices] = nan_arr[i, missing_indices]

            # Set the corresponding entries in the mask to True
            mask[i, missing_indices] = True

    else:
        print("Error: invalid missing scenario")

    return corrupted_input, mask

def draw(file):
    """
    Plot the data as daily values.

    Args:
    - file: xlsx file containing a pandas dataframe with columns [['timestamp', 'real_data', 'data_corrupted', 'data_imputed']
    """

    # Load the data
    df = pd.read_excel(file, engine='openpyxl')

    # Reshape the data into a 2D array with shape (num_days, num_measurements_per_day)
    data = df['real_data'].values.reshape(-1, 48)
    data_corrupted = df['data_corrupted'].values.reshape(-1, 48)
    data_imputed = df['data_imputed'].values.reshape(-1, 48)
    
    # Select the first two days
    data = data[:4]
    data_corrupted = data_corrupted[:4]
    data_imputed = data_imputed[:4]

    # Create a flattened copy of the real data and imputed data for plotting
    data_ = data.flatten()
    data_imputed_ = data_imputed.flatten()

    # Create an array of x-axis values for the plot
    x1 = np.arange(len(data_corrupted.flatten()))

    # Create an array of x-axis values for the tick marks to represent days
    x2 = np.arange(0, len(data_corrupted.flatten()), 48)

    # Plot the real data
    plt.plot(x1, data_corrupted.flatten(), label='Real data')

    # Plot the corrupted data with last known values
    data_[~np.isnan(data_corrupted.flatten())] = np.nan
    nan = 0
    for i in range(len(data_)):
        if np.isnan(data_[i]) == False and nan == 0:
            data_[i-1] = data_corrupted.flatten()[i-1]
            nan = 1
        elif np.isnan(data_[i]) == True and nan == 1:
            data_[i] = data_corrupted.flatten()[i]
            nan = 0
    plt.plot(x1, data_, '--b', label='Corrupted data')

    # Plot the imputed data with last known values
    data_imputed_[~np.isnan(data_corrupted.flatten())] = np.nan
    nan = 0
    for i in range(len(data_imputed_)):
        if np.isnan(data_imputed_[i]) == False and nan == 0:
            data_imputed_[i-1] = data_corrupted.flatten()[i-1]
            nan = 1
        elif np.isnan(data_imputed_[i]) == True and nan == 1:
            data_imputed_[i] = data_corrupted.flatten()[i]
            nan = 0
    plt.plot(x1, data_imputed_, '--r', label='Imputed data')

    # Set the x-axis ticks to represent days and set the x- and y-axis labels and tick parameters
    plt.xticks(x2)
    plt.xticks(x2, np.arange(len(data_corrupted)))
    plt.ylabel('Measured temperature [deg C]', color='black')
    plt.xlabel('Days [-]', color='black')
    plt.tick_params(axis="x")
    plt.tick_params(axis="y")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Display the plot
    plt.show()
    plt.close()