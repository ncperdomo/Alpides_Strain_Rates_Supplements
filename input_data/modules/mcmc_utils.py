# Import required libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sb
import corner
import pandas as pd

# Coded by Nicolas Castro-Perdomo for the Statistical Computing (STAT-S610) class final project.
# Github link: https://github.com/ncperdomo/GNSS_Elastic_Dislocation_MCMC.git
# Reach out if you find any bugs! Email: jcastrop [at] indiana [dot] edu  
# Initial version: Fall, 2022
# Last updated: Jan, 2025. Indiana University

############################################################################################################
############################################## MCMC functions ##############################################
############################################################################################################

############################### Elastic antiplane screw dislocation models #################################
def elastic_model(x, a, v_0, D, x_0):
    return a + v_0 * np.arctan((x - x_0) / D) / np.pi

def shallow_creep_model(x, a, v_0, D, x_0, v_c, d_c):
    delta = 0.0001  # Small constant
    return (a + v_0 * np.arctan((x - x_0) / D) / np.pi 
            - v_c * np.arctan((x - x_0) / d_c) / np.pi 
            + v_c * np.arctan((x - x_0) / delta) / np.pi) # Check Paul Segall's 2010 book for the creep model, signs can be tricky

def log_target_distribution(y, x, thetam, sigma, model_type='elastic'):
    if model_type == 'elastic':
        y_m = elastic_model(x, *thetam)
    elif model_type == 'shallow_creep':
        y_m = shallow_creep_model(x, *thetam)
    return np.sum(-(y - y_m) * (y - y_m) / (2 * sigma * sigma))
############################################################################################################

# Function to load input data for MCMC modelling from a CSV file
def load_data(file_path):
    data = np.loadtxt(file_path, dtype=float, delimiter='\t')
    x_data = data[:, 0]
    v_data = data[:, 1]
    return x_data, v_data

# Main MCMC routine to sample the posterior distribution of the model parameters
def run_mcmc(x_data, v_data, initial_params, n_steps, s_rw, model_type='elastic', priors=None):
    samples = []
    theta = initial_params
    acc_step = 0
    rmse_mcmc = []
    rng = np.random.default_rng(12345) # Random number generator with a seed for reproducibility

    # Set default priors if none are provided
    if priors is None:
        if model_type == 'elastic':
            priors = {'a': (-5, 5), 'v_0': (0, 50), 'D': (0, 50), 'x_0': (-10, 10)} # San Andreas Fault
        elif model_type == 'shallow_creep':
            priors = {'a': (-50, 50), 'v_0': (0, 10), 'D': (0, 30), 'x_0': (-5, 5), 'v_c': (0, 10), 'd_c': (0, 10)} # Dead Sea Creeping Fault

    # Print priors for debugging
    print("-----------------------------------")
    print("Priors used in MCMC routine:")
    for param, bounds in priors.items():
        print(f"{param}: {bounds}")
    print("-----------------------------------")
    
    start_time = time.time()
    for step in range(n_steps):
        samples.append(theta)
        theta1 = theta + s_rw * rng.multivariate_normal(np.zeros(len(theta)), np.identity(len(theta)))

        if model_type == 'elastic':
            a, v_0, D, x_0 = theta1
            if (priors['a'][0] <= a <= priors['a'][1] and priors['v_0'][0] <= v_0 <= priors['v_0'][1] and
                priors['D'][0] <= D <= priors['D'][1] and priors['x_0'][0] <= x_0 <= priors['x_0'][1]):
                alpha = min(1, np.exp(log_target_distribution(v_data, x_data, theta1, 1, model_type) -
                                      log_target_distribution(v_data, x_data, theta, 1, model_type)))
                if alpha > rng.uniform(0, 1):
                    theta = theta1
                    acc_step += 1
                rmse_mcmc.append(rmse(x_data, v_data, theta, model_type))
        elif model_type == 'shallow_creep':
            a, v_0, D, x_0, v_c, d_c = theta1
            if (priors['a'][0] <= a <= priors['a'][1] and priors['v_0'][0] <= v_0 <= priors['v_0'][1] and
                priors['D'][0] <= D <= priors['D'][1] and priors['x_0'][0] <= x_0 <= priors['x_0'][1] and
                priors['v_c'][0] <= v_c <= priors['v_c'][1] and priors['d_c'][0] <= d_c <= priors['d_c'][1]):
                alpha = min(1, np.exp(log_target_distribution(v_data, x_data, theta1, 1, model_type) -
                                      log_target_distribution(v_data, x_data, theta, 1, model_type)))
                if alpha > rng.uniform(0, 1):
                    theta = theta1
                    acc_step += 1
                rmse_mcmc.append(rmse(x_data, v_data, theta, model_type))

    execution_time = time.time() - start_time
    return np.array(samples), rmse_mcmc, acc_step / n_steps, execution_time

# Function to calculate the root mean square error (RMSE) 
def rmse(x, y, thetam, model_type='elastic'):
    if model_type == 'elastic':
        model_func = elastic_model
    elif model_type == 'shallow_creep':
        model_func = shallow_creep_model
    Ny = len(y)
    sigma = 1
    sum_y_ym = np.sum((y - model_func(x, *thetam)) ** 2 / sigma)
    return np.sqrt(sum_y_ym / Ny)

############################################################################################################
############################################ MCMC plotting functions #######################################
############################################################################################################

# Plotting functions for the MCMC results
sb.set_theme(style='white')

def plot_mcmc_results(samples, rmse_mcmc, burn_in_index, x, x_data, v_data, model_type='elastic', output_corner_plot=None, output_trace_plots=None, save_fig=True, vel_legend_loc=1):
    labels = [r"$a$", r"$v_0$", r"$D$", r"$x_0$"] if model_type == 'elastic' else [r"$a$", r"$v_0$", r"$D$", r"$x_0$", r"$v_c$", r"$d_c$"]
    fig = corner.corner(samples[burn_in_index:], labels=labels,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    if save_fig:
        plt.savefig(output_corner_plot, dpi=600)
    plt.show()

    n_params = len(labels)
    n_plots = n_params + 3  # Additional plots for best fit, RMSE histogram, and RMSE over iterations
    n_cols = 3  # Number of columns for the subplots
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division to determine the number of rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))

    # Flatten axs if it is a 2D array for easier indexing
    if isinstance(axs, np.ndarray) and axs.ndim > 1:
        axs = axs.flatten()

    # First plot: Observations vs. MCMC best fit
    if model_type == 'elastic':
        axs[0].plot(x, elastic_model(x, *samples.mean(axis=0)), label='MCMC best fit', color='lightcoral')
    elif model_type == 'shallow_creep':
        axs[0].plot(x, shallow_creep_model(x, *samples.mean(axis=0)), label='MCMC best fit', color='lightcoral')
    axs[0].axvline(x=samples.mean(axis=0)[3], linestyle='--', color='grey', label='Fault location')
    axs[0].scatter(x_data, v_data, s=3, color='steelblue', label='GNSS observations', zorder=10) # plot observed velocities on top
    axs[0].set_xlabel('Across-fault distance (km)')
    axs[0].set_ylabel('Fault parallel velocity (mm/yr)')
    axs[0].legend(loc=vel_legend_loc, prop={'size': 6})

    # Second plot: RMSE histogram
    axs[1].hist(rmse_mcmc, bins=20, color='darkgrey', ec='grey') 
    axs[1].set_yscale('log')
    axs[1].set_xlabel('RMSE')
    axs[1].set_ylabel('Count')

    # Third plot: RMSE over iterations
    axs[2].plot(rmse_mcmc, color='darkgrey')
    axs[2].axvline(x=burn_in_index, linestyle='--', color='red', label='Burn-in limit', linewidth=1, alpha=0.5)
    axs[2].legend(loc=1, prop={'size': 7})
    axs[2].set_xlabel('MCMC iteration steps')
    axs[2].set_ylabel('RMSE')

    # Plotting each parameter over iterations
    for i in range(n_params):
        axs[i + 3].plot(samples[:, i], color='darkgrey')
        axs[i + 3].axvline(x=burn_in_index, linestyle='--', color='red', label='Burn-in limit', linewidth=1, alpha=0.5)
        mean_param = np.median(samples[burn_in_index:, i])
        axs[i + 3].axhline(y=mean_param, color='black', linestyle='--', label='Median', linewidth=1, alpha=0.5)
        axs[i + 3].legend(loc=1, prop={'size': 7})
        axs[i + 3].set_xlabel('MCMC iteration steps')
        axs[i + 3].set_ylabel(labels[i])

    # Hide any unused subplots
    for j in range(n_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(pad=1.5)
    if save_fig:
        plt.savefig(output_trace_plots, dpi=600)
    plt.show()

############################################################################################################

def perform_mcmc_inversion(file_path, output_mcmc_file, output_corner_plot, output_trace_plots, model_type, initial_params=None, priors=None, n_steps=500000, s_rw=0.3, input_vel_profile_half_length=100, burn_in_percentage=0.1, vel_legend_loc=1, save_fig=True):
    """
    Perform MCMC analysis based on the specified model type (elastic or shallow_creep) and parameters.
    
    Parameters:
        file_path (str): Path to the input data file containing observation points.
        output_mcmc_file (str): Path to save the MCMC output file with predicted velocities.
        model_type (str): The type of model to run ('elastic' or 'shallow_creep'). Default is 'shallow_creep'.
        initial_params (list): List of initial parameter values for the MCMC simulation. Should match the model type.
        priors (dict): Dictionary of parameter names as keys and their prior ranges as values.
        n_steps (int): Number of steps for the MCMC simulation. Default is 500,000.
        s_rw (float): Random walk standard deviation for parameter updates in MCMC. Default is 0.3.
        
    Returns:
        pd.DataFrame: DataFrame containing the median parameters of the MCMC analysis.
    """
    
    # Load the data
    x_data, v_data = load_data(file_path)
    
    # Set default initial parameters and priors if not provided
    if initial_params is None or priors is None:
        if model_type == 'elastic':
            initial_params = [-12, -25, 20, 0]  # [a, v_0, D, x_0]
            priors = {'a': (-20, 0), 'v_0': (-50, 0), 'D': (0.0001, 50), 'x_0': (-15, 15)}
        elif model_type == 'shallow_creep':
            initial_params = [-12, -25, 20, 0, -8, 3]  # [a, v_0, D, x_0, v_c, d_c]
            priors = {'a': (-20, 0), 'v_0': (-50, 0), 'D': (0.0001, 50), 'x_0': (-5, 5), 'v_c': (-15, 0), 'd_c': (0.0001, 12)}

    # Run the MCMC simulation
    samples, rmse_mcmc, acceptance_rate, execution_time = run_mcmc(x_data, v_data, initial_params, n_steps, s_rw, model_type, priors)
    
    print(f"MCMC Execution Time: {np.round(execution_time, 2)} seconds")
    print(f"Acceptance rate: {np.round(acceptance_rate, 2) * 100}%")
    
    # Determine the burn-in period
    burn_in_index = n_steps // int(burn_in_percentage*100)  # Discard first (burn_in_percentage*100)% of samples as burn-in

    # Plot the MCMC results
    x = np.linspace(-input_vel_profile_half_length, input_vel_profile_half_length, 1000)
    plot_mcmc_results(samples, rmse_mcmc, burn_in_index, x, x_data, v_data, model_type, output_corner_plot, output_trace_plots, save_fig=save_fig, vel_legend_loc=vel_legend_loc)

    # Calculate median values of parameters after burn-in
    median_params = np.median(samples[burn_in_index:], axis=0)

    # 2 Decimal places
    median_params = np.round(median_params, 2)

    # Assign median parameters to variables
    if model_type == 'elastic':
        a, v_0, D, x_0 = median_params
        # save model predicted velocities to be plotted later on a PyGMT map
        velocities = elastic_model(x, a, v_0, D, x_0)
    elif model_type == 'shallow_creep':
        a, v_0, D, x_0, v_c, d_c = median_params
        # save model predicted velocities to be plotted later on a PyGMT map
        velocities = shallow_creep_model(x, a, v_0, D, x_0, v_c, d_c)

    # Format and round to 2 decimal places
    x = np.round(x, 2)
    velocities = np.round(velocities, 2)
    np.set_printoptions(suppress=True)

    # Save distances and model predicted velocities to a text file (tab-separated)
    np.savetxt(output_mcmc_file, np.column_stack((x, velocities)), delimiter='\t', fmt='%.2f', header='', comments='')

    # create dataframe with model predicted velocities
    model_predicted_velocities_df = pd.DataFrame(np.column_stack((x, velocities)), columns=['distance', 'velocity'])

    # Convert median parameters to DataFrame and assign column names
    median_params_df = pd.DataFrame(median_params.reshape(1, -1))
    if model_type == 'elastic':
        median_params_df.columns = ['a', 'v_0', 'D', 'x_0']
        print("-----------------------------------")
        print(f'Best fit parameters:\n{median_params_df.to_string(index=False)}')
        print("-----------------------------------")
    elif model_type == 'shallow_creep':
        median_params_df.columns = ['a', 'v_0', 'D', 'x_0', 'v_c', 'd_c']
        print("-----------------------------------")
        print(f'Best fit parameters:\n{median_params_df.to_string(index=False)}')
        print("-----------------------------------")

    return median_params_df, model_predicted_velocities_df
