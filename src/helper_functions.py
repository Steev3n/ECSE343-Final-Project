import math
import numpy as np
import matplotlib.pyplot as plt
from circuit_simulator import CircuitSimulator
import pickle
from scipy.stats import qmc
from tqdm.auto import tqdm

def plot_data(x_test, tpoints):
    # Create the figure and the first axis (for Volts)
    _ ,ax1 = plt.subplots(figsize=(10, 6))

    # Plot V_1, V_2, and V_3 on the primary y-axis
    ax1.plot(tpoints, x_test[:, 0], label='$V_1$')
    ax1.plot(tpoints, x_test[:, 1], label='$V_2$', linestyle='--')
    ax1.plot(tpoints, x_test[:, 2], label='$V_3$', linestyle='--')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Volt (V)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a twin axis for I_E (for mA)
    ax2 = ax1.twinx()
    # Scale I_E to mA
    ie_ma = x_test[:, 3] * 1000
    ax2.plot(tpoints, ie_ma, label='$I_E$', color='red', linestyle='--')
    ax2.set_ylabel("Current (mA)")

    # Calculate the ratio of the zero position relative to the range
    # We force the zero to be at the same proportional height on both axes
    def align_zeros(ax_ref, ax_target):
        ymin_ref, ymax_ref = ax_ref.get_ylim()
        rat = ymax_ref / (ymax_ref - ymin_ref)
        ymin_tar, ymax_tar = ax_target.get_ylim()
        if abs(ymin_tar) > abs(ymax_tar):
            new_ymax = ymin_tar * rat / (rat - 1)
            ax_target.set_ylim(ymin_tar, new_ymax)
        else:
            new_ymin = ymax_tar * (rat - 1) / rat
            ax_target.set_ylim(new_ymin, ymax_tar)

    # Apply the alignment
    align_zeros(ax1, ax2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Voltage and Current vs. Time")
    plt.tight_layout()
    plt.show()

"""
inputs: number of samples, amplitude, frequency, delta_time, Period T, noise 


"""



def create_dataset(num_samples, amplitude, f, delta_t, T, noise):
    x = [] # To store the simulated transient responses (features)
    y = [] # To store the ground truth R and C values (labels)

    print("Creating dataset...")
    for i in tqdm(range(num_samples)):

        # Randomly sample resistance (1000 to 5k Ohms) and capacitance (0.1 to 10 microFarads)
        if i == 0:

            sampler = qmc.LatinHypercube(d=2)
            lhs = sampler.random(n=num_samples)

            create_dataset.lhs_scaled = qmc.scale(
                lhs,
                [1.0, 0.1e-6],
                [2500.0, 5e-6]
            )

        R = create_dataset.lhs_scaled[i, 0]
        C = create_dataset.lhs_scaled[i, 1]

        # Initialize the Modified Nodal Analysis (MNA) simulator with current parameters
        mna = CircuitSimulator(amplitude, f, R, C)
        y.append([math.log(R), math.log(C)])

        # Initialize state variables 
        x_init = np.zeros((4,))

        # Perform simulation using the Backward Euler method
        transient, _ = mna.BEuler(x_init, delta_t, T, noise = noise)
        # Scale only the 4th signal (current) by 1000 (convert A to mA)
        transient[:, 3] *= 1000
        x.append(transient)

    # Convert lists to NumPy arrays for easier manipulation in ML frameworks
    x = np.array(x)
    y = np.array(y)

    print("Dataset created successfully")
    return x, y


def save_dataset(x, y):
    """
    Serializes the generated data and targets into a pickle file.
    """
    data_to_save = {
        'data': x,    # The time-series simulation results
        'target': y   # The corresponding R and C values
    }
    with open(f'data/dataset.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

def plot_rc_distribution(y):
    """
    Plots the frequency distribution of the real R and C values.
    """
    # y contains natural log values, use np.exp to recover real values
    R_real = np.exp(y[:, 0])
    C_real = np.exp(y[:, 1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.hist(R_real, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Resistance (R)')
    ax1.set_xlabel('Resistance (Ohms)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Scale to microFarads for readability
    ax2.hist(C_real * 1e6, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution of Capacitance (C)')
    ax2.set_xlabel('Capacitance (\u03bcF)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Scatter plot of R vs C
    ax3.scatter(R_real, C_real * 1e6, alpha=0.6, color='coral', edgecolor='black', s=15)
    ax3.set_title('R vs C Joint Distribution')
    ax3.set_xlabel('Resistance (Ohms)')
    ax3.set_ylabel('Capacitance (\u03bcF)')
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
