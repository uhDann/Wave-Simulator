import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Sources
from WaveSimulation2D import WaveSimulation2D


class Experiment:
    def __init__(self, grid_size, ds, dt, c, noise=None, boundary="mur", t_type="sin", rows=2, cols=3, start_time=0, total_time=10, plot_path=None):
        '''
        Initialize the experiment with the simulation parameters and plot parameters.

        Parameters:
            grid_size: Tuple (nx, ny)
                Spatial grid dimensions.
            ds: float
                Spatial step size.
            dt: float
                Time step size.
            c: float
                Speed of sound in the medium.
            noise: str
                Type of noise to add. Options are 'white', 'speckle', 'gaussian', 'perlin' or None as default.
            boundary: str
                Boundary condition type ("reflective", "pml" or "mur").
            t_type: str
                Type of source ("impulse" or "sin").
            rows: int
                Number of rows in the subplot grid.
            cols: int
                Number of columns in the subplot grid.
            start_time: float
                Time to start plotting.
            total_time: float
                Total time to run the simulation.
            plot_path: str
                Path to save the plot. If None, the plot is animated.

        Availiable Experiments:
            1. plot_row_transducers
                Plots the wave field for a row of choosen transducers.
            2. plot_2_transducers
                Plots the wave field for two transducers in a random central location.
            3. plot_var_transducers
                Plots the wave field for a row of transducers with varying frequencies.
            4. pml_test
                Plots the wave field for two transducers in a random central location with PML boundary.
            5. error_test
                Plots the wave field for different values of the error parameter (dt or ds).
            6. plot_interference
                Plots the interference pattern of the wave field. Must be used after running one of the previous experiments.
            7. plot_1_noise_transducers
                Plots the wave field and finds the threshold of when noise is too disruptive.

            '''

        # Initialize the simulation environment
        self.sim = WaveSimulation2D(
            grid_size, ds, dt, c, noise, boundary=boundary)

        # Save the experiment parameters
        self.nx, self.ny = grid_size
        self.grid_size = grid_size
        self.rows = rows
        self.cols = cols
        self.start_time = start_time
        self.total_time = total_time
        if t_type not in ["impulse", "sin", "both"]:
            raise ValueError("Invalid source type. Use 'impulse' or 'sin'.")
        self.t_type = t_type
        self.plot_path = plot_path

        if plot_path is None:
            print("[INFO] No plot_path passed, where possible the animation will be displayed. For some tests, only the plot will be displayed.")

    def _animate(self):
        '''Internal method to animate the simulation.'''
        self.sim.run_simulation(
            steps=int(self.total_time / 0.01) + 1, vmin=-1, vmax=1, plot_interval=1)

    def _plot(self):
        '''Internal method to plot the simulation.'''

        # Number of subplots
        num_subplots = self.rows * self.cols
        time_step = (self.total_time - self.start_time) / (num_subplots - 1)
        self.start_time /= 0.01

        fig, axes = plt.subplots(self.rows, self.cols,
                                 figsize=(20, 8), constrained_layout=True)
        axes = axes.flatten()
        plot_step = int(time_step / 0.01)  # Ensure plot_step is an integer

        # Step manually and plot
        for i in tqdm(range(int(self.total_time / self.sim.dt) + 1)):  # Include the last step
            if self.sim.noise is not None:
                self.sim.step(addNoise=True, noise_amplitude=0.001)
            else:
                self.sim.step()

            if i >= self.start_time and i % plot_step == 0:
                subplot_index = i // plot_step
                if subplot_index < num_subplots:
                    im = self.sim.plot(ax=axes[subplot_index])

        # Add a colorbar one for all subplots
        fig.colorbar(im, ax=axes, orientation='vertical',
                     fraction=0.05, pad=0.02).set_label('Wave Amplitude')
        plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_row_transducers(self):
        '''Plots the wave field for a row of choosen transducers.'''

        # Add the selecred sources to the simulation
        if self.t_type == "impulse":
            for i in range(1, 10, 3):
                self.sim.add_source(
                    Sources.create_impulse_source_2D(10000, i, 1, 0.02))
        elif self.t_type == "sin":
            for i in range(0, 11, 2):
                self.sim.add_source(
                    Sources.create_sinusoidal_source_2D(10, 1, i, 0))

        if self.plot_path is not None:
            self._plot()
        else:
            self._animate()

    def plot_2_transducers(self):
        '''Plots the wave field for two transducers in a random central location.'''

        # Add the selecred sources to the simulation
        if self.t_type == "impulse":
            self.sim.add_source(
                Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
            self.sim.add_source(
                Sources.create_impulse_source_2D(10000, 6, 6, 0.02))
        elif self.t_type == "sin":
            self.sim.add_source(
                Sources.create_sinusoidal_source_2D(10, 1, 4, 4))
            self.sim.add_source(
                Sources.create_sinusoidal_source_2D(10, 1, 6, 6))

        if self.plot_path is not None:
            self._plot()
        else:
            self._animate()

    def plot_var_transducers(self):
        '''Plots the wave field for a row of transducers with varying frequencies.'''

        # Add the sources to the simulation
        self.sim.add_source(
            Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
        self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 2, 2))
        self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 6, 6))

        if self.plot_path is not None:
            self._plot()
        else:
            self._animate()

    def plot_1_noise_transducers(self):
        """Plots the wave field and finds the threshold of when the third sinusoidal source becomes too disruptive."""

        def calculate_snr(signal, noise_amplitude):
            """
            Calculate the Signal-to-Noise Ratio (SNR) for a given signal and noise level.

            Parameters:
                signal (np.ndarray): The wave field or image signal.
                noise_amplitude (float): Amplitude of the added third sinusoidal source.

            Returns:
                float: Signal-to-noise ratio (SNR) in decibels.
            """
            signal_power = np.mean(signal**2)  # Mean power of the signal
            noise_power = noise_amplitude**2  # Power of the noise (third source amplitude)
            snr = 10 * np.log10(signal_power / noise_power)

            return snr

        noise_levels = np.linspace(0, 1, 10)  # Range of amplitudes for the third sinusoidal source
        snr_values = []  # Store SNR values for graphing
        thresholds = []

        for noise_amplitude in noise_levels:
            # Reset the simulation
            self.sim.u.fill(0)
            self.sim.u_prev.fill(0)
            self.sim.u_next.fill(0)

            # Add the two fixed sinusoidal sources
            self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 4, 4))  # Source 1
            self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 6, 6))  # Source 2

            # Add the third sinusoidal source with dynamic amplitude
            self.sim.add_source(Sources.create_sinusoidal_source_2D(noise_amplitude, 1, 2, 4))  # Third source

            # Run the simulation
            print(f"Running simulation with third source amplitude {noise_amplitude}...")

            for _ in tqdm(range(int(self.total_time / self.sim.dt) + 1)):
                self.sim.step(addNoise=False)

            # Calculate SNR
            signal_field = self.sim.u
            snr = calculate_snr(signal_field, noise_amplitude)
            snr_values.append((noise_amplitude, snr))  # Record for plotting
            print(f"SNR: {snr:.4f} dB")

            # Record the threshold if SNR falls below a critical value (e.g., 1 dB)
            if snr < 1:
                thresholds.append((noise_amplitude, snr))

        # Plot the SNR vs. third source amplitude graph
        noise_amplitudes, snrs = zip(*snr_values)
        plt.figure(figsize=(10, 6))
        plt.plot(noise_amplitudes, snrs, marker='o', label='SNR vs. Third Source Amplitude')
        plt.axhline(y=1, color='r', linestyle='--', label='SNR Threshold (1 dB)')
        plt.title("SNR vs. Third Impulse Source Amplitude")
        plt.xlabel("Third Source Amplitude")
        plt.ylabel("SNR (dB)")
        plt.legend()
        plt.grid(True)
        plt.show()
        if thresholds:
            print(f"Noise threshold found at amplitude {thresholds[0][0]} with SNR {thresholds[0][1]:.4f} dB.")
        else:
            print("No disruptive noise threshold found in the tested range.")

    def pml_test(self):
        '''Plots the pml profile and wave field for two transducers in a random central location with PML boundary.'''

        # Plot the pml profile
        self.sim.plot_pml_profile()

        # Add the sources and run the simulation with PML boundary
        self.plot_2_transducers()

    def error_test(self, test_subject):
        '''Plots the wave field for different values of the error parameter (dt or ds).

        Parameters:
            test_subject: str'''

        def run_simulation_and_plot(param_name, param_values, axes_pressure_field, axes_difference):
            '''Internal method to run the simulation for different values of the error parameter and plot the results.

            Parameters:
                param_name: str
                    Name of the error parameter.
                param_values: array-like
                    Values of the error parameter to test.
                axes_pressure_field: array-like
                    Array of axes to plot the pressure_field results.
                axes_difference: array-like
                    Array of axes to plot the difference results.

            Returns:
                im: matplotlib.image.AxesImage
                    Image object for the colorbar.
            '''
            benchmark_pressure_field = None
            assert param_values.any()
            for idx, param in enumerate(param_values):
                print(f"Running for {param_name} = {param}")

                if param_name == "dt":
                    new_env = WaveSimulation2D(self.grid_size, self.sim.ds, param, self.sim.c,
                                               self.sim.noise, boundary=self.sim.boundary, stability_check=False)
                elif param_name == "ds":
                    new_grid = (
                        int(self.grid_size[0] * (self.sim.ds / param)), int(self.grid_size[1] * (self.sim.ds / param)))
                    new_env = WaveSimulation2D(new_grid, param, self.sim.dt, self.sim.c,
                                               self.sim.noise, boundary=self.sim.boundary, stability_check=False)
                else:
                    raise ValueError(
                        "Invalid test subject. Use 'dt' or 'ds'.")

                if self.t_type == "impulse":
                    new_env.add_source(
                        Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
                    new_env.add_source(
                        Sources.create_impulse_source_2D(10000, 6, 6, 0.02))
                elif self.t_type == "sin":
                    new_env.add_source(
                        Sources.create_sinusoidal_source_2D(10, 1, 4, 4))
                    new_env.add_source(
                        Sources.create_sinusoidal_source_2D(10, 1, 6, 6))
                elif self.t_type == "both":
                    new_env.add_source(
                        Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
                    new_env.add_source(
                        Sources.create_sinusoidal_source_2D(10, 1, 6, 6))

                # Run the simulation up to 3 seconds
                for _ in tqdm(range(int(3 / new_env.dt))):
                    new_env.step()

                # Plot at 3 seconds
                im_pressure_field = new_env.plot(ax=axes_pressure_field[idx])
                axes_pressure_field[idx].set_title(
                    f"Simulation step: {param_name} = {param:.4f}")

                if idx == 0:
                    benchmark_pressure_field = new_env.u.copy()
                    im_difference = new_env.plot(ax=axes_difference[idx])
                    axes_difference[idx].set_title(
                        f"Benchmark pressure field\n{param_name} = {param:.4f}")
                else:
                    if param_name == "ds":
                        new_env.u = new_env.rescale_pressure_field(
                            (self.nx, self.ny))
                        new_env.nx = self.nx
                        new_env.ny = self.ny

                    assert benchmark_pressure_field is not None
                    new_env.u = np.abs(new_env.u - benchmark_pressure_field)

                    im_difference = new_env.plot(ax=axes_difference[idx])
                    axes_difference[idx].set_title(
                        f"Absolute difference from benchmark\n{param_name} = {param:.4f}")

                del new_env
            return im_pressure_field, im_difference

        fig_pressure_field, axes_pressure_field = plt.subplots(2, 3, figsize=(
            20, 8), constrained_layout=True)

        fig_difference, axes_difference = plt.subplots(
            2, 3, figsize=(20, 8), constrained_layout=True)

        axes_pressure_field = axes_pressure_field.flatten()
        axes_difference = axes_difference.flatten()

        if test_subject == "dt":
            param_values = np.linspace(self.sim.dt, 0.1, 6)
            im_pressure_field, im_difference = run_simulation_and_plot(
                "dt", param_values, axes_pressure_field, axes_difference)
        elif test_subject == "ds":
            param_values = np.linspace(self.sim.ds, 0.5, 6)
            im_pressure_field, im_difference = run_simulation_and_plot(
                "ds", param_values, axes_pressure_field, axes_difference)
        else:
            raise ValueError(
                "Invalid test subject. Use 'dt' or 'ds'.")

        fig_pressure_field.colorbar(im_pressure_field, ax=axes_pressure_field, orientation='vertical',
                                    fraction=0.05, pad=0.02).set_label('Wave Amplitude')

        fig_difference.colorbar(im_difference, ax=axes_difference, orientation='vertical',
                                fraction=0.05, pad=0.02).set_label('Benchmark Difference')

        if self.plot_path is not None:
            fig_pressure_field.savefig(
                self.plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_interference(self, additoonal_path=None):
        '''Plots the interference pattern of the wave field. Must be used after running one of the previous experiments.

        Parameters:
            additoonal_path: str
                Path to save the plot. If None, the plot is displayed.
        '''

        # Check if the simulation has been run
        if self.sim.u is None:
            raise ValueError(
                "Run the simulation before plotting interference. Select from the available experiments.")

        # Get the wave field data
        combined_wave_field = self.sim.u

        # Calculate positive and negative interference
        positive_interference = combined_wave_field > 0
        negative_interference = combined_wave_field < 0
        canceling_interference = np.isclose(combined_wave_field, 0, atol=1e-2)

        # Create a figure and three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        # Plot positive interference
        ax1.imshow(positive_interference, extent=(0, self.sim.nx * self.sim.ds,
                   0, self.sim.ny * self.sim.ds), origin='lower', cmap='Reds')
        ax1.set_title('Positive Interference')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')

        # Plot negative interference
        ax2.imshow(negative_interference, extent=(0, self.sim.nx * self.sim.ds,
                   0, self.sim.ny * self.sim.ds), origin='lower', cmap='Blues')
        ax2.set_title('Negative Interference')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')

        # Plot canceling interference
        ax3.imshow(canceling_interference, extent=(0, self.sim.nx * self.sim.ds,
                   0, self.sim.ny * self.sim.ds), origin='lower', cmap='Greens')
        ax3.set_title('Canceling Interference')
        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')

        red_patch = plt.Line2D([0], [0], color='red',
                               lw=4, label='Positive Interference')
        blue_patch = plt.Line2D([0], [0], color='blue',
                                lw=4, label='Negative Interference')
        green_patch = plt.Line2D(
            [0], [0], color='green', lw=4, label='Canceling Interference')
        ax1.legend(handles=[red_patch])
        ax2.legend(handles=[blue_patch])
        ax3.legend(handles=[green_patch])

        plt.tight_layout()

        if self.plot_path is not None:
            plt.savefig(self.plot_path[:-4]+"_interf.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

