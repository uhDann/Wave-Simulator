import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import Sources

class WaveSimulation2D:
    def __init__(self, grid_size, ds, dt, c, noise=None, pml_width=10, boundary="pml", stability_check=True):
        """
        Initialize the simulation.

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
            Type of noise to add. Options are None, "white", "speckle", "gaussian", "perlin"
        pml_width: int
            Width of the perfectly matched layer (PML) in grid points.
        boundary: str
            Boundary condition type ("reflective" or "absorbing").
        stability_check: bool
            Whether to run the Courant condition check.
        """
        self.nx, self.ny = grid_size

        self.ds = ds
        self.dt = dt
        self.noise = noise

        self.c = c

        # Initiatie the PML damping profile and boundary conditions
        self.boundary = boundary
        self.pml_width = pml_width

        # Damping coefficient σ(x, y)
        self.sigma = np.zeros((self.nx, self.ny))
        self._initialize_pml()

        # Wave function at t
        self.u = np.zeros((self.nx, self.ny))
        if self.noise is not None:
            self.simulate_noise(amplitude=0.01, noise_type=self.noise)

        # Wave function at t-1
        self.u_prev = np.zeros((self.nx, self.ny))
        # Wave function at t+1
        self.u_next = np.zeros((self.nx, self.ny))

        # List of source functions
        self.sources = []

        self.time = 0

        # Stability condition (Courant condition)
        if stability_check:
            self.stability_limit = c * dt / ds
            if self.stability_limit > (1 / 2):
                raise ValueError(
                    "Stability condition violated. Reduce dt or increase ds.")

    def _initialize_pml(self):
        """Define the PML damping profile."""
        # Initialize sigma_x and sigma_y to handle damping separately along axes
        sigma_x = np.zeros((self.nx, 1))  # Along x-axis
        sigma_y = np.zeros((1, self.ny))  # Along y-axis

        # Quadratic damping profile for x-direction
        for i in range(self.nx):
            dx = min(i, self.nx - 1 - i) / self.pml_width
            if dx < 1.0:
                sigma_x[i, 0] = (1 - dx) ** 2

        # Quadratic damping profile for y-direction
        for j in range(self.ny):
            dy = min(j, self.ny - 1 - j) / self.pml_width
            if dy < 1.0:
                sigma_y[0, j] = (1 - dy) ** 2

        # Combine damping profiles: Sum or take max for uniform edge damping
        self.sigma = sigma_x + sigma_y  # Uniform damping near edges

    def add_source(self, source_function):
        """
        Add a source to the simulation.

        Parameters:
        source_function: function
            A function f(x, y, t) defining the source.
        """
        self.sources.append(source_function)

    def perlin_noise(shape, scale=10.0, amplitude=1.0):
        noise = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                noise[i, j] = pnoise2(i / scale, j / scale)
        return noise * amplitude

    def simulate_noise(self, amplitude=0.01, noise_type=NOISE):
        """
        Add simulated noise to the wave field.

        Parameters:
        amplitude: float
            The amplitude of the noise to scale the random values.
        noise_type: str
            Type of noise to add. Options are 'gaussian' (default) or 'uniform'.
        """
        # Environment/system noise
        # Fine-grained random variations, overall roughness/grain
        if noise_type == "white":
            noise = np.random.uniform(
                low=-amplitude, high=amplitude, size=self.u.shape)
        # Creates complex, multi-scale textures (rough terrains, turbulent patterns)
        elif noise_type == "speckle":
            noise = self.u * \
                np.random.normal(1.0, amplitude, size=self.u.shape)

        # Texture noise
        # Smooth random variations, overall smoothness
        elif noise_type == "gaussian":
            orig_noise = np.random.normal(0, amplitude, size=self.u.shape)
            noise = gaussian_filter(orig_noise, sigma=2)
        # Natural looking textures (clouds, terrain, waves)
        elif noise_type == "perlin":
            noise = self.perlin_noise(
                self.u.shape, scale=20.0, amplitude=amplitude)
        else:
            raise ValueError(
                "Invalid noise type. Use 'white', 'speckle', 'gaussian', 'perlin'.")

        self.u += noise

    def calculate_snr(self, signal, noise_amplitude):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for a given signal and noise level.

        Parameters:
            signal (np.ndarray): The wave field or image signal.
            noise_amplitude (float): Amplitude of the noise added.

        Returns:
            float: Signal-to-noise ratio (SNR) in decibels.
        """
        signal_power = np.mean(signal**2)  # Mean power of the signal
        noise_power = noise_amplitude**2   # Power of the noise
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    # TODO: "absorbing" and "pml" do not work as expected yet
    def step(self, addNoise=False, noise_amplitude=0.001):
        """
        Perform one time step of the simulation.
        """
        c2 = self.c ** 2
        dt2 = self.dt ** 2
        ds2 = self.ds ** 2

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                laplacian = (
                    self.u[i + 1, j] + self.u[i - 1, j]
                    + self.u[i, j + 1] + self.u[i, j - 1]
                    - 4 * self.u[i, j]
                ) / ds2

                source_contribution = 0
                for source in self.sources:
                    source_contribution += source(
                        i * self.ds, j * self.ds, self.time)

                self.u_next[i, j] = (
                    2 * self.u[i, j]
                    - self.u_prev[i, j]
                    + c2 * dt2 * (laplacian + source_contribution)
                )

        # Apply boundary conditions
        if self.boundary == "reflective":
            self.u_next[0, :] = self.u_next[1, :]
            self.u_next[-1, :] = self.u_next[-2, :]
            self.u_next[:, 0] = self.u_next[:, 1]
            self.u_next[:, -1] = self.u_next[:, -2]
        elif self.boundary == "pml":
            self.u_next *= np.exp(-self.sigma * self.dt)
        elif self.boundary == "mur":
            # Apply Mur absorbing boundary condition
            self.u_next[0, :] = self.u[1, :] + (self.c * self.dt - self.ds) / (
                self.c * self.dt + self.ds) * (self.u_next[1, :] - self.u[0, :])
            self.u_next[-1, :] = self.u[-2, :] + (self.c * self.dt - self.ds) / (
                self.c * self.dt + self.ds) * (self.u_next[-2, :] - self.u[-1, :])
            self.u_next[:, 0] = self.u[:, 1] + (self.c * self.dt - self.ds) / (
                self.c * self.dt + self.ds) * (self.u_next[:, 1] - self.u[:, 0])
            self.u_next[:, -1] = self.u[:, -2] + (self.c * self.dt - self.ds) / (
                self.c * self.dt + self.ds) * (self.u_next[:, -2] - self.u[:, -1])

        # Add noise if specified
        if addNoise:
            self.simulate_noise(amplitude=noise_amplitude)

        # Update time step
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev
        self.time += self.dt

    def plot_pml_profile(self):
        """Plot the PML damping profile."""
        plt.figure()
        plt.imshow(self.sigma, extent=(0, self.nx * self.ds, 0,
                   self.ny * self.ds), cmap='viridis', origin='lower')
        plt.colorbar(label="Damping Coefficient")
        plt.title("PML Damping Profile")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("MSFigures/2D/pml_profile.png")
        plt.show()

    def plot(self, ax=None, vmin=-1.0, vmax=1.0):
        """
        Plot the current field with a fixed color range.

        Parameters:
        ax: matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new figure and axes are created.
        vmin: float
            Minimum value for the color scale.
        vmax: float
            Maximum value for the color scale.
        """
        # Default to the current range of the current pressure field
        if vmin is None or vmax is None:
            vmin, vmax = np.min(self.u), np.max(self.u)

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(
            self.u, extent=(0, self.nx * self.ds, 0, self.ny * self.ds),
            cmap='viridis', origin='lower', vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Time: {self.time-self.dt:.2f} s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax is None:
            plt.colorbar(im, ax=ax, label="Pressure")
            plt.show()

        return im

    def run_simulation(self, steps, vmin=None, vmax=None, plot_interval=1):
        """
        Run the wave simulation and update the plot in real-time.

        Parameters:
        steps: int
            Number of time steps to run.
        vmin: float
            Minimum value for the color scale.
        vmax: float
            Maximum value for the color scale.
        plot_interval: int
            Number of steps between plot updates.
        """
        # Set up the figure for real-time plotting
        plt.ion()

        fig, ax = plt.subplots()

        im = ax.imshow(
            self.u, extent=(0, self.nx * self.ds, 0, self.ny * self.ds),
            cmap='viridis', origin='lower', vmin=vmin, vmax=vmax
        )

        plt.colorbar(im, ax=ax, label="Pressure")

        ax.set_title("Time: 0.00 s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()

        # Run the simulation
        for step in range(steps):
            if self.noise is not None:
                self.step(addNoise=True, noise_amplitude=0.001)
            else:
                self.step(addNoise=False)

            if step % plot_interval == 0:
                # Update the image data
                im.set_data(self.u)

                im.set_clim(vmin, vmax)

                ax.set_title(f"Time: {self.time:.2f} s")

                # Pause to update the plot
                # If you comment then it doesn't work
                plt.pause(0.0001)

        plt.ioff()
        plt.show()


class Experiment:
    def __init__(self, grid_size, ds, dt, noise, c, boundary="mur", t_type="sin", rows=2, cols=3, start_time=0, total_time=10, plot_path=None):
        '''
        Initialize the experiment with the simulation parameters and plot parameters.

        Parameters:
            grid_size: Tuple (nx, ny)
                Spatial grid dimensions.
            ds: float
                Spatial step size.
            dt: float
                Time step size.
            noise: str
                Type of noise to add. Options are 'white', 'speckle', 'gaussian', 'per
            c: float
                Speed of sound in the medium.
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

            '''

        # Initialize the simulation environment
        self.sim = WaveSimulation2D(
            grid_size, ds, dt, c, noise, boundary=boundary)

        # Save the experiment parameters
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

        # Create figure and subplots
        fig, axes = plt.subplots(self.rows, self.cols,
                                 figsize=(20, 8), constrained_layout=True)
        axes = axes.flatten()
        plot_step = int(time_step / 0.01)  # Ensure plot_step is an integer

        # Step manually and plot
        for i in tqdm(range(int(self.total_time / 0.01) + 1)):  # Include the last step
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
        """Plots the wave field and finds the threshold of when noise is too disruptive."""
        noise_levels = np.linspace(0.001, 0.1, 10)  # Define a range of noise amplitudes
        thresholds = []

        for noise_amplitude in noise_levels:
            # Reset the simulation
            self.sim.u.fill(0)
            self.sim.u_prev.fill(0)
            self.sim.u_next.fill(0)

            # Add sources
            if self.t_type == "impulse":
                self.sim.add_source(Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
                self.sim.add_source(Sources.create_impulse_source_2D(10000, 6, 6, 0.02))
                self.sim.add_source(Sources.create_impulse_source_2D(500, 2, 4, 0.02))
            elif self.t_type == "sin":
                self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 4, 4))
                self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, 6, 6))
                self.sim.add_source(Sources.create_sinusoidal_source_2D(3, 1, 2, 4))

            # Run the simulation
            for _ in range(100):  # Run for a fixed number of steps
                self.sim.step(addNoise=True, noise_amplitude=noise_amplitude)

            # Calculate SNR
            signal_field = self.sim.u
            snr = self.sim.calculate_snr(signal_field, noise_amplitude)

            # Record the threshold if SNR falls below a critical value (e.g., 1 dB)
            if snr < 1:
                thresholds.append((noise_amplitude, snr))
                break  # Stop once threshold is found

        if thresholds:
            print(f"Noise threshold found at amplitude {thresholds[0][0]} with SNR {thresholds[0][1]:.2f} dB.")
        else:
            print("No disruptive noise threshold found in the tested range.")

        # Optionally, plot the final wave field for the threshold case
        if thresholds:
            self.sim.plot()

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

        def run_simulation_and_plot(param_name, param_values, axes):
            '''Internal method to run the simulation for different values of the error parameter and plot the results.

            Parameters:
                param_name: str
                    Name of the error parameter.
                param_values: array-like
                    Values of the error parameter to test.
                axes: array-like
                    Array of axes to plot the results.

            Returns:
                im: matplotlib.image.AxesImage
                    Image object for the colorbar.
            '''
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
                elif param_name == "noise":
                    new_env = WaveSimulation2D(self.grid_size, self.sim.ds, self.sim.dt, self.sim.c,
                                               self.sim.noise, boundary=self.sim.boundary, stability_check=False)

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

                # Plot at 2 seconds
                im = new_env.plot(ax=axes[idx])
                axes[idx].set_title(f"Simulation step: {param_name} = {param:.4f}")

                del new_env
            return im

        fig, axes = plt.subplots(2, 3, figsize=(
            20, 8), constrained_layout=True)
        axes = axes.flatten()

        if test_subject == "dt":
            param_values = np.linspace(self.sim.dt, 0.1, 6)
            im = run_simulation_and_plot("dt", param_values, axes)
        elif test_subject == "ds":
            param_values = np.linspace(self.sim.ds, 0.5, 6)
            im = run_simulation_and_plot("ds", param_values, axes)
        elif test_subject == "noise":
            param_values = np.linspace(0.0, 10, 6)
            im = run_simulation_and_plot("noise", param_values, axes)

        fig.colorbar(im, ax=axes, orientation='vertical',
                     fraction=0.05, pad=0.02).set_label('Wave Amplitude')
        if self.plot_path is not None:
            plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
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
        
if __name__ == "__main__":
    # General usage:
    # 1. Specify simulation parameters
    # 2. Create a simulation object
    # 3. Add sources to the simulation, there are some define in Sources.py
    # 4. Either manually step the simulation or run with animatd plotting
    # 5. Alternatively, run the simulation for the desired experiment

    ##################### SETUP THE ENVIRONMENT PARAMETERS #####################
    # Dimensions of grid
    grid_size = (100, 100)
    # Difference in distance between grid points
    ds = 0.1
    # Time step
    dt = 0.01
    # Speed of sound in medium
    c = 1.0

    ########################### SETUP THE EXPERIMENT ###########################

    experiment_type = "dt_error"
    t_type = "sin"  # "impulse" or "sin"
    total_time = 3      # Recomended: 6 sec for impulse, 10 sec for sin
    NOISE = None      # None, "white", "speckle", "gaussian", "perlin"
    experiment_type = "node"
    t_type = "impulse"  # "impulse" or "sin"
    total_time = 15      # Recomended: 6 sec for impulse, 10 sec for sin
    NOISE = None      # "white", "speckle", "gaussian", "perlin"

    # Pass None to animate
    plot_path = f"MSFigures/2D/WS_{experiment_type}_{t_type}_{total_time}s.png"
    # plot_path = None

    row_impulse = Experiment(grid_size, ds, dt, NOISE, c, boundary="mur",
                             t_type=t_type, total_time=total_time, plot_path=plot_path)
    row_impulse = Experiment(grid_size, ds, dt, NOISE, c, boundary="mur", t_type=t_type, total_time=total_time, plot_path=plot_path)
    disruption_test = Experiment(grid_size, ds, dt, NOISE, c, boundary="mur", t_type="sin")

    ############################## Experiment 1-6 ##############################

    # "2_transducers" Experiment
    # row_impulse.plot_2_transducers()

    # "row_transducers" Experiment
    # row_impulse.plot_row_transducers()

    # "var_transducers" Experiment
    #row_impulse.plot_var_transducers()

    # "pml_test" Experiment
    # row_impulse.pml_test()

    # # The interference plot must only be used to plot the interference pattern of the previous experiments
    # additiona_path = f"MSFigures/2D/WS_{experiment_type}_{t_type}_{total_time}s.png"
    # row_impulse.plot_interference(additoonal_path=additiona_path)

    # "error_test" Experiment
    #test_subject = "noise"  # "ds", "dt", or "noise"
    #row_impulse.error_test(test_subject)

    # "1_noise_transducers" Experiment
    row_impulse.plot_1_noise_transducers()

    # "noise_threshold" Experiment
    # row_impulse.noise_threshold()
