import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Sources


class WaveSimulation3D:
    def __init__(self, grid_size, ds, dt, c, noise=None, pml_width=10, boundary="pml", stability_check=True):
        """
        Initialize the simulation.

        Parameters:
        grid_size: Tuple (nx, ny, nz)
            Spatial grid dimensions.
        ds: float
            Spatial step size.
        dt: float
            Time step size.
        c: float
            Speed of sound in the medium.
        boundary: str
            Boundary condition type ("reflective" or "absorbing").
        """
        self.nx, self.ny, self.nz = grid_size

        self.ds = ds
        self.dt = dt

        self.c = c

        # Initiatie the PML damping profile and boundary conditions
        self.boundary = boundary
        self.pml_width = pml_width
        self.noise = noise

        # Damping coefficient σ(x, y)
        self.sigma = np.zeros((self.nx, self.ny, self.nz))
        self._initialize_pml()

        # Wave function at t
        # u[y_i, x_i, z_i]
        self.u = np.zeros((self.nx, self.ny, self.nz))
        # Wave function at t-1
        self.u_prev = np.zeros((self.nx, self.ny, self.nz))
        # Wave function at t+1
        self.u_next = np.zeros((self.nx, self.ny, self.nz))

        # List of source functions
        self.sources = []

        self.time = 0

        # Stability condition (Courant condition)
        if stability_check:
            self.stability_limit = c * dt / ds
            if self.stability_limit > (1 / 3):
                raise ValueError(
                    "Stability condition violated. Reduce dt or increase ds.")

    def _initialize_pml(self):
        """Define the PML damping profile."""
        # TODO: Is this correct indexing?
        # Initialize sigma_x, sigma_y, and sigma_z to handle damping separately along axes
        sigma_x = np.zeros((self.nx, 1, 1))  # Along x-axis
        sigma_y = np.zeros((1, self.ny, 1))  # Along y-axis
        sigma_z = np.zeros((1, 1, self.nz))  # Along z-axis

        # Quadratic damping profile for x-direction
        for i in range(self.nx):
            dx = min(i, self.nx - 1 - i) / self.pml_width
            if dx < 1.0:
                sigma_x[i, 0, 0] = (1 - dx) ** 2

        # Quadratic damping profile for y-direction
        for j in range(self.ny):
            dy = min(j, self.ny - 1 - j) / self.pml_width
            if dy < 1.0:
                sigma_y[0, j, 0] = (1 - dy) ** 2

        # Quadratic damping profile for z-direction
        for k in range(self.nz):
            dz = min(k, self.nz - 1 - k) / self.pml_width
            if dz < 1.0:
                sigma_z[0, 0, k] = (1 - dz) ** 2

        # Combine damping profiles: Sum or take max for uniform edge damping
        self.sigma = sigma_x + sigma_y + sigma_z  # Uniform damping near edges

    def add_source(self, source_function):
        """
        Add a source to the simulation.

        Parameters:
        source_function: function
            A function f(x, y, z, t) defining the source.
        """
        self.sources.append(source_function)

    def step(self):
        """
        Perform one time step of the simulation.
        """
        c2 = self.c ** 2
        dt2 = self.dt ** 2
        ds2 = self.ds ** 2

        # Compute the Laplacian in 3D
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                for k in range(1, self.nz - 1):
                    laplacian = (
                        self.u[i + 1, j, k] + self.u[i - 1, j, k] +
                        self.u[i, j + 1, k] + self.u[i, j - 1, k] +
                        self.u[i, j, k + 1] + self.u[i, j, k - 1] -
                        6 * self.u[i, j, k]
                    ) / ds2

                    source_contribution = 0
                    for source in self.sources:
                        source_contribution += source(
                            i * self.ds, j * self.ds, k * self.ds, self.time)

                    self.u_next[i, j, k] = (
                        2 * self.u[i, j, k]
                        - self.u_prev[i, j, k]
                        + c2 * dt2 * (laplacian + source_contribution)
                    )

        # Apply boundary conditions
        if self.boundary == "reflective":
            self.u_next[0, :, :] = self.u_next[1, :, :]
            self.u_next[-1, :, :] = self.u_next[-2, :, :]
            self.u_next[:, 0, :] = self.u_next[:, 1, :]
            self.u_next[:, -1, :] = self.u_next[:, -2, :]
            self.u_next[:, :, 0] = self.u_next[:, :, 1]
            self.u_next[:, :, -1] = self.u_next[:, :, -2]
        elif self.boundary == "mur":
            c = self.c
            dt = self.dt
            ds = self.ds
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    for k in range(1, self.nz - 1):
                        self.u_next[0, j, k] = self.u[1, j, k] + (c * dt - ds) / (c * dt + ds) * (self.u_next[1, j, k] - self.u[0, j, k])
                        self.u_next[-1, j, k] = self.u[-2, j, k] + (c * dt - ds) / (c * dt + ds) * (self.u_next[-2, j, k] - self.u[-1, j, k])
                        self.u_next[i, 0, k] = self.u[i, 1, k] + (c * dt - ds) / (c * dt + ds) * (self.u_next[i, 1, k] - self.u[i, 0, k])
                        self.u_next[i, -1, k] = self.u[i, -2, k] + (c * dt - ds) / (c * dt + ds) * (self.u_next[i, -2, k] - self.u[i, -1, k])
                        self.u_next[i, j, 0] = self.u[i, j, 1] + (c * dt - ds) / (c * dt + ds) * (self.u_next[i, j, 1] - self.u[i, j, 0])
                        self.u_next[i, j, -1] = self.u[i, j, -2] + (c * dt - ds) / (c * dt + ds) * (self.u_next[i, j, -2] - self.u[i, j, -1])

            # Update time step
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev
        self.time += self.dt

    def plot_pml_profile(self, z_slice):
        """
        Plot the PML damping profile.
        Parameters:
        z_slice: int
            Index of the grid to plot a slice at.
        """
        plt.figure()
        plt.imshow(self.sigma, extent=(0, self.nx * self.ds, 0,
                   self.ny * self.ds), cmap='viridis', origin='lower')
        plt.colorbar(label="Damping Coefficient")
        plt.title("PML Damping Profile")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("MSFigures/2D/pml_profile.png")
        plt.show()

    def plot(self, z_slice, ax=None, vmin=-1.0, vmax=1.0):
        """
        Plot slice of the current field with a fixed color range at the given at the given z index.

        Parameters:
        z_slice: int
            Index of the grid to plot a slice at.
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
            self.u[:, :, z_slice], extent=(
                0, self.nx * self.ds, 0, self.ny * self.ds),
            cmap='viridis', origin='lower', vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Time: {self.time:.2f} s at z = {z_slice * self.ds:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if ax is None:
            plt.colorbar(im, ax=ax, label="Pressure")
            plt.show()

        return im

    def run_simulation(self, steps, z_slice, vmin=None, vmax=None, plot_interval=1):
        """
        Run the wave simulation and update the plot in real-time.

        Parameters:
        steps: int
            Number of time steps to run.
        z_slice: int
            Index of the grid to plot a slice at.
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
            self.u[:, :, z_slice], extent=(
                0, self.nx * self.ds, 0, self.ny * self.ds),
            cmap='viridis', origin='lower', vmin=vmin, vmax=vmax
        )

        plt.colorbar(im, ax=ax, label="Pressure")

        ax.set_title("Time: 0.00 s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()

        # Run the simulation
        for step in range(steps):
            self.step()

            if step % plot_interval == 0:
                # Update the image data
                im.set_data(self.u[:, :, z_slice])

                im.set_clim(vmin, vmax)

                ax.set_title(f"Time: {self.time:.2f} s")

                # Pause to update the plot
                # If you comment then it doesn't work
                plt.pause(0.0001)

        plt.ioff()
        plt.show()


class Experiment3D:
    def __init__(self, grid_size, ds, dt, noise, c, boundary="mur", t_type="sin", rows=2, cols=3, start_time=0, total_time=10, plot_path=None):
        '''
        Initialize the experiment with the simulation parameters and plot parameters.
        
        Parameters:
            grid_size: Tuple (nx, ny, nz)
                Spatial grid dimensions.
            ds: float
                Spatial step size.
            dt: float
                Time step size.
            noise: str
                Type of noise to add. Options are 'white', 'speckle', 'gaussian', 'perlin'.
            c: float
                Speed of sound in the medium.
            boundary: str
                Boundary condition type ("reflective", "absorbing" or "mur").
            t_type: str
                Type of source ("impulse" or "sin").
            start_time: float
                Time to start plotting.
            total_time: float
                Total time to run the simulation.
            plot_path: str
                Path to save the plot. If None, the plot is animated.
        '''
        
        # Initialize the simulation environment
        self.sim = WaveSimulation3D(grid_size, ds, dt, c, boundary=boundary)

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
        self.sim.run_simulation(steps=int(self.total_time / self.sim.dt) + 1, z_slice=self.grid_size[2] // 2, vmin=-1, vmax=1, plot_interval=1)

    def _plot(self):
        '''Internal method to plot the simulation.'''

        # Number of subplots
        num_subplots = self.rows * self.cols
        time_step = (self.total_time - self.start_time) / (num_subplots - 1)
        self.start_time /= 0.01

        # Create figure and subplots
        # Create figure and subplots
        fig, axes = plt.subplots(self.rows, self.cols, figsize=(20, 8), constrained_layout=True)
        axes = axes.flatten()
        plot_step = int(time_step / 0.01)  # Ensure plot_step is an integer

        for i in tqdm(range(int(self.total_time / 0.01) + 1)):  # Include the last step
            if self.sim.noise is not None:
                self.sim.step(addNoise=True, noise_amplitude=0.001)
            else:
                self.sim.step()

            if i >= self.start_time and i % plot_step == 0:
                subplot_index = i // plot_step
                if subplot_index < num_subplots:
                    im = self.sim.plot(ax=axes[subplot_index], z_slice=self.grid_size[2] // 2)
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02).set_label('Wave Amplitude')
        plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_2_transducers(self):
        '''Plots the wave field for two transducers in a random central location.'''

        # Add the selected sources to the simulation
        if self.t_type == "impulse":
            self.sim.add_source(Sources.create_impulse_source_3D(10000, 4, 4, 0, 0.02))
            self.sim.add_source(Sources.create_impulse_source_3D(10000, 6, 6, 0, 0.02))
        elif self.t_type == "sin":
            self.sim.add_source(Sources.create_sinusoidal_source_3D(10, 1, 4, 4, 0))
            self.sim.add_source(Sources.create_sinusoidal_source_3D(10, 1, 6, 6, 0))

        if self.plot_path is not None:
            self._plot()
        else:
            self._animate()

    def error_test(self, test_subject):
        '''Plots the wave field for different values of the error parameter (dt or ds).
        
        Parameters:
            test_subject: str'''

        def run_simulation_and_plot(param_name, param_values, ax):
            '''Internal method to run the simulation for different values of the error parameter and plot the results.
            
            Parameters:
                param_name: str
                    Name of the error parameter.
                param_values: array-like
                    Values of the error parameter to test.
                ax: matplotlib.axes.Axes
                    Axes to plot the results.
        
            Returns:
                im: matplotlib.image.AxesImage
                    Image object for the colorbar.
            '''
            for param in param_values:
                print(f"Running for {param_name} = {param}")

                if param_name == "dt":
                    new_env = WaveSimulation3D(self.grid_size, self.sim.ds, param, self.sim.c, boundary=self.sim.boundary)
                elif param_name == "ds":
                    new_env = WaveSimulation3D(self.grid_size, param, self.sim.dt, self.sim.c, boundary=self.sim.boundary)

                if self.t_type == "impulse":
                    new_env.add_source(Sources.create_impulse_source_3D(10000, 4, 4, 4, 0.02))
                    new_env.add_source(Sources.create_impulse_source_3D(10000, 6, 6, 6, 0.02))
                elif self.t_type == "sin":
                    new_env.add_source(Sources.create_sinusoidal_source_3D(10, 1, 4, 4, 4))
                    new_env.add_source(Sources.create_sinusoidal_source_3D(10, 1, 6, 6, 6))
                elif self.t_type == "both":
                    new_env.add_source(Sources.create_impulse_source_3D(10000, 4, 4, 4, 0.02))
                    new_env.add_source(Sources.create_sinusoidal_source_3D(10, 1, 6, 6, 6))

                # Run the simulation up to 3 seconds
                for _ in tqdm(range(int(3 / new_env.dt))):
                    new_env.step()

                # Plot at 2 seconds
                im = new_env.plot(ax=ax, z_slice=10)
                ax.set_title(f"Simulation step: {param_name} = {param:.3f}")

                del new_env
            return im
        
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

        if test_subject == "dt":
            param_values = np.linspace(0.01, 1, 6)
            im = run_simulation_and_plot("dt", param_values, ax)
        elif test_subject == "ds":
            param_values = np.linspace(0.01, 0.7, 6)
            im = run_simulation_and_plot("ds", param_values, ax)

        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.02).set_label('Wave Amplitude')
        if self.plot_path is not None:
            plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # General usage:
    # 1. Specify simulation parameters
    # 2. Create a simulation object
    # 3. Add sources to the simulation, there are some define in Sources.py
    # 4. Either manually step the simulation or run with animated plotting
    # 5. Alternatively, run the simulation for the desired experiment

    ##################### SETUP THE ENVIRONMENT PARAMETERS #####################
    # Dimensions of grid
    grid_size = (100, 100, 20)
    # Difference in distance between grid points
    ds = 0.1
    # Time step
    dt = 0.01
    # Speed of sound in medium
    c = 1.0

    ########################### SETUP THE EXPERIMENT ###########################

    experiment_type = "2transducers"  # "dt_error", "ds_error"
    t_type = "sin"  # "impulse" or "sin"
    total_time = 3      # Recommended: 6 sec for impulse, 10 sec for sin
    NOISE = None     # "white", "speckle", "gaussian", "perlin"

    plot_path = f"MSFigures/3D/WS_{experiment_type}_{t_type}_{total_time}s.png" # Pass None to animate
    #plot_path = None

    experiment = Experiment3D(grid_size, ds, dt, NOISE, c, boundary="mur", t_type=t_type, total_time=total_time, plot_path=plot_path)

    ############################## Experiment ##############################

    # "2_transducers" Experiment
    experiment.plot_2_transducers()

    # # "error_test" Experiment
    # test_subject = "dt"  # "ds" or "dt"
    # experiment.error_test(test_subject)

# if __name__ == "__main__":
#     # General usage:
#     # 1. Specify simulation parameters
#     # 2. Create a simulation object
#     # 3. Add sources to the simulation, there are some define in Sources.py
#     # 4. Either manually step the simulation or run with animatd plotting

#     # Dimensions of grid
#     grid_size = (100, 100, 100)

#     # Difference in distance between grid points
#     ds = 0.1

#     # Time step
#     dt = 0.01

#     # Speed of sound in medium
#     c = 1.0

#     sim = WaveSimulation3D(grid_size, ds, dt, c, boundary="absorbing")

#     sim.add_source(Sources.create_sinusoidal_source_3D(
#         amplitude=10.0, frequency=1.0, x0=5, y0=2, z0=5))

#     sim.run_simulation(steps=300, z_slice=50, vmin=-
#                        1, vmax=1, plot_interval=1)

#     # Step manually and plot
#     # for _ in tqdm(range(300)):
#     #     sim.step()
#     #     if _ % 30 == 0:
#     #         # Plotting is blocking
#     #         # Plot at the middle z-slice
#     #         sim.plot(z_slice=50)
