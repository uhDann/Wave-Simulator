import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Sources


class WaveSimulation2D:
    def __init__(self, grid_size, ds, dt, c, pml_width=10, boundary="pml"):
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
        boundary: str
            Boundary condition type ("reflective" or "absorbing").
        """
        self.nx, self.ny = grid_size

        self.ds = ds
        self.dt = dt

        self.c = c

        # Initiatie the PML damping profile and boundary conditions
        self.boundary = boundary
        self.pml_width = pml_width

        # Damping coefficient σ(x, y)
        self.sigma = np.zeros((self.nx, self.ny))
        self._initialize_pml()

        # Wave function at t
        self.u = np.zeros((self.nx, self.ny))
        # Wave function at t-1
        self.u_prev = np.zeros((self.nx, self.ny))
        # Wave function at t+1
        self.u_next = np.zeros((self.nx, self.ny))

        # List of source functions
        self.sources = []

        self.time = 0

        # Stability condition (Courant condition)
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

    # TODO: "absorbing" and "pml" do not work as expected yet
    def step(self):
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
        elif self.boundary == "absorbing":
            self.u_next[0, :] = 0
            self.u_next[-1, :] = 0
            self.u_next[:, 0] = 0
            self.u_next[:, -1] = 0
        elif self.boundary == "pml":
            self.u_next *= np.exp(-self.sigma * self.dt)
        elif self.boundary == "mur":
            # Apply Mur absorbing boundary condition
            self.u_next[0, :] = self.u[1, :] + (self.c * self.dt - self.ds) / (self.c * self.dt + self.ds) * (self.u_next[1, :] - self.u[0, :])
            self.u_next[-1, :] = self.u[-2, :] + (self.c * self.dt - self.ds) / (self.c * self.dt + self.ds) * (self.u_next[-2, :] - self.u[-1, :])
            self.u_next[:, 0] = self.u[:, 1] + (self.c * self.dt - self.ds) / (self.c * self.dt + self.ds) * (self.u_next[:, 1] - self.u[:, 0])
            self.u_next[:, -1] = self.u[:, -2] + (self.c * self.dt - self.ds) / (self.c * self.dt + self.ds) * (self.u_next[:, -2] - self.u[:, -1])


        # Update time step
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev
        self.time += self.dt

    def plot_pml_profile(self):
        """Plot the PML damping profile."""
        plt.figure()
        plt.imshow(self.sigma, extent=(0, self.nx * self.ds, 0, self.ny * self.ds), cmap='viridis', origin='lower')
        plt.colorbar(label="Damping Coefficient")
        plt.title("PML Damping Profile")
        plt.xlabel("x")
        plt.ylabel("y")
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
        ax.set_title(f"Time: {self.time-0.01:.2f} s")
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
            self.step()

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
    def __init__(self, grid_size, ds, dt, c, boundary="mur", t_type="sin", rows=2, cols=3, start_time=0, total_time=10):
        self.sim = WaveSimulation2D(grid_size, ds, dt, c, boundary=boundary)

        # Save the plot parameters
        self.rows = rows
        self.cols = cols
        self.start_time = start_time
        self.total_time = total_time
        if t_type not in ["impulse", "sin"]:
            raise ValueError("Invalid source type. Use 'impulse' or 'sin'.")
        self.t_type = type

    def _plot(self):
        # Number of subplots
        num_subplots = self.rows * self.cols
        time_step = (self.total_time - self.start_time) / (num_subplots - 1)
        self.start_time /= 0.01

        # Create figure and subplots
        fig, axes = plt.subplots(self.rows, self.cols, figsize=(20, 8), constrained_layout=True)
        # Flatten the axes array for easy indexing
        axes = axes.flatten()
        plot_step = int(time_step / 0.01)  # Ensure plot_step is an integer

        # Step manually and plot
        for i in tqdm(range(int(self.total_time / 0.01) + 1)):  # Include the last step
            self.sim.step()
            if i >= self.start_time and i % plot_step == 0:
                subplot_index = i // plot_step
                if subplot_index < num_subplots:
                    im = self.sim.plot(ax=axes[subplot_index])


        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02).set_label('Wave Amplitude')
        plt.savefig("MSFigures/2D/WaveSimulation2D_test.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_row_transducers(self):

        if self.t_type == "impulse":
            for i in range(1, 9, 3):
                self.sim.add_source(Sources.create_impulse_source_2D(10000, i, 1, 0.02))
        elif self.t_type == "sin":
            for i in range(0, 10, 2):
                self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 1, i, 0))

        self._plot()
    
    def plot_2_trans(self):

        if self.t_type == "impulse":
            self.sim.add_source(Sources.create_impulse_source_2D(10000, 4, 4, 0.02))
            self.sim.add_source(Sources.create_impulse_source_2D(10000, 6, 6, 0.02))
        elif self.t_type == "sin":
            self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 4, 4, 0))
            self.sim.add_source(Sources.create_sinusoidal_source_2D(10, 6, 6, 0))

        self._plot()

        


        
        


if __name__ == "__main__":
    # General usage:
    # 1. Specify simulation parameters
    # 2. Create a simulation object
    # 3. Add sources to the simulation, there are some define in Sources.py
    # 4. Either manually step the simulation or run with animatd plotting

    # Dimensions of grid
    grid_size = (100, 100)

    # Difference in distance between grid points
    ds = 0.1

    # Time step
    dt = 0.01

    # Speed of sound in medium
    c = 1.0

    # Set the transducer type to "impulse" or "sin"
    t_type = "sin"

    row_impulse = Experiment(grid_size, ds, dt, c, t_type=t_type, total_time=6)

    row_impulse.plot_row_transducers()


    
    # sim = WaveSimulation2D(grid_size, ds, dt, c, boundary="mur")
    # sim.add_source(Sources.create_impulse_source_2D(10000, 1, 0, 0))
    # sim.add_source(Sources.create_impulse_source_2D(10000, 3, 0, 0))
    # sim.run_simulation(steps=1000, vmin=-1, vmax=1, plot_interval=1)