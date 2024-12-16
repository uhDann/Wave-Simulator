import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Sources


class WaveSimulation3D:
    def __init__(self, grid_size, ds, dt, c, boundary="absorbing"):
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

        self.boundary = boundary

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
        self.stability_limit = c * dt / ds
        if self.stability_limit > (1 / 3):
            raise ValueError(
                "Stability condition violated. Reduce dt or increase ds.")

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
                    self.u_next[i, j, k] = (
                        2 * self.u[i, j, k]
                        - self.u_prev[i, j, k]
                        + c2 * dt2 * laplacian
                    )

        # Add sources
        for source in self.sources:
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        self.u_next[i, j, k] += self.dt ** 2 * \
                            source(i * self.ds, j * self.ds,
                                   k * self.ds, self.time)

        # Apply boundary conditions
        if self.boundary == "reflective":
            self.u_next[0, :, :] = self.u_next[1, :, :]
            self.u_next[-1, :, :] = self.u_next[-2, :, :]
            self.u_next[:, 0, :] = self.u_next[:, 1, :]
            self.u_next[:, -1, :] = self.u_next[:, -2, :]
            self.u_next[:, :, 0] = self.u_next[:, :, 1]
            self.u_next[:, :, -1] = self.u_next[:, :, -2]
        elif self.boundary == "absorbing":
            self.u_next[0, :, :] = 0
            self.u_next[-1, :, :] = 0
            self.u_next[:, 0, :] = 0
            self.u_next[:, -1, :] = 0
            self.u_next[:, :, 0] = 0
            self.u_next[:, :, -1] = 0

        # Update time step
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev
        self.time += self.dt

    def plot(self, z_slice, vmin=-1.0, vmax=1.0):
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

        plt.imshow(self.u[:, :, z_slice], extent=(
            0, self.nx * self.ds, 0, self.ny * self.ds), cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)

        plt.colorbar(label="Pressure")
        plt.title(f"Time: {self.time:.2f} s at z = {z_slice * self.ds:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

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


if __name__ == "__main__":
    # General usage:
    # 1. Specify simulation parameters
    # 2. Create a simulation object
    # 3. Add sources to the simulation, there are some define in Sources.py
    # 4. Either manually step the simulation or run with animatd plotting

    # Dimensions of grid
    grid_size = (100, 100, 100)

    # Difference in distance between grid points
    ds = 0.1

    # Time step
    dt = 0.01

    # Speed of sound in medium
    c = 1.0

    sim = WaveSimulation3D(grid_size, ds, dt, c, boundary="absorbing")

    sim.add_source(Sources.create_sinusoidal_source_3D(
        amplitude=10.0, frequency=1.0, x0=5, y0=2, z0=5))

    sim.run_simulation(steps=300, z_slice=50, vmin=-
                       1, vmax=1, plot_interval=1)

    # Step manually and plot
    # for _ in tqdm(range(300)):
    #     sim.step()
    #     if _ % 30 == 0:
    #         # Plotting is blocking
    #         # Plot at the middle z-slice
    #         sim.plot(z_slice=50)
