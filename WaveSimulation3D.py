import matplotlib.pyplot as plt
import numpy as np

class WaveSimulation3D:
    def __init__(self, grid_size, ds, dt, c, boundary="mur", stability_check=True):
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
        stability_check: bool
            True to enable the stability condition check.
        """
        self.nx, self.ny, self.nz = grid_size

        self.ds = ds
        self.dt = dt
        self.boundary = boundary

        self.c = c

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


    def plot(self, z_slice=None, point1=None, point2=None, ax=None, vmin=-1.0, vmax=1.0):
        """
        Plot slice of the current field with a fixed color range at the given z index or along a plane created by two points.

        Parameters:
        z_slice: int, optional
            Index of the grid to plot a slice at along the z-axis.
        point1: tuple of int, optional
            First point (x, y) to define a vertical plane.
        point2: tuple of int, optional
            Second point (x, y) to define a vertical plane.
        vmin: float, optional
            Minimum value for the color scale.
        vmax: float, optional
            Maximum value for the color scale.

        """
        if z_slice is not None:
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
            ax.set_title(f"t: {self.time-self.dt:.2f}s at z = {z_slice * self.ds:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if ax is None:
                plt.colorbar(im, ax=ax, label="Pressure")
                plt.show()

            return im

        elif point1 is not None and point2 is not None:
            # Extract the coordinates
            x1, y1 = point1
            x2, y2 = point2

            # Calculate the plane equation coefficients (ax + by = d)
            a = y1 - y2
            b = x2 - x1
            d = a * x1 + b * y1

            # Create a grid for the plane
            xx, zz = np.meshgrid(np.arange(self.nx), np.arange(self.nz))
            yy = (d - a * xx) / b

            # Interpolate the field values on the plane
            plane_slice = np.zeros_like(xx, dtype=float)
            for i in range(self.nx):
                for j in range(self.nz):
                    if 0 <= yy[j, i] < self.ny:
                        plane_slice[j, i] = self.u[i, int(yy[j, i]), j]

            if vmin is None or vmax is None:
                vmin, vmax = np.min(plane_slice), np.max(plane_slice)

            if ax is None:
                fig, ax = plt.subplots()

            im = ax.imshow(
                plane_slice, extent=(
                    0, self.nx * self.ds, 0, self.nz * self.ds),
                cmap='viridis', origin='lower', vmin=vmin, vmax=vmax
            )
            ax.set_title(f"t: {self.time-self.dt:.2f}s along plane def by {point1} and {point2}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            if ax is None:
                plt.colorbar(im, ax=ax, label="Pressure")
                plt.show()

            return im
        else:
            raise ValueError("Either z_slice or both point1 and point2 must be provided.")

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