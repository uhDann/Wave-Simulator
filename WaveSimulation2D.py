import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from scipy.ndimage import gaussian_filter


class WaveSimulation2D:
    def __init__(self, grid_size, ds, dt, c, noise=None, pml_width=10, boundary="mur", stability_check=True):
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
            Boundary condition type ("reflective" or "pml" or "mur").
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
            self.simulate_noise(amplitude=0.01)

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

    def simulate_noise(self, amplitude=0.01):
        """
        Add simulated noise to the wave field.

        Parameters:
        amplitude: float
            The amplitude of the noise to scale the random values.

        """
        # Environment/system noise
        # Fine-grained random variations, overall roughness/grain
        if self.noise == "white":
            noise = np.random.uniform(
                low=-amplitude, high=amplitude, size=self.u.shape)
        # Creates complex, multi-scale textures (rough terrains, turbulent patterns)
        elif self.noise == "speckle":
            noise = self.u * \
                np.random.normal(1.0, amplitude, size=self.u.shape)

        # Texture noise
        # Smooth random variations, overall smoothness
        elif self.noise == "gaussian":
            orig_noise = np.random.normal(0, amplitude, size=self.u.shape)
            noise = gaussian_filter(orig_noise, sigma=2)
        # Natural looking textures (clouds, terrain, waves)
        elif self.noise == "perlin":
            noise = self.perlin_noise(
                self.u.shape, scale=20.0, amplitude=amplitude)
        elif self.noise == None:
            noise = np.zeros(self.u.shape)
        else:
            raise ValueError(
                "Invalid noise type. Use 'white', 'speckle', 'gaussian', 'perlin', or None.")

        self.u += noise
    

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

