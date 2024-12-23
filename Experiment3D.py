import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

from WaveSimulation3D import WaveSimulation3D
import Sources

class Experiment:
    def __init__(self, grid_size, ds, dt, c, boundary="mur", t_type="sin", rows=2, cols=3, start_time=0, total_time=10, plot_path=None):
        '''
        Initialize the experiment with the simulation parameters and plot parameters.
        
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
                Boundary condition type ("reflective", "absorbing" or "mur").
            t_type: str
                Type of source ("impulse" or "sin").
            rows: int
                Number of rows in the plot.
            cols: int
                Number of columns in the plot.
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
            raise ValueError("Invalid source type. Use 'impulse' or 'sin' or 'both'.")
        self.t_type = t_type
        self.plot_path = plot_path

        if plot_path is None:
            print("[INFO] No plot_path passed, where possible the animation will be displayed. For some tests, only the plot will be displayed.")

    def _animate(self, z_slice=None):
        '''Internal method to animate the simulation.'''

        if z_slice is None:
            z_slice = self.grid_size[2] // 2
        self.sim.run_simulation(steps=int(self.total_time / self.sim.dt) + 1, z_slice=z_slice, vmin=-1, vmax=1, plot_interval=1)

    def _plot(self, z_slice=None, point1=None, point2=None):
        '''Internal method to plot the simulation.'''

        if z_slice is None:
            z_slice = self.grid_size[2] // 2

        # Number of subplots
        num_subplots = self.rows * self.cols
        time_step = (self.total_time - self.start_time) / (num_subplots - 1)
        self.start_time /= 0.01

        if point1 is not None and point2 is not None:
            # Create figure with GridSpec for z-slice and plane slice
            fig = plt.figure(figsize=(20, 16), constrained_layout=True)
            spec = GridSpec(2 * self.rows, self.cols, figure=fig)
        else:
            # Create figure with GridSpec for z-slice only
            fig = plt.figure(figsize=(20, 8), constrained_layout=True)
            spec = GridSpec(self.rows, self.cols, figure=fig)

        plot_step = int(time_step / 0.01)  # Ensure plot_step is an integer

        im_z = None
        im_plane = None

        axes = []
        # Top row for z-slice
        for row in range(self.rows):
            for col in range(self.cols):
                axes.append(fig.add_subplot(spec[row, col]))

        # Bottom row for plane slice (if applicable)
        if point1 is not None and point2 is not None:
            for row in range(self.rows, 2 * self.rows):
                for col in range(self.cols):
                    axes.append(fig.add_subplot(spec[row, col]))

        for i in tqdm(range(int(self.total_time / 0.01) + 1)):  # Include the last step
            
            self.sim.step()

            if i >= self.start_time and i % plot_step == 0:
                subplot_index = i // plot_step
                if subplot_index < num_subplots:
                    im_z = self.sim.plot(ax=axes[subplot_index], z_slice=z_slice, vmin=-1, vmax=1)
                    if point1 is not None and point2 is not None:
                        im_plane = self.sim.plot(ax=axes[subplot_index + num_subplots], point1=point1, point2=point2, vmin=-1, vmax=1)

        # Add shared colorbar for consistency
        cbar = fig.colorbar(im_z, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('Wave Amplitude')

        # Save or display
        if self.plot_path is not None and point1 is not None and point2 is not None:
            plt.savefig(self.plot_path.replace('.png', '_combined.png'), dpi=300, bbox_inches='tight')
        elif self.plot_path is not None:
            plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_2_transducers(self):
        '''Plots the wave field for two transducers in a random central location.'''

        point1 = [4, 4, 0]
        point2 = [2, 2, 0]

        # Add the selected sources to the simulation
        if self.t_type == "impulse":
            self.sim.add_source(Sources.create_impulse_source_3D(10e39, point1[0], point1[1], point1[2]+0.01, 0.02))
            self.sim.add_source(Sources.create_impulse_source_3D(10e39, point2[0], point2[1], point2[2]+0.01, 0.02))
        elif self.t_type == "sin":
            self.sim.add_source(Sources.create_sinusoidal_source_3D(50, 1, point1[0], point1[1], point1[2]))
            self.sim.add_source(Sources.create_sinusoidal_source_3D(50, 1, point2[0], point2[1], point2[2]))
        elif self.t_type == "both":
            self.sim.add_source(Sources.create_impulse_source_3D(10e39, 4, 4, 0.01, 0.02))
            self.sim.add_source(Sources.create_sinusoidal_source_3D(50, 1, 2, 2, 0))

        if self.plot_path is not None:
            self._plot(z_slice=self.grid_size[2] // 4, point1=point1[:-1], point2=point2[:-1])
        else:
            self._animate()

    def plot_grid_transducers(self):
        '''Plots the wave field for a grid of chosen transducers.'''

        if self.t_type == "both":
            raise ValueError("Cannot use 'both' source type for this experiment.")
        
        point1 = [4, 4, 0]
        point2 = [2, 2, 0]
        
        # Add the selected sources to the simulation
        if self.t_type == "impulse":
            for i in range(1, 5):
                for j in range(1, 5):
                    self.sim.add_source(
                        Sources.create_impulse_source_3D(10e39, i, j, 0.01, 0.02))
        elif self.t_type == "sin":
            for i in range(0, 6):
                for j in range(0, 6):
                    self.sim.add_source(
                        Sources.create_sinusoidal_source_3D(50, 1, i, j, 0))

        if self.plot_path is not None:
            self._plot(z_slice=self.grid_size[2] // 4, point1=point1[:-1], point2=point2[:-1])
        else:
            self._animate()
