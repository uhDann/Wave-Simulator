from Experiment3D import Experiment

# General usage:
# 1. Specify simulation parameters
# 2. Create a simulation object
# 3. Add sources to the simulation, there are some define in Sources.py
# 4. Either manually step the simulation or run with animated plotting
# 5. Alternatively, run the simulation for the desired experiment

##################### SETUP THE ENVIRONMENT PARAMETERS #####################
# Dimensions of grid
grid_size = (50, 50, 50)
# Difference in distance between grid points
ds = 0.1
# Time step
dt = 0.01
# Speed of sound in medium
c = 1.0

########################### SETUP THE EXPERIMENT ###########################

experiment_type = "combined"  # "dt_error", "ds_error"
t_type = "both"  # "impulse" or "sin" or "both"
total_time = 0.5      # Recommended: 6 sec for impulse, 10 sec for sin

plot_path = f"MSFigures/3D/WS_{experiment_type}_{t_type}_{total_time}s.png" # Pass None to animate
#plot_path = None

experiment = Experiment(grid_size, ds, dt, c, boundary="mur", t_type=t_type, total_time=total_time, plot_path=plot_path)

############################## Experiment ##############################

# "2_transducers" Experiment
experiment.plot_2_transducers()