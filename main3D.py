from Experiment3D import Experiment

'''
General usage:

1. Specify simulation parameters
2. Specify the experiment type
3. Create the experiment object
4. Pass the path to save the plot or pass None to animate
5. Run the desired experiment

'''

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

experiment_type = "combined"        # Specify the experiment type
t_type = "sin"                      # "impulse" or "sin" or "both"
total_time = 6                      # Recommended: 3 sec for impulse, 5 sec for sin

plot_path = f"MSFigures/3D/WS_{experiment_type}_{t_type}_{total_time}s.png" # Pass None to animate
#plot_path = None

experiment = Experiment(grid_size, ds, dt, c, boundary="mur", 
                        t_type=t_type, total_time=total_time, plot_path=plot_path)

############################## Experiment ##############################

# "2_transducers" Experiment
experiment.plot_2_transducers()

# "grid_transducers" Experiment
# "both" is not specified
# experiment.plot_grid_transducers()