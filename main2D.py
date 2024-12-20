from Experiment2D import Experiment

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
grid_size = (100, 100)
# Difference in distance between grid points
ds = 0.1
# Time step
dt = 0.01
# Speed of sound in medium
c = 1.0

########################### SETUP THE EXPERIMENT ###########################

experiment_type = "combined"        # Specify the experiment type
t_type = "sin"                      # "impulse" or "sin" or "both"
total_time = 6                      # Recommended: 6 sec
Noise = None                        # None, "white", "speckle", "gaussian", "perlin"

# Pass None to animate
plot_path = f"MSFigures/2D/WS_{experiment_type}_{t_type}_{total_time}s.png"
# plot_path = None

experiment = Experiment(grid_size, ds, dt, c, noise=Noise, boundary="mur",
                        t_type=t_type, total_time=total_time, plot_path=plot_path)

############################### Experiments ################################

# "2_transducers" Experiment
# experiment.plot_2_transducers()

# "row_transducers" Experiment
#experiment.plot_row_transducers()

# "var_transducers" Experiment
# experiment.plot_var_transducers()

# "pml_test" Experiment
# experiment.pml_test()

# # The interference plot must only be used to plot the interference pattern of the previous experiments
# additiona_path = f"MSFigures/2D/WS_{experiment_type}_{t_type}_{total_time}s.png"
# experiment.plot_interference(additoonal_path=additiona_path)

# "error_test" Experiment
# test_subject = "noise"  # "ds", "dt"
# experiment.error_test(test_subject)

# "1_noise_transducers" Experiment
experiment.plot_1_noise_transducers()

# "noise_threshold" Experiment
# experiment.noise_threshold()
