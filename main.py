from functions import *

# Experimental parameters
experiment_params = [
    (-2, 1, 3, -4),  # Experiment 1
    (-1, 2, -3, 4),  # Experiment 2
    (0.5, -1.5, 2, -2.5),  # Experiment 3
    (1, -1, 1.5, -1.5),  # Experiment 4
    (-2.5, 2.5, -0.5, 0.5),  # Experiment 5
    (2, -2, 3, -3),  # Experiment 6
    (-1.5, 1, 2.5, -2),  # Experiment 7
    (1.5, -1, -2, 2),  # Experiment 8
    (-3, 3, -1, 1),  # Experiment 9
    (2.5, -2.5, 0.5, -0.5)  # Experiment 10
]

# Running experiments
for i, params in enumerate(experiment_params, start=1):
    conduct_experiment(*params, experiment_id=i)
