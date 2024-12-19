# Wave Simulator Project

## Overview
The Wave Simulator project aims to develop a comprehensive model for simulating wave phenomena in various environments. This project includes the implementation of simulation tools, derivation references, and a detailed description of the process involved in modeling and running simulations.

## Team Members
- **Misha Rudchenko**
- **Danila Kozlov**
- **Charlene Chen**

## Table of Contents

- Installation
- Usage
  - 2D Wave Simulation
  - 3D Wave Simulation
- Experiments
- Contributing
- License

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/uhDann/Wave-Simulator.git
    cd wave-simulator
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 2D Wave Simulation

To run a 2D wave simulation, you can use the [`main2D.py`](main2D.py ) script. The script allows you to specify various simulation parameters and run different types of experiments.

Example usage:
```sh
python main2D.py
```

### 3D Wave Simulation

To run a 3D wave simulation, you can use the `main3D.py` script. The script allows you to specify various simulation parameters and run different types of experiments.

Example usage:
```sh
python main3D.py
```

## Experiments

The framework supports various experiments that can be configured in the `main2D.py` and `main3D.py` scripts. Some of the available experiments include:

- `plot_row_transducers`: Plots the wave field for a row of chosen transducers.
- `plot_2_transducers`: Plots the wave field for two transducers in a random central location.
- `plot_var_transducers`: Plots the wave field for a row of transducers with varying frequencies.
- `pml_test`: Plots the wave field for two transducers in a random central location with PML boundary.
- `error_test`: Plots the wave field for different values of the error parameter (dt or ds).
- `plot_interference`: Plots the interference pattern of the wave field.
- `plot_1_noise_transducers`: Plots the wave field and finds the threshold of when noise is too disruptive.

## License

This project is licensed under the MIT License. 

## Contact

For any questions or concerns, please open an issue in the repository or contact the project maintainers.