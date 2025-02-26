# Standard library imports
import os
import sys
import time

# To ignore initilization message from Pygame
sys.stdout = open(os.devnull, 'w')

# External library imports
import pygame as pg

# Reactivate the standard output
sys.stdout = sys.__stdout__

# Internal module imports
import model
import display

def analyze_arguments(args, parameters={}):
    """
    Parses command-line arguments and updates the parameters dictionary.
    
    Parameters:
    args (list): List of command-line arguments.
    parameters (dict): Dictionary to store parsed parameters.

    Raises:
    SyntaxError: If an expected argument value is missing or an unknown argument is encountered.
    """
    if len(args) == 0:
        return parameters
    
    key = args[0]
    
    if key in ("-l", "--length"):
        if len(args) < 2:
            raise SyntaxError("A value is expected for the terrain length!")
        parameters["terrain_size"] = float(args[1])
        analyze_arguments(args[2:], parameters)
        return
    
    if key in ("-n", "--number_of_cells"):
        if len(args) < 2:
            raise SyntaxError("A value is expected for the number of cells per direction!")
        parameters["grid_size"] = int(args[1])
        analyze_arguments(args[2:], parameters)
        return
    
    if key in ("-w", "--wind"):
        if len(args) < 2:
            raise SyntaxError("A pair of values X,Y is expected for the wind vector!")
        wx, wy = map(float, args[1].split(","))
        parameters["wind_vector"] = (wx, wy)
        analyze_arguments(args[2:], parameters)
        return
    
    if key in ("-s", "--start"):
        if len(args) < 2:
            raise SyntaxError("A pair of indices (row, column) is expected for the fire start position!")
        fi, fj = map(int, args[1].split(","))
        parameters["fire_start"] = (fi, fj)
        analyze_arguments(args[2:], parameters)
        return
    
    raise SyntaxError(f"Unknown argument: {key}")

def display_help():
    """
    Displays usage information and exits the program.
    """
    header = 50*"=" + " Simulation help " + 50*"="
    print("\n" + header)
    print("""
        Usage: simulation [OPTIONS]
          
        Launches the fire simulation considering the provided options.
          
        Options:
            -h, --help              Provides this help.
            -l, --length            Defines the terrain size (default: 1.0km).
            -n, --number_of_cells   Number of cells per direction for discretization (default:20).
            -w, --wind              Defines the wind velocity vector (default: (1.0,1.0), no wind).
            -s, --start             Defines the indices where the fire starts (default: (10,10)).
    """)
    print(len(header)*"=" + "\n")

    exit(1)

def check_parameters(params):
    """
    Validates the simulation parameters.
    
    Parameters:
    params (dict): Dictionary containing simulation parameters.
    
    Returns:
    bool: True if parameters are valid, otherwise raises an error.

    Raises:
    ValueError: If any parameter value is invalid.
    """
    if params["terrain_size"] <= 0:
        raise ValueError("[FATAL ERROR] Terrain length must be positive!")
    if params["grid_size"] <= 0:
        raise ValueError("[FATAL ERROR] Number of cells per direction must be positive!")
    if not (0 <= params["fire_start"][0] < params["grid_size"] and 0 <= params["fire_start"][1] < params["grid_size"]):
        raise ValueError("[FATAL ERROR] Invalid indices for fire start position!")
    return True

def display_parameters(params):
    """
    Displays the defined parameters for the simulation.
    
    Parameters:
    params (dict): Dictionary containing simulation parameters.
    """
    header = 20*"=" + " Simulation parameters " + 20*"="
    print("\n" + header + "\n")
    print(f"\tTerrain size: {params['terrain_size']}km")
    print(f"\tGrid size: {params['grid_size']}x{params['grid_size']} ")
    print(f"\tWind vector: {params['wind_vector']}")
    print(f"\tFire start position: {params['fire_start']}")
    print("\n" + len(header)*"=" + "\n")

def initialize_simulation(parse_args):
    """
    Initializes the simulation with the parsed arguments and returns the parameters.
    
    Parameters:
    parse_args (list): List of command-line arguments.
    
    Returns:
    dict: Dictionary containing simulation parameters.
    """
    params = {
        "terrain_size": 1.0,
        "grid_size": 20,
        "wind_vector": (1.0, 1.0),
        "fire_start": (10, 10)
    }
    
    if "--help" in parse_args or "-h" in parse_args:
        display_help()
    
    analyze_arguments(parse_args, params)
    display_parameters(params)
    check_parameters(params)

    return params

def run_simulation(parse_args, n_iterations=100):
    """
    Runs the fire simulation using the provided parameters.
    
    Parameters:
    parse_args (list): List of command-line arguments.
    """
    params = initialize_simulation(parse_args)

    pg.init()
    fire_model = model.FireSpreadModel(params["terrain_size"], params["grid_size"], params["wind_vector"], params["fire_start"])
    fire_display = display.DisplayFire(params["grid_size"])
    fire_display.update(fire_model.fire_map, fire_model.vegetation_map)

    header = 20*"=" + " Running simulation " + 20*"="
    print("\n" + header + "\n")

    # Files to save the computing and rendering performances
    with open("../results/computing_times.txt", "w") as compute_file, open("../results/rendering_times.txt", "w") as render_file:
        compute_file.write("Computing time\n")
        render_file.write("Rendering time\n")

        must_continue = True
        while must_continue and fire_model.time_step < n_iterations:
            # Fire update computing time
            start_time = time.time()
            fire_update = fire_model.update_fire()
            end_time = time.time()
            compute_time = end_time - start_time
            compute_file.write(f"{compute_time*1000:.6f}\n") # Computing time in ms

            if fire_update: # If the fire is still burning
                # Fire display rendering time
                start_time = time.time()
                fire_display.update(fire_model.fire_map, fire_model.vegetation_map)
                end_time = time.time()
                render_time = end_time - start_time
                render_file.write(f"{render_time:.6f}\n") # Rendering time in ms

                if fire_model.time_step % 10 == 0:
                    print(f"\tTimestep {fire_model.time_step}:")
                    print(f"\t\tComputing time: {compute_time*1000:.6f}ms")
                    print(f"\t\tRendering time: {render_time*1000:.6f}ms\n")
                
                # Stop the simulation if the user closed the display window
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        must_continue = False
                        pg.quit()

    footer = 20*"=" + " End of simulation " + 20*"="
    print(footer + "\n")

if __name__ == "__main__":
    run_simulation(sys.argv[1:])
