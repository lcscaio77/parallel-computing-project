# Standard library imports
import sys
import time

# External library imports
import pygame as pg

# Internal module imports
import model
import display

def analyze_arguments(args, parameters={}):
    """
    Parses command-line arguments and updates the parameters dictionary.
    
    Parameters:
    args (list): List of command-line arguments.
    parameters (dict): Dictionary to store parsed parameters.
    """
    if len(args) == 0:
        return parameters
    
    key = args[0]
    
    if key in ("-l", "--length"):
        if len(args) < 2:
            raise SyntaxError("A value is expected for the terrain length!")
        parameters["length"] = float(args[1])
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
        parameters["wind"] = (wx, wy)
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
    print("""
Usage: simulation [option(s)]
  Launches the fire simulation considering the provided options.
  Options:
    -l, --length=VALUE      Defines the terrain size (float, km).
    -n, --number_of_cells=N   Number of cells per direction for discretization.
    -w, --wind=VX,VY          Defines the wind velocity vector (default: no wind).
    -s, --start=ROW,COL       Defines the indices where the fire starts (default: (10,10)).
    """)
    exit(1)

def check_parameters(params):
    """
    Validates the simulation parameters.
    
    Parameters:
    params (dict): Dictionary containing simulation parameters.
    
    Returns:
    bool: True if parameters are valid, otherwise raises an error.
    """
    if params["length"] <= 0:
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
    print("Simulation parameters:")
    print(f"\tTerrain size: {params['length']} km")
    print(f"\tGrid size: {params['grid_size']} cells per direction")
    print(f"\tWind vector: {params['wind']}")
    print(f"\tFire start position: {params['fire_start']}")

# Default parameters
params = {
    "length": 1.0,
    "grid_size": 20,
    "wind": (1.0, 1.0),
    "fire_start": (10, 10)
}

# Parse command-line arguments
parse_args = sys.argv[1:]
if "--help" in parse_args or "-h" in parse_args:
    display_help()
analyze_arguments(parse_args, params)
display_parameters(params)

# Validate parameters
check_parameters(params)

# Initialize Pygame
pg.init()

# Initialize fire model and display
fire_model = model.FireSpreadModel(params["length"], params["grid_size"], params["wind"], params["fire_start"])
fire_display = display.DisplayFire(params["grid_size"])

# Update display with initial state
fire_display.update(fire_model.fire_map, fire_model.vegetation_map)

must_continue = True
start_time = time.time()
while fire_model.update_fire() and must_continue:
    compute_time = time.time()
    print(f"Computation time: {compute_time - start_time}")
    if fire_model.time_step % 32 == 0:
        print(f"Time step {fire_model.time_step}\n==============")
    start_time = time.time()
    
    fire_display.update(fire_model.fire_map, fire_model.vegetation_map)
    render_time = time.time()
    print(f"Rendering time: {render_time - start_time}")
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            must_continue = False
            pg.quit()

print("Simulation complete")
