# Standard library imports
import os
import sys
import time

# To ignore initilization message from Pygame
sys.stdout = open(os.devnull, 'w')

# External library imports
import numpy as np
import pygame as pg
from mpi4py import MPI

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
            -l, --length            Defines the terrain nbp (default: 1.0km).
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
    print(f"\tTerrain nbp: {params['terrain_size']}km")
    print(f"\tGrid nbp: {params['grid_size']}x{params['grid_size']} ")
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

def run_simulation(parse_args, n_iterations=250):
    """
    Runs the fire simulation using the provided parameters.
    
    Parameters:
    parse_args (list): List of command-line arguments.
    """
    # Process 0 manages the display (and prints)
    if rank == 0:
        params = initialize_simulation(parse_args)
    else:
        params = None

    # Process 0 sends the parameters of the simulation to the other processes
    params = comm.bcast(params, root=0)
    
    grid_size = params["grid_size"] # Size of the grid
    n_model_procs = nbp - 1 # Number of processes managing the model

    if rank == 0: # Process 0 divides the grid in chunks for each computing process
        chunk = grid_size // n_model_procs
        reste = grid_size % n_model_procs
        subdomains = []
        start_row = 0
        for r in range(1, nbp): 
            length = chunk + (1 if r <= reste else 0)
            subdomains.append((start_row, start_row+length))
            start_row += length
    else:
        subdomains = None

    # Process 0 sends the division to the other processes
    subdomains = comm.bcast(subdomains, root=0)

    # Each computing process manage a division of the grid
    if rank > 0:
        row_start, row_end = subdomains[rank-1]
        fire_model = model.FireSpreadModel(
            terrain_size=params["terrain_size"],
            grid_size=grid_size,
            wind_vector=params["wind_vector"],
            fire_start_position=params["fire_start"],
            max_wind_speed=60.0,
            row_start=row_start, # Beginning of the division
            row_end=row_end # End of the division
        )
    else:
        fire_model = None

    if rank == 0:
        # Display initialization
        pg.init()
        fire_display = display.DisplayFire(grid_size)
        
        # Initialization of the "reduced model" for process 0
        full_fire_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
        full_vege_map = np.zeros((grid_size, grid_size), dtype=np.uint8)

        header = 20*"=" + " Running simulation " + 20*"="
        print("\n" + header + "\n", flush=True)

    if rank == 0:
        # File to save the rendering performances
        render_file = open(f"../results/rendering_times_par_step_3.txt", "w")
        render_file.write("Rendering time\n")
        
        # File to save the computing performances
        compute_file = open(f"../results/computing_times_par_step_3.txt", "w")
        compute_file.write("Computing time\n")
    else:
        # File to save the local (for each process) computing performances
        compute_file = open(f"../results/computing_times_par_proc_{rank}.txt", "w")
        compute_file.write("Computing time\n")
        
    
    # Start of the simulation
    must_continue = True
    t = 0
    if rank == 0:
            simulation_time_start = time.time()
    while must_continue and t < n_iterations:
        # Local update status default initialization
        local_update = False
        
        if rank == 0:
            iteration_time_start = time.time()

        if rank > 0:
            # Computing processes exchange informations on their ghost cells
            fire_model.exchange_ghost_cells(comm, rank, nbp)

            # Local model update
            local_model_update_time_start = time.time()
            local_update = fire_model.update_fire()

            # Local model update computing time
            local_model_update_time = time.time() - local_model_update_time_start
            compute_file.write(f"{local_model_update_time*1000:.3f}\n") # Local computing time in ms

        comm.Barrier()

        if rank == 0:
            iteration_time = time.time() - iteration_time_start
            compute_file.write(f"{iteration_time*1000:.6f}\n")  # Computing time of a global iteration in ms

        # Combining the local updates of the fire
        global_update = comm.allreduce(local_update, op=MPI.LOR)

        if rank > 0:
            # Extracting the maps without the ghost cells
            local_fire = fire_model.fire_map[1:fire_model.local_height+1, :].copy()
            local_vege = fire_model.vegetation_map[1:fire_model.local_height+1, :].copy()

            # Sending each local maps to the process 0
            comm.send(local_fire, dest=0, tag=10)
            comm.send(local_vege, dest=0, tag=11)
        elif rank == 0:
            if global_update: # If the fire is still burning
                
                # Updating the maps
                full_fire_map[:] = 0
                full_vege_map[:] = 0
                for computing_proc in range(1, nbp):
                    # Getting the division of each computing process
                    (row_start, row_end) = subdomains[computing_proc - 1]

                    # Receiving the local maps of each computing process
                    sub_fire = comm.recv(source=computing_proc, tag=10)
                    sub_vege = comm.recv(source=computing_proc, tag=11)

                    # Adding the local maps to the global one
                    full_fire_map[row_start:row_end, :] = sub_fire
                    full_vege_map[row_start:row_end, :] = sub_vege

                # Display update
                display_update_time_start = time.time()
                fire_display.update(full_fire_map, full_vege_map)

                # Fire display update time
                display_update_time = time.time() - display_update_time_start
                render_file.write(f"{display_update_time:.3f}\n") # Rendering time in ms

                if t % 10 == 0: # Every 10 time steps we print the computing time of the last update
                    print(f"\tTimestep {t}:", flush=True)
                    print(f"\t\tIteration time: {iteration_time*1000:.3f}ms", flush=True)
                    print(f"\t\tRendering time: {display_update_time*1000:.3f}ms\n", flush=True)

                # Stop the simulation if the user closed the display window
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        must_continue = False
                        pg.quit()
        
        t += 1
        comm.Barrier()

    if rank == 0:
        simulation_time = time.time() - simulation_time_start
        print(f"\tTotal simulation time : {simulation_time}s", flush=True)

        footer = 20*"=" + " End of simulation " + 20*"="
        print(footer + "\n", flush=True)
            
        render_file.close()
        compute_file.close()
    elif rank > 0:
        compute_file.close()

if __name__ == "__main__":
    # Initializing MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    run_simulation(sys.argv[1:], 250)
