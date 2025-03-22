# Standard library imports
import copy
from math import log

# External library imports
import numpy as np
from numpy.linalg import norm

# Constants for accessing row and column indices
ROW = 1
COLUMN = 0

def pseudo_random(index: int, time_step: int) -> float:
    """
    Generates a pseudo-random number based on an index and a time step.
    This ensures that the simulation remains deterministic across executions.
    
    Parameters:
        index (int): Unique index for the calculation.
        time_step (int): The current simulation time step.
    
    Returns:
        float: A pseudo-random number in the range [0,1].
    """
    seed = index * (time_step + 1)
    rand_val = (48271 * seed) % 2147483647
    return rand_val / 2147483646.

def log_factor(value: int) -> float:
    """
    Computes a logarithmic scaling factor for probability calculations.
    
    Parameters:
        value (int): The input value to scale.
    
    Returns:
        float: Logarithmic scaling factor.
    """
    return log(1. + value) / log(256)

class FireSpreadModel:
    """
    A fire propagation model in homogeneous vegetation.
    """
    def __init__(self, 
                 terrain_size: float, 
                 grid_size: int, 
                 wind_vector, 
                 fire_start_position, 
                 max_wind_speed: float = 60.0,
                 row_start: int = 0,
                 row_end: int = 0):
        """
        Initializes the fire propagation model.
        
        Parameters:
            terrain_size (float): Length of the square domain in km.
            grid_size (int): Number of discretized grid cells in each direction.
            wind_vector (array-like): Wind direction and magnitude as a 2D vector.
            fire_start_position (tuple): Lexicographic indices where the fire starts.
            max_wind_speed (float): Maximum wind speed (km/h) beyond which fire cannot propagate against the wind.
        """
        if grid_size <= 0:
            raise ValueError("Grid size must be greater than zero.")

        self.terrain_size = terrain_size
        self.grid_size = grid_size
        self.cell_size = terrain_size / grid_size
        self.wind_vector = np.array(wind_vector, dtype=float)
        self.wind_speed = norm(self.wind_vector)
        self.max_wind_speed = max_wind_speed

        # Local map division
        self.row_start = row_start
        self.row_end   = row_end
        self.local_height = row_end - row_start

        # Initialize local vegetation and fire maps
        shape_local = (self.local_height + 2, grid_size)
        self.vegetation_map = np.full(shape_local, 255, dtype=np.uint8)
        self.fire_map       = np.zeros(shape_local,     dtype=np.uint8)

        # Set initial fire location if it is in the local division
        self.fire_front = {}
        start_r, start_c = fire_start_position
        if row_start <= start_r < row_end:
            local_r = start_r - row_start + 1
            self.fire_map[local_r, start_c] = 255
            self.fire_front[(local_r, start_c)] = 255

        # Wind influence coefficients
        ALPHA0 = 0.452790762
        ALPHA1 = 0.000958264437
        ALPHA2 = 0.0000361499382

        self.propagation_prob = (
            ALPHA0 
            + ALPHA1 * min(self.wind_speed, self.max_wind_speed) 
            + ALPHA2 * min(self.wind_speed, self.max_wind_speed) ** 2
        )
        self.extinction_prob = 0.3

        # Wind direction influence factors
        self.alpha_east_west  = abs(self.wind_vector[COLUMN] / self.max_wind_speed) + 1
        self.alpha_west_east  = 1.0 - abs(self.wind_vector[COLUMN] / self.max_wind_speed)
        self.alpha_south_north= abs(self.wind_vector[ROW]    / self.max_wind_speed) + 1
        self.alpha_north_south= 1.0 - abs(self.wind_vector[ROW]    / self.max_wind_speed)

        self.time_step = 0

    def get_global_index(self, local_r, c) -> int:
        """
        Returns a unique index for lexicographic coordinates.
        
        Parameters:
            coordinates (tuple): The (column, row) position in the grid.
        
        Returns:
            int: A unique index corresponding to the given coordinates.
        """
        global_r = self.row_start + (local_r - 1)
        return global_r * self.grid_size + c

    def _neighbors(self, r, c):
        """
        Return the list of direct neighbors of a given cell along with 
        the corresponding wind influence factor.

        Parameters:
            r (int): Row index (local coordinates, between 1 and local_height).
            c (int): Column index (between 0 and grid_size - 1).

        Returns:
            list[tuple[tuple[int, int], float]]: A list of neighbor coordinates and their
            associated wind factor, in the order: Up, Down, Left, Right.
        """
        return [
            ((r - 1, c), self.alpha_north_south),  # Up
            ((r + 1, c), self.alpha_south_north),  # Down
            ((r, c - 1), self.alpha_west_east),    # Left
            ((r, c + 1), self.alpha_east_west)     # Right
        ]

    def update_fire(self) -> bool:
        """
        Update the fire and vegetation maps based on the fire spread rules.

        Returns:
            bool: True if there are still burning cells, False otherwise.
        """
        next_fire_front = copy.deepcopy(self.fire_front)

        # Build an extended list of burning cells:
        # - active fire cells (self.fire_front)
        # - ghost cells received from neighbors (value == 255)
        ghost_front = []
        for c in range(self.grid_size):
            if self.fire_map[0, c] == 255:
                ghost_front.append((0, c))
            if self.fire_map[self.local_height + 1, c] == 255:
                ghost_front.append((self.local_height + 1, c))

        # Combine active and ghost fire cells
        combined_front = list(self.fire_front.items()) + [(g, 255) for g in ghost_front]

        # Fire propagation
        for (r, c), fire_intensity in combined_front:
            power = log_factor(fire_intensity)
            for (nr, nc), wind_factor in self._neighbors(r, c):
                # Only update active area (exclude ghost zones)
                if 1 <= nr <= self.local_height and 0 <= nc < self.grid_size:
                    index = self.get_global_index(r, c)
                    rand = pseudo_random(index * 13427 + self.time_step, self.time_step)
                    green_power = self.vegetation_map[nr, nc]
                    correction = power * log_factor(green_power)

                    if rand < wind_factor * self.propagation_prob * correction:
                        self.fire_map[nr, nc] = 255
                        next_fire_front[(nr, nc)] = 255

        # Fire extinction or intensity reduction (active area only)
        for (r, c), intensity in list(next_fire_front.items()):
            index = self.get_global_index(r, c)
            if intensity == 255:
                rand = pseudo_random(index * 52513 + self.time_step, self.time_step)
                if rand < self.extinction_prob:
                    self.fire_map[r, c] >>= 1
                    next_fire_front[(r, c)] >>= 1
            else:
                self.fire_map[r, c] >>= 1
                next_fire_front[(r, c)] >>= 1
                if next_fire_front[(r, c)] == 0:
                    next_fire_front.pop((r, c))

        # Vegetation depletion in burning cells
        for (r, c) in next_fire_front:
            if self.vegetation_map[r, c] > 0:
                self.vegetation_map[r, c] -= 1

        # Update fire front and time step
        self.fire_front = next_fire_front
        self.time_step += 1

        return len(self.fire_front) > 0

    def exchange_ghost_cells(self, comm, rank, size):
        """
        Exchange the top and bottom active rows with neighboring ranks.

        Parameters:
            comm (MPI.Comm): MPI communicator.
            rank (int): Rank of the current process (1 to size - 1).
            size (int): Total number of processes.
        """
        top_rank = rank - 1
        bottom_rank = rank + 1

        # Exchange with top neighbor (if any)
        if top_rank >= 1:
            # Send active row 1 to top neighbor
            comm.send(self.fire_map[1, :].copy(), dest=top_rank, tag=100)
            comm.send(self.vegetation_map[1, :].copy(), dest=top_rank, tag=101)
            
            # Receive ghost row from top neighbor
            self.fire_map[0, :] = comm.recv(source=top_rank, tag=200)
            self.vegetation_map[0, :] = comm.recv(source=top_rank, tag=201)
        else:
            # No top neighbor
            self.fire_map[0, :] = 0
            self.vegetation_map[0, :] = 0

        # Exchange with bottom neighbor (if any)
        if bottom_rank <= size - 1:
            # Send active bottom row to bottom neighbor
            comm.send(self.fire_map[self.local_height, :].copy(), dest=bottom_rank, tag=200)
            comm.send(self.vegetation_map[self.local_height, :].copy(), dest=bottom_rank, tag=201)

            # Receive ghost row from bottom neighbor
            self.fire_map[self.local_height + 1, :] = comm.recv(source=bottom_rank, tag=100)
            self.vegetation_map[self.local_height + 1, :] = comm.recv(source=bottom_rank, tag=101)
        else:
            # No bottom neighbor
            self.fire_map[self.local_height + 1, :] = 0
            self.vegetation_map[self.local_height + 1, :] = 0
