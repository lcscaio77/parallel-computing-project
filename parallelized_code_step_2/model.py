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
    def __init__(self, terrain_size: float, grid_size: int, wind_vector, 
                 fire_start_position, max_wind_speed: float = 60.0):
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
        self.wind_vector = np.array(wind_vector)
        self.wind_speed = norm(self.wind_vector)
        self.max_wind_speed = max_wind_speed
        
        # Initialize vegetation and fire maps
        self.vegetation_map = np.full((grid_size, grid_size), 255, dtype=np.uint8)
        self.fire_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Set initial fire location
        self.fire_map[fire_start_position[COLUMN], fire_start_position[ROW]] = np.uint8(255)
        self.fire_front = {(fire_start_position[COLUMN], fire_start_position[ROW]): np.uint8(255)}
        
        # Wind influence coefficients
        ALPHA0 = 0.452790762
        ALPHA1 = 0.000958264437
        ALPHA2 = 0.0000361499382
        
        self.propagation_prob = (
            ALPHA0 + ALPHA1 * min(self.wind_speed, self.max_wind_speed) + 
            ALPHA2 * min(self.wind_speed, self.max_wind_speed) ** 2
        )
        self.extinction_prob = 0.3
        
        # Wind direction influence factors
        self.alpha_east_west = abs(self.wind_vector[COLUMN] / self.max_wind_speed) + 1
        self.alpha_west_east = 1.0 - abs(self.wind_vector[COLUMN] / self.max_wind_speed)
        self.alpha_south_north = abs(self.wind_vector[ROW] / self.max_wind_speed) + 1
        self.alpha_north_south = 1.0 - abs(self.wind_vector[ROW] / self.max_wind_speed)
        
        self.time_step = 0

    def get_global_index(self, coordinates) -> int:
        """
        Returns a unique index for lexicographic coordinates.
        
        Parameters:
        coordinates (tuple): The (column, row) position in the grid.
        
        Returns:
        int: A unique index corresponding to the given coordinates.
        """
        return coordinates[ROW] * self.grid_size + coordinates[COLUMN]

    def update_fire(self) -> bool:
        """
        Updates the fire and vegetation maps by computing fire spread.
        
        Returns:
        bool: True if the fire is still burning, False otherwise.
        """
        next_fire_front = copy.deepcopy(self.fire_front)
        
        for coord, fire_intensity in self.fire_front.items():
            power = log_factor(fire_intensity)
            
            # Check adjacent cells for fire propagation
            neighbors = [
                ((coord[COLUMN], coord[ROW] + 1), self.alpha_south_north),  # Up
                ((coord[COLUMN], coord[ROW] - 1), self.alpha_north_south),  # Down
                ((coord[COLUMN] + 1, coord[ROW]), self.alpha_east_west),    # Right
                ((coord[COLUMN] - 1, coord[ROW]), self.alpha_west_east)     # Left
            ]
            
            for neighbor, wind_factor in neighbors:
                if 0 <= neighbor[ROW] < self.grid_size and 0 <= neighbor[COLUMN] < self.grid_size:
                    random_value = pseudo_random(self.get_global_index(coord) * 13427 + self.time_step, self.time_step)
                    green_power = self.vegetation_map[neighbor[COLUMN], neighbor[ROW]]
                    correction = power * log_factor(green_power)
                    if random_value < wind_factor * self.propagation_prob * correction:
                        self.fire_map[neighbor[COLUMN], neighbor[ROW]] = np.uint8(255)
                        next_fire_front[neighbor] = np.uint8(255)
            
            # Fire intensity reduction (extinguishing process)
            if fire_intensity == 255:
                random_value = pseudo_random(self.get_global_index(coord) * 52513 + self.time_step, self.time_step)
                if random_value < self.extinction_prob:
                    self.fire_map[coord[COLUMN], coord[ROW]] >>= 1
                    next_fire_front[coord] >>= 1
            else:
                self.fire_map[coord[COLUMN], coord[ROW]] >>= 1
                next_fire_front[coord] >>= 1
                if next_fire_front[coord] == 0:
                    next_fire_front.pop(coord)
        
        # Update fire front and reduce vegetation at burning locations
        self.fire_front = next_fire_front
        for coord in self.fire_front:
            if self.vegetation_map[coord[COLUMN], coord[ROW]] > 0:
                self.vegetation_map[coord[COLUMN], coord[ROW]] -= 1
        
        self.time_step += 1
        return len(self.fire_front) > 0
