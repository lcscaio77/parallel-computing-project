# External library imports
import pygame as pg
import numpy as np

class DisplayFire:
    """
    Handles the visualization of fire spread and vegetation density.
    
    The red color component represents fire intensity (0-255), while the green component 
    represents vegetation density (0-255). Each discretized cell corresponds to a pixel 
    in the display window.
    """
    def __init__(self, grid_size: int):
        """
        Initializes the display window for fire and vegetation visualization.
        
        Parameters:
        grid_size (int): Number of cells per direction in the simulation grid.
        """
        self.width = grid_size
        self.height = grid_size
        self.screen = pg.display.set_mode((self.width, self.height))
        self.pixels = None

    def update(self, fire_map: np.ndarray, vegetation_map: np.ndarray):
        """
        Updates the visualization based on the current fire and vegetation maps.
        
        Parameters:
        fire_map (np.ndarray): 2D array representing fire intensity (0-255) at each cell.
        vegetation_map (np.ndarray): 2D array representing vegetation density (0-255) at each cell.
        """
        if self.pixels is None:
            self.pixels = pg.surfarray.pixels2d(self.screen)
        
        # Compute the color map: Red for fire intensity, Green for vegetation density
        color_map = (65536 * fire_map.astype(np.int32)) + (256 * vegetation_map.astype(np.int32))
        
        self.pixels[:, :] = color_map[:, :]
        pg.display.update()
