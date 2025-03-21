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
    if value <= 0: 
        return 0.0
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

        # --- AJOUT : mémoriser le sous-domaine ---
        self.row_start = row_start
        self.row_end   = row_end
        self.local_height = row_end - row_start

        # --- On crée des tableaux LOCAUX, 
        #     mais avec 2 lignes fantômes : shape = (local_height + 2, grid_size).
        #     Indice 0 = fantôme du haut, 
        #     Indice [1 .. local_height] = vraies lignes,
        #     Indice local_height+1 = fantôme du bas.
        # ---
        shape_local = (self.local_height + 2, grid_size)
        self.vegetation_map = np.full(shape_local, 255, dtype=np.uint8)
        self.fire_map       = np.zeros(shape_local,     dtype=np.uint8)

        # --- On initialise le front du feu localement ---
        self.fire_front = {}

        # --- On place le feu de départ S'IL se trouve dans nos lignes locales ---
        start_r, start_c = fire_start_position
        if row_start <= start_r < row_end:
            # conversion : la ligne start_r globale devient (start_r - row_start + 1) en local
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
        # local_r (1..local_height) => global_r = row_start + (local_r -1)
        global_r = self.row_start + (local_r - 1)
        # on peut construire un index style global_r * grid_size + c
        return global_r * self.grid_size + c

    def _neighbors(self, r, c):
        """
        Retourne la liste des voisins directs, 
        avec le facteur vent correspondant (wind_factor).
        r,c sont en coordonnées LOCALES (dans la zone [1..local_height, 0..grid_size-1]).
        """
        return [
            ((r - 1, c), self.alpha_north_south),  # Up    (local row-1)
            ((r + 1, c), self.alpha_south_north),  # Down
            ((r, c - 1), self.alpha_west_east),    # Left
            ((r, c + 1), self.alpha_east_west)     # Right
        ]

    def update_fire(self) -> bool:
        """
        Updates the fire and vegetation maps by computing fire spread.
        
        Returns:
        bool: True if the fire is still burning, False otherwise.
        """
        next_fire_front = copy.deepcopy(self.fire_front)

        # 1) Construire une "liste étendue" des cellules en feu :
        #    - Les cellules actives qui sont en feu (self.fire_front.items())
        #    - Les ghost cells qui seraient en feu (=255) reçues du voisin
        #      (mais qui ne sont pas dans self.fire_front puisqu'elles
        #       n'appartiennent pas à ce sous-domaine)
        ghost_front = []
        # Ajouter la ligne fantôme du haut
        for c in range(self.grid_size):
            if self.fire_map[0, c] == 255:
                ghost_front.append((0, c))
        # Ajouter la ligne fantôme du bas
        for c in range(self.grid_size):
            if self.fire_map[self.local_height+1, c] == 255:
                ghost_front.append((self.local_height+1, c))

        # Combiner : cellules actives en feu (avec leur intensité)
        # + ghost cells en feu (intensité=255)
        combined_front = list(self.fire_front.items()) + [ (g, 255) for g in ghost_front ]

        # 2) Propager le feu depuis chacune de ces cellules
        for (r, c), fire_intensity in combined_front:
            power = log_factor(fire_intensity)
            # Parcours de ses voisins
            for (nr, nc), wind_factor in self._neighbors(r, c):
                # => Si c'est un ghost cell, on n'y écrit pas
                #    on ne met à jour *que* la zone active
                if 1 <= nr <= self.local_height and 0 <= nc < self.grid_size:
                    random_value = pseudo_random(self.get_global_index(r, c) * 13427 + self.time_step,
                                                self.time_step)
                    green_power = self.vegetation_map[nr, nc]
                    correction = power * log_factor(green_power)

                    if random_value < wind_factor * self.propagation_prob * correction:
                        self.fire_map[nr, nc] = 255
                        next_fire_front[(nr, nc)] = 255

        # 3) Gérer l’extinction / réduction de l’intensité, 
        #    uniquement sur les cellules actives qui sont en feu
        #    (on exclut les ghost cells ici)
        for (r, c), fire_intensity in list(next_fire_front.items()):
            # r,c ∈ [1..local_height, 0..grid_size-1]
            if fire_intensity == 255:
                random_value = pseudo_random(self.get_global_index(r, c) * 52513 + self.time_step,
                                            self.time_step)
                if random_value < self.extinction_prob:
                    self.fire_map[r, c] >>= 1  # /2
                    next_fire_front[(r, c)] >>= 1
            else:
                self.fire_map[r, c] >>= 1
                next_fire_front[(r, c)] >>= 1
                if next_fire_front[(r, c)] == 0:
                    next_fire_front.pop((r, c))

        # 4) Réduire la végétation dans les cellules actives en feu
        for (r, c) in next_fire_front:
            if self.vegetation_map[r, c] > 0:
                self.vegetation_map[r, c] -= 1

        # Mise à jour du front + temps
        self.fire_front = next_fire_front
        self.time_step += 1

        return len(self.fire_front) > 0


    # --- AJOUT : fonction d'échange des lignes fantômes ---
    def exchange_ghost_cells(self, comm, rank, size):
        """
        Échange la première et la dernière ligne *active* 
        (indices local 1 et local_height) avec les rangs voisins.
        
        - rank s’étend de 1..(size-1).
        - rank=1 n’a pas de voisin du dessus,
        - rank=(size-1) n’a pas de voisin du dessous.
        """

        top_rank = rank - 1
        bottom_rank = rank + 1

        # 1) Envoi/réception avec le voisin du "dessus" (top_rank)
        if top_rank >= 1:
            # on envoie la première ligne active (index=1) vers top_rank
            top_active_fire = self.fire_map[1, :].copy()
            top_active_vege = self.vegetation_map[1, :].copy()
            comm.send(top_active_fire, dest=top_rank, tag=100)
            comm.send(top_active_vege, dest=top_rank, tag=101)
            # print(f"Proc {rank} envoie {top_active_fire} au proc {top_rank}.", flush=True)
            
            # on reçoit la ligne fantôme du dessus (index=0) de top_rank
            ghost_fire = comm.recv(source=top_rank, tag=200)
            ghost_vege = comm.recv(source=top_rank, tag=201)
            # print(f"Proc {rank} reçoit {ghost_fire} du proc {top_rank}.", flush=True)
            self.fire_map[0, :] = ghost_fire
            self.vegetation_map[0, :] = ghost_vege
        else:
            # rank=1 => pas de voisin au-dessus
            self.fire_map[0, :] = 0
            self.vegetation_map[0, :] = 0

        # 2) Envoi/réception avec le voisin du "dessous" (bottom_rank)
        if bottom_rank <= (size - 1):
            # on envoie la dernière ligne active (index=local_height) vers bottom_rank
            bottom_active_fire = self.fire_map[self.local_height, :].copy()
            bottom_active_vege = self.vegetation_map[self.local_height, :].copy()
            comm.send(bottom_active_fire, dest=bottom_rank, tag=200)
            comm.send(bottom_active_vege, dest=bottom_rank, tag=201)
            # print(f"Proc {rank} envoie {bottom_active_fire} au proc {bottom_rank}.", flush=True)
            
            # on reçoit la ligne fantôme du dessous (index=local_height+1)
            ghost_fire = comm.recv(source=bottom_rank, tag=100)
            ghost_vege = comm.recv(source=bottom_rank, tag=101)
            # print(f"Proc {rank} reçoit {ghost_fire} du proc {bottom_rank}.", flush=True)
            self.fire_map[self.local_height+1, :] = ghost_fire
            self.vegetation_map[self.local_height+1, :] = ghost_vege
        else:
            # rank=(size-1) => pas de voisin en-dessous
            self.fire_map[self.local_height+1, :] = 0
            self.vegetation_map[self.local_height+1, :] = 0
