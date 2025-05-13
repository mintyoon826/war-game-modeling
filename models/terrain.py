import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from PIL import Image

@dataclass
class TerrainEffects:
    movement_speed: float  # Movement speed multiplier
    detection_prob: float  # Detection probability multiplier
    kill_prob: float      # Kill probability multiplier

class TerrainSystem:
    def __init__(self, xyz_file: str = "database/xyz_coordinates.csv"):
        # Load elevation data from xyz_coordinates.csv
        # 이미 행이 y좌표(0~448), 열이 x좌표(0~748)인 형태로 저장되어 있음
        self.dem_data = pd.read_csv(xyz_file, header=0, index_col=0).values
        
        # Set DEM dimensions based on actual data size
        self.dem_height, self.dem_width = self.dem_data.shape  # y: 0-448, x: 0-748
        
        # Load background image dimensions
        bg_img = Image.open("results/background.png")
        self.img_width, self.img_height = bg_img.size
        
        # Terrain thresholds
        self.mountain_threshold = 50.0  # meters
        self.water_threshold = 40.0     # meters
        
        # Terrain effect weights
        self.decay_weight_mount = 0.5   # Mountain effect weight
        self.decay_weight_water = 0.3   # Water effect weight
        
        # Unit type specific effects
        self.unit_terrain_effects = {
            "DRONE": TerrainEffects(1.0, 1.0, 1.0),      # Drones unaffected by terrain
            "TANK": TerrainEffects(0.7, 0.8, 1.0),       # Tanks affected by terrain
            "ANTI_TANK": TerrainEffects(0.8, 0.9, 1.0),  # Anti-tank units
            "INFANTRY": TerrainEffects(0.9, 0.9, 1.0),   # Infantry
            "COMMAND_POST": TerrainEffects(0.8, 0.8, 1.0), # Command post
            "ARTILLERY": TerrainEffects(0.7, 0.8, 1.0)   # Artillery
        }
    

    def get_elevation(self, x: int, y: int) -> float:
        """Get elevation at given coordinates"""
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.dem_width - 1))
        y = max(0, min(y, self.dem_height - 1))
        return self.dem_data[y, x]
    
    def get_terrain_effects(self, x: int, y: int, unit_type: str) -> TerrainEffects:
        """Get terrain effects for a unit at given coordinates"""
        elevation = self.get_elevation(x, y)
        base_effects = self.unit_terrain_effects[unit_type]
        
        # Calculate terrain modifiers
        if elevation > self.mountain_threshold:
            mountain_modifier = (elevation - self.mountain_threshold) * self.decay_weight_mount
            return TerrainEffects(
                movement_speed=base_effects.movement_speed * (1 - mountain_modifier),
                detection_prob=base_effects.detection_prob * (1 - mountain_modifier),
                kill_prob=base_effects.kill_prob * (1 - mountain_modifier)
            )
        elif elevation < self.water_threshold:
            water_modifier = (self.water_threshold - elevation) * self.decay_weight_water
            return TerrainEffects(
                movement_speed=base_effects.movement_speed * (1 - water_modifier),
                detection_prob=base_effects.detection_prob * (1 - water_modifier),
                kill_prob=base_effects.kill_prob * (1 - water_modifier)
            )
        
        return base_effects
    
    def check_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if there is line of sight between two points"""
        x1, y1 = start
        x2, y2 = end
        
        # Bresenham's line algorithm to check points between start and end
        points = self._get_line_points(x1, y1, x2, y2)
        
        # Get elevation at start and end points
        start_elev = self.get_elevation(x1, y1)
        end_elev = self.get_elevation(x2, y2)
        
        # Check if any point along the line is higher than both endpoints
        for x, y in points:
            if x == x1 and y == y1 or x == x2 and y == y2:
                continue
                
            point_elev = self.get_elevation(x, y)
            if point_elev > max(start_elev, end_elev):
                return False
        
        return True
    
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> list:
        """Get all points along a line using Bresenham's algorithm"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            points.append((x, y))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return points 