from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict
from .unit import UnitType

class Phase(Enum):
    FIRE_SUPPRESSION = 1  # 적 화력 박멸
    ARMOR_ENGAGEMENT = 2  # 기갑 교전
    CLOSE_COMBAT = 3      # 근접전투

@dataclass
class Command:
# To do list : 
# 1. Maneuver objective : For red team If none, the maneuver objective is the current position of each unit.
# 2. tir : when dimension is not 1 (2개인 경우 까다로움)
    phase: Phase
    team: str  # "RED" or "BLUE"
    tir: Tuple[float, float]  # Drone reconnaissance area
    fire_priority: List[int]  # Fire support priority
    maneuver_objective: Tuple[float, float]  # Phase-specific maneuver objective
    
    @classmethod
    def create_phase_1_command(cls, team: str):
        if team == "RED":
            return cls(
                phase=Phase.FIRE_SUPPRESSION,
                team=team,
                tir=(600, 200),  # Red team's reconnaissance area
                fire_priority=[1, 2, 3, 4, 5],  # Artillery > Command > Armor > Anti-tank > Infantry
                maneuver_objective=(450, 280)  # Red team's objective
            )
        else:  # BLUE team
            return cls(
                phase=Phase.FIRE_SUPPRESSION,
                team=team,
                tir=(50, 350),  # Blue team's reconnaissance area
                fire_priority=[1, 2, 3, 4, 5],  # Artillery > Command > Armor > Anti-tank > Infantry
                maneuver_objective=(590, 250)  # Blue team's objective
            )
    
    @classmethod
    def create_phase_2_command(cls, team: str):
        if team == "RED":
            return cls(
                phase=Phase.ARMOR_ENGAGEMENT,
                team=team,
                tir=(550, 200),  # Red team's reconnaissance area
                fire_priority=[3, 2, 1, 4, 5],  # Armor > Command > Artillery > Anti-tank > Infantry
                maneuver_objective=(450, 280)  # Red team's objective
            )
        else:  # BLUE team
            return cls(
                phase=Phase.ARMOR_ENGAGEMENT,
                team=team,
                tir=(350, 300),  # Blue team's reconnaissance area
                fire_priority=[3, 2, 1, 4, 5],  # Armor > Command > Artillery > Anti-tank > Infantry
                maneuver_objective=(450, 250)  # Blue team's objective
            )
    
    @classmethod
    def create_phase_3_command(cls, team: str):
        if team == "RED":
            return cls(
                phase=Phase.CLOSE_COMBAT,
                team=team,
                tir=(450, 250),  # Red team's reconnaissance area
                fire_priority=[3, 4, 2, 5, 1],  # Armor > Anti-tank > Command > Infantry > Artillery
                maneuver_objective=(450, 280)  # Red team's objective
            )
        else:  # BLUE team
            return cls(
                phase=Phase.CLOSE_COMBAT,
                team=team,
                tir=(350, 300),  # Blue team's reconnaissance area
                fire_priority=[3, 4, 2, 5, 1],  # Armor > Anti-tank > Command > Infantry > Artillery
                maneuver_objective=(350, 300)  # Blue team's objective
            )

class CommandSystem:
# To do list : 
# 1. 검토 필요 ()
    def __init__(self):
        self.red_command: Command = None
        self.blue_command: Command = None
        self.initialize_commands()
    
    def initialize_commands(self):
        self.red_command = Command.create_phase_1_command("RED")
        self.blue_command = Command.create_phase_1_command("BLUE")
    
    def evaluate_dc(self, team: str) -> bool:
        """Evaluate Decision Criteria for phase transition"""
        command = self.red_command if team == "RED" else self.blue_command
        
        if command.phase == Phase.FIRE_SUPPRESSION:
            # DC1: Check if 4 or more enemy artillery units are F-kill or K-kill
            return self._check_dc1(team)
        elif command.phase == Phase.ARMOR_ENGAGEMENT:
            # DC2: Check if 2 or more enemy tanks are F-kill or K-kill
            return self._check_dc2(team)
        return False
    
    def _check_dc1(self, team: str) -> bool:
        """Check Decision Criteria 1: Enemy artillery destruction"""
        # TODO: Implement actual check
        return False
    
    def _check_dc2(self, team: str) -> bool:
        """Check Decision Criteria 2: Enemy tank destruction"""
        # TODO: Implement actual check
        return False
    
    def transition_phase(self, team: str):
        """Transition to next phase"""
        command = self.red_command if team == "RED" else self.blue_command
        
        if command.phase == Phase.FIRE_SUPPRESSION:
            command = Command.create_phase_2_command(team)
        elif command.phase == Phase.ARMOR_ENGAGEMENT:
            command = Command.create_phase_3_command(team)
        
        if team == "RED":
            self.red_command = command
        else:
            self.blue_command = command 