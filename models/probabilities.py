import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple

class TargetState(Enum):
    ES = "ES(노출/정지)"  # Exposed/Stationary
    EM = "EM(노출/기동)"  # Exposed/Moving
    DS = "DS(차폐/정지)"  # Defilade/Stationary
    DM = "DM(차폐/기동)"  # Defilade/Moving

class DamageType(Enum):
    MINOR = "Minor (경상)"
    SERIOUS = "Serious (중상)"
    CRITICAL = "Critical (치명상)"
    FATAL = "Fetal (사망)"

class TankDamageType(Enum):
    MOBILITY = "기동력 파괴확률"
    FIREPOWER = "화력 파괴확률"
    TURRET = "솟 파괴확률"
    COMPLETE = "완파 확률"

class ProbabilitySystem:
    def __init__(self):
        # Load probability data
        self.rifle_hit_prob = pd.read_csv('database/rifle_at_commander_hit.csv')
        self.rifle_damage_prob = pd.read_csv('database/rifle_at_commander_kh.csv')
        self.tank_hit_prob = pd.read_csv('database/tank_artillery_hit.csv')
        self.tank_damage_prob = pd.read_csv('database/tank_artillery_kh.csv')
    
    def get_hit_probability(self, weapon_type: str, distance: float, target_state: TargetState) -> float:
        """Get hit probability based on weapon type, distance, and target state"""
        if weapon_type == "rifle":
            df = self.rifle_hit_prob
        else:  # tank
            df = self.tank_hit_prob
        
        # Find the closest distance in the table
        distances = df['Distance (m)'].values
        closest_idx = np.abs(distances - distance).argmin()
        
        # Get probability for the target state
        return df.iloc[closest_idx][target_state.value]
    
    def get_rifle_damage_probability(self, distance: float, target_state: TargetState) -> Dict[DamageType, float]:
        """Get rifle damage probabilities for different damage types"""
        # Find the closest distance in the table
        distances = self.rifle_damage_prob['거리(m)'].unique()
        closest_distance = distances[np.abs(distances - distance).argmin()]
        
        # Filter for the specific distance and target state
        mask = (self.rifle_damage_prob['거리(m)'] == closest_distance) & \
               (self.rifle_damage_prob['표적 상태'] == target_state.value)
        
        row = self.rifle_damage_prob[mask].iloc[0]
        
        return {
            DamageType.MINOR: row[DamageType.MINOR.value],
            DamageType.SERIOUS: row[DamageType.SERIOUS.value],
            DamageType.CRITICAL: row[DamageType.CRITICAL.value],
            DamageType.FATAL: row[DamageType.FATAL.value]
        }
    
    def get_tank_damage_probability(self, target_state: TargetState) -> Dict[TankDamageType, float]:
        """Get tank damage probabilities for different damage types"""
        # Filter for the specific target state
        mask = self.tank_damage_prob['표적 상태'] == target_state.value
        
        probabilities = {}
        for damage_type in TankDamageType:
            row = self.tank_damage_prob[mask & (self.tank_damage_prob['손상 유형'] == damage_type.value)].iloc[0]
            probabilities[damage_type] = row['P_{k/h}']
        
        return probabilities
    
    def determine_rifle_damage(self, distance: float, target_state: TargetState) -> DamageType:
        """Determine rifle damage type based on probabilities"""
        probs = self.get_rifle_damage_probability(distance, target_state)
        
        # Create cumulative probabilities
        cum_probs = np.cumsum(list(probs.values()))
        
        # Generate random number
        r = np.random.random()
        
        # Determine damage type
        for i, (damage_type, cum_prob) in enumerate(zip(probs.keys(), cum_probs)):
            if r <= cum_prob:
                return damage_type
        
        return DamageType.MINOR  # Default to minor damage
    
    def determine_tank_damage(self, target_state: TargetState) -> TankDamageType:
        """Determine tank damage type based on probabilities"""
        probs = self.get_tank_damage_probability(target_state)
        
        # Create cumulative probabilities
        cum_probs = np.cumsum(list(probs.values()))
        
        # Generate random number
        r = np.random.random()
        
        # Determine damage type
        for i, (damage_type, cum_prob) in enumerate(zip(probs.keys(), cum_probs)):
            if r <= cum_prob:
                return damage_type
        
        return TankDamageType.MOBILITY  # Default to mobility damage 