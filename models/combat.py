from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import numpy as np
from .unit import Unit, UnitType
from .probabilities import ProbabilitySystem, TargetState, DamageType, TankDamageType

class UnitState(Enum):
    ALIVE = "Alive"
    M_KILL = "M-kill"  # Mobility kill
    F_KILL = "F-kill"  # Firepower kill
    K_KILL = "K-kill"  # Complete kill

class Action(Enum):
    MANEUVER = "Maneuver"
    FIRE = "Fire"
    STOP = "Stop"

# 탐지 DB Table (유닛 타입별)
DETECT_DB = {
    UnitType.INFANTRY:    {"detect_range": 1000, "detectability": 0.8,   "mountain_prob": 0.2},
    UnitType.ANTI_TANK:   {"detect_range": 1000, "detectability": 0.8,   "mountain_prob": 0.2},
    UnitType.TANK:        {"detect_range": 3000, "detectability": 2.0,   "mountain_prob": 0.5},
    UnitType.ARTILLERY:   {"detect_range": 2000, "detectability": 1.5,   "mountain_prob": 0.5},
    UnitType.COMMAND_POST:{"detect_range": 1500, "detectability": 1.0,   "mountain_prob": 0.2},
    UnitType.DRONE:       {"detect_range": 5000, "detectability": 0.0001,"mountain_prob": 0.3},
}

@dataclass
class CombatUnit:
    unit: Unit
    state: UnitState = UnitState.ALIVE
    action: Action = Action.STOP
    target_list: Optional[Set[Unit]] = None
    eligible_target_list: Optional[Set[Unit]] = None
    is_moving: bool = False
    is_defilade: bool = False
    # 탐지 관련 필드
    detect_range: float = 1000
    detectability: float = 1.0
    mountain_detect_prob: float = 0.2
    failed_detect_count: int = 0

    def __post_init__(self):
        if self.target_list is None:
            self.target_list = set()
        if self.eligible_target_list is None:
            self.eligible_target_list = set()
        # 유닛 타입에 따라 탐지 DB 값 자동 할당
        db = DETECT_DB.get(self.unit.unit_type)
        if db:
            self.detect_range = db["detect_range"]
            self.detectability = db["detectability"]
            self.mountain_detect_prob = db["mountain_prob"]

    def get_target_state(self) -> TargetState:
        """Get the current target state based on movement and defilade status"""
        if self.is_defilade:
            return TargetState.DS if not self.is_moving else TargetState.DM
        return TargetState.ES if not self.is_moving else TargetState.EM

class CombatSystem:
    """
    전투 시스템의 핵심 로직을 담당하는 클래스.
    - 유닛 탐지, 사격, 피해 처리, 명중 확률 계산 등 전투 관련 기능 제공
    """
    # 무기별 사거리(픽셀 단위, 실제 시뮬레이션에 맞게 조정)
    WEAPON_RANGES = {
        UnitType.INFANTRY: 500,
        UnitType.TANK: 2000,
        UnitType.ANTI_TANK: 1500,
        UnitType.ARTILLERY: 5000,
        UnitType.DRONE: 1000,
        UnitType.COMMAND_POST: 100
    }
    # 무기별 데미지 값
    DAMAGE_VALUES = {
        'rifle': {
            DamageType.MINOR: 10,
            DamageType.SERIOUS: 30,
            DamageType.CRITICAL: 60,
            DamageType.FATAL: 100
        },
        'tank': {
            TankDamageType.MOBILITY: 30,
            TankDamageType.FIREPOWER: 50,
            TankDamageType.TURRET: 70,
            TankDamageType.COMPLETE: 100
        }
    }

    def __init__(self, probability_system: Optional[ProbabilitySystem] = None):
        self.prob_system = probability_system or ProbabilitySystem()
        self.distance_rescale = 1.0

    def set_distance_rescale(self, scale: float):
        """거리 환산 계수 설정 (시뮬레이션 해상도에 맞게)"""
        self.distance_rescale = scale

    def los(self, pos1: Tuple[float, float], pos2: Tuple[float, float], terrain=None) -> bool:
        """Line of Sight(시야) 분석 (지형 반영 가능)

        observer와 상대 사이 거리를 10으로 나눠 각 지점에서 높이 z가
                해당 지점에서의 terrain_height z보다 높으면 탐지 가능
        
        """
        # terrain.check_line_of_sight() 활용 (없으면 True)
        if terrain is not None:
            x1, y1 = int(pos1[0]), int(pos1[1])
            x2, y2 = int(pos2[0]), int(pos2[1])
            return terrain.check_line_of_sight((x1, y1), (x2, y2))
        return True

    def detect(self, observer: CombatUnit, all_units: List[CombatUnit], terrain, z_threshold=None):
        """DB Table/산악지형/누적 탐지 실패 반영 탐지 로직
        observer의 Detect_range * 상대의 Detectability
        """
        observer.target_list.clear()
        if z_threshold is None:
            z_threshold = getattr(terrain, 'mountain_threshold', 50.0)
        for target in all_units:
            if target is observer:
                continue
            # 1. 거리 계산
            distance = self._calculate_distance(observer.unit.position, target.unit.position)
            # 2. 탐지 거리 계산
            detect = observer.detect_range * target.detectability
            if distance > detect:
                continue
            # 3. LOS 체크
            if not self.los(observer.unit.position, target.unit.position, terrain):
                continue
            # 4. 산악지형 확률적 탐지
            x, y = int(target.unit.position[0]), int(target.unit.position[1])
            elevation = terrain.get_elevation(x, y)
            if elevation > z_threshold:
                # 누적 실패 보정
                prob = target.mountain_detect_prob + 0.1 * target.failed_detect_count
                if np.random.random() > prob:
                    target.failed_detect_count += 1
                    continue
                else:
                    target.failed_detect_count = 0
            # 상대 유닛 상태. 화력까지 포함된 후에 보완. 완전 파괴되면 target list에서 아예 제외하고 시작 등
            # 5. 상대팀이면 target_list에 추가
            if observer.unit.team != target.unit.team:
                observer.target_list.add(target.unit)

    def available_target(self, unit: CombatUnit):
        """무기 특성에 따라 공격 가능한 적 유닛 선별"""
        unit.eligible_target_list.clear()
        for target in unit.target_list:
            if self._is_valid_target(unit.unit, target):
                unit.eligible_target_list.add(target)

    def _is_valid_target(self, attacker: Unit, target: Unit) -> bool:
        """공격자의 무기 특성에 따라 유효 타겟인지 판정"""
        if attacker.unit_type == UnitType.ANTI_TANK:
            return target.unit_type in [UnitType.TANK, UnitType.INFANTRY]
        return True

    def finding_target(self, unit: CombatUnit):
        """공격 가능한 타겟 선정 및 사격 이벤트 예약"""
        if unit.state in [UnitState.ALIVE, UnitState.M_KILL] and unit.action == Action.STOP:
            if unit.eligible_target_list:
                unit.action = Action.FIRE
                target = self._select_target(unit)
                if target:
                    self._schedule_fire(unit, target)

    def _select_target(self, unit: CombatUnit) -> Optional[Unit]:
        """유닛 타입/우선순위에 따라 타겟 선정"""
        if not unit.eligible_target_list:
            return None
        if unit.unit.unit_type == UnitType.ARTILLERY:
            return self._select_artillery_target(unit)
        else:
            return self._select_direct_fire_target(unit)

    def _select_artillery_target(self, unit: CombatUnit) -> Unit:
        """곡사화기(포병)용 타겟 선정 (우선순위/아군 피해 고려 가능)"""
        # TODO: 우선순위/아군 피해 고려 구현
        return next(iter(unit.eligible_target_list))

    def _select_direct_fire_target(self, unit: CombatUnit) -> Unit:
        """직사화기(직접 조준) 무기용 타겟 선정 (가장 가까운 적)"""
        return min(unit.eligible_target_list, key=lambda t: self._calculate_distance(unit.unit.position, t.position))

    @staticmethod
    def _calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """두 위치(픽셀 좌표) 간 유클리드 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _schedule_fire(self, unit: CombatUnit, target: Unit):
        """FEL(이벤트 큐)에 사격 이벤트 예약 (확장 가능)"""
        # TODO: 실제 이벤트 큐 구현
        pass

    def fire(self, unit: CombatUnit, target: Unit):
        """사격 이벤트 실행 (직사/곡사 구분)"""
        if target not in unit.eligible_target_list:
            return
        if unit.unit.unit_type == UnitType.ARTILLERY:
            self._artillery_fire(unit, target)
        else:
            self._direct_fire(unit, target)
        unit.action = Action.STOP

    def _artillery_fire(self, unit: CombatUnit, target: Unit):
        """곡사화기 사격 (추후 구현)"""
        pass

    def _direct_fire(self, unit: CombatUnit, target: Unit):
        """직사화기 사격 및 피해 처리"""
        distance = self._calculate_distance(unit.unit.position, target.position)
        weapon_type = "rifle" if unit.unit.unit_type == UnitType.INFANTRY else "tank"
        hit_prob = self.prob_system.get_hit_probability(
            weapon_type,
            distance,
            target.get_target_state()
        )
        if np.random.random() <= hit_prob:
            if weapon_type == "rifle":
                self._process_rifle_damage(unit, target, distance)
            else:
                self._process_tank_damage(unit, target)

    def _process_rifle_damage(self, unit: CombatUnit, target: Unit, distance: float):
        """소총류 피해 처리 (피해 타입별로 상태 변경)"""
        damage_type = self.prob_system.determine_rifle_damage(
            distance,
            target.get_target_state()
        )
        if damage_type == DamageType.FATAL:
            target.state = UnitState.K_KILL
        elif damage_type == DamageType.CRITICAL:
            target.state = UnitState.F_KILL
        elif damage_type == DamageType.SERIOUS:
            target.state = UnitState.M_KILL

    def _process_tank_damage(self, unit: CombatUnit, target: Unit):
        """전차류 피해 처리 (피해 타입별로 상태 변경)"""
        damage_type = self.prob_system.determine_tank_damage(
            target.get_target_state()
        )
        if damage_type == TankDamageType.COMPLETE:
            target.state = UnitState.K_KILL
        elif damage_type == TankDamageType.FIREPOWER:
            target.state = UnitState.F_KILL
        elif damage_type == TankDamageType.MOBILITY:
            target.state = UnitState.M_KILL

    def is_in_range(self, unit: CombatUnit, target: CombatUnit) -> bool:
        """무기별 사거리 내에 있는지 판정"""
        distance = self._calculate_distance(unit.unit.position, target.unit.position)
        return distance <= self.WEAPON_RANGES[unit.unit.unit_type]

    def calculate_hit_probability(self, unit: CombatUnit, target: CombatUnit) -> float:
        """명중 확률 계산 (무기/거리/상태 반영)"""
        distance = self._calculate_distance(unit.unit.position, target.unit.position)
        target_state = target.get_target_state()
        weapon_type = "rifle" if unit.unit.unit_type == UnitType.INFANTRY else "tank"
        return self.prob_system.get_hit_probability(weapon_type, distance, target_state)

    def process_damage(self, unit: CombatUnit, target: CombatUnit) -> float:
        """피해량 계산 및 반환 (실제 체력 감소는 외부에서)"""
        distance = self._calculate_distance(unit.unit.position, target.unit.position)
        target_state = target.get_target_state()
        if unit.unit.unit_type == UnitType.INFANTRY:
            damage_type = self.prob_system.determine_rifle_damage(distance, target_state)
            return self.DAMAGE_VALUES['rifle'][damage_type]
        else:
            damage_type = self.prob_system.determine_tank_damage(target_state)
            return self.DAMAGE_VALUES['tank'][damage_type] 