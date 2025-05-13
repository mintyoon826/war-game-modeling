import numpy as np
import os
import yaml
from models.terrain import TerrainSystem
from models.combat import CombatSystem, CombatUnit
from models.probabilities import ProbabilitySystem
from models.events import EventQueue, SimulationEvent
from models.logging import SimulationLogger, Event, StateSnapshot
from models.visualization import SimulationVisualizer
from models.unit import Unit, UnitType
import time
from matplotlib.image import imread
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_units(cfg):
    units = []
    initial_positions = cfg.get('initial_positions', {})

    # 팀별 유닛 사양: (cfg 키, 기본 위치)
    team_specs = {
        'RED': {
            'DRONE': ('num_drone_red',    (100, 100)),
            'TANK': ('num_tank_red',      (120, 120)),
            'ANTI_TANK': ('num_at_red',    (140, 140)),
            'INFANTRY': ('num_infantry_red', (160, 160)),
            'COMMAND_POST': ('num_cp_red', (180, 180)),
            'ARTILLERY': ('num_artillery_red', (200, 200)),
        },
        'BLUE': {
            'DRONE': ('num_drone_blue',    (300, 300)),
            'TANK': ('num_tank_blue',      (320, 320)),
            'ANTI_TANK': ('num_at_blue',    (340, 340)),
            'INFANTRY': ('num_infantry_blue', (360, 360)),
            'COMMAND_POST': ('num_cp_blue', (380, 380)),
            'ARTILLERY': ('num_artillery_blue', (400, 400)),
        },
    }

    # 카운터 초기화
    unit_counters = {
        team: {unit_type: 1 for unit_type in specs}
        for team, specs in team_specs.items()
    }

    def get_positions(team, unit_type, count, default_pos):
        pos_list = initial_positions.get(team, {}).get(unit_type)
        if pos_list and len(pos_list) >= count:
            return [tuple(p) for p in pos_list[:count]]
        return [default_pos] * count

    def add_units(team, unit_type, cfg_key, default_pos):
        # 드론 옵션 체크
        if unit_type == 'DRONE' and not cfg.get('with_drone', True):
            return
        
        count = cfg.get(cfg_key, 0)
        positions = get_positions(team, unit_type, count, default_pos)
        
        for i, pos in enumerate(positions):
            unit = Unit(getattr(UnitType, unit_type), team, pos)
            idx = unit_counters[team][unit_type]
            unit.id = f"{team}_{unit_type}_{idx}"
            units.append(CombatUnit(unit))
            unit_counters[team][unit_type] += 1

    # 각 팀과 유닛 타입에 대해 유닛 생성
    for team, specs in team_specs.items():
        for unit_type, (cfg_key, default_pos) in specs.items():
            add_units(team, unit_type, cfg_key, default_pos)

    return units

def run_simulation(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    terrain_system = TerrainSystem()
    probability_system = ProbabilitySystem()
    combat_system = CombatSystem(probability_system)
    if 'distance_rescale' in cfg:
        combat_system.set_distance_rescale(cfg['distance_rescale'])
    event_queue = EventQueue()
    logger = SimulationLogger(log_file=os.path.join(cfg['output_dir'], "simulation_log.json"))
    logger2 = open(os.path.join(cfg['output_dir'], "log.txt"), "w") # 전투 기록만 표시하는 로거
    visualizer = SimulationVisualizer(config=cfg)
    units = create_test_units(cfg)
    current_time = 0.0
    time_step = 1.0
    max_time = cfg['max_time']
    print(f"Starting simulation... (with_drone={cfg.get('with_drone', True)}, max_time={max_time}, output_dir={cfg['output_dir']})")
    
    # Get background image size for bounds
    bg_img = imread(os.path.join("results", "background.png"))
    img_height, img_width = bg_img.shape[0], bg_img.shape[1]
    
    # 시뮬레이션 루프
    while current_time < max_time:
        # 현재 상태 스냅샷 저장
        state_snapshot = StateSnapshot(
            timestamp=current_time,
            units=[{
                'id': unit.unit.id,
                'type': unit.unit.unit_type.name,
                'team': unit.unit.team,
                'position': unit.unit.position,
                'status': unit.state.value,
                'action': unit.action.value if hasattr(unit.action, 'value') else unit.action,
                'health': unit.unit.health,
                'target_list': [t.id for t in getattr(unit, 'target_list', [])],
                'eligible_target_list': [t.id for t in getattr(unit, 'eligible_target_list', [])],
            } for unit in units],
            terrain_state={'timestamp': current_time},
            combat_state={'timestamp': current_time}
        )
        logger.log_state(state_snapshot)
        
        # === 탐지 로직 적용 ===
        for unit in units:
            if not unit.unit.is_alive():
                continue
            combat_system.detect(unit, units, terrain_system)
        
        # 전투 처리
        attack_results = []
        for unit in units:
            if not unit.unit.is_alive():
                continue
                
            # 탐지된 적 유닛만 대상으로 전투
            enemies = [u for u in units if u.unit in unit.target_list and u.unit.team != unit.unit.team and u.unit.is_alive()]
            if not enemies:
                continue
                
            # 가장 가까운 적 찾기
            closest_enemy = min(enemies, key=lambda e: combat_system._calculate_distance(unit.unit.position, e.unit.position))
            
            # 사거리와 시야 확인
            if combat_system.is_in_range(unit, closest_enemy) and terrain_system.check_line_of_sight(unit.unit.position, closest_enemy.unit.position):
                # 명중 확률 계산 및 공격
                hit_prob = combat_system.calculate_hit_probability(unit, closest_enemy)
                unit.action = unit.action.FIRE if hasattr(unit.action, 'FIRE') else 'FIRE'
                
                if np.random.random() < hit_prob:
                    damage = combat_system.process_damage(unit, closest_enemy)
                    attack_results.append((closest_enemy, damage))
                    
                    # 전투 이벤트 로깅
                    combat_event = Event(
                        timestamp=current_time,
                        event_type="COMBAT",
                        actor_id=unit.unit.id,
                        action="FIRE",
                        target_id=closest_enemy.unit.id,
                        details={
                            'hit': True,
                            'damage': damage,
                            'hit_probability': hit_prob
                        }
                    )
                    logger.log_event(combat_event)
                    print(f"Time {current_time:.1f}: {unit.unit.id} HIT {closest_enemy.unit.id} for {damage} damage (p={hit_prob:.2f})")
                    logger2.write(f"Time {current_time:.1f}: {unit.unit.id} HIT {closest_enemy.unit.id} for {damage} damage (p={hit_prob:.2f})\n")
                else:
                    # 빗나감 이벤트 로깅
                    combat_event = Event(
                        timestamp=current_time,
                        event_type="COMBAT",
                        actor_id=unit.unit.id,
                        action="FIRE",
                        target_id=closest_enemy.unit.id,
                        details={
                            'hit': False,
                            'damage': 0,
                            'hit_probability': hit_prob
                        }
                    )
                    logger.log_event(combat_event)
                    print(f"Time {current_time:.1f}: {unit.unit.id} MISS {closest_enemy.unit.id} (p={hit_prob:.2f})")
                    logger2.write(f"Time {current_time:.1f}: {unit.unit.id} MISS {closest_enemy.unit.id} (p={hit_prob:.2f})\n")

        # 공격 결과 적용
        for target, damage in attack_results:
            target.unit.take_damage(damage)
            if not target.unit.is_alive():
                target.state = target.state.K_KILL if hasattr(target.state, 'K_KILL') else 'K_KILL'
        
        # 유닛 상태 업데이트
        for unit in units:
            if unit.action == (unit.action.FIRE if hasattr(unit.action, 'FIRE') else 'FIRE'):
                unit.action = unit.action.STOP if hasattr(unit.action, 'STOP') else 'STOP'
        
        # 이동 처리
        for unit in units:
            if not unit.unit.is_alive() or unit.unit.unit_type == UnitType.COMMAND_POST:
                continue
                
            # 랜덤 이동
            new_x = unit.unit.position[0] + np.random.randint(-10, 11)
            new_y = unit.unit.position[1] + np.random.randint(-10, 11)
            
            # 경계 체크 및 제한
            new_x = max(0, min(new_x, terrain_system.dem_width - 1))
            new_y = max(0, min(new_y, terrain_system.dem_height - 1))
            
            unit.unit.position = (new_x, new_y)
            unit.action = unit.action.MOVE if hasattr(unit.action, 'MOVE') else 'MOVE'
            
            # 이동 이벤트 로깅
            movement_event = Event(
                timestamp=current_time,
                event_type="MOVEMENT",
                actor_id=unit.unit.id,
                action="MOVE",
                details={
                    'old_position': unit.unit.position,
                    'new_position': (new_x, new_y)
                }
            )
            logger.log_event(movement_event)
        
        # 시간 업데이트
        current_time += time_step
        
        # 승리 조건 확인
        red_units = [u for u in units if u.unit.team == 'RED' and u.unit.is_alive()]
        blue_units = [u for u in units if u.unit.team == 'BLUE' and u.unit.is_alive()]
        
        if not red_units:
            print(f"\nTime {current_time:.1f}: Blue Team Wins!")
            break
        if not blue_units:
            print(f"\nTime {current_time:.1f}: Red Team Wins!")
            break
    
    # 시뮬레이션 종료
    logger.save_logs()
    logger2.close()
    print("Simulation completed. Generating visualizations...")
    
    # 최종 유닛 상태 저장
    final_units = [{
        'id': unit.unit.id,
        'type': unit.unit.unit_type.name,
        'team': unit.unit.team,
        'position': unit.unit.position,
        'status': unit.state.value if hasattr(unit.state, 'value') else unit.state,
        'health': unit.unit.health
    } for unit in units]
    
    # 시각화 생성
    print("visualizer.img_height, visualizer.img_width: ", visualizer.img_height, visualizer.img_width)
    visualizer.create_map_visualization(final_units, current_time, output_file=os.path.join(cfg['output_dir'], "final_state.png"))
    
    # 배경 이미지 저장
    bg_path = os.path.join("results", "background.png")
    if not os.path.exists(bg_path):
        visualizer.save_background_image(bg_path)

    visualizer.create_animation(
        logger.state_snapshots,
        output_file=os.path.join(cfg['output_dir'], "simulation.mp4"),
        events=[e.to_dict() for e in logger.events],
        fps=cfg['fps']
    )
    visualizer.plot_metrics(logger.log_file)

if __name__ == "__main__":
    config = load_config('config.yaml')
    run_simulation(config) 