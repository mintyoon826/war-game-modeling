import numpy as np
import os
import yaml
from models.terrain import TerrainSystem
from models.combat import CombatSystem, CombatUnit, UnitState, Action
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
    command_system = CommandSystem()
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
        # ── 예약된 MOVE/FIRE 이벤트 처리 ────────────────────────────
        while event_queue.peek_next_time() is not None and event_queue.peek_next_time() <= current_time:
            evt = event_queue.get_next_event()
            if evt.action == "Maneuver":
                new_pos = evt.details["new_position"]
                evt.actor.unit.move_to(new_pos)
                evt.actor.action = Action.MANEUVER
                logger.log_event(Event(
                    timestamp=current_time,
                    event_type="MOVEMENT",
                    actor_id=evt.actor.unit.id,
                    action="MOVE",
                    details={"new_position": new_pos}
                ))

            elif evt.action == "Fire":
                # 1) 화력 실행
                for tgt in evt.details["targets"]:
                    combat_system.fire(evt.actor, tgt)
                evt.actor.action = Action.FIRE
                # 전투 로그
                for tgt in evt.details["targets"]:
                    logger.log_event(Event(
                        timestamp=current_time,
                        event_type="COMBAT",
                        actor_id=evt.actor.unit.id,
                        action="FIRE",
                        target_id=tgt.unit.id,
                        details={
                            "hit": tgt.state in (UnitState.M_KILL, UnitState.F_KILL, UnitState.K_KILL),
                            "resulting_state": tgt.state.value
                        }
                    ))
                # 2) 교전 후 상태에 따른 분류: 기동 정지 혹은 경로 탐색
                st = evt.actor.state
                if st in [UnitState.ALIVE, UnitState.F_KILL]:
                    # 살아 있거나 화력만 파괴된 경우: 기존 목표로 기동 재예약
                    combat_system.maneuver(
                        unit=evt.actor,
                        goal=evt.actor.current_goal,
                        event_queue=event_queue,
                        current_time=current_time,
                        terrain=terrain_system,
                        all_units=units
                    )
                else:
                    # 기동 불능(M_KILL) 또는 완파(K_KILL): 더 이상 이동하지 않도록 목표 초기화
                    evt.actor.current_goal = None


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

        # 액션 리셋
        for u in units:
            if u.action in (Action.FIRE, Action.MANEUVER):
                u.action = Action.STOP
        
        # 페이즈 전환 평가 -> phase 전환 조건을 만족했는지 확인
        # 검토 필요
        if command_system.evaluate_dc("RED"):
            command_system.transition_phase("RED")
        if command_system.evaluate_dc("BLUE"):
            command_system.transition_phase("BLUE")


        # === 탐지 로직 적용 ===
        # 탐지 및 화력 예약 -> 코드 수정
        # 정지 상태(ES)에서 적을 감지하면 곧바로 Fire 이벤트(FEL)에 예약 -> 맞는지?
        for unit in units:
            if not unit.unit.is_alive():
                continue
            combat_system.detect(unit, units, terrain_system)
            combat_system.available_target(unit)
            combat_system.finding_target(unit, event_queue, current_time)
        
        
        
        # ── 기동 명령 예약 ───────────────────────────────────
        for unit in units:
            if not unit.unit.is_alive() or unit.unit.unit_type == UnitType.COMMAND_POST:
                continue

            # 페이즈별 목표 가져오기
            goal = (command_system.red_command if unit.unit.team == "RED"
                    else command_system.blue_command).maneuver_objective

            unit.current_goal = goal
            combat_system.maneuver(
                unit=unit,
                goal=goal,
                event_queue=event_queue,
                current_time=current_time,
                terrain=terrain_system,
                all_units=units
            )
        
        
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