import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from .logging import StateSnapshot
import rasterio
from rasterio.plot import show as rio_show
from matplotlib.colors import LightSource
import matplotlib.lines as mlines
import contextily as ctx
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm
import matplotlib.image as mpimg
import os
from rasterio.windows import from_bounds
from matplotlib import gridspec
import matplotlib.font_manager as fm
import matplotlib.widgets as mwidgets  # for interactive slider
import plotly.graph_objects as go
import base64
from PIL import Image
import io

# Configure Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows Korean font
# Fallback fonts if Malgun Gothic is not available
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

class SimulationVisualizer:
    def __init__(self, config=None):
        self.bg_path = os.path.join("results", "background.png")
        self.bg_img = mpimg.imread(self.bg_path)
        self.img_height, self.img_width = self.bg_img.shape[0], self.bg_img.shape[1]
        self.fig = plt.figure(figsize=(16, 9))
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        self.ax_map = self.fig.add_subplot(self.gs[0, 0])
        self.ax_panel = self.fig.add_subplot(self.gs[0, 1])
        self.ax_panel.axis('off')
        # set output directory for visualizations
        if config and 'output_dir' in config:
            self.output_dir = config['output_dir']
        else:
            self.output_dir = os.path.dirname(self.bg_path)
            
        # DEM dimensions
        self.dem_width = 750  # x: 0-749
        self.dem_height = 450  # y: 0-449

    def _scale_coordinates(self, x: float, y: float) -> tuple:
        """Scale DEM coordinates to image coordinates"""
        scaled_x = (x / self.dem_width) * self.img_width
        scaled_y = (y / self.dem_height) * self.img_height
        return scaled_x, scaled_y

    def create_map_visualization(self, units: List[Dict], timestamp: float, output_file: str = None):
        plt.figure(figsize=(16, 9))
        plt.imshow(self.bg_img)
        for unit in units:
            x, y = unit['position']
            # Scale coordinates
            x, y = self._scale_coordinates(x, y)
            color = 'red' if unit['team'] == 'RED' else 'blue'
            marker = self._get_marker(unit['type'])

            # 상태별 시각화
            if unit.get('health', 1) <= 0 or unit.get('status', '').upper() in ['K_KILL', 'DESTROYED']:
                facecolor = 'gray'
                alpha = 0.5
            elif unit.get('action', '').upper() == 'FIRE':
                facecolor = 'yellow'
                alpha = 1.0
            else:
                facecolor = color
                alpha = 1.0
            plt.scatter(x, y, c=facecolor, marker=marker, s=120, edgecolors='k', alpha=alpha, label=f"{unit['team']}_{unit['type']}")
        
        # Legend
        handles = [
            mlines.Line2D([], [], color='red', marker='o', markerfacecolor='red', label='RED', markersize=10, linestyle='None'),
            mlines.Line2D([], [], color='blue', marker='o', markerfacecolor='blue', label='BLUE', markersize=10, linestyle='None'),
            mlines.Line2D([], [], color='gray', marker='o', markerfacecolor='gray', label='DESTROYED', markersize=10, linestyle='None', markeredgecolor='k', alpha=0.5),
            mlines.Line2D([], [], color='yellow', marker='o', markerfacecolor='yellow', label='FIRE', markersize=10, linestyle='None', markeredgecolor='k'),
        ]
        plt.legend(handles=handles, loc='upper right', fontsize='small', ncol=2)
        plt.axis('off')
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close()

    def _get_marker(self, unit_type):
        return {
            'DRONE': '*',
            'TANK': 's',
            'ANTI_TANK': 'P',
            'INFANTRY': 'o',
            'COMMAND_POST': 'X',
            'ARTILLERY': '^',
        }.get(unit_type, 'o')

    def create_animation(self, state_snapshots: List[StateSnapshot], output_file: str = 'simulation.mp4', fps: int = 20, n_frames: int = None, events: list = None, unit_type_list=None, team_list=None, total_counts=None, kill_log_duration: float = 3.0):
        if n_frames is None:
            n_frames = len(state_snapshots)
        frame_indices = np.linspace(0, len(state_snapshots) - 1, n_frames).astype(int)
        marker_map = {
            'DRONE': '*',
            'TANK': 's',
            'ANTI_TANK': 'P',
            'INFANTRY': 'o',
            'COMMAND_POST': 'X',
            'ARTILLERY': '^',
        }
        # 한글 매핑
        type_kor = {
            'DRONE': '드론',
            'TANK': '탱크',
            'ANTI_TANK': '대전차',
            'INFANTRY': '보병',
            'COMMAND_POST': '지휘소',
            'ARTILLERY': '곡사포',
        }
        team_kor = {'RED': '레드', 'BLUE': '블루'}
        team_color = {'RED': 'red', 'BLUE': 'blue'}
        status_kor = {'ALIVE': '생존', 'FIRE': '공격', 'DESTROYED': '파괴'}
        bg_img = self.bg_img
        pbar = tqdm(total=n_frames, desc="Rendering animation frames")
        if unit_type_list is None:
            unit_type_list = ['DRONE', 'TANK', 'ANTI_TANK', 'INFANTRY', 'COMMAND_POST', 'ARTILLERY']
        if team_list is None:
            team_list = ['RED', 'BLUE']
        if total_counts is None:
            total_counts = {team: {ut: 0 for ut in unit_type_list} for team in team_list}
            for unit in state_snapshots[0].units:
                total_counts[unit['team']][unit['type']] += 1
        # Prepare kill log with timestamps
        kill_events = []
        if events is not None:
            for e in events:
                if e.get('event_type') == 'COMBAT' and e.get('details', {}).get('hit') and e.get('details', {}).get('damage', 0) > 0:
                    kill_events.append({
                        'timestamp': e['timestamp'],
                        'actor_id': e['actor_id'],
                        'target_id': e.get('target_id', ''),
                        'damage': e['details']['damage']
                    })
        def parse_unit_kor(unit_id):
            # e.g. BLUE_INFANTRY_1 -> 블루_보병_1
            parts = unit_id.split('_')
            if len(parts) >= 3:
                team = parts[0]
                typ = parts[1]
                idx = '_'.join(parts[2:])  # in case index itself contains underscores
                return f"{team_kor.get(team, team)}_{type_kor.get(typ, typ)}_{idx}"
            elif len(parts) == 2:
                team, typ = parts
                return f"{team_kor.get(team, team)}_{type_kor.get(typ, typ)}"
            else:
                return unit_id
        def get_kill_log(snapshot_time):
            # 이벤트 로그 기반 파괴 이벤트 집계
            kill_events = []
            destroyed_targets = set()
            for e in events or []:
                if (
                    e.get('event_type') == 'COMBAT'
                    and e.get('details', {}).get('hit')
                    and e.get('details', {}).get('damage', 0) > 0
                    and e.get('timestamp', 0) <= snapshot_time
                ):
                    target_id = e.get('target_id', '')
                    if target_id and target_id not in destroyed_targets:
                        destroyed_targets.add(target_id)
                        actor = parse_unit_kor(e['actor_id'])
                        target = parse_unit_kor(target_id)
                        # 팀에 따라 색상 설정
                        actor_color = 'red' if 'RED' in e['actor_id'] else 'blue'
                        kill_events.append((f"{actor}이 {target}를 파괴.", actor_color))
            return kill_events[-5:]  # 최근 5개만 표시
        
        def update(frame_idx):
            self.ax_map.clear()
            self.ax_panel.clear()
            self.ax_panel.axis('off')
            snapshot = state_snapshots[frame_indices[frame_idx]]
            self.ax_map.imshow(bg_img)

            for unit in snapshot.units:
                x, y = unit['position']
                # Scale coordinates
                x, y = self._scale_coordinates(x, y)
                unit_type = unit['type']
                team = unit['team']
                marker = marker_map.get(unit_type, 'o')

                # 상태별 시각화
                if unit.get('health', 1) <= 0 or unit.get('status', '').upper() in ['K_KILL', 'DESTROYED']:
                    facecolor = 'gray'
                    edgecolor = team_color[team]
                    alpha = 0.7
                elif unit.get('action', '').upper() == 'FIRE':
                    facecolor = 'yellow'
                    edgecolor = team_color[team]
                    alpha = 1.0
                else:
                    facecolor = team_color[team]
                    edgecolor = 'none'
                    alpha = 1.0
                self.ax_map.scatter(x, y, c=facecolor, marker=marker, s=180, edgecolors=edgecolor, linewidths=2.5, alpha=alpha, zorder=10)
            
            self.ax_map.set_title(f'Time: {snapshot.timestamp:.1f}')
            self.ax_map.axis('off')
            
            # 무기 종류
            weapon_legend = [
                mlines.Line2D([], [], color='black', marker=marker_map[ut], markerfacecolor='black', markeredgecolor='black', markersize=14, label=type_kor[ut], linestyle='None')
                for ut in unit_type_list
            ]
            # 상태(색상)
            status_legend = [
                mlines.Line2D([], [], color='red', marker='o', markerfacecolor='red', markeredgecolor='red', markersize=14, label='생존(Red)', linestyle='None'),
                mlines.Line2D([], [], color='yellow', marker='o', markerfacecolor='yellow', markeredgecolor='red', markersize=14, label='공격(Red)', linestyle='None'),
                mlines.Line2D([], [], color='gray', marker='o', markerfacecolor='gray', markeredgecolor='red', markersize=14, label='파괴(Red)', linestyle='None', alpha=0.7),
                mlines.Line2D([], [], color='blue', marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=14, label='생존(Blue)', linestyle='None'),
                mlines.Line2D([], [], color='yellow', marker='o', markerfacecolor='yellow', markeredgecolor='blue', markersize=14, label='공격(Blue)', linestyle='None'),
                mlines.Line2D([], [], color='gray', marker='o', markerfacecolor='gray', markeredgecolor='blue', markersize=14, label='파괴(Blue)', linestyle='None', alpha=0.7),
            ]
            
            # 무기 종류 범례 (왼쪽 위)
            weapon_legend_artist = self.ax_map.legend(
                handles=weapon_legend, loc='upper left', fontsize=13, title='무기 종류',
                frameon=True, bbox_to_anchor=(0.01, 0.99)
            )
            weapon_legend_artist.get_frame().set_facecolor('white')
            weapon_legend_artist.get_frame().set_alpha(0.5)
            self.ax_map.add_artist(weapon_legend_artist)
            
            # 상태(색상) 범례 (오른쪽 위)
            status_legend_artist = self.ax_map.legend(
                handles=status_legend, loc='upper right', fontsize=13, title='상태(색상)',
                frameon=True, bbox_to_anchor=(0.99, 0.99)
            )
            status_legend_artist.get_frame().set_facecolor('white')
            status_legend_artist.get_frame().set_alpha(0.5)
            
            # 우측 패널(1/4): kill log + 현황판
            kill_log = get_kill_log(snapshot.timestamp)
            y0 = 0.95
            self.ax_panel.text(0.5, y0, 'Kill Log', fontsize=16, fontweight='bold', ha='center', va='top', transform=self.ax_panel.transAxes)
            
            for i, (msg, color) in enumerate(get_kill_log(snapshot.timestamp)):
                self.ax_panel.text(0.5, y0 - 0.07 * (i + 1), msg, fontsize=13, color=color, ha='center', va='top', transform=self.ax_panel.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            
            curr_counts = {team: {ut: 0 for ut in unit_type_list} for team in team_list}
            for unit in snapshot.units:
                if unit.get('health', 1) > 0:
                    curr_counts[unit['team']][unit['type']] += 1
            table_data = []
            for ut in unit_type_list:
                row = [type_kor[ut]]
                for team in team_list:
                    row.append(f"{curr_counts[team][ut]}/{total_counts[team][ut]}")
                table_data.append(row)
            col_labels = ["무기 종류"] + [team_kor[t] for t in team_list]
            table = self.ax_panel.table(cellText=table_data, colLabels=col_labels, loc='lower center', cellLoc='center', bbox=[0.05, 0.05, 0.9, 0.45])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            self.ax_panel.axis('off')
            pbar.update(1)
        
        # Create animation
        anim = FuncAnimation(
            self.fig, update,
            frames=n_frames,
            interval=1000 / fps,
            repeat=False
        )
        anim.save(output_file, writer='ffmpeg', fps=fps)
        pbar.close()

    def plot_metrics(self, log_file: str):
        with open(log_file, 'r') as f:
            log_data = json.load(f)

        # 1) 시간순으로 스냅샷 정렬
        snaps = sorted(log_data['state_snapshots'], key=lambda s: s['timestamp'])
        timestamps = [s['timestamp'] for s in snaps]

        # 2) 전체 Alive 유닛 수
        total_red  = [sum(1 for u in s['units']
                        if u['team']=='RED' and u.get('status')=='Alive')
                    for s in snaps]
        total_blue = [sum(1 for u in s['units']
                        if u['team']=='BLUE' and u.get('status')=='Alive')
                    for s in snaps]

        # 3) 전투 이벤트 타임스탬프
        combat_events = [e['timestamp'] for e in log_data['events']
                        if e['event_type']=='COMBAT']

        # 4) 유형별 Alive 유닛 수 집계
        unit_types = ['ARTILLERY','DRONE','TANK','ANTI_TANK','INFANTRY','COMMAND_POST']
        red_counts  = {t: [] for t in unit_types}
        blue_counts = {t: [] for t in unit_types}
        for s in snaps:
            for t in unit_types:
                red_counts[t].append(
                    sum(1 for u in s['units']
                        if u['team']=='RED' and u.get('type')==t and u.get('status')=='Alive')
                )
                blue_counts[t].append(
                    sum(1 for u in s['units']
                        if u['team']=='BLUE' and u.get('type')==t and u.get('status')=='Alive')
                )

        # 5) 플롯: 4행(전체, 전투, RED 스택, BLUE 스택)
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

        # (0) 전체 유닛 선그래프
        ax0.plot(timestamps, total_red,  'r-', label='Total RED')
        ax0.plot(timestamps, total_blue, 'b-', label='Total BLUE')
        ax0.set_ylabel('Units')
        ax0.set_title('Total Alive Units Over Time')
        ax0.legend()

        # (1) 전투 이벤트 히스토그램
        ax1.hist(combat_events, bins=20, color='gray')
        ax1.set_ylabel('#Events')
        ax1.set_title('Combat Event Distribution')

        # 색상 팔레트(Set2, 6가지)
        palette = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F']

        # (2) RED 팀 스택바
        bottom = np.zeros(len(snaps))
        for idx, t in enumerate(unit_types):
            vals = np.array(red_counts[t])
            ax2.bar(timestamps, vals, bottom=bottom, 
                    width=(timestamps[1]-timestamps[0])*0.8,
                    color=palette[idx], label=t)
            bottom += vals
        ax2.set_ylabel('RED Units')
        ax2.set_title('RED Team Composition Over Time')
        ax2.legend(loc='upper right')

        # (3) BLUE 팀 스택바
        bottom = np.zeros(len(snaps))
        for idx, t in enumerate(unit_types):
            vals = np.array(blue_counts[t])
            ax3.bar(timestamps, vals, bottom=bottom, 
                    width=(timestamps[1]-timestamps[0])*0.8,
                    color=palette[idx], label=t)
            bottom += vals
        ax3.set_ylabel('BLUE Units')
        ax3.set_title('BLUE Team Composition Over Time')
        ax3.legend(loc='upper right')

        ax3.set_xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'simulation_metrics.png'))

    def create_interactive_visualization(self, state_snapshots: List[StateSnapshot], events: list = None, unit_type_list=None, team_list=None, total_counts=None, kill_log_duration: float = 3.0):
        """Interactive visualization with a slider to move through time frames."""
        # Prepare defaults
        n_frames = len(state_snapshots)
        if unit_type_list is None:
            unit_type_list = ['DRONE', 'TANK', 'ANTI_TANK', 'INFANTRY', 'COMMAND_POST', 'ARTILLERY']
        if team_list is None:
            team_list = ['RED', 'BLUE']
        if total_counts is None:
            total_counts = {team: {ut: 0 for ut in unit_type_list} for team in team_list}
            for unit in state_snapshots[0].units:
                total_counts[unit['team']][unit['type']] += 1
        # Mapping for visualization
        marker_map = {
            'DRONE': '*', 'TANK': 's', 'ANTI_TANK': 'P', 'INFANTRY': 'o', 'COMMAND_POST': 'X', 'ARTILLERY': '^'
        }
        type_kor = {
            'DRONE': '드론', 'TANK': '탱크', 'ANTI_TANK': '대전차', 'INFANTRY': '보병', 'COMMAND_POST': '지휘소', 'ARTILLERY': '곡사포'
        }
        team_kor = {'RED': '레드', 'BLUE': '블루'}
        team_color = {'RED': 'red', 'BLUE': 'blue'}
        # Prepare kill events
        kill_events = []
        if events is not None:
            for e in events:
                if e.get('event_type') == 'COMBAT' and e.get('details', {}).get('hit') and e.get('details', {}).get('damage', 0) > 0:
                    kill_events.append({
                        'timestamp': e['timestamp'], 'actor_id': e['actor_id'], 'target_id': e.get('target_id', ''), 'damage': e['details']['damage']
                    })
        # Helpers
        def parse_unit_kor(unit_id):
            parts = unit_id.split('_')
            if len(parts) >= 3:
                team, typ, idx = parts[0], parts[1], '_'.join(parts[2:])
                return f"{team_kor.get(team, team)}_{type_kor.get(typ, typ)}_{idx}"
            elif len(parts) == 2:
                team, typ = parts
                return f"{team_kor.get(team, team)}_{type_kor.get(typ, typ)}"
            return unit_id
        def get_kill_log(snapshot_time):
            # 이벤트 로그 기반 파괴 이벤트 집계
            kill_events = []
            destroyed_targets = set()
            for e in events or []:
                if (
                    e.get('event_type') == 'COMBAT'
                    and e.get('details', {}).get('hit')
                    and e.get('details', {}).get('damage', 0) > 0
                    and e.get('timestamp', 0) <= snapshot_time
                ):
                    target_id = e.get('target_id', '')
                    if target_id and target_id not in destroyed_targets:
                        destroyed_targets.add(target_id)
                        actor = parse_unit_kor(e['actor_id'])
                        target = parse_unit_kor(target_id)
                        kill_events.append(f"{actor}이 {target}를 파괴.")
            return kill_events[-5:]  # 최근 5개만 표시
        # Adjust layout for slider
        self.fig.subplots_adjust(bottom=0.15)
        # Draw frame function
        def draw_frame(idx):
            snapshot = state_snapshots[idx]
            self.ax_map.clear(); self.ax_panel.clear(); self.ax_panel.axis('off')
            # Map
            self.ax_map.imshow(self.bg_img)
            for unit in snapshot.units:
                x, y = unit['position']; team = unit['team']; ut = unit['type']
                marker = marker_map.get(ut, 'o')
                # determine style
                if unit.get('health', 1) <= 0 or unit.get('status', '').upper() in ['K_KILL','DESTROYED']:
                    facecolor='gray'; edgecolor=team_color[team]; alpha=0.7
                elif unit.get('action','').upper()=='FIRE':
                    facecolor='yellow'; edgecolor=team_color[team]; alpha=1.0
                else:
                    facecolor=team_color[team]; edgecolor='none'; alpha=1.0
                self.ax_map.scatter(x, y, c=facecolor, marker=marker, s=180, edgecolors=edgecolor, linewidths=2.5, alpha=alpha, zorder=10)
            self.ax_map.set_title(f'Time: {snapshot.timestamp:.1f}'); self.ax_map.axis('off')
            # Legends
            weapon_legend = [mlines.Line2D([],[],color='gray',marker=marker_map[ut],markerfacecolor='gray',markeredgecolor='gray',markersize=14, label=type_kor[ut], linestyle='None') for ut in unit_type_list]
            status_legend = [
                mlines.Line2D([],[],color=team_color[t],marker='o',markerfacecolor=team_color[t],markeredgecolor=team_color[t],markersize=14,label=f"생존({team_kor[t]})",linestyle='None') for t in team_list
            ] + [
                mlines.Line2D([],[],color='yellow',marker='o',markerfacecolor='yellow',markeredgecolor=team_color[t],markersize=14,label=f"공격({team_kor[t]})",linestyle='None') for t in team_list
            ] + [
                mlines.Line2D([],[],color='gray',marker='o',markerfacecolor='gray',markeredgecolor=team_color[t],markersize=14,label=f"파괴({team_kor[t]})",linestyle='None',alpha=0.7) for t in team_list
            ]
            wl = self.ax_map.legend(handles=weapon_legend, loc='upper left', fontsize=13, title='무기 종류', frameon=True, bbox_to_anchor=(0.01,0.99))
            wl.get_frame().set_facecolor('white'); wl.get_frame().set_alpha(0.5); self.ax_map.add_artist(wl)
            sl = self.ax_map.legend(handles=status_legend, loc='upper right', fontsize=13, title='상태(색상)', frameon=True, bbox_to_anchor=(0.99,0.99))
            sl.get_frame().set_facecolor('white'); sl.get_frame().set_alpha(0.5)
            # Panel: kill log + status table
            y0=0.95
            self.ax_panel.text(0.5,y0,'Kill Log',fontsize=16,fontweight='bold',ha='center',va='top',transform=self.ax_panel.transAxes)
            for i,msg in enumerate(get_kill_log(snapshot.timestamp)):
                self.ax_panel.text(0.5,y0-0.07*(i+1),msg,fontsize=13,color='black',ha='center',va='top',transform=self.ax_panel.transAxes,bbox=dict(facecolor='white',alpha=0.8,edgecolor='none'))
            curr_counts={t:{ut:0 for ut in unit_type_list} for t in team_list}
            for unit in snapshot.units:
                if unit.get('health',1)>0:
                    curr_counts[unit['team']][unit['type']]+=1
            table_data=[[type_kor[ut]]+ [f"{curr_counts[t][ut]}/{total_counts[t][ut]}" for t in team_list] for ut in unit_type_list]
            cols=["무기 종류"]+[team_kor[t] for t in team_list]
            tbl=self.ax_panel.table(cellText=table_data,colLabels=cols,loc='lower center',cellLoc='center',bbox=[0.05,0.05,0.9,0.45])
            tbl.auto_set_font_size(False); tbl.set_fontsize(11); self.ax_panel.axis('off')
            self.fig.canvas.draw_idle()
        # Initial draw
        draw_frame(0)
        # Slider
        slider_ax = self.fig.add_axes([0.2,0.05,0.6,0.03])
        slider = mwidgets.Slider(slider_ax, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
        slider.on_changed(lambda val: draw_frame(int(val)))
        plt.show()