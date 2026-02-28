import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from matplotlib.patches import Rectangle


def trajectories_plot(ep_i,
                      x_min,
                      x_max,
                      time_step,
                      CAV_PR,
                      CF_model,
                      control_strategy,
                      RL_agent,
                      lane_name,      # 保留参数但不再使用
                      lane_length,    # 保留参数但不再使用
                      light):

    file_path = r'./Road_Network/fcd_output.xml'
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 两条车道的 y 中心值（按你给的）
    lane_y_centers = [95.2, 98.4]

    def y_to_group(y_val: float) -> int:
        """根据距离哪个车道中心更近进行分类：返回 0 或 1"""
        return int(np.argmin([abs(y_val - c) for c in lane_y_centers]))

    # group_tracks[group_id][vehicle_id] = {'time':[], 'x':[], 'type':[]}
    group_tracks = defaultdict(lambda: defaultdict(lambda: {
        'time': [],
        'x': [],
        'type': []
    }))

    # 读取 XML 并按 y 分类
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            vtype = vehicle.get('type')
            y = float(vehicle.get('y'))
            x = float(vehicle.get('x'))

            g = y_to_group(y)
            group_tracks[g][vehicle_id]['time'].append(time)
            group_tracks[g][vehicle_id]['x'].append(x)
            group_tracks[g][vehicle_id]['type'].append(vtype)

    # 横向子图（1行2列）
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 计算最大时间用于画信号灯
    max_t = 0.0
    for g in [0, 1]:
        for data in group_tracks[g].values():
            if data['time']:
                max_t = max(max_t, max(data['time']))
    signal_times = np.arange(0, max_t + 1, 1)

    # 信号灯条固定在 y(位置x) = 395~405
    band_ymin, band_ymax = 392, 398
    band_h = band_ymax - band_ymin

    for g in [0, 1]:
        ax = axes[g]
        # 画信号灯色带（用 Rectangle，y 用真实数据坐标，所以一定显示）
        for t in signal_times:
            time_in_cycle = t % light.cycle_length

            if time_in_cycle < light.green_duration:
                dur = light.green_duration
                color = 'green'
            elif time_in_cycle < light.green_duration + light.yellow_duration:
                dur = light.yellow_duration
                color = 'yellow'
            else:
                dur = light.red_duration
                color = 'red'

            rect = Rectangle(
                (t, band_ymin),  # 左下角 (time, y=位置x)
                dur,             # 宽度=持续时间
                band_h,          # 高度=405-395
                facecolor=color,
                edgecolor='none',
                alpha=0.8,
                zorder=10
            )
            ax.add_patch(rect)
            
        vehicles = group_tracks[g]

        # 画轨迹（x vs time）
        for vehicle_id, data in vehicles.items():
            if len(data['time']) < 2:
                continue

            for j in range(1, len(data['time'])):
                if data['time'][j] - data['time'][j - 1] > time_step:
                    continue

                ax.plot(
                    data['time'][j - 1:j + 1],
                    data['x'][j - 1:j + 1],
                    color='blue' if data['type'][j] == 'CAV' else 'black',
                    linewidth=1,
                    zorder=2
                )

        # 图形设置
        ax.set_title(
            f'{control_strategy} - Episode {ep_i} - '
            f'Lane Y≈{lane_y_centers[g]} - CAV_PR {CAV_PR}'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylim(200, 450)  # y轴显示0到车道长度
        ax.set_ylabel('Position (x)')
        ax.set_xlim(x_min, 400)

        # 确保 395~405 不会被裁掉（如你本来ylim覆盖它，这行也不会破坏）
        y0, y1 = ax.get_ylim()
        ax.set_ylim(min(y0, band_ymin - 5), max(y1, band_ymax + 5))

        ax.grid(False)

    # 保存图片
    output_dir = r'./Output_trajectories/{}_{}_{}'.format(control_strategy, RL_agent, CF_model)
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_output_{ep_i}_{CAV_PR}.png'))
    plt.close()