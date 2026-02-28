import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np

from collections import defaultdict

def trajectories_plot(ep_i,
                      x_min,
                      x_max,
                      time_step,
                      CAV_PR,
                      CF_model,
                      control_strategy,
                      RL_agent,
                      lane_name,
                      lane_length,
                      light):
    # 修改为你的文件路径
    file_path = r'F:\Research\Paper\[Experiment]TR_PartC_MARL_Eco_driving\Code\Road_Network\fcd_output.xml'

    # 解析 trajectory_output.xml 文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 创建一个字典来保存每个车道内每辆车的时间与位置数据
    lane_tracks = defaultdict(lambda: defaultdict(lambda: {'time': [], 'x': [], 'type': []}))

    # 创建一个字典来保存每个车辆的历史车道信息
    vehicle_lanes = defaultdict(list)

    # 遍历每个 <timestep> 元素
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))  # 获取时间
        # 遍历该时间步中的每辆车
        for vehicle in timestep.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            lane_id = vehicle.get('lane')  # 获取车道ID
            type = vehicle.get('type')  # 获取车辆类型
            y = float(vehicle.get('y')) # 获取y坐标
            x = float(vehicle.get('x'))  # 获取车辆的 x 坐标

            # 检查是否发生换道（即车辆车道发生变化）
            if vehicle_id not in vehicle_lanes:
                vehicle_lanes[vehicle_id].append(lane_id)  # 初始化车辆的车道记录
            elif vehicle_lanes[vehicle_id][-1] != lane_id:
                vehicle_lanes[vehicle_id].append(lane_id)  # 记录换道行为

            # 存储时间与位置数据，按车道分组
            lane_tracks[lane_id][vehicle_id]['time'].append(time)
            lane_tracks[lane_id][vehicle_id]['x'].append(x)
            lane_tracks[lane_id][vehicle_id]['type'].append(type)  # 存储车辆类型

    # 只选择 'WE_0' 和 'WE_1' 车道
    selected_lanes = ['WE_0', 'WE_1']
    filtered_lane_tracks = {lane_id: lane_tracks[lane_id] for lane_id in selected_lanes if lane_id in lane_tracks}

    # 创建子图的数量等于选择的车道数
    num_lanes = len(filtered_lane_tracks)
    fig, axes = plt.subplots(num_lanes, 1, figsize=(10, 6 * num_lanes))

    # 如果只有一个车道，axes 只是一个单独的子图，因此要处理它
    if num_lanes == 1:
        axes = [axes]

    # 绘制每个车道的车辆轨迹
    for i, (lane_id, vehicles) in enumerate(filtered_lane_tracks.items()):
        ax = axes[i]  # 获取对应车道的轴
        lane_id_i = lane_name.index(lane_id)
        lane_length_i = lane_length[lane_id_i]  # 获取车道长度
        # 在保存车辆轨迹之前检查车辆换道行为
        for vehicle_id, data in vehicles.items():
            # 确保车辆的轨迹有足够的数据点来进行换道检查
            if len(vehicle_lanes[vehicle_id]) > 1:  # 确保有多个车道记录
                for j in range(1, len(data['time'])):
                    # 只有在车道数据足够时才进行换道检查
                    if data['time'][j] - data['time'][j - 1] > time_step:  # 确保时间间隔大于 0.5 秒
                        continue
                    if j < len(vehicle_lanes[vehicle_id]) and vehicle_lanes[vehicle_id][j] != vehicle_lanes[vehicle_id][
                        j - 1]:
                        ax.plot(data['time'][j - 1:j + 1], data['x'][j - 1:j + 1],
                                label=f'Vehicle {vehicle_id}',
                                color='blue' if data['type'][j] == 'CAV' else 'black')
                    else:
                        ax.plot(data['time'][j - 1:j + 1], data['x'][j - 1:j + 1],
                                label=f'Vehicle {vehicle_id}',
                                color='blue' if data['type'][j] == 'CAV' else 'black')
            else:
                # 如果车辆没有换道行为，直接绘制轨迹
                ax.plot(data['time'], data['x'], label=f'Vehicle {vehicle_id}',
                        color='blue' if data['type'][0] == 'CAV' else 'black')

        signal_times = np.arange(0, max(
            max(data['time']) for vehicles in filtered_lane_tracks.values() for data in vehicles.values()) + 1, 1)
        for time in signal_times:
            time_in_cycle = time % light.cycle_length
            if time_in_cycle < light.green_duration:
                phase = 'Green'
            elif time_in_cycle < light.green_duration + light.yellow_duration:
                phase = 'Yellow'
            else:
                phase = 'Red'

            if phase == 'Green':
                ax.axvspan(time, time + light.green_duration, ymin= 0.99, ymax=1.0, color='green', alpha=0.8,
                           label='Green Light' if time == 0 else "")
            elif phase == 'Yellow':
                ax.axvspan(time, time + light.yellow_duration, ymin= 0.99, ymax=1.0, color='yellow', alpha=0.8,
                           label='Yellow Light' if time == 0 else "")
            elif phase == 'Red':
                ax.axvspan(time, time + light.red_duration, ymin= 0.99, ymax=1.0, color='red', alpha=0.8,
                           label='Red Light' if time == 0 else "")

        # 设置标题和标签

        ax.set_title(f'{control_strategy} - Eposide {ep_i} - Lane {lane_id} - CAV_PR {CAV_PR} Vehicle Position vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (x)')
        ax.grid(False)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(200, 200 + lane_length_i)


    output_dir = r'F:\Research\Paper\[Experiment]TR_PartC_MARL_Eco_driving\Code\Output_trajectories\{}_{}_{}'.format(control_strategy, RL_agent, CF_model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存图片
    plt.savefig(os.path.join(output_dir, f'trajectory_output_{ep_i}_{CAV_PR}.png'))
    # # 调整布局，避免重叠
    # plt.tight_layout()
    # plt.show()

    plt.close()