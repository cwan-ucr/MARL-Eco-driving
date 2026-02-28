import math
import random
import os
import numpy as np
import sys

from sumolib import checkBinary

def set_sumo(gui, sumocfg_file_name, max_steps):
    # SUMO仿真基础设置包括:
    # gui: 是否仿真可视化, True为开启可视化, False为不开启
    # sumocfg_file_name: .rou.xml, net.xml等文件的文件夹地址索引
    # max_steps: SUMO仿真最大等待仿真步长, 单位为秒(s)
    # CF_model: 仿真策略, 'IDM', 'GLOSA'

    # SUMO系统环境检查, 查看系统是否构建SUMO_HOME环境变量
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # 设置sumo-gui
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # 设置sumo命令行, 其中：
    # --no-step-log表示不打印sumo仿真输出
    # --waiting-time-memory表示最大启动等待时间
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_name,
                "--no-step-log", "true",
                "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

def generate_cfg_file(time_step):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file_name = 'test.sumocfg'
    rou_file_name = 'test.rou.xml'
    net_file_name = 'test.net.xml'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    rou_file = os.path.join(curr_path, rou_file_name)
    net_file = os.path.join(curr_path, net_file_name)

    with open(cfg_file, "w") as route:
        print("""<configuration>
    <input>
        <net-file value="{}"/>
        <route-files value="{}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="600"/>
        <step-length value="{}"/>
    </time>
    <output>
        <fcd-output value="fcd_output.xml" />
    </output>  
</configuration>""".format(net_file, rou_file, time_step), file=route)

def generate_rou_file(simulation_steps, volume_per_leg, CAV_PR, warmup_time, CF_model, control_strategy, seed):
    random.seed(seed)
    np.random.seed(seed)

    car_count_per_leg = np.array(volume_per_leg) * simulation_steps / 3600
    car_count_total = car_count_per_leg.sum().astype(int).item()
    car_count_per_leg = car_count_per_leg.astype(int).tolist()

    timings = []
    car_gen_steps = []
    min_old = []
    max_old = []
    min_new = 0
    max_new = simulation_steps

    for i in range(len(volume_per_leg)):
        timings.append(np.random.weibull(2, int(car_count_per_leg[i])))
        timings[i] = np.cumsum(timings[i])

        car_gen_steps.append([])
        min_old.append([])
        max_old.append([])

        if car_count_per_leg[i] < 1:
            continue

        min_old[i] = timings[i][0]
        max_old[i] = timings[i][-1]

        for j in range(car_count_per_leg[0]):
            car_gen_steps[i].append(((max_new - min_new)
                                     / (max_old[i] - min_old[i])) * (timings[i][j] - min_old[i]))

        car_gen_steps[i] = warmup_time + np.rint(car_gen_steps[i])  # 对时间进行取整

    curr_path = os.path.dirname(os.path.abspath(__file__))
    rou_file_name = 'test.rou.xml'
    rou_cfg_file = os.path.join(curr_path, rou_file_name)
    # print(rou_cfg_file)

    with open(rou_cfg_file, "w") as route:

        print("""<routes>

        <vType id = 'CAV' vClass="private" tau="1.6" accel="4.0" decel="4.0" color="#00FF00" speedFactor="1.0" sigma="0.2" length="5.0" minGap="1.0" maxSpeed="18.00" guiShape="passenger"/>
        <vType id = 'HDV' vClass="private" tau="1.6" accel="4.0" decel="4.0" color="#FF0000" speedFactor="1.0" sigma="0.2" length="5.0" minGap="1.0" maxSpeed="18.00" guiShape="passenger"/>

        <route id="N2S" edges="NS -SN"/>
        <route id="S2N" edges="SN -NS"/>
        <route id="W2E" edges="WE -EW"/>
        <route id="E2W" edges="EW -WE"/>""", file=route)

        depart_list = []

        for i in range(len(volume_per_leg)):
            if car_count_per_leg[i] < 1:
                continue

            if i == 0:
                route_i = 'W2E'
            elif i == 1:
                route_i = 'E2W'
            elif i == 2:
                route_i = 'N2S'
            elif i == 3:
                route_i = 'S2N'

            for j in range(car_count_per_leg[i]):
                depart_list.append([route_i, car_gen_steps[i][j]])

        depart_list = sorted(depart_list, key=lambda x: x[1])

        for i in range(car_count_total):
            veh_type = 'HDV'
            if np.random.rand() <= CAV_PR:
                veh_type = 'CAV'

            if veh_type == 'HDV' or CF_model == 'IDM':
                print('        <vehicle id="%s_%i" type="%s" route="%s" depart="%i" departSpeed="10" departLane="best"/>'
                      % (depart_list[i][0], i + 1, veh_type, depart_list[i][0], depart_list[i][1]), file=route)
            else:
                print('        <vehicle id="%s_%i" type="%s" route="%s" depart="%i" departSpeed="10" departLane="best">'
                      % (depart_list[i][0], i + 1, veh_type, depart_list[i][0], depart_list[i][1]), file=route)
                print('            <param key="has.glosa.device" value="true"/>', file=route)
                print('            <param key="device.glosa.range" value="170"/>', file=route)
                print('        </vehicle>', file=route)


        print('</routes>', file=route)

    random.seed(seed)
    np.random.seed(seed)