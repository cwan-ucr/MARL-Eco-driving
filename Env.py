import numpy as np
import traci


class SUMOEnv:
    def __init__(self,
                 sumo_cmd: str,
                 strategy: str,
                 need_transition: bool,
                 init_seed: int,
                 warmup_time: float,
                 dangerous_time: float,
                 time_step: float,
                 expert_episode: int,
                 lc_min_speed: float,
                 TTC_min: float,
                 TTC_max: float,
                 action_bound: float):
        """
        :param sumo_cmd: sumo启动命令
        :param strategy: 环境策略, 'IDM', 'GLOSA', 'RL'
        :param need_transition: 是否需要状态转移
        :param init_seed: 环境初始化随机种子
        :param warmup_time: 环境预热时间
        :param time_step: 单位时间步长
        :param dangerous_time: 危险状态持续时间
        :param expert_episode: 专家策略预训练周期
        :param lc_min_speed: 允许换道的最小速度
        :param TTC_min: TTC最小值
        :param TTC_max: TTC最大值
        """

        # 环境参数
        self.cmd = sumo_cmd
        self.strategy = strategy
        self.need_transition = need_transition
        self.time_step = time_step
        self.dangerous_time = dangerous_time
        self.init_seed = init_seed
        self.expert_episode = expert_episode
        self.TTC_min = TTC_min
        self.TTC_max = TTC_max
        self.lane_changing_min_speed = lc_min_speed
        self.action_bound = action_bound

        # 环境初始化
        self.max_nodes = int(360 / 6)
        self.runing_time = 0
        self.iteration = 0
        self.direction = ['WE', 'EW', 'NS', 'SN']
        self.entering_lanes = ['WE_0', 'WE_1', 'EW_0', 'EW_1',
                               'NS_0', 'SN_0']
        self.entering_cycle_pass_count = [0, 0, 0, 0, 0, 0]
        self.departing_lanes = ['-EW_0', '-EW_1', '-WE_0', '-WE_1',
                                '-SN_0', '-NS_0']
        self.conflict_lanes = [':J1_4_0', ':J1_4_1', ':J1_1_0', ':J1_1_1',
                               ':J1_0_0', ':J1_3_0']
        self.lanes_entering_length = []
        self.lanes_entering_Max_speed = []
        self.lanes_conflict_length = []
        self.lanes_departing_length = []
        self.lanes_minimal_travel_time = []
        self.num_lanes = len(self.entering_lanes)

        self.directions_state = []
        self.directions_updated_state = []
        self.directions_veh_names = []
        self.directions_veh_types = []

        # 信号灯初始化
        self.signal = ['J1']
        self.light = light()
        self.warmup_time = warmup_time
        self.green_duration = 27
        self.yellow_duration = 3
        self.red_duration = 27
        self.cycle_length = self.green_duration \
                            + 2 * self.yellow_duration \
                            + self.red_duration
        self.light.green_duration = self.green_duration
        self.light.yellow_duration = self.yellow_duration
        self.light.red_duration = self.red_duration
        self.light.cycle_length = self.cycle_length

        # 无量纲时间步
        self.undemon_time_step = self.time_step / self.cycle_length
        self.undemon_dangerous_time = self.dangerous_time / self.cycle_length
        self.undemon_end_lost_time = 2.0 / self.cycle_length

        # 评价指标初始化
        self.fuel_CAV = 0
        self.stop_CAV = 0
        self.tet_CAV = 0
        self.tit_CAV = 0
        self.comfort_CAV = 0
        self.TT_CAV = 0

        self.fuel_consumption = 0
        self.desired_pass_error = 0
        self.stop_time = 0
        self.tet = 0
        self.comfort = 0
        self.tit = 0
        self.total_travel_time = 0
        self.discharge_number = 0
        self.CAV_number = 0
        
        self.fuel_min = self.get_energy_consumption(np.array([0.0]), np.array([0.0]))
        self.fuel_max = self.get_energy_consumption(np.array([18.0]), np.array([4.0]))

    def reset(self):

        np.random.seed(self.init_seed + self.iteration + 1)

        traci.start(self.cmd, label="master")

        # 指标初始化
        self.LC_CAV = 0
        self.fuel_CAV = 0
        self.stop_CAV = 0
        self.tet_CAV = 0
        self.tit_CAV = 0
        self.comfort_CAV = 0
        self.TT_CAV = 0

        self.LC = 0
        self.fuel_consumption = 0
        self.stop_time = 0
        self.tet = 0
        self.tit = 0
        self.comfort = 0
        self.total_travel_time = 0
        self.desired_pass_error = 0
        self.discharge_number = 0
        self.CAV_number = 0

        # 车道信息获取, 包括车道长度、最大速度
        self.light.reset()

        for lane_id in range(self.num_lanes):
            entering_lane_name = self.entering_lanes[lane_id]
            conflict_lane_name = self.conflict_lanes[lane_id]
            departing_lane_name = self.departing_lanes[lane_id]

            self.light.cycle_pass_count.append([])
            self.light.lane_pass_count.append([])

            self.lanes_entering_length.append(
                traci.lane.getLength(entering_lane_name))
            self.lanes_entering_Max_speed.append(
                traci.lane.getMaxSpeed(entering_lane_name))
            self.lanes_minimal_travel_time.append(
                traci.lane.getLength(entering_lane_name) / traci.lane.getMaxSpeed(entering_lane_name))
            self.lanes_conflict_length.append(
                traci.lane.getLength(conflict_lane_name))
            self.lanes_departing_length.append(
                traci.lane.getLength(departing_lane_name))

        # 基于进口道的动态图构建
        if len(self.signal) == 1:
            for i, direction in enumerate(self.direction):

                if self.iteration == 0:
                    self.directions_state.append(direction)
                    self.directions_updated_state.append(direction)
                    self.directions_veh_names.append(direction)
                    self.directions_veh_types.append(direction)

        self.iteration += 1
        # 跳过无车条件下的仿真时间步
        while (not traci.vehicle.getIDList()) or (self.runing_time < self.warmup_time):
            traci.simulationStep()
            self.runing_time = traci.simulation.getTime()

        # 获取道路车辆
        for leg_id in range(len(self.direction)):
            leg_name = self.direction[leg_id]

            vehicle_name_list_last = []

            (leg_state_i, *_, leg_veh_names_i, leg_veh_types_i) = (
                self.get_current_leg_state(leg_name, vehicle_name_list_last))

            self.directions_state[leg_id] = leg_state_i
            self.directions_veh_names[leg_id] = leg_veh_names_i
            self.directions_veh_types[leg_id] = leg_veh_types_i

        # 图结构更新
        states = [item for item in self.directions_state]
        veh_names = [item for item in self.directions_veh_names]
        veh_types = [item for item in self.directions_veh_types]

        return states, veh_names, veh_types

    def step(self, states, actions, veh_names, veh_types):
        # 根据动作更新环境状态，进口道
        current_lane_index = veh_names.copy()
        for leg_i in range(veh_names.__len__()):
            veh_names_leg_i = veh_names[leg_i]
            leg_name = self.direction[leg_i]
            if veh_names_leg_i.__len__() == 0 or True not in veh_types[leg_i]:
                continue

            action_leg_i = actions[leg_i]
            current_lane_index_leg_i = []
            for i, veh_name_i in enumerate(veh_names_leg_i):
                action_i = action_leg_i[i]
                ego_speed = traci.vehicle.getSpeed(veh_name_i)
                ego_lane_index = traci.vehicle.getLaneID(veh_name_i)
                ego_position = traci.vehicle.getLanePosition(veh_name_i)

                lc_motivation = action_i[1].item()
                lc_availability = action_i[2].item()
                lane_changing_direction = 'straight'

                if ego_lane_index == 'WE_0':
                    lane_changing_direction = 'straight' if lc_motivation < lc_availability + 0.25 else 'left'
                elif ego_lane_index == 'WE_1':
                    lane_changing_direction = 'right' if lc_motivation > lc_availability + 0.25 else 'straight'

                current_lane_index_i = traci.vehicle.getLaneIndex(veh_name_i)
                current_lane_index_leg_i.append(current_lane_index_i)
                target_lane_id = self.get_target_lane_id(leg_name,
                                                         current_lane_index_i,
                                                         lane_changing_direction,
                                                         ego_speed)

                if (traci.vehicle.getTypeID(veh_name_i) == 'CAV'
                        and self.strategy != 'SUMO'
                        and self.iteration > self.expert_episode
                        and ego_position > 15):
                    traci.vehicle.setSpeed(veh_name_i,
                                           ego_speed + action_i[0].item() * self.time_step)
                    traci.vehicle.changeLane(veh_name_i, target_lane_id, 1.0)


            current_lane_index[leg_i] = current_lane_index_leg_i

        # 根据IDM更新环境状态，出口道
        for lane_id, lane_name in enumerate(self.departing_lanes):
            veh_names_lane = traci.lane.getLastStepVehicleIDs(lane_name)
            if veh_names_lane.__len__() == 0:
                continue

            for i, veh_name_i in enumerate(veh_names_lane):
                traci.vehicle.setSpeed(veh_name_i, -1)

        # 根据IDM更新环境状态，交叉口内部
        for lane_id, lane_name in enumerate(self.conflict_lanes):
            veh_names_lane = traci.lane.getLastStepVehicleIDs(lane_name)
            if veh_names_lane.__len__() == 0:
                continue

            for i, veh_name_i in enumerate(veh_names_lane):
                traci.vehicle.setSpeed(veh_name_i, -1)

        # 进行一步仿真
        traci.simulationStep()

        # 更新加速度
        action_Env = actions.copy()
        for leg_i in range(veh_names.__len__()):
            veh_names_leg_i = veh_names[leg_i]
            current_lane_index_leg_i = current_lane_index[leg_i]
            if veh_names_leg_i.__len__() == 0 or True not in veh_types[leg_i]:
                continue

            action_Env_leg_i = np.copy(actions[leg_i])
            for i, veh_name_i in enumerate(veh_names_leg_i):
                action_Env_leg_i[i][0] = traci.vehicle.getAcceleration(veh_name_i)
                action_Env_leg_i[i][1] = self.get_lane_changing_decision(veh_name_i, action_Env_leg_i[i][1],
                                                                         current_lane_index_leg_i[i])

            action_Env[leg_i] = action_Env_leg_i

        # 获取观察值、奖励和完成标志
        reward = veh_names.copy()
        done = veh_names.copy()

        for leg_i in range(veh_names.__len__()):
            # 获取道路车辆
            leg_name = self.direction[leg_i]
            veh_names_leg_i = veh_names[leg_i]

            if veh_names_leg_i.__len__() == 0:
                continue

            (leg_state_i,
             leg_updated_state_i,
             leg_veh_names_i,
             leg_veh_types_i) = self.get_current_leg_state(leg_name, veh_names[leg_i])

            self.directions_state[leg_i] = leg_state_i
            self.directions_updated_state[leg_i] = leg_updated_state_i
            self.directions_veh_names[leg_i] = leg_veh_names_i
            self.directions_veh_types[leg_i] = leg_veh_types_i

            # 更新车辆名称列表

        # POMDP
        #reward = self.calculate_reward(self.directions_veh_names, action_Env)
        # MDP
        reward, done = self.calculate_reward(veh_names, veh_types, states, action_Env, actions)
        
        self.update_metrics()

        return (self.directions_state,
                self.directions_updated_state,
                action_Env,
                done,
                reward,
                self.directions_veh_names,
                self.directions_veh_types)

    def calculate_reward(self, veh_names, veh_types, states_last, action_Env, actions):
        """
        :param veh_names: 车辆名称列表
        :param veh_types: 车辆类型列表, 0为普通车，1为自动驾驶车
        :param states_last: 上一时刻状态向量
        :param action_Env: 环境反馈后的动作向量，考虑安全约束
        :param actions: actor输出的动作向量
        """
        # 根据当前状态计算奖励
        reward = veh_names.copy()
        done = veh_names.copy()

        for leg_i in range(veh_names.__len__()):
            reward_leg_i, done_leg_i, info_leg_i = (self.get_leg_reward
                                                   (leg_i,
                                                    veh_names[leg_i],
                                                    veh_types[leg_i],
                                                    states_last[leg_i],
                                                    action_Env[leg_i],
                                                    actions[leg_i]))

            if info_leg_i == 'continue':
                continue

            reward[leg_i] = reward_leg_i
            done[leg_i] = done_leg_i

        return reward, done

    def get_current_leg_state(self, leg_name, vehicle_name_list_last):
        """
        :param leg_name: 进口道名称,'WE', 'EW', 'NS', 'SN'
        :param vehicle_name_list_last: 上一时刻车辆名称列表
        获取当前状态信息，返回一个状态向量state
        状态向量state: N*M, N为车辆数量，M为状态向量维度，
        车辆名称列表Vehicle_name_list: 返回该进口道所有车辆名称列表
        车辆类型列表Vehicle_type_list: 返回该进口道所有车辆类型列表, 0为普通车，1为自动驾驶车
        :returns: state, Vehicle_name_list, Vehicle_type_list
        """

        vehicle_name_list = traci.edge.getLastStepVehicleIDs(leg_name)

        # expected leg state condition
        if vehicle_name_list.__len__() == 0:
            if vehicle_name_list_last.__len__() == 0:
                return [], [], [], [], [], []
            else:
                state_leg = []
                lane_name = traci.vehicle.getLaneID(vehicle_name_list_last[0])
                if lane_name in self.conflict_lanes:
                    lane_name_last = self.entering_lanes[self.conflict_lanes.index(lane_name)]
                elif lane_name in self.departing_lanes:
                    lane_name_last = self.entering_lanes[self.departing_lanes.index(lane_name)]
                else:
                    lane_name_last = lane_name

                lane_id_last = self.entering_lanes.index(lane_name_last)
                state_leg.append(self.get_car_state(vehicle_name_list_last[0], lane_id_last, leg_name))
                state_leg = np.array(state_leg, dtype=np.float32)
                state_leg = np.concatenate((state_leg,
                                            np.zeros((self.max_nodes - 1, state_leg.shape[1]))), axis=0)

                vehicle_type_list = np.zeros(self.max_nodes)
                vehicle_type_list.astype(bool)
                vehicle_type_list[0] = (traci.vehicle.getTypeID(vehicle_name_list_last[0]) == 'CAV')

                return np.array(state_leg, dtype=np.float32), np.array(state_leg,
                                                                       dtype=np.float32), [], vehicle_type_list.tolist()

        # 获取当前时间步车辆状态（包括刚进入的车辆状态，不包括驶出的车辆）
        state_leg, vehicle_type_list = self.get_new_leg_state(leg_name, vehicle_name_list)

        # 获取上一时间步更新后的车辆状态（不包括新进入的车辆，不包括驶出的车辆）
        state_leg_last, _ = self.get_new_leg_state(leg_name, vehicle_name_list_last)

        return state_leg, state_leg_last, vehicle_name_list, vehicle_type_list

    def get_new_leg_state(self, leg_name, vehicle_name_list):
        """
        :param vehicle_name_list: 当前时刻车辆名称列表
        :returns: state, vehicle_name_list, vehicle_type_list
        """
        vehicle_type_list = []
        vehicle_lane_id_list = []

        # State initialization
        state_leg = []

        # 获取当前车辆状态
        for i, vehicle_name_i in enumerate(vehicle_name_list):
            vehicle_type_i = traci.vehicle.getTypeID(vehicle_name_i)
            vehicle_lane_name_i = traci.vehicle.getLaneID(vehicle_name_i)
            vehicle_i_position = traci.vehicle.getLanePosition(vehicle_name_i)

            if vehicle_lane_name_i in self.conflict_lanes:
                lane_id = self.conflict_lanes.index(vehicle_lane_name_i)
            elif vehicle_lane_name_i in self.departing_lanes:
                lane_id = self.departing_lanes.index(vehicle_lane_name_i)
            else:
                lane_id = self.entering_lanes.index(vehicle_lane_name_i)

            vehicle_state = self.get_car_state(vehicle_name_i, lane_id, leg_name)

            state_leg.append(vehicle_state)
            vehicle_type_list.append(vehicle_type_i == 'CAV')
            vehicle_lane_id_list.append(lane_id)

        # 节点填充，节点数量小于max_nodes将补全至max_nodes
        if len(state_leg) < self.max_nodes:
            for i in range(len(state_leg), self.max_nodes):
                state_leg.append(np.array([0]*17, dtype=np.float32))
                vehicle_type_list.append(False)

        return np.array(state_leg, dtype=np.float32), vehicle_type_list

    def get_leg_reward(self, leg_id, veh_names_leg_i, veh_types_leg_i, states_last_leg_i, action_Env_leg_i,
                       action_actor_leg_i):
        """
        获取当前进口道奖励
        :param leg_id: 进口道编号, 0: 'WE', 1: 'EW', 2: 'NS', 3: 'SN'
        :param veh_names_leg_i: 车辆名称列表
        :param veh_types_leg_i: 车辆类型列表
        :param action_Env_leg_i: 动作向量
        :param action_actor_leg_i: actor输出的动作向量
        :returns: reward, done, info
        """
        info = 'not continue'

        # 获取当前进口道最低行驶时间
        if leg_id == 0:
            minimal_tt = self.lanes_minimal_travel_time[0]
            max_speed = self.lanes_entering_Max_speed[0]
            green_cycle_ratio = (self.green_duration + self.yellow_duration) / self.cycle_length
        elif leg_id == 1:
            minimal_tt = self.lanes_minimal_travel_time[2]
            max_speed = self.lanes_entering_Max_speed[2]
            green_cycle_ratio = (self.green_duration + self.yellow_duration) / self.cycle_length
        elif leg_id == 2:
            minimal_tt = self.lanes_minimal_travel_time[4]
            max_speed = self.lanes_entering_Max_speed[4]
            green_cycle_ratio = (self.red_duration + self.yellow_duration) / self.cycle_length
        else:
            minimal_tt = self.lanes_minimal_travel_time[5]
            max_speed = self.lanes_entering_Max_speed[5]
            green_cycle_ratio = (self.red_duration + self.yellow_duration) / self.cycle_length

        if veh_names_leg_i.__len__() == 0:
            reward = []
            done = []
            info = 'continue'
            return reward, done, info

        current_veh_num = veh_names_leg_i.__len__()

        # 奖励定义,包括: 1. 绿灯通过奖励，2. 速度奖励，3. 停止时间奖励, 4. 安全奖励
        stop_reward_leg_i = np.zeros(self.max_nodes)
        safe_reward_leg_i = np.zeros(self.max_nodes)

        # 获取当前进口道状态
        if self.need_transition:
            mask_leg_i = np.array(veh_types_leg_i, dtype=int)
            state_leg_i = self.directions_updated_state[leg_id]
        else:
            mask_leg_i = np.array(self.directions_veh_types[leg_id], dtype=int)
            state_leg_i = self.directions_state[leg_id]

        position_leg_i = state_leg_i[:, 0]  # 归一化车辆位置, position_i = x/lane_length
        speed_leg_i = state_leg_i[:, 1] + 1e-6  # 归一化车辆速度, speed_i = v/lane_max_speed
        speed_last_leg_i = states_last_leg_i[:, 1] + 1e-6  # 归一化车辆速度, speed_i = v/lane_max_speed
        acc_last_leg_i = state_leg_i[:, 2] + 1e-6
        dx_leg_i = state_leg_i[:, 3]  # 归一化间距, dx_i = dx/lane_length
        dv_leg_i = state_leg_i[:, 4] + 1e-6  # 归一化速度差(前车减后车), dv_i = dv/lane_max_speed
        acc_leg_i = action_Env_leg_i[:, 0]  # 纵向加速度
        leader_leg_i = state_leg_i[:, 7]  # 车辆前车类型, leader_i = 0/1, 0为无前车，1为有前车

        """------------------------------不停车奖励-----------------------------------"""
        # 计算绿灯通过奖励:
        green_pass_reward_leg_i = -mask_leg_i

        """------------------------------能耗奖励-----------------------------------"""
        # 计算能耗奖励: reward_energy_vehicle_i = 0.1 * (1 - exp(-0.001 * energy_i))
        energy_leg_i = self.get_energy_consumption((speed_leg_i + speed_last_leg_i) * max_speed / 2, acc_leg_i)
        energy_reward_leg_i = -(energy_leg_i - self.fuel_min) * mask_leg_i

        """------------------------------速度奖励-----------------------------------"""
        # 计算速度奖励: reward_speed_vehicle_i = (0.1 + 0.9 * log(1 + position_i)) * speed_i
        # speed_reward_leg_i = (0.1 + 0.9 * np.sin(position_leg_i * np.pi / 2)) * speed_leg_i
        speed_reward_leg_i = speed_leg_i
        speed_reward_leg_i = speed_reward_leg_i * mask_leg_i

        """------------------------------停车奖励-----------------------------------"""
        # 计算停止时间奖励 reward_stop_vehicle_i = -1.0 * stop_time_i
        stop_reward_leg_i[speed_leg_i < 0.1] = -1
        stop_reward_leg_i = stop_reward_leg_i * mask_leg_i

        """------------------------------安全奖励-----------------------------------"""
        # 计算安全奖励 reward_safe_vehicle_i =
        # -1 if ttc < TTC_min
        # (TTC_max - ttc)/ (TTC_max - TTC_min) elif TTC_min < ttc < TTC_max else 0
        # ttc = dx / (dv + 1e-6) if dv > 0 else 0
        ttc_leg_i = -minimal_tt * dx_leg_i / dv_leg_i
        ttc_leg_i[dv_leg_i >= 0] = np.inf

        ttc_range_1 = ttc_leg_i < self.TTC_min
        ttc_range_2 = (ttc_leg_i > self.TTC_min) & (ttc_leg_i < self.TTC_max)
        safe_reward_leg_i[ttc_range_1] = -1
        safe_reward_leg_i[ttc_range_2] = -(self.TTC_max - ttc_leg_i[ttc_range_2]) / (self.TTC_max - self.TTC_min)
        safe_reward_leg_i = safe_reward_leg_i * mask_leg_i 

        """------------------------------舒适度奖励-----------------------------------"""
        # 计算舒适度奖励 reward_comfort_vehicle_i = -1 * (1 - exp(-0.001 * (speed_i - 1.0) ^ 2))
        comfort_leg_i = (acc_leg_i / 4) ** 2
        comfort_reward_leg_i = -comfort_leg_i * mask_leg_i

        r_g, r_v, r_f, r_s, r_c, r_u, r_t = self.reward_coefficient()

        reward_leg_i = (speed_reward_leg_i * r_v +
                        stop_reward_leg_i * r_s +
                        energy_reward_leg_i * r_f +
                        green_pass_reward_leg_i * r_g +
                        safe_reward_leg_i * r_c +
                        comfort_reward_leg_i * r_u) * self.time_step

        pass_leg_i = state_leg_i[:, 0] >= 1
        done_leg_i = (state_leg_i[:, 0] >= 1) & (mask_leg_i == 1)

        reward_leg_i = self.get_done_reward(veh_names_leg_i, reward_leg_i, pass_leg_i, state_leg_i[:, 1], mask_leg_i)

        # 指标计算
        ttc_count = (ttc_range_1 | ttc_range_2)
        self.fuel_CAV += (energy_leg_i * mask_leg_i).sum() * self.time_step
        self.stop_CAV += (-stop_reward_leg_i).sum() * self.time_step
        self.comfort_CAV += (-comfort_reward_leg_i).sum() * self.time_step
        self.TT_CAV += mask_leg_i.sum() * self.time_step
        self.tet_CAV += (ttc_count & (mask_leg_i == 1)).sum() * self.time_step
        self.tit_CAV += (self.TTC_max - ttc_leg_i[ttc_count & (mask_leg_i == 1)]).sum() * self.time_step

        self.fuel_consumption += (energy_leg_i[0:current_veh_num]).sum() * self.time_step
        self.stop_time += (speed_leg_i[0:current_veh_num] < 0.1).sum() * self.time_step
        self.tet += ttc_count[0:current_veh_num].sum() * self.time_step
        self.tit += (self.TTC_max - ttc_leg_i[ttc_count]).sum() * self.time_step
        self.comfort += comfort_leg_i[0:current_veh_num].sum() * self.time_step
        self.total_travel_time += current_veh_num * self.time_step
        self.discharge_number += (pass_leg_i).sum()
        self.CAV_number += done_leg_i.sum()

        return reward_leg_i, done_leg_i, info

    def get_energy_consumption(self, v, a):
        """
        :param v: 车辆速度
        :param a: 车辆加速度
        :returns: energy_consumption_leg_i
        """
        # ARRB Fuel consumption model
        # https://www.sciencedirect.com/science/article/abs/pii/0191261589900143
        # F(v, a) = alpha + max{beta1*v*Rt + beta2*m*v*a^2, 0} if a > 0 else max{beta1*v*Rt, 0}
        # R_t = b1 + b2 *v^2 + m*a + m*g*G
        m = 1600  # kg
        g = 9.81  # m/s^2
        G = 0  # Grade
        alpha = 0.666
        beta1 = 0.0717
        beta2 = 0.0344
        b1 = 0.269
        b2 = 0.0171
        b3 = 0.000672

        Rt = b1 + b2 * v  + b3 * v ** 2 + m * a / 1000 + m * g * G / 100000
        acc_Energy = beta1 * v * Rt + beta2 * m * v * a ** 2 / 1000
        if a.size > 1:
            acc_Energy[a < 0] = beta1 * v[a < 0] * Rt[a < 0]
        else:
            if a < 0: acc_Energy = beta1 * v * Rt

        energy_consumption_leg_i = alpha + np.clip(acc_Energy, 0, 200)

        return energy_consumption_leg_i

    def get_green_pass_reward(self, leg_id, minimal_tt, green_cycle_ratio, position_leg_i, speed_leg_i):
        """
        获取当前进口道奖励
        :param leg_id: 进口道编号, 0: 'WE', 1: 'EW', 2: 'NS', 3: 'SN'

        :returns: reward, done, info
        """
        """------------------------------绿灯奖励-----------------------------------"""
        # 计算绿灯通过奖励 reward_green_pass_vehicle_i =
        # -1 if vehicle_i pass with the red light,
        # 0.1 if vehicle_i pass with the green light
        # 0 if vehicle_i pass with dangerous light
        green_pass_reward_leg_i = 0.1 * np.ones(self.max_nodes)
        green_remain_ratio, green_sw_ratio = self.get_signal_state(self.direction[leg_id])
        pass_time_ratio_leg_i = minimal_tt * ((1 - position_leg_i) / speed_leg_i) / self.cycle_length

        if green_remain_ratio > (self.undemon_time_step - 1e-4):  # 绿灯剩余时间大于0
            flag_r1 = (pass_time_ratio_leg_i > green_remain_ratio)  # 车辆通过时间大于绿灯剩余时间, 红灯驶出
            flag_r2 = (pass_time_ratio_leg_i < green_sw_ratio)  # 车辆通过时间小于绿灯切换时间, 红灯驶出
            flag_r = flag_r1 & flag_r2

            flag_y1 = (pass_time_ratio_leg_i < green_remain_ratio)  # 车辆通过时间小于剩余时间
            flag_y2 = (pass_time_ratio_leg_i > (
                        green_remain_ratio - self.undemon_end_lost_time - self.undemon_dangerous_time))  # 车辆通过时间大于黄灯剩余时间-danger_time, 黄灯驶出
            flag_y = flag_y1 & flag_y2

            flag_g1 = (pass_time_ratio_leg_i > green_sw_ratio)  # 车辆通过时间大于下一绿灯结束时间, 绿灯驶出
            flag_g2 = (pass_time_ratio_leg_i < (
                        green_sw_ratio + self.undemon_dangerous_time))  # 车辆通过时间小于绿灯切换时间+danger_time, 绿灯驶出
            flag_g = flag_g1 & flag_g2
        else:  # 绿灯剩余时间小于0
            flag_r1 = (pass_time_ratio_leg_i < green_sw_ratio)  # 车辆通过时间小于绿灯切换时间, 红灯驶出
            flag_r21 = (pass_time_ratio_leg_i > (green_sw_ratio + green_cycle_ratio))  # 车辆通过时间大于下一绿灯结束时间, 红灯驶出
            flag_r22 = (pass_time_ratio_leg_i < (green_sw_ratio + 1))  # 车辆通过时间小于下一红灯结束时间, 红灯驶出
            flag_r2 = flag_r21 & flag_r22
            flag_r = flag_r1 | flag_r2

            flag_y1 = (pass_time_ratio_leg_i > (green_sw_ratio +
                                                green_cycle_ratio -
                                                self.undemon_end_lost_time -
                                                self.undemon_dangerous_time))  # 车辆通过时间小于绿灯剩余时间+danger_time, 黄灯驶出
            flag_y2 = (pass_time_ratio_leg_i < (green_sw_ratio +
                                                green_cycle_ratio))  # 车辆通过时间大于绿灯剩余时间, 黄灯驶出
            flag_y = flag_y1 & flag_y2

            flag_g1 = (pass_time_ratio_leg_i > green_sw_ratio)  # 车辆通过时间大于下一绿灯结束时间, 绿灯
            flag_g2 = (pass_time_ratio_leg_i < (
                        green_sw_ratio + self.undemon_dangerous_time))  # 车辆通过时间小于绿灯切换时间+danger_time, 绿灯驶出
            flag_g = flag_g1 & flag_g2

        green_pass_reward_leg_i[flag_r] = -1
        green_pass_reward_leg_i[flag_y] = -1
        green_pass_reward_leg_i[flag_g] = 0

        green_pass_reward_leg_i[pass_time_ratio_leg_i > 1] = -1

        return green_pass_reward_leg_i
    
    def update_metrics(self):
        """
        更新指标, 考虑departure车辆移动能耗，停驶时间，舒适度，安全性等指标
        """
        veh_state = []
        for lane_id, lane_name in enumerate(self.departing_lanes + self.conflict_lanes):
            veh_names_lane = traci.lane.getLastStepVehicleIDs(lane_name)
            if veh_names_lane.__len__() == 0:
                continue
            
            for i, veh_name_i in enumerate(veh_names_lane):
                speed = traci.vehicle.getSpeed(veh_name_i)
                acceleration = traci.vehicle.getAcceleration(veh_name_i)
                position = traci.vehicle.getLanePosition(veh_name_i)
                leader = traci.vehicle.getLeader(veh_name_i, dist=150)
                speed_last = speed - acceleration * self.time_step
                mask = (traci.vehicle.getTypeID(veh_name_i) == 'CAV')
                if lane_name in self.departing_lanes and position > 50:
                    continue
                
                if leader is not None:
                    leader_name = leader[0]
                    space_gap = leader[1] + traci.vehicle.getLength(leader_name) + traci.vehicle.getMinGap(leader_name)
                    speed_error = traci.vehicle.getSpeed(leader_name) - speed
                else:
                    space_gap = 150
                    speed_error = 18 - speed
                    
                veh_state.append([(speed + speed_last)/2, acceleration, space_gap, speed_error, mask])

        veh_state = np.array(veh_state)
        if veh_state.shape[0] == 0:
            return
        energy_usage = self.get_energy_consumption(veh_state[:, 0], veh_state[:, 1])
        comfort = (veh_state[:, 1] / 4) ** 2
        stop = (veh_state[:, 0] / 18.0) < 0.1
        mask = veh_state[:, 4].astype(int)
        ttc = -veh_state[:, 2] / (veh_state[:, 3] + 1e-6)
        ttc[ttc < 0] = np.inf
        ttc_count = (ttc < self.TTC_min) | ((ttc > self.TTC_min) & (ttc < self.TTC_max))
        tit = self.TTC_max - ttc
        tit[~ttc_count] = 0
        
        # all vehicle performance
        self.fuel_consumption += energy_usage.sum() * self.time_step
        self.tet += ttc_count.sum() * self.time_step
        self.tit += tit.sum() * self.time_step
        self.comfort += comfort.sum() * self.time_step
        self.stop_time += stop.sum() * self.time_step
        self.total_travel_time += len(veh_state) * self.time_step
        
        # CAV performance
        self.fuel_CAV += (energy_usage*mask).sum() * self.time_step
        self.comfort_CAV += (comfort*mask).sum() * self.time_step
        self.tet_CAV += (ttc_count*mask).sum() * self.time_step
        self.tit_CAV += (tit*mask).sum() * self.time_step
        self.stop_CAV += (stop*mask).sum() * self.time_step
        self.TT_CAV += mask.sum() * self.time_step
        
        return
       

    def get_done_reward(self, Veh_ID, reward, pass_index, lane_index, mask):
        """
        根据终点状态计算终止奖励
        :param reward: 瞬时奖励
        :param state: 当前状态
        :param mask: 车辆类型
        :return: reward
        """

        if pass_index.sum() < 1:
            return reward

        pass_lane = (lane_index[pass_index] * self.num_lanes).astype(np.int32)
        pass_name = [Veh_ID[index] for index, value in enumerate(pass_index) if value]
        done_reward = np.zeros(pass_lane.shape, dtype=np.float32)

        for i, lane_id in enumerate(pass_lane):

            veh_name = pass_name[i]
            depart_delay = traci.vehicle.getDepartDelay(veh_name)

            self.desired_pass_error += depart_delay

        reward[pass_index] += done_reward

        return reward * mask
    
    def get_car_state(self, vehicle_name, lane_id, direction):
        """
        获取当前状态信息，返回一个状态向量
        :param vehicle_name: 车辆名称
        :param lane_id: 车道ID编号
        :param direction: 进口道方向，'WE', 'EW', 'NS', 'SN'
        :return: 车辆状态向量，
        包括位置/道路长度、速度/限速、车头间距/道路长度、速度差/限速、绿灯余弦编码、绿灯余弦编码微分、跟驰模式
        """

        # Road and signal state
        Lane_length = self.lanes_entering_length[lane_id]
        max_speed = self.lanes_entering_Max_speed[lane_id]
        green_remain_ratio, green_sw_ratio = self.get_signal_state(direction)

        if green_remain_ratio > (self.undemon_time_step - 1e-4):
            map = (np.pi +
                   np.pi * round(green_remain_ratio * self.cycle_length) /
                   (self.yellow_duration + self.green_duration))
        else:
            map = np.pi * round((1 - green_sw_ratio) * self.cycle_length) / \
                  (self.yellow_duration + self.red_duration)

        light_cosine = np.cos(map)
        light_cosine_d = -np.sin(map)

        # Ego vehicle state
        ego_lane_name = traci.vehicle.getLaneID(vehicle_name)
        ego_position = traci.vehicle.getLanePosition(vehicle_name)
        ego_speed = traci.vehicle.getSpeed(vehicle_name)
        ego_acceleration = traci.vehicle.getAcceleration(vehicle_name)
        ego_type = 1 if traci.vehicle.getTypeID(vehicle_name) == 'CAV' else 0

        # Leader and Follower state at the current lane
        ego_leader = traci.vehicle.getLeader(vehicle_name, dist=150)
        ego_follower = traci.vehicle.getFollower(vehicle_name, dist=150)

        if ego_lane_name not in self.entering_lanes and ego_lane_name in self.conflict_lanes:
            ego_position = Lane_length + ego_position
        elif ego_lane_name not in self.entering_lanes and ego_lane_name in self.departing_lanes:
            ego_position = Lane_length + self.lanes_conflict_length[lane_id] + ego_position

        if ego_leader is not None:
            leader_name = ego_leader[0]
            space_gap = ego_leader[1] + traci.vehicle.getLength(leader_name) + traci.vehicle.getMinGap(leader_name)
            speed_error = traci.vehicle.getSpeed(leader_name) - ego_speed
            ego_leader_type = 0 if traci.vehicle.getTypeID(leader_name) == 'BUS' else 1
            if traci.vehicle.getTypeID(leader_name) == 'CAV':
                ego_leader_type = 1
            elif traci.vehicle.getTypeID(leader_name) == 'HDV':
                ego_leader_type = 2
        else:
            space_gap = 150
            speed_error = max_speed - ego_speed
            ego_leader_type = 3
            if green_remain_ratio < (self.undemon_time_step - 1e-4) and green_sw_ratio > (
                    self.undemon_time_step + 1e-4):
                space_gap = Lane_length - ego_position + traci.vehicle.getLength(vehicle_name) + traci.vehicle.getMinGap(vehicle_name)
                speed_error = -ego_speed

        if ego_follower[1] != -1.0:
            follower_name = ego_follower[0]
            space_gap_f = ego_follower[1] + traci.vehicle.getLength(follower_name) + traci.vehicle.getMinGap(follower_name)
            speed_error_f = ego_speed - traci.vehicle.getSpeed(follower_name)
        else:
            space_gap_f = ego_position
            speed_error_f = ego_speed

        # one-hot coding, [0, 0, 0], 'Leader_CAV', 'CAV-CAV', 'CAV-HDV'
        ego_driving_mode = np.zeros(3, dtype=np.float32)
        if ego_type == 1:
            ego_driving_mode[0] = 1
            if ego_leader_type == 1:
                ego_driving_mode[1] = 1
                ego_driving_mode[0] = 0
            elif ego_leader_type == 2:
                ego_driving_mode[2] = 1
                ego_driving_mode[0] = 0

        # Leader and Follower status at the other lane
        if lane_id == 0:
            ego_leader_opt = traci.vehicle.getLeftLeaders(vehicle_name, blockingOnly=False)
            ego_follower_opt = traci.vehicle.getLeftFollowers(vehicle_name, blockingOnly=False)
        elif lane_id == 1:
            ego_leader_opt = traci.vehicle.getRightLeaders(vehicle_name, blockingOnly=False)
            ego_follower_opt = traci.vehicle.getRightFollowers(vehicle_name, blockingOnly=False)

        if ego_leader_opt.__len__() > 0:
            ego_leader_opt = ego_leader_opt[0]
            leader_opt_name = ego_leader_opt[0]
            space_opt_gap = ego_leader_opt[1] + traci.vehicle.getLength(leader_opt_name) + traci.vehicle.getMinGap(leader_opt_name)
            speed_opt_error = traci.vehicle.getSpeed(leader_opt_name) - ego_speed
        else:
            space_opt_gap = 150
            speed_opt_error = max_speed - ego_speed
            if green_remain_ratio < (self.undemon_time_step - 1e-4) and green_sw_ratio > (
                    self.undemon_time_step + 1e-4):
                space_opt_gap = Lane_length - ego_position + traci.vehicle.getLength(vehicle_name) + traci.vehicle.getMinGap(vehicle_name)
                speed_opt_error = -ego_speed

        if ego_follower_opt.__len__() > 0:
            ego_follower_opt = ego_follower_opt[0]
            follower_opt_name = ego_follower_opt[0]
            space_opt_f_gap = ego_follower_opt[1] + traci.vehicle.getLength(follower_opt_name) + traci.vehicle.getMinGap(follower_opt_name)
            speed_opt_f_error = traci.vehicle.getSpeed(follower_opt_name) - ego_speed
        else:
            space_opt_f_gap = ego_position
            speed_opt_f_error = ego_speed

        state = [ego_position / Lane_length,
                 ego_speed / max_speed,
                 ego_acceleration / 4,
                 space_gap / Lane_length,
                 speed_error / max_speed,
                 space_gap_f / Lane_length,
                 speed_error_f / max_speed,
                 space_opt_gap / Lane_length,
                 speed_opt_error / max_speed,
                 space_opt_f_gap / Lane_length,
                 speed_opt_f_error / max_speed,
                 lane_id,
                 light_cosine,
                 light_cosine_d]

        state = np.concatenate([state, ego_driving_mode], dtype=np.float32)


        return state    

    def get_vehicle_state(self, vehicle_name, lane_id, direction):
        """
        获取当前状态信息，返回一个状态向量
        :param vehicle_name: 车辆名称
        :param lane_id: 车道ID编号
        :param direction: 进口道方向，'WE', 'EW', 'NS', 'SN'
        :return: 车辆状态向量，
        包括位置/道路长度、速度/限速、车头间距/道路长度、速度差/限速、绿灯余弦编码、绿灯余弦编码微分、跟驰模式
        """

        Lane_length = self.lanes_entering_length[lane_id]
        max_speed = self.lanes_entering_Max_speed[lane_id]

        ego_lane_name = traci.vehicle.getLaneID(vehicle_name)
        ego_position = traci.vehicle.getLanePosition(vehicle_name)
        ego_speed = traci.vehicle.getSpeed(vehicle_name)
        ego_acceleration = traci.vehicle.getAcceleration(vehicle_name)
        ego_leader = traci.vehicle.getLeader(vehicle_name, dist=150)
        ego_type = 1 if traci.vehicle.getTypeID(vehicle_name) == 'CAV' else 0

        # one-hot coding, [0, 0, 0], 'Leader_CAV', 'CAV-CAV', 'CAV-HDV'
        ego_driving_mode = np.zeros(3, dtype=np.float32)

        green_remain_ratio, green_sw_ratio = self.get_signal_state(direction)

        if ego_lane_name not in self.entering_lanes and ego_lane_name in self.conflict_lanes:
            ego_position = Lane_length + ego_position
        elif ego_lane_name not in self.entering_lanes and ego_lane_name in self.departing_lanes:
            ego_position = Lane_length + self.lanes_conflict_length[lane_id] + ego_position

        if ego_leader is not None:
            leader_name = ego_leader[0]
            space_gap = ego_leader[1] + traci.vehicle.getLength(leader_name) +\
                        traci.vehicle.getMinGap(leader_name)
            speed_error = traci.vehicle.getSpeed(leader_name) - ego_speed
            ego_leader_type = 1 if traci.vehicle.getTypeID(leader_name) == 'CAV' else 0
        else:
            space_gap = 150
            speed_error = max_speed - ego_speed
            ego_leader_type = 2
            if green_remain_ratio < (self.undemon_time_step - 1e-4) and green_sw_ratio > (
                    self.undemon_time_step + 1e-4):
                space_gap = Lane_length - ego_position + traci.vehicle.getLength(vehicle_name) +\
                            traci.vehicle.getMinGap(vehicle_name)
                speed_error = -ego_speed

        if ego_type == 1:
            ego_driving_mode[0] = 1
            if ego_leader_type == 1:
                ego_driving_mode[1] = 1
                ego_driving_mode[0] = 0
            elif ego_leader_type == 0:
                ego_driving_mode[2] = 1
                ego_driving_mode[0] = 0

        if green_remain_ratio > (self.undemon_time_step - 1e-4):
            map = (np.pi +
                   np.pi * round(green_remain_ratio * self.cycle_length) /
                   (self.yellow_duration + self.green_duration))
        else:
            map = np.pi * round((1 - green_sw_ratio) * self.cycle_length) / \
                  (self.yellow_duration + self.red_duration)

        light_cosine = np.cos(map)
        light_cosine_d = -np.sin(map)

        state = [ego_position / Lane_length,
                 ego_speed / max_speed,
                 ego_acceleration / 4,
                 space_gap / Lane_length,
                 speed_error / max_speed,
                 light_cosine,
                 light_cosine_d]
        state = np.concatenate([state, ego_driving_mode],dtype=np.float32)

        return state

    def get_signal_state(self, direction):
        """
        获取信号状态
        :param direction: 进口道方向，'WE', 'EW', 'NS', 'SN'
        :return: 绿灯剩余时间比、下一绿灯切换时间比
        """

        # 获取信号状态
        signal_id = self.signal[0]
        current_signal = traci.trafficlight.getPhase(signal_id)

        time_next_phase = traci.trafficlight.getNextSwitch(signal_id)
        current_time = traci.simulation.getTime()
        Time_lasting = time_next_phase - current_time

        if current_signal == 0:
            green_remain_ratio = (Time_lasting + self.yellow_duration + self.time_step) / \
                                 self.cycle_length
            green_sw_ratio = (green_remain_ratio +
                              (self.yellow_duration + self.red_duration) / self.cycle_length)
        elif current_signal == 1:
            green_remain_ratio = (Time_lasting + self.time_step) / \
                                 self.cycle_length
            green_sw_ratio = (green_remain_ratio +
                              (self.yellow_duration + self.red_duration) / self.cycle_length)
        elif current_signal == 2:
            green_remain_ratio = 0
            green_sw_ratio = (Time_lasting + self.yellow_duration + self.time_step) / \
                             self.cycle_length
        else:
            green_remain_ratio = 0
            green_sw_ratio = (Time_lasting + self.time_step) / \
                             self.cycle_length

        if direction == 'NS' or direction == 'SN':
            return green_sw_ratio, green_remain_ratio
        elif direction == 'WE' or direction == 'EW':
            return green_remain_ratio, green_sw_ratio

    def get_lane_changing_decision(self, veh_name, lane_changing_decision, current_lane_id):
        """
        获取车道改变决策
        :param veh_name: 车辆名称
        :param lane_changing_decision: 车道改变决策
        :param current_lane_id: 当前车道id
        :return: 车道改变决策
        """
        lane_index_changed = traci.vehicle.getLaneIndex(veh_name)
        if lane_index_changed == current_lane_id:
            lane_changing_decision = 0
        else:
            lane_changing_decision = 1

        return lane_changing_decision

    def get_target_lane_id(self, current_leg_name, current_lane_index, lane_changing_direction, speed):
        """
        获取目标车道id
        :param current_leg_name: 当前进口道名称
        :param current_lane_index: 当前车道id
        :param lane_changing_direction: 车道改变方向, 'left', 'right', 'straight'
        :param speed: 纵向速度
        :return: 目标车道id
        """

        if current_leg_name == 'NS' or current_leg_name == 'SN':
            target_lane_id = current_lane_index
            return target_lane_id

        if ((lane_changing_direction == 'left')
                and (speed > self.lane_changing_min_speed)):

            target_lane_id = 1
        elif ((lane_changing_direction == 'right')
              and (speed > self.lane_changing_min_speed)):

            target_lane_id = 0
        else:
            target_lane_id = current_lane_index

        return target_lane_id

    def reward_coefficient(self):
        """
        获取奖励系数: r_g → green_pass, r_v → velocity, r_f → fuel, r_s → stop, r_c → collision
        :returns: r_g, r_v, r_f, r_s, r_c, r_u, r_total
        """
        r_g = 0.05
        r_v = 0.8
        r_f = 0.8 / (self.fuel_max - self.fuel_min)
        r_s = 1.0
        r_c = 1.0
        r_u = 0.2

        r_total = r_g + r_v + r_f + r_s + r_c

        return r_g, r_v, r_f, r_s, r_c, r_u, r_total

    def action_sampling(self):
        """
        生成动作
        :returns: actions
        """
        actions = (2 * self.action_bound.detach().cpu().numpy() *
                   (np.random.rand(self.max_nodes, 2) - 0.5))
        return actions.astype(np.float32)

    @staticmethod
    def close():
        traci.close()


class light:
    def __init__(self):
        """
        创建信号灯
        """
        self.green_duration = 0
        self.yellow_duration = 0
        self.red_duration = 0
        self.cycle_length = 0
        self.saturated_headway = 1.5
        self.cycle_pass_count = []
        self.lane_pass_count = []

    def update_lane_pass_state(self, lane_index):
        current_time = traci.simulation.getTime()
        current_cycle = int(current_time // self.cycle_length)

        pass_state = [current_time, current_cycle]

        self.cycle_pass_count[lane_index].append(pass_state)

    def reset(self):
        self.cycle_pass_count = []
        self.lane_pass_count = []

