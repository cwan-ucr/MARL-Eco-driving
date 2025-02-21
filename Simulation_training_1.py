import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import traci

from Env import SUMOEnv
from Agent.SAC_agent import SAC_agent
from Road_Network.sumocfg import set_sumo, generate_cfg_file, generate_rou_file
from Trajectories_visualization import trajectories_plot

# 环境设置
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--seed', type=int, default=2668, help='random seed')
parser.add_argument('--training', type=bool, default=False, help='training or testing')
parser.add_argument('--Env_name', type=str, default='SUMO_RL', help='name of simulation environment')
parser.add_argument('--control_strategy', type=str, default='SUMO', help='Longitudinal control strategy for CAV, include SUMO, and RL')
parser.add_argument('--CF_model', type=str, default='IDM', help='choose the CF model, include Random, IDM, and GLOSA')
parser.add_argument('--Need_transition_state', type=bool, default=True, help='whether need to transition state, True or False')

# SUMO交通流参数设置
parser.add_argument('--volume_per_leg', type=tuple, default=[2000, 0, 0, 0], help='Traffic volume per lane, in 3600 steps, 600: W2E, 0: E2W, 0: N2S, and 0: S2N')
parser.add_argument('--time_step', type=float, default=1.0, help='update interval for environment, steps per second')
parser.add_argument('--CAV_PR', type=float, default=0.6, help='CAV penetration rate')
parser.add_argument('--warmup_time', type=int, default=40, help='Warmup steps before applying simulation, in second')
parser.add_argument('--perception_region', type=float, default=25.0, help='Perception of CAVs on-board sensor, information of HDVs can be shared if in the region, in miles')
parser.add_argument('--dangerous_time', type=float, default=0.0, help='Dangerous time when green start and yellow end, in seconds')
parser.add_argument('--simulation_time', type=int, default=340, help='Simulation steps per episode, in steps')
parser.add_argument('--refresh_progress_interval', type=int, default=40, help='Refresh the simulation progress in every K steps, in steps')

# 车辆参数设置
parser.add_argument('--lc_min_speed', type=float, default=4.0, help='Max wsimulation steps per episode, in miles per second')
parser.add_argument('--TTC_min', type=float, default=0.8, help='Safety car-following time headway, in seconds')
parser.add_argument('--TTC_max', type=float, default=3.0, help='Safety car-following time headway, in seconds')

# RL训练设置
parser.add_argument('--RL_agent', type=str, default='M_VDN', help='Policy of MARL, include ISAC, VDN, and M_VDN')
parser.add_argument('--NN_reset', type=int, default=4000, help='Model reset at K-th episode, in iterations')
parser.add_argument('--Max_episode', type=int, default=1, help='Max training episode')
parser.add_argument('--save_episode', type=int, default=100, help='Model saving interval, in iterations.')
parser.add_argument('--expert_episode', type=int, default=0, help='Max pretraining episode')
parser.add_argument('--Max_ep_steps', type=int, default=600, help='Max training steps per episode, in steps')
parser.add_argument('--eval_interval', type=int, default=100, help='Model evaluating interval, trajectories plot, in episodes.')
parser.add_argument('--Dynamic_target_entropy', type=bool, default=False, help='Whether need to dynamic target entropy TE = -dim(A_eff), True or False')
parser.add_argument('--feature_dim', type=int, default=7, help='Feature dim of each node')
parser.add_argument('--action_dim', type=int, default=2, help='Action dim of each node, longitudinal and lateral')

# DNN 设置
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
parser.add_argument('--beta', type=float, default=0.2, help='Beta for the leaky_relu')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden net width, s_dim-hidden_dim-hidden_dim-a_dim')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

# SAC_agent设置
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping value')
parser.add_argument('--policy_delay', type=int, default=1, help='Policy decay: update N times of critic then update actor, in steps')
parser.add_argument('--actor_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--critic_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--alpha_lr', type=float, default=3e-4, help='Entropy coefficient')
parser.add_argument('--tau', type=float, default=5e-3, help='Soft update value of Target Critic')
parser.add_argument('--adaptive_alpha', type=bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--buffer_size', type=int, default=100000, help='Capacity of replay buffer')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size of training')


opt = parser.parse_args()
opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')  # from str to torch.device

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)

# 单个episode运行
def run_simulations(env, agent, total_steps, ep_i, opt):

    # 初始化
    ep_reward = 0
    actor_loss = 0
    critic_loss = 0
    alpha_loss = 0

    ep_i_step = 0
    time_in = time.time()

    states, veh_names, veh_types = env.reset()

    # 训练
    while traci.simulation.getTime() < opt.simulation_time \
            and ep_i_step <= opt.Max_ep_steps:

        actions = veh_names.copy()
        mask = veh_types.copy()

        for i in range(0, len(env.direction)):
            veh_names_direction_i = veh_names[i]
            if veh_names_direction_i.__len__() == 0:
                continue

            states_direction_i = states[i].copy()
            veh_types_direction_i = veh_types[i]
            mask_direction_i = np.array(veh_types_direction_i, dtype=int)

            mask[i] = mask_direction_i

            # 随机采样
            if ep_i < opt.expert_episode and opt.training and opt.CF_model == 'Random':
                actions_direction_i = env.action_sampling()
                actions[i] = actions_direction_i
                continue

            actions_direction_i = agent.take_action(states_direction_i, mask_direction_i)

            actions[i] = actions_direction_i

        # 环境交互
        # return: state_next, Adj_next, action_Env, done, reward, veh_names, veh_types
        (states_next,
         update_last_states,
         actions_env,
         dones,
         rewards,
         veh_names_next,
         veh_types_next) = env.step(states, actions, veh_names, veh_types)

        # 经验回放
        # add(self, state, Adj, mask, action, reward, state_next, Adj_next, mask_next, done):
        for i in range(0, len(env.direction)):
            if veh_names[i].__len__() == 0 or sum(mask[i]) == 0:
                continue
            try :
                actions[i].astype(np.float32)
            except:
                print('error')

            # Add to replay buffer: transition state or future state
            if opt.Need_transition_state:
                agent.replay_buffer.add(states[i],
                                        mask[i],
                                        actions[i].astype(np.float32),
                                        np.array(rewards[i], dtype=np.float32),
                                        update_last_states[i],
                                        np.array(veh_types[i], dtype=int),
                                        np.array(dones[i], dtype=bool))
            else:
                agent.replay_buffer.add(states[i],
                                        mask[i],
                                        actions[i].astype(np.float32),
                                        np.array(rewards[i], dtype=np.float32),
                                        states_next[i],
                                        np.array(veh_types_next[i], dtype=int),
                                        np.array(dones[i], dtype=bool))
            if veh_names_next[i].__len__() == 0:
                states[i] = []
            ep_reward += rewards[i].sum() / mask[i].sum() if sum(mask[i]) > 0 else 0

        # 状态更新
        states = states_next.copy()
        veh_names = veh_names_next.copy()
        veh_types = veh_types_next.copy()
        ep_i_step += 1

        if (ep_i >= opt.expert_episode
            and opt.training
            and total_steps + ep_i_step > opt.batch_size
            and opt.control_strategy == 'RL'):

            agent.update(ep_i_step)

        if ep_i_step % opt.refresh_progress_interval == 0:
            time_out = time.time() - time_in
            progress = ( ep_i_step * opt.time_step / (opt.simulation_time - opt.warmup_time) ) * 100
            sys.stdout.write(f"\rCurrent Eposide: {ep_i - opt.expert_episode}"
                             f" SUMO Simulation Progress: {progress:.2f}%"
                             f" Time Usage: {time_out:.2f}s")
            sys.stdout.flush()  # 强制刷新输出
            time.sleep(0.001)  # 可选：加一个小的延时让进度更平滑，避免过快更新

    env.close()

    # 每eval_interval个episode输出仿真轨迹
    if ep_i % opt.eval_interval == 0:
        trajectories_plot(ep_i - opt.expert_episode,
                          opt.warmup_time,
                          opt.simulation_time,
                          opt.time_step,
                          opt.CAV_PR,
                          opt.CF_model,
                          opt.control_strategy,
                          opt.RL_agent,
                          env.entering_lanes,
                          env.lanes_entering_length,
                          env.light)

    # 在NN_reset个episode重置SAC智能体
    if (ep_i - opt.expert_episode) - opt.NN_reset == 0 and ep_i > opt.expert_episode:
        agent.reset()
        print('SAC agent has been reset')

    # 每save_episode个episode保存SAC智能体
    if ep_i % opt.save_episode == 0 and opt.training:
        agent.save(int(ep_i),
                   opt.CAV_PR,
                   opt.time_step,
                   opt.control_strategy,
                   opt.RL_agent)

    total_steps += ep_i_step

    if agent.alpha_loss.__len__() > 0:
        critic_loss = agent.critic_loss[-1]
        actor_loss = agent.actor_loss[-1]
        alpha_loss = agent.alpha_loss[-1]

    ep_fuel = env.fuel_consumption / env.discharge_number
    ep_stop = env.stop_time / env.discharge_number
    ep_tet = env.tet / env.discharge_number
    ep_tit = env.tit / env.discharge_number
    ep_tt = env.total_travel_time / env.discharge_number
    ep_dt = env.desired_pass_error / env.discharge_number
    ep_q = env.discharge_number

    ep_matrix = np.array([ep_fuel, ep_stop, ep_tet, ep_tit, ep_dt, ep_tt, ep_q])

    return ep_reward, ep_matrix, actor_loss, critic_loss, alpha_loss, total_steps


def train(opt):

    opt.state_dim = 7
    opt.action_dim = 2
    opt.action_bound = torch.tensor([4, 1]).to(opt.device)
    opt.max_e_steps = 1e3

    # SUMO环境搭建
    ############################################################################
    cfg_file_name = 'Road_Network/test.sumocfg'
    cfg_file = os.path.join(curr_path, cfg_file_name)
    sumo_cmd = set_sumo(gui=False, sumocfg_file_name=cfg_file, max_steps=600)
    env = SUMOEnv(sumo_cmd=sumo_cmd,
                  strategy=opt.control_strategy,
                  need_transition=opt.Need_transition_state,
                  init_seed=opt.seed,
                  warmup_time=opt.warmup_time,
                  time_step=opt.time_step,
                  dangerous_time=opt.dangerous_time,
                  expert_episode=opt.expert_episode,
                  lc_min_speed=opt.lc_min_speed,
                  TTC_min=opt.TTC_min,
                  TTC_max=opt.TTC_max,
                  action_bound=opt.action_bound)
    ############################################################################
    opt.max_nodes = env.max_nodes

    # 仿真种子生成
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    # 构建SAC智能体
    agent = SAC_agent(**vars(opt))
    if not opt.training and opt.control_strategy == 'RL':
        agent.load("500",
                    opt.CAV_PR,
                    opt.time_step,
                    opt.control_strategy,
                    opt.RL_agent)
        agent.eval()

    # SAC训练/验证
    total_steps = 0
    training_curve = []

    for ep_i in range(opt.Max_episode):

        env_seed += 1
        generate_cfg_file(time_step = opt.time_step)
        if ep_i > opt.expert_episode and opt.control_strategy == 'RL':
            opt.CF_model = 'IDM'

        generate_rou_file(opt.Max_ep_steps, opt.volume_per_leg, opt.CAV_PR, opt.warmup_time, opt.CF_model, opt.control_strategy, env_seed)

        (ep_reward,
         ep_matrix,
         actor_loss,
         critic_loss,
         alpha_loss,
         total_steps) = run_simulations(env, agent, total_steps, ep_i, opt)

        if ep_reward.item() != 0:
            print('\n')
            print('Episode: ', ep_i - opt.expert_episode,
                  ' Reward:', round(ep_reward.item(), 2),
                  ' actor_loss:', round(actor_loss, 2),
                  ' critic_loss:', round(critic_loss, 2),
                  ' alpha_loss:', round(alpha_loss, 2) )
            print('Fuel (mL/v):', round(ep_matrix[0], 2),
                  ' Stop (s/v):', round(ep_matrix[1], 2),
                  ' TET (s/v):', round(ep_matrix[2], 2),
                  ' TIT (s/v):', round(ep_matrix[3], 2),
                  ' Dt (s/v):', round(ep_matrix[4], 2),
                  ' Pass (v):', round(ep_matrix[6], 2),
                  ' ATT (s/v):', round(ep_matrix[5], 2))
            print('--------------------------------------------------------------------------------------------------------------------------------------')

        training_curve.append((ep_reward,
                               ep_matrix[0],
                               ep_matrix[1],
                               ep_matrix[2],
                               ep_matrix[3],
                               ep_matrix[4],
                               ep_matrix[5],
                               ep_matrix[6],
                               actor_loss,
                               critic_loss,
                               alpha_loss,))

    # if opt.training:
    #     agent.save('final',
    #                         opt.CAV_PR,
    #                         opt.time_step,
    #                         opt.control_strategy,
    #                         opt.RL_agent)

    return training_curve


if __name__ == "__main__":
    Volume_List = [2000]
    CAV_PR_list = [0.2, 0.6, 1.0]
    for CAV_PR in CAV_PR_list:

        opt.CAV_PR = CAV_PR
        list = ['IDM', 'GLOSA', 'M_VDN']

        for i, CF_model in enumerate(list):
            if CF_model == 'IDM' or CF_model == 'GLOSA':
                opt.control_strategy = 'SUMO'
                opt.CF_model = CF_model
            else:
                opt.control_strategy = 'RL'
                opt.CF_model = 'IDM'
                opt.RL_agent = CF_model

            training_curve = train(opt)
            # 保存训练曲线数据: .csv文件，
            np.savetxt('testing_{}_CAVPR_{}_Volume_{}_gamma_{}_agent_{}_eposide_{}.csv'
                      .format(CF_model,
                              opt.CAV_PR,
                              opt.volume_per_leg[0],
                              opt.gamma,
                              opt.RL_agent,
                              opt.Max_episode),
                              np.array(training_curve), delimiter=',')