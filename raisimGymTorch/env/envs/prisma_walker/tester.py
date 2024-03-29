from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin import prisma_walker
from raisimGymTorch.env.bin.prisma_walker import NormalSampler
from raisimGymTorch.env.bin.prisma_walker import RaisimGymEnv
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import webbrowser
import numpy as np
import torch
import datetime
import argparse
from datetime import datetime
from pickle import load
import matplotlib.pyplot as plt


#import matplotlib as mpl
#simport matplotlib.pyplot as plt


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
parser.add_argument('-v', '--velocity', nargs='+', help = 'command velocity for the quadruped', type=float, default = [1, 0, 0.1]) #nargs take more than 1 argument

args = parser.parse_args()
task_name = "prisma_walker_locomotion"
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
max_simulation_duration = cfg['environment']['max_time']

env = VecEnv(prisma_walker.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
#env = VecEnv(prisma_walker.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
control_dt = 0.01

weight_path = "/home/claudio/raisim_ws/raisimlib/raisimGymTorch/data/prisma_walker/noActions_noImuData/full_500.pt"
#weight_path = "/home/claudio/Downloads/materiale_tesi_ANTONIO_ZAMPA_PRISMA_WALKER/Materiale da consegnare/Gym_torch_urdf/raisimGymTorch/raisimGymTorch/data/prisma_walker_locomotion/best_train/y_0_yaw_0_full_0_y_maggiore_di_0_full_40_y_e_yaw_vanno_a_0/full_40.pt"

actualTorque_x = []
actualTorque_y = []
actualTorque_z = []
motorTorque_x = []
motorTorque_y = []
motorTorque_z = []

iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
#flag = env.motors_check_()
if (weight_path == ""):# or ( flag == 0):
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0
   

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim, cfg['environment']['num_envs'])
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    #env.command_vel(*args.velocity) #l'asterisco * serve a transformare [1,1,1] in 1,1,1"""

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 5000 ## 10 secs
    current_time=0
    counter = 0
    gc=[]
    bodyAngularVel_x = []
    bodyAngularVel_y = []
    bodyAngularVel_z = []

    q_1 = []
    q_2 = []
    q_3 = []

    dotq_1 = []
    dotq_2 = []
    dotq_3 = []
    obs_list = []
    #obs_l = torch.from_numpy(obs)
    for step in range(max_steps):
        if current_time == 0:
            time.sleep(1)
        else:
            time.sleep(0.01)
        if step == 0:
            time.sleep(3)

        obs = env.observe(False)
        obs_list.append(*obs)
       
        #obs_l = torch.cat(obs_l, obs)
        bodyAngularVel_x.append(*obs[:,0])
        bodyAngularVel_y.append(*obs[:,1])
        bodyAngularVel_z.append(*obs[:,2])

        q_1.append(*obs[:,6])
        q_2.append(*obs[:,7])
        q_3.append(*obs[:,8])

        dotq_1.append(*obs[:,9])
        dotq_2.append(*obs[:,10])
        dotq_3.append(*obs[:,11])

        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        actualTorque_x.append(env.getActualTorques()[0])
        motorTorque_x.append(env.getMotorTorques()[0])
 
        #gc = env.getError_vector()
        current_time = current_time + 0.01
      
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    time = np.arange(0, max_steps/100, control_dt, dtype='float32') 

    plt.figure()
    plt.plot(time, actualTorque_x, label="torque with PD")
    plt.plot(time, motorTorque_x, label="torque computed manually")
    plt.title('Torque varying the control mode')
    plt.xlabel('time')
    plt.ylabel('torque')
    plt.grid()
    plt.legend()
    plt.show()


    #print(obs_l.shape)
    print(obs_list[0][:3])

    plt.figure()
    plt.plot(time, q_1, label="q_1")
    plt.plot(time, q_2, label="q_2")
    #plt.plot(time, q_3, label="q_3")
    plt.title('joint positions')
    plt.xlabel('time')
    plt.ylabel('rad')
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(time, dotq_1, label="dotq_1")
    plt.plot(time, dotq_2, label="dotq_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('joint velocities')
    plt.xlabel('time')
    plt.ylabel('rad/s')
    plt.grid()
    plt.legend()
    plt.show()
