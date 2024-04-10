from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.prisma_walker import NormalSampler
from raisimGymTorch.env.bin.prisma_walker import RaisimGymEnv
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import numpy as np
import torch
import datetime
import argparse
import pickle 
import matplotlib.pyplot as plt
from threading import Thread


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

env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
control_dt = 0.01

weight_path = "/home/claudio/raisim_ws/raisimlib/raisimGymTorch/data/prisma_walker/2joint2/full_3000.pt"

actualTorque_x = []
motorTorque_x = []


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
    max_steps = 4000
    current_time= 0
    counter = 0

    bodyAngularVel_x = []
    bodyAngularVel_y = []
    bodyAngularVel_z = []

    traj_x = []
    with open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor1.txt") as file:
        traj_x = [line.strip() for line in file]

    traj_y = []
    with open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor2.txt") as file:
        traj_y = [line.strip() for line in file]

    q_1 = []
    q_2 = []

    dotq_1 = []
    dotq_2 = []

    ddotq_1 = []
    ddotq_2 = []

    ddotq_1 = []
    ddotq_2 = []
    
    pTarge_x = []
    pTarge_y = []
    obs_list = []

    yaw = []
    pitch = []
    bodyAngularVel = []
    currentAction = []

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

        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())

        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        q_1.append(env.getPositions()[0])
        q_2.append(env.getPositions()[1])

        dotq_1.append(env.getVelocities()[0])
        dotq_2.append(env.getVelocities()[1])

        ddotq_1.append(env.getAccelerations()[0])
        ddotq_2.append(env.getAccelerations()[1])

        actualTorque_x.append(env.getActualTorques()[0])
        motorTorque_x.append(env.getMotorTorques()[0])

        pTarge_x.append(env.getReferences()[0])
        #pTarge_x.append(traj_x[step])
        pTarge_y.append(env.getReferences()[1])

        bodyAngularVel.append(env.getAngularVel()[0])
        currentAction.append(env.getCurrentAction()[0])
        pitch.append(env.getPitch()[0])
        yaw.append(env.getYaw()[0])

        #gc = env.getError_vector()
        current_time = current_time + 0.01
        #time.sleep(0.01)
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    time = np.arange(0, max_steps/100, control_dt, dtype='float32') 

    """SAVE INTO A FILE
    with open(r'/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor1.txt', 'w') as fp:
        for item in pTarge_x:
            fp.write("%s\n" % item)
    
    with open(r'/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor2.txt', 'w') as fp:
        for item in pTarge_y:
            fp.write("%s\n" % item)"""

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
    plt.plot(time, q_1, label="$q_1$")
    plt.plot(time, pTarge_x, label="$\hat{q}_{1}$")
    #plt.plot(time, q_3, label="q_3")
    plt.title('joint positions')
    plt.xlabel('time')
    plt.ylabel('rad')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, q_2, label="$q_2$")
    plt.plot(time, pTarge_y, label="$\hat{q}_{2}$")
    plt.title('joint reference')
    plt.xlabel('time')
    plt.ylabel('rad')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, dotq_1, label="\dot{q}_1")
    plt.plot(time, dotq_2, label="\dot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('joint velocities')
    plt.xlabel('time')
    plt.ylabel('rad/s')
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(time, ddotq_1, label="\ddot{q}_1")
    plt.plot(time, ddotq_2, label="\ddot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('joint accelerations')
    plt.xlabel('time')
    plt.ylabel('rad/$s^2$')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, yaw, label="$\theta$")
    #plt.plot(time, ddotq_2, label="\ddot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('y axis direction')
    plt.xlabel('time')
    plt.ylabel('rad/$s^2$')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, pitch, label="$\phi$")
    #plt.plot(time, ddotq_2, label="\ddot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('z axis direction')
    plt.xlabel('time')
    plt.ylabel('rad/$s^2$')
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(time, currentAction, label="$a_t$")
    #plt.plot(time, ddotq_2, label="\ddot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('currentAction')
    plt.xlabel('time')
    plt.ylabel('rad/$s^2$')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, bodyAngularVel, label="$\omega_x$")
    #plt.plot(time, ddotq_2, label="\ddot{q}_2")
    #plt.plot(time, dotq_3, label="dotq_3")
    plt.title('angular velocity')
    plt.xlabel('time')
    plt.ylabel('rad/$s^2$')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Thread(target = plotFunction).start()
    Thread(target = func2).start()