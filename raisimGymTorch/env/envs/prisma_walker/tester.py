from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.prisma_walker import RaisimGymEnv
#from raisimGymTorch.env.bin.prisma_walker import GenCoordFetcher
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
import matplotlib 
from threading import Thread
matplotlib.use('agg')
from matplotlib import pyplot as plt

actualTorque_x = []
motorTorque_x = []
q_1 = []
q_2 = []

pTarge_x = []
pTarge_y = []

dotq_1 = []
dotq_2 = []

ddotq_1 = []
ddotq_2 = []

obs_list = []

yaw = []
pitch = []
bodyAngularVel = []
currentAction = []

#simulation_time = np.arange(0, max_steps/100, 0.01, dtype='float32') 
simulation_time = []
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
plot_dir = task_path + "/plots/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def runNN(max_steps):
  
    if (weight_path == ""):# or ( flag == 0):
        print("Can't find trained weight, please provide a trained weight with --weight switch\n")
    else:
        print("Loaded weight from {}\n".format(weight_path))
        #start = time.time()
        env.reset()
        reward_ll_sum = 0
        start_step_id = 0

        print("Visualizing and evaluating the policy: ", weight_path)
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim, cfg['environment']['num_envs'])
        loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

        #env.command_vel(*args.velocity) #l'asterisco * serve a transformare [1,1,1] in 1,1,1"""

        env.load_scaling(weight_dir, int(iteration_number))
        env.turn_on_visualization()

        current_time= 0
 
        """traj_x = []
        with open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor1.txt") as file:
            traj_x = [line.strip() for line in file]

        traj_y = []
        with open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor2.txt") as file:
            traj_y = [line.strip() for line in file]"""

        #obs_l = torch.from_numpy(obs)
        for step in range(max_steps):
            if current_time == 0:
                time.sleep(1)
            else:
                time.sleep(0.01)
            if step == 0:
                time.sleep(1)

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
            simulation_time.append(current_time)
            #time.sleep(0.01)
            if dones or step == max_steps - 1:
                print('----------------------------------------------------')
                print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
                print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
                print('----------------------------------------------------\n')
                start_step_id = step + 1
                reward_ll_sum = 0.0


        """SAVE INTO A FILE
        with open(r'/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor1.txt', 'w') as fp:
            for item in pTarge_x:
                fp.write("%s\n" % item)
        
        with open(r'/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/trajectory_motor2.txt', 'w') as fp:
            for item in pTarge_y:
                fp.write("%s\n" % item)"""


def plotVariables(max_steps):

    plotDatas(simulation_time, actualTorque_x, motorTorque_x, "implicitly integrated torque", "explitly integrated torque", "joints torques", "$N/m$")
    plotDatas(simulation_time, q_1, pTarge_x, "$q_1$", "$\hat{q}_{1}$", "m1 joint positions", "rad")
    plotDatas(simulation_time, q_2, pTarge_y, "$q_2$", "$\hat{q}_{2}$", "m2 joint positions", "rad")
    plotDatas(simulation_time, dotq_1, dotq_2, "$\dot{q}_1$", "$\dot{q}_2$", "joint velocities", "rad/s")
    plotDatas(simulation_time, ddotq_1, ddotq_1, "$\ddot{q}_1$", "$\ddot{q}_2$", "joint accelerations", "rad/$s^2$")

def plotDatas(t, x, y, label1, label2, title, ylabel):
  
    plt.figure()
    plt.plot(t, x, label=label1)
    plt.plot(t, y, label=label2)
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(plot_dir + title + ".png")



if __name__ == '__main__':
        # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
    parser.add_argument('-v', '--velocity', nargs='+', help = 'command velocity for the quadruped', type=float, default = [0.0]) #nargs take more than 1 argument
    parser.add_argument('-env', '--environment',  help='if the stairs spawn or not', type=bool, default=False)
    parser.add_argument('-h', '--stepHeight',  help='step height', type=float, default=[0.01])

    args = parser.parse_args()

    print(args.velocity)
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
    env.command_vel(*args.velocity)
    if(args.environment == True):
        env.select_terrain_from_tester()
        env.select_heightMap(*args.stepHeight)

    # shortcuts
    ob_dim = env.num_obs
    act_dim = env.num_acts
    control_dt = 0.01

    weight_path = "/home/claudio/raisim_ws/raisimlib/raisimGymTorch/data/prisma_walker_server/full_600.pt"

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    #duration in ms of the simulation
    max_steps = 4000
    nnthread = Thread(target = runNN, args= (max_steps, ))
    plotThread = Thread(target = plotVariables, args= (max_steps, ))

    try: 
        nnthread.start()
        nnthread.join()
    except KeyboardInterrupt:
        plotThread.start()
        plotThread.join()