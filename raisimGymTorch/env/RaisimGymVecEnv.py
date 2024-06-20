# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        self.vector = np.zeros(10)
        self.success = 0
        
        self.actualTorques = np.zeros(3, dtype=np.float32)
        self.motorTorques = np.zeros(3, dtype=np.float32)
        self.q_ref = np.zeros(3, dtype=np.float32)
        self.q = np.zeros(3, dtype=np.float32)
        self.dotq = np.zeros(3, dtype=np.float32)
        self.ddotq = np.zeros(3, dtype=np.float32)
        self.yaw = np.zeros(3, dtype=np.float32)
        self.pitch = np.zeros(3, dtype=np.float32)
        self.bodyAngularVel = np.zeros(3, dtype=np.float32)
        self.currentAction = np.zeros(3, dtype=np.float32)
 

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def getError_vector(self):
        self.vector = self.wrapper.getError()
        return self.vector
    def motors_check_(self):
        self.success = self.wrapper.motors_check()
        return self.success
    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()
    
    def command_vel(self, omega_z):
        self.wrapper.command_vel(omega_z)

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def getMotorTorques(self):
        self.wrapper.getMotorTorques(self.motorTorques)
        return self.motorTorques

    def getActualTorques(self):
         self.wrapper.getActualTorques(self.actualTorques)
         return self.actualTorques

    def getReferences(self):
         self.wrapper.getpTarget(self.q_ref)
         return self.q_ref
    
    def getPositions(self):
         self.wrapper.getJointPositions(self.q)
         return self.q

    def getVelocities(self):
         self.wrapper.getJointVelocities(self.dotq)
         return self.dotq    

    def getAccelerations(self):
         self.wrapper.getJointAccelerations(self.ddotq)
         return self.ddotq   

    def getPitch(self):
         self.wrapper.getPitch(self.pitch)
         return self.pitch
    
    def getYaw(self):
         self.wrapper.getYaw(self.yaw)
         return self.yaw

    def getAngularVel(self):
         self.wrapper.getAngularVel(self.bodyAngularVel)
         return self.bodyAngularVel  

    def getCurrentAction(self):
         self.wrapper.getCurrentAction(self.currentAction)
         return self.currentAction 

        
    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
