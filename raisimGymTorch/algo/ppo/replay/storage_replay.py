import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import random

class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, start_curriculum, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_obs = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.rewards = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.bool)

        # For PPO
        self.actions_log_prob = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)

        #REPLAY
        self.critic_obs_rep = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs_rep = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.rewards_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions_rep = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.bool)
        self.actions_log_prob_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages_rep = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu_rep = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma_rep = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)

        #REPLAY STAIRS
        self.critic_obs_rep_stairs = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs_rep_stairs = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.rewards_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions_rep_stairs = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.bool)
        self.actions_log_prob_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages_rep_stairs = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu_rep_stairs = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma_rep_stairs = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)


        # torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        # REPLY torch variables
        self.critic_obs_rep_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_rep_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_rep_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_rep_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_rep_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_rep_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_rep_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_rep_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_rep_tc = torch.from_numpy(self.sigma).to(self.device)

        #TORCH STAIRS
        self.critic_obs_rep_stairs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_rep_stairs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_rep_stairs_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_rep_stairs_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_rep_stairs_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_rep_stairs_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_rep_stairs_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_rep_stairs_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_rep_stairs_tc = torch.from_numpy(self.sigma).to(self.device)


        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device
        
        self.start_curriculum = start_curriculum
        self.current_max_return = 0 
        self.isStairs = False
        
        self.step = 0
        self.update = 0
        self.last_values_rep = 0

    def add_transitions(self, actor_obs, update, critic_obs, actions, mu, sigma, rewards, dones, actions_log_prob):
        
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step] = critic_obs
        self.actor_obs[self.step] = actor_obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)
        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        self.update = update
        self.step += 1

    def clear(self):
        self.step = 0



    def compute_returns(self, last_values, critic, gamma, lam, update, average_ll_performance):
        with torch.no_grad():
            self.values = critic.predict(torch.from_numpy(self.critic_obs).to(self.device)).cpu().numpy()

        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.cpu().numpy()
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

                next_is_not_terminal = 1.0 - self.dones[step]
                delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
                advantage = delta + next_is_not_terminal * gamma * lam * advantage
                self.returns[step] = advantage + self.values[step]
        

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Convert to torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)


        if update >= self.start_curriculum:
            with torch.no_grad():
                self.values_rep = critic.predict(torch.from_numpy(self.critic_obs_rep).to(self.device)).cpu().numpy()


            advantage_rep = 0
            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    #next_values_rep = self.last_values_rep 
                    next_values_rep = last_values.cpu().numpy()
                    # next_is_not_terminal = 1.0 - self.dones[step].float()
                else:
                    next_values_rep = self.values_rep[step + 1]
                    # next_is_not_terminal = 1.0 - self.dones[step+1].float()

                next_is_not_terminal_rep = 1.0 - self.dones_rep[step]
                delta_rep = self.rewards_rep[step] + next_is_not_terminal_rep * gamma * next_values_rep - self.values_rep[step]
                advantage_rep = delta_rep + next_is_not_terminal_rep * gamma * lam * advantage_rep
                self.returns_rep[step] = advantage_rep + self.values_rep[step]


            self.advantages_rep = self.returns_rep - self.values_rep
            self.advantages_rep = (self.advantages_rep - self.advantages_rep.mean()) / (self.advantages_rep.std() + 1e-8)

            #VARIABLES UPDATES DURING THE BATCH REPLAY
            self.values_rep_tc = torch.from_numpy(self.values_rep).to(self.device)
            self.returns_rep_tc = torch.from_numpy(self.returns_rep).to(self.device)
            self.advantages_rep_tc = torch.from_numpy(self.advantages_rep).to(self.device)

        
        #At the end of the training episode, i say "mmmh, is the tensor that I filled worth to be replayed? If yes, I save it in this torch tensor, otherwise not"
        #In this If there are the variables that I save AND NEVER again re-update
        if update >=50 and update <= (self.start_curriculum):
            if(average_ll_performance > self.current_max_return):
                print("Best episode! Save this experiences for replay")
                self.current_max_return = average_ll_performance
                print("Max returns log: ", self.current_max_return)

                #COPY THIS BATCH OF EXPERIENCE TO REPLAY AFTERWARDS
                self.critic_obs_rep = self.critic_obs.copy()
                self.actor_obs_rep = self.actor_obs.copy()
                self.actions_rep = self.actions.copy()
                self.rewards_rep = self.rewards.copy().reshape(-1, 1)
                self.dones_rep = self.dones.copy().reshape(-1, 1)
                self.actions_log_prob_rep = self.actions_log_prob.copy().reshape(-1, 1)
                self.mu_rep = self.mu.copy()
                self.sigma_rep = self.sigma.copy()

                #COPY THE SAME BATCH ON TORCH VARIABLES
                self.critic_obs_rep_tc = torch.from_numpy(self.critic_obs_rep).to(self.device)
                self.actor_obs_rep_tc = torch.from_numpy(self.actor_obs_rep).to(self.device)
                self.actions_rep_tc = torch.from_numpy(self.actions_rep).to(self.device)
                self.rewards_rep_tc = torch.from_numpy(self.rewards_rep).to(self.device)
                self.actions_log_prob_rep_tc = torch.from_numpy(self.actions_log_prob_rep).to(self.device)
                self.mu_rep_tc = torch.from_numpy(self.mu_rep).to(self.device)
                self.sigma_rep_tc =torch.from_numpy(self.sigma_rep).to(self.device)
                
                #self.last_values_rep = last_values.cpu().numpy()


        #At the end of the day, is this pytorch tensor that is used to update the parameters of the neural network

        """
        if(self.isStairs == True):
            print("Replay on stairs")
            self.critic_obs_rep_stairs_tc = torch.from_numpy(self.critic_obs).to(self.device)
            self.actor_obs_rep_stairs_tc = torch.from_numpy(self.actor_obs).to(self.device)
            self.actions_rep_stairs_tc = torch.from_numpy(self.actions).to(self.device)
            self.rewards_rep_stairs_tc = torch.from_numpy(self.rewards).to(self.device)
            self.actions_log_prob_rep_stairs_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
            self.mu_rep_stairs_tc = torch.from_numpy(self.mu).to(self.device)
            self.sigma_rep_stairs_tc =torch.from_numpy(self.sigma).to(self.device)
            self.values_rep_stairs_tc = torch.from_numpy(self.values).to(self.device)
            self.returns_rep_stairs_tc = torch.from_numpy(self.returns).to(self.device)
            self.advantages_rep_stairs_tc = torch.from_numpy(self.advantages).to(self.device)
        """


    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches  
        #Campiona randomicamente, da una batch di 40000 elementi, una mini-batch di esperienza che e' 10000
        #Dato un vettore (400000,100) ti campiona una matrice (100000,100) randomicamente
        if self.update < 100000000: #self.start_curriculum:
            for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
                actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
                critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
                actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
                sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
                mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
                values_batch = self.values_tc.view(-1, 1)[indices]
                returns_batch = self.returns_tc.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
                advantages_batch = self.advantages_tc.view(-1, 1)[indices]
                yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch
        else:
            for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
                if random.randint(0,4) != 3:#In questo modo, per ogni epoca, 
                    actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
                    critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
                    actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
                    sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
                    mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
                    values_batch = self.values_tc.view(-1, 1)[indices]
                    returns_batch = self.returns_tc.view(-1, 1)[indices]
                    old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
                    advantages_batch = self.advantages_tc.view(-1, 1)[indices]
                    yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch
                else:
                    print('replay')
                    actor_obs_batch = self.actor_obs_rep_tc.view(-1, *self.actor_obs_rep_tc.size()[2:])[indices]
                    critic_obs_batch = self.critic_obs_rep_tc.view(-1, *self.critic_obs_rep_tc.size()[2:])[indices]
                    actions_batch = self.actions_rep_tc.view(-1, self.actions_rep_tc.size(-1))[indices]
                    sigma_batch = self.sigma_rep_tc.view(-1, self.sigma_rep_tc.size(-1))[indices]
                    mu_batch = self.mu_rep_tc.view(-1, self.mu_rep_tc.size(-1))[indices]
                    values_batch = self.values_rep_tc.view(-1, 1)[indices]
                    returns_batch = self.returns_rep_tc.view(-1, 1)[indices]
                    old_actions_log_prob_batch = self.actions_log_prob_rep_tc.view(-1, 1)[indices]
                    advantages_batch = self.advantages_tc.view(-1, 1)[indices]
                    yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch




    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        self.start_curriculum = 1000000

        if self.update < self.start_curriculum:
            for batch_id in range(num_mini_batches):
                yield self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.actions_tc.view(-1, self.actions_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.sigma_tc.view(-1, self.sigma_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.mu_tc.view(-1, self.mu_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.values_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.advantages_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.returns_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                    self.actions_log_prob_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
        else:
            for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
                if random.randint(0,9) != 3:#In questo modo, per ogni epoca, 
                    actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
                    critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
                    actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
                    sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
                    mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
                    values_batch = self.values_tc.view(-1, 1)[indices]
                    returns_batch = self.returns_tc.view(-1, 1)[indices]
                    old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
                    advantages_batch = self.advantages_tc.view(-1, 1)[indices]
                    yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch
                else:
                    print('replay')
                    actor_obs_batch = self.actor_obs_rep_tc.view(-1, *self.actor_obs_rep_tc.size()[2:])[indices]
                    critic_obs_batch = self.critic_obs_rep_tc.view(-1, *self.critic_obs_rep_tc.size()[2:])[indices]
                    actions_batch = self.actions_rep_tc.view(-1, self.actions_rep_tc.size(-1))[indices]
                    sigma_batch = self.sigma_rep_tc.view(-1, self.sigma_rep_tc.size(-1))[indices]
                    mu_batch = self.mu_rep_tc.view(-1, self.mu_rep_tc.size(-1))[indices]
                    values_batch = self.values_rep_tc.view(-1, 1)[indices]
                    returns_batch = self.returns_rep_tc.view(-1, 1)[indices]
                    old_actions_log_prob_batch = self.actions_log_prob_rep_tc.view(-1, 1)[indices]
                    advantages_batch = self.advantages_tc.view(-1, 1)[indices]
                    yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch
