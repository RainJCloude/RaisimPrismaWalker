import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture #This is the object of the class. From this you need the attribute architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None
    
    def sample(self, obs):

        self.action_mean = self.architecture.architecture(obs).cpu().numpy()
        #print("obs: ", obs[0,:])
        #print("action mean: ", self.action_mean)

        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions) #actions are the batch of action that we stored. action_mean are the actual output of the network when you insert the batch of obs

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class ActorRnn(Actor):
    def __init__(self, architecture, distribution, device='cpu'):
        super(ActorRnn, self).__init__(architecture, distribution, device=device)


    def sample(self, obs):
        #compute logit
        "obs is (num_envs, num_obs)"
        obs.reshape(obs[0], num_seq, -1) #(batch, Lenght, input_size)
        h0 = torch.zeros(1, obs[0], shape[0]) #need a forward method
        self.action_mean = self.architecture.architecture(obs).cpu().numpy()
        #print("obs: ", obs[0,:])
        #print("action mean: ", self.action_mean)

        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob
    

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape



class Reshape(nn.Module): #here i define or a complete NN or a set of layer to give to sequences
    def __init__(self, shape, num_envs):
        super(Reshape, self).__init__()
        self.num_envs = num_envs
        self.shape = shape
        self.batch = nn.BatchNorm1d(shape)

    def forward(self, obs): #here obs is the output of the layer before this. If i print its shape i get (100, 256) where 256 is the number of output features from the second hidden layer
        #the obs here is not the input, but the output of the previous layer
        if(len(obs.shape) == 2):
            if(obs.shape[0] == 7):
                zeros = torch.zeros([99, self.shape])
                x = torch.cat((obs, zeros), 0) #concatenate along channel zero
                return self.batch(x)
            else:
                return self.batch(obs)
        else:
            obs = obs.view(-1, obs.shape[-1]) #(800, 100, 256)
            return self.batch(obs)

class ReturnShape(nn.Module): #here i define or a complete NN or a set of layer to give to sequences
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, obs): #Forward is called during inference
        #print(obs.shape)  #the obs here is not the input, but the output of the previous layer
        if(obs.shape[0] == 150): #forwarding pass
            return obs
        elif(obs.shape[0] == 75000): #for the computation of the value loss term
            obs = obs.view(-1, 150, obs.shape[-1])
            return obs
        elif(obs.shape[0] == 18750): #training in whihc split the dataset
            obs = obs.view(-1, obs.shape[-1])
            return obs
 
class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, num_envs):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]

        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))  #linear can receive any tensor in input, the important is that its last channel has the correct dimension
            #modules.append(Reshape(shape[idx+1], num_envs))
            modules.append(self.activation_fn())
            modules.append(nn.Dropout(0.4))#After the activation, unless you use RELU. In that case is influent
            #Dropouts scales the output of the NN to preserve the value of the output if we didn' kill any perceptron 
            scale.append(np.sqrt(2))


        modules.append(nn.Linear(shape[-1], output_size))
        #modules.append(ReturnShape(output_size,  num_envs))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class RNN(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, num_envs):
        super().__init__()
        modules = [nn.RNN(input_size, shape[0], num_layers = 1, batch_first = True), self.activation_fn()]

        ho = torch.zeros(1, num_envs, shape[0]) #num_layers, batch, hidden_size
        c0 = torch.zeros(1, num_envs, shape[0])

        scale = [np.sqrt(2)]
        for idx in range(len(shape)-1):
            modules.append(nn.RNN(shape[idx], shape[idx+1], num_layers = 1, batch_first = True))  #linear can receive any tensor in input, the important is that its last channel has the correct dimension
            modules.append(self.activation_fn())
            modules.append(nn.Dropout(0.4))#After the activation, unless you use RELU. In that case is influent
            #Dropouts scales the output of the NN to preserve the value of the output if we didn' kill any perceptron 
            scale.append(np.sqrt(2))
        
        #output Layer
        modules.append(nn.Linear(shape[-1], output_size))

        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]



class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits): #for the step forward of the NN uses the custom function to sample from the distribution. The sample is given from the mean plus a random noise * the standard deviation
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob) #logprob return log(pi(s | logits))
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs): #For the backpropagation uses the Normal class provided by pytorch
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
