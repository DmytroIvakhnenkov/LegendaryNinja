import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LegendaryNinja_v0 import LegendaryNinja_v0
import random
from collections import deque
import os
from tqdm import tqdm
total_rewards = []
import matplotlib.pyplot as plt

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.input_state = 18  # the dimension of state space
        self.num_actions = 4  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 512)  # input layer
        self.fc2 = nn.Linear(512, 512)  # hidden layer
        self.fc3 = nn.Linear(512, 512)  # hidden layer
        self.fc4 = nn.Linear(512, 512)  # hidden layer
        self.fc5 = nn.Linear(512, 512)  # hidden layer
        self.fc6 = nn.Linear(512, self.num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        
        Parameter:
            states: a batch size of states
        
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        q_values = self.fc6(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=1e-4, GAMMA=0.999, batch_size=64, capacity=10000):
        """ 
        The agent learning how to control the action of the cart pole.
        
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 4  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net()  # the evaluate network
        self.target_net = Net()  # the target network

        soft_update(self.target_net, self.evaluate_net, 1)

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            None (Don't need to return anything)
        '''
        #if self.count % 20 == 0:
        #    self.target_net.load_state_dict(self.evaluate_net.state_dict())
        soft_update(self.target_net, self.evaluate_net, 0.005)

        # Begin your code
        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        #print(rewards)
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(actions)
        next_observations = torch.FloatTensor(np.array(next_observations))
        rewards = torch.FloatTensor(rewards)
        
        actions = actions.unsqueeze(-1)
        state_action_values  = self.evaluate_net.forward(observations) 
        state_action_values  = state_action_values.gather(1, actions)
        
        next_state_values = self.target_net(next_observations).max(1)[0].detach()
        for i in range(len(done)):
            if(done[i]):
                next_state_values[i] = torch.FloatTensor([0])

        expected_state_action_values = (next_state_values * self.gamma) + rewards
        

        state_action_values  = state_action_values.squeeze()
        MSE = nn.MSELoss()
        loss = MSE(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.evaluate_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # End your code

        return loss
    

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            action: the chosen action.
        """

        with torch.no_grad():
            # Begin your code
            explore = np.random.choice([0, 1], p=[1-self.epsilon, self.epsilon])
            if(explore):
                return random.choice(env.action_space)
            else:
                Q = self.target_net.forward(torch.FloatTensor(state)).squeeze(0).detach()
                action = int(torch.argmax(Q).numpy())
                return action
            
            # End your code

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        Q = self.target_net.forward(
                torch.FloatTensor(self.env.reset())).squeeze(0).detach()

        return float(torch.max(Q).numpy())
        # End your code


def train(env):
    """
    Train the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    episode = 100
    rewards = []
    loss = []
    
    for i in tqdm(range(episode)):
        #if(i % 20 == 19):
         #   agent.epsilon = agent.epsilon - 0.01
         #   agent.learning_rate = agent.learning_rate * 0.5
        state = env.reset()
        count = 0
        while True:
            agent.count += 1
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            count += reward
            agent.buffer.insert(state, action, reward,
                                next_state, int(done))
            if agent.count >= 1000:
                loss.append(agent.learn().detach().numpy())
            if done or count > 200:
                rewards.append(count)
                break
            state = next_state
            #if count > 50 and agent.learning_rate == 1e-4:
            #    agent.learning_rate = agent.learning_rate/10
            #if count > 100 and agent.learning_rate == 1e-5:
           #     agent.learning_rate = agent.learning_rate/10
            #agent.target_net.load_state_dict(agent.evaluate_net.state_dict())

    torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")
    
    plt.figure("Loss")
    plt.plot(loss)
    plt.figure("Rewards")
    plt.plot(rewards)
    plt.show()


def test(env):
    """
    Test the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(10):
        state = env.reset()
        count = 0
        while True:
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, r, done = env.step(action)
            count = count + r
            if done:
                rewards.append(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    '''
    The main funtion
    '''
    env = LegendaryNinja_v0(render=False)

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
    #testing section:
    env = LegendaryNinja_v0(render=True)
    test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()