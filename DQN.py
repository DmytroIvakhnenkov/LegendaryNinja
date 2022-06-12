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
import cv2

START_EPSILON = 1
END_EPSILON = 0.1
START_LEARNING_RATE = 1e-4
END_LEARNING_RATE = 1e-7
NUMBER_OF_ITERATIONS = 100000

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class replay_buffer():

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.num_actions = 4  # the dimension of action space
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(2592, 256)  # output layer
        self.fc2 = nn.Linear(256, self.num_actions)  # output layer

    def forward(self, states):
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = torch.reshape(x, (-1, 2592))
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=START_EPSILON, learning_rate=START_LEARNING_RATE, GAMMA=0.99, batch_size=32, capacity=100000):

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

        if self.count % 20 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())


        # Begin your code
        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        
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
    

    def choose_action(self, stack):

        with torch.no_grad():
            # Begin your code
            explore = np.random.choice([0, 1], p=[1-self.epsilon, self.epsilon])
            if(explore):
                return random.choice(env.action_space)
            else:
                Q = self.target_net.forward(torch.FloatTensor(stack)).squeeze(0).detach()
                action = int(torch.argmax(Q).numpy())
                return action
            
            # End your code

    def check_max_Q(self, starting_stack):
        # Begin your code
        Q = self.target_net.forward(
                torch.FloatTensor(starting_stack)).squeeze(0).detach()

        return float(torch.max(Q).numpy())
        # End your code


def train(env):

    agent = Agent(env)
    episode = 300
    steps = []
    loss = []
    maxQ = []

    state = env.reset()
    # resize the input and convert it to gray 
    state  = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
    # create variables to stack last 4 frames
    stack = np.stack((state, state, state, state), axis=0)
    starting_stack = stack
    next_stack = np.stack((state, state, state, state), axis=0)
    total_steps = 0
    
    for i in tqdm(range(NUMBER_OF_ITERATIONS+1000)):

        agent.count += 1
        action = agent.choose_action(stack)
        next_state, reward, done = env.step(action)
        total_steps += 1
        # resize the input and convert it to gray
        next_state  = cv2.cvtColor(cv2.resize(next_state, (84, 84)), cv2.COLOR_BGR2GRAY)
        next_state = np.reshape(next_state, (1, 84, 84))
        # add a new frame to the stack
        next_stack = np.append(next_state, next_stack[:3, :, :], axis=0)
        # insert a memory
        agent.buffer.insert(stack, action, reward, next_stack, int(done))
        # start learning, if there is enough memories 
        if agent.count >= 1000:
            loss.append(agent.learn().detach().numpy())
            # reduce the exploration rate
            agent.epsilon = agent.epsilon * pow((END_EPSILON/START_EPSILON), 1/NUMBER_OF_ITERATIONS)
            # reduce the learning rate
            agent.learning_rate = agent.learning_rate * pow((END_LEARNING_RATE/START_LEARNING_RATE), 1/NUMBER_OF_ITERATIONS)
        # record the maximum Q of the starting state
        maxQ.append(agent.check_max_Q(starting_stack))
        # next state is now the current state
        stack = next_stack
        # reset the game, if it is game over 
        if done:
            steps.append(total_steps)

            # RESET
            state = env.reset()
            # resize the input and convert it to gray 
            state  = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
            # create variables to stack last 4 frames
            stack = np.stack((state, state, state, state), axis=0)
            starting_stack = stack
            next_stack = np.stack((state, state, state, state), axis=0)
            total_steps = 0


    torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")
    
    plt.figure("MaxQ")
    plt.plot(maxQ)
    plt.figure("Loss")
    plt.plot(loss)
    plt.figure("Total Steps")
    plt.plot(steps)
    plt.show()


def test(env):

    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(3):
        state = env.reset()
        # resize the input and convert it to gray 
        state  = cv2.cvtColor(cv2.resize(state, (84, 84)), cv2.COLOR_BGR2GRAY)
        # create variables to stack last 4 frames
        stack = np.stack((state, state, state, state), axis=0)
        next_stack = np.stack((state, state, state, state), axis=0)
        count = 0
        while True:
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(stack)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done = env.step(action)
            # resize the input and convert it to gray
            next_state  = cv2.cvtColor(cv2.resize(next_state, (84, 84)), cv2.COLOR_BGR2GRAY)
            next_state = np.reshape(next_state, (1, 84, 84))
            # add a new frame to the stack
            next_stack = np.append(next_state, next_stack[:3, :, :], axis=0)
            count = count + 1
            if done:
                rewards.append(count)
                break
            stack = next_stack

    print(f"reward: {np.mean(rewards)}")



if __name__ == "__main__":

    env = LegendaryNinja_v0()

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
    #testing section:
    env = LegendaryNinja_v0()
    test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()