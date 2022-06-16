import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LegendaryNinja_v0 import LegendaryNinja_v0
import random
from collections import deque
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

total_rewards = []

START_EPSILON = 1
END_EPSILON = 0.01
START_LEARNING_RATE = 1e-6
END_LEARNING_RATE = 1e-6
NUMBER_OF_START_ITERATINS = 10000
NUMBER_OF_ITERATIONS = 200000
NUMBER_OF_ITERATIONS_C = 100000

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
        self.input_state = 18  # the dimension of state space
        self.num_actions = 4  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 512)  # input layer
        self.fc2 = nn.Linear(512, 512)  # hidden layer
        self.fc3 = nn.Linear(512, 512)  # hidden layer
        self.fc4 = nn.Linear(512, self.num_actions)  # output layer

    def forward(self, states):

        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class Agent():
    def __init__(self, env, device, epsilon=START_EPSILON, learning_rate=START_LEARNING_RATE, GAMMA=0.99, batch_size=32, capacity=200000):
        self.device = device
        self.env = env
        self.n_actions = 4  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net().to(device)  # the evaluate network
        self.target_net = Net().to(device)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):

        if self.count % 20 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        #soft_update(self.target_net, self.evaluate_net, 0.005)

        # Begin your code
        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
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


        with torch.no_grad():
            # Begin your code
            explore = np.random.choice([0, 1], p=[1-self.epsilon, self.epsilon])
            if(explore):
                return random.choice(env.action_space)
            else:
                Q = self.target_net.forward(torch.FloatTensor(state).to(self.device)).squeeze(0).detach()
                action = int(torch.argmax(Q).cpu().numpy())
                return action
            
            # End your code

    def check_max_Q(self, state):

        # Begin your code
        Q = self.target_net.forward(
                torch.FloatTensor(state).to(self.device)).squeeze(0).detach()

        return float(torch.max(Q).cpu().numpy())
        # End your code


def train(env, device):

    agent = Agent(env, device)
    steps = []
    loss = []
    maxQ = []
    state = env.reset()
    starting_state = state
    total_steps = 0
    e_reduction = pow((END_EPSILON/START_EPSILON), 1/NUMBER_OF_ITERATIONS)
    lr_reduction = pow((END_LEARNING_RATE/START_LEARNING_RATE), 1/NUMBER_OF_ITERATIONS)
    for i in tqdm(range(NUMBER_OF_ITERATIONS+NUMBER_OF_START_ITERATINS+NUMBER_OF_ITERATIONS_C)):
        agent.count += 1
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_steps += 1
        # insert a memory
        agent.buffer.insert(state, action, reward, next_state, int(done))
        # start learning, if there is enough memories 
        if agent.count >= NUMBER_OF_START_ITERATINS and agent.count < NUMBER_OF_START_ITERATINS+NUMBER_OF_ITERATIONS:
            loss.append(agent.learn().detach().cpu().numpy())
            # reduce the exploration rate
            agent.epsilon = agent.epsilon * e_reduction
            # reduce the learning rate
            agent.learning_rate = agent.learning_rate * lr_reduction
        if agent.count >= NUMBER_OF_START_ITERATINS+NUMBER_OF_ITERATIONS:
            loss.append(agent.learn().detach().cpu().numpy())
            if agent.count % 20 == 0 and is_game_solved(env, device, agent.target_net):
                break
        # record the maximum Q of the starting state
        maxQ.append(agent.check_max_Q(starting_state))
        # next state is now the current state
        state = next_state
        # reset the game, if it is game over 
        if done:
            steps.append(total_steps)
            # RESET
            state = env.reset()
            total_steps = 0


    torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")
    
    plt.figure("MaxQ")
    plt.ylabel("Q-value of the starting state")
    plt.xlabel("Iterations")
    plt.plot(maxQ)
    plt.savefig('maxQ.png')

    plt.figure("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.plot(loss)
    plt.savefig('Loss.png')

    plt.figure("Total Steps")
    plt.ylabel("Total steps (max:2000)")
    plt.xlabel("Game number")
    plt.plot(steps)
    plt.savefig('Total Steps.png')

    plt.show()

def is_game_solved(env, device, target_net):
    env.save()
    state = env.reset()
    c = 0
    testing_agent = Agent(env, device)
    testing_agent.target_net.load_state_dict(target_net.state_dict())
    testing_agent.epsilon = 0
    while True:
        action = testing_agent.choose_action(state)
        next_state, _, done = env.step(action)
        c += 1
        if done:
            break
        state = next_state
        
    env.load()

    if (c > 2000):
        return True
    return False

def test(env, device):

    count = 0
    testing_agent = Agent(env, device)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))

    state = env.reset()
    while True:
        Q = testing_agent.target_net.forward(
            torch.FloatTensor(state).to(device)).squeeze(0).detach()
        action = int(torch.argmax(Q).cpu().numpy())
        next_state, _, done = env.step(action)
        count = count + 1
        if done:
            break
        state = next_state

    print(f"Steps: {(count)}")


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = LegendaryNinja_v0(render=False)

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env, device)
    #testing section:
    env = LegendaryNinja_v0(render=True)
    test(env, device)


    env.close()