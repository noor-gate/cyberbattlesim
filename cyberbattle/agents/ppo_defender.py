import logging
import sys
from typing import Optional, Tuple
from gym import Env
import numpy as np
import progressbar
import torch
import torch.cuda
from cyberbattle._env import cyberbattle_env
from cyberbattle.agents.agent_ppo import ActorCritic, RolloutBuffer
from cyberbattle.agents.baseline.agent_dql import ChosenActionMetadata
import torch.nn as nn

from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, EnvironmentBounds, Feature_reimaged_node, Verbosity
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle._env.defender import DefenderAgent
from cyberbattle.agents.baseline.learner import Breakdown, PolicyStats
from cyberbattle.simulation import model
from cyberbattle.simulation.actions import DefenderAgentActions
from cyberbattle.simulation.model import Environment, NodeID

device = torch.device('cpu')
#if (torch.cuda.is_available()):
#    device = torch.device('cuda:0')
#    torch.cuda.empty_cache()


TRAIN_EPISODE_COUNT = 10
ITERATION_COUNT = 500


class PPODefender():
    def __init__(self,
                 ep: EnvironmentBounds, gamma: float, env: cyberbattle_env.CyberBattleEnv):
        self.ep = ep
        self.state_space = Feature_reimaged_node(ep)
        self.action_space = 2 * len(list(env.environment.network.nodes))
        self.state_space_size = 2 * len(list(env.environment.network.nodes))
        # print("state space", self.state_space)
        self.policy_old = ActorCritic(self.state_space_size, self.action_space)
        self.policy = ActorCritic(self.state_space_size, self.action_space)
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.K_epochs = 40
        self.eps_clip = 0.2
        lr_actor = 0.000003       # learning rate for actor network
        lr_critic = 0.001
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

    def step(self, wrapped_env: AgentWrapper, observation, actions: DefenderAgentActions):
        
        with torch.no_grad():
            state = self.state_space.get(wrapped_env.state, node=None)
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state).to(device)
            abstract_action, action_logprob, state_val = self.policy_old.act(state)
            # print(abstract_action)

        self.buffer.states.append(torch.tensor(state).to(device))
        self.buffer.actions.append(abstract_action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        space = 0.5 * self.action_space

        #print(abstract_action)
        if abstract_action < space:
            node_id = wrapped_env.env.get_nodeid_from_index(abstract_action)
            node_info = wrapped_env.env.get_node(node_id)
            if node_info.status == model.MachineStatus.Running:
                logging.info(f"Defender detected malware, reimaging node {abstract_action}")
                actions.fix_vulnerability(node_id)
                #  print("reimging", abstract_action)
                return node_info.value
        elif abstract_action >= space and abstract_action < self.action_space:
            abstract_action = int(abstract_action % space)
            node_id = wrapped_env.env.get_nodeid_from_index(abstract_action)
            node_info = wrapped_env.env.get_node(node_id)
            if node_info.status == model.MachineStatus.Running and node_info.agent_installed:
                if node_info.reimagable:
                    logging.info(f"Defender detected malware, reimaging node {abstract_action}")
                    actions.reimage_node(node_id)
                    # print("node reimage")
                    return node_info.value
        return 0

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * nn.functional.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            # print(f"loss={loss.mean()}")
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
