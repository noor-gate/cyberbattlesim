import sys
import numpy as np
from numpy import ndarray
import progressbar
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from cyberbattle._env import cyberbattle_env
from typing import Optional, Tuple
from .baseline.plotting import PlotTraining
import logging

from cyberbattle._env.cyberbattle_env import EnvironmentBounds, Observation
from cyberbattle.agents.baseline.agent_dql import ChosenActionMetadata, CyberBattleStateActionModel
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, Verbosity
from cyberbattle.agents.baseline.learner import Breakdown, Learner, Outcomes, PolicyStats, Stats, TrainedLearner
import cyberbattle.agents.baseline.agent_wrapper as w

device = torch.device('cpu')
"""if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()"""


def print_stats(stats):
    """Print learning statistics"""
    def print_breakdown(stats):
        def ratio(kind: str) -> str:
            x, y = stats['reward'][kind], stats['noreward'][kind]
            sum = x + y
            if sum == 0:
                return 'NaN'
            else:
                return f"{(x / sum):.2f}"

        def print_kind(kind: str):
            print(
                f"    {kind}: {stats['reward'][kind]}/{stats['noreward'][kind]} "
                f"({ratio(kind)})")
        print_kind('local')
        print_kind('remote')
        print_kind('connect')

    print("  Breakdown [Reward/NoReward (Success rate)]")
    print_breakdown(stats)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """The Deep Neural Network used to estimate the Q function"""

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOLearner(Learner):
    def __init__(self,
                 ep: EnvironmentBounds, gamma: float):
        self.model = CyberBattleStateActionModel(ep)
        self.policy_old = ActorCritic(len(self.model.state_space.dim_sizes), self.model.action_space.flat_size()).to(device)
        self.policy = ActorCritic(len(self.model.state_space.dim_sizes), self.model.action_space.flat_size()).to(device)
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.K_epochs = 40
        self.eps_clip = 0.2
        lr_actor = 0.1      # learning rate for actor network
        lr_critic = 0.001
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        raise NotImplementedError

    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        raise NotImplementedError

    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata) -> None:
        raise NotImplementedError

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate((np.array(global_state, dtype=np.float32),
                               np.array(actor_features, dtype=np.float32)))

    def metadata_from_gymaction(self, wrapped_env, gym_action):
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.model.node_specific_features.get(wrapped_env.state, actor_node)
        abstract_action = self.model.action_space.abstract_from_gymaction(gym_action)
        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features))

    def try_exploit_at_candidate_actor_states(
            self,
            wrapped_env,
            current_global_state,
            actor_features,
            abstract_action):

        actor_state = self.get_actor_state_vector(current_global_state, actor_features)

        action_style, gym_action, actor_node = self.model.implement_action(
            wrapped_env, actor_features, abstract_action)

        if gym_action:
            assert actor_node is not None, 'actor_node should be set together with gym_action'

            return gym_action, ChosenActionMetadata(
                abstract_action=abstract_action,
                actor_node=actor_node,
                actor_features=actor_features,
                actor_state=actor_state)
        else:
            return None, None

    def select_action(self, wrapped_env, observation):
        # with torch.no_grad():
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)

        # Gather the features of all the current active actors (i.e. owned nodes)
        active_actors_features = [
            self.model.node_specific_features.get(wrapped_env.state, from_node)
            for from_node in w.owned_nodes(observation)
        ]

        unique_active_actors_features = list(np.unique(active_actors_features, axis=0))

        # array of actor state vector for every possible set of node features
        actor_state_vector = [
            self.get_actor_state_vector(current_global_state, node_features)
            for node_features in unique_active_actors_features]

        with torch.no_grad():
            state = torch.tensor(actor_state_vector).to(device)
            abstract_actions, actions_logprobs, state_vals = self.policy_old.act(state)

        candidate_indices = list(range(len(actor_state_vector)))

        for actor_index in candidate_indices:
            abstract_action = abstract_actions[actor_index]
            action_logprob = actions_logprobs[actor_index]
            state_val = state_vals[actor_index]

            actor_features = unique_active_actors_features[actor_index]
            actor_state = self.get_actor_state_vector(current_global_state, actor_features)

            gym_action, metadata = self.try_exploit_at_candidate_actor_states(
                wrapped_env,
                current_global_state,
                actor_features,
                abstract_action)

            self.buffer.states.append(torch.tensor(actor_state).to(device))
            self.buffer.actions.append(abstract_action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            if gym_action:
                return gym_action, metadata

            self.buffer.rewards.append(0.0)
            self.buffer.is_terminals.append(False)

        return None, None

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
            #print(f"loss={loss.mean()}")
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
