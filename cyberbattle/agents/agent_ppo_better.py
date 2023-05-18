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
import gym.spaces as spaces
from torch.distributions import Categorical, Distribution

from cyberbattle.agents.icm import ICM

from .baseline.plotting import PlotTraining
import logging

from cyberbattle._env.cyberbattle_env import EnvironmentBounds, Observation
from cyberbattle.agents.baseline.agent_dql import ChosenActionMetadata, CyberBattleStateActionModel
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, Verbosity
from cyberbattle.agents.baseline.learner import Breakdown, Learner, Outcomes, PolicyStats, Stats, TrainedLearner
import cyberbattle.agents.baseline.agent_wrapper as w

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


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

def linear_decay_lr(optimizer, n_step, max_step=1e7, max_lr=3e-4, min_lr=1e-5):
    if n_step >= max_step:
        optimizer.param_groups[0]['lr'] = min_lr
    else:
        optimizer.param_groups[0]['lr'] = (min_lr - max_lr) / max_step * n_step + max_lr


def linear_decay_beta(n_step, max_step=1e7, max_b=1e-2, min_b=1e-5):
    if n_step >= max_step:
        return min_b
    else:
        return (min_b - max_b) / max_step * n_step + max_b


def linear_decay_eps(n_step, max_step=1e7, max_e=0.2, min_e=0.1):
    if n_step >= max_step:
        return min_e
    else:
        return (min_e - max_e) / max_step * n_step + max_e


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


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid()(x)


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


class PPOLearnerBetter(Learner):
    def __init__(self,
                 ep: EnvironmentBounds, gamma: float):
        self.model = CyberBattleStateActionModel(ep)
        self.policy_old = ActorCritic(len(self.model.state_space.dim_sizes), self.model.action_space.flat_size())
        self.policy = ActorCritic(len(self.model.state_space.dim_sizes), self.model.action_space.flat_size())
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.K_epochs = 40
        self.eps_clip = 0.2
        self.intr_reward_strength = 0.02
        self.lambd = 0.95
        self.ppo_batch_size = 1
        self.icm_epochs = 1
        self.icm_batch_size = 1
        self.icm = ICM(len(self.model.state_space.dim_sizes), self.model.action_space.flat_size()).to(device)
        lr_actor = 0.0003       # learning rate for actor network
        lr_critic = 0.001
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        betas = (0.9, 0.999)
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=lr_actor, betas=betas)

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
    
    def update(self, timestep):
        # Convert lists from memory to tensors
        self.timestep = timestep
        old_states = torch.stack(self.buffer.states).to(device).detach()
        #old_states = torch.transpose(old_states, 0, 1)
        old_actions = torch.stack(self.buffer.actions).T.to(device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).T.to(device).detach()
        # Finding s, n_s, a, done, reward:
        curr_states = old_states[:-1, :]
        next_states = old_states[1:, :]
        actions = old_actions[:-1].long()
        rewards = torch.tensor(self.buffer.rewards[:-1]).T.to(device).detach()
        mask = (~torch.tensor(self.buffer.is_terminals).T.to(device).detach()[:-1]).type(torch.long)
        with torch.no_grad():
            intr_reward, _, _ = self.icm(actions, curr_states, next_states, mask)
        intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)


        """print('Mean_intr_reward_per_1000_steps',
                               intr_rewards.mean() * 1000,
                               self.timestep
                               )"""

        # Finding comulitive advantage
        with torch.no_grad():
            state_values = torch.squeeze(self.policy.critic(curr_states))
            next_state_values = torch.squeeze(self.policy.critic(next_states))
            td_target = (rewards + intr_rewards) / 2 + self.gamma * next_state_values * mask
            delta = td_target - state_values

            """print('maxValue',
                                   state_values.max(),
                                   timestep
                                   )
            print('meanValue',
                                   state_values.mean(),
                                   self.timestep
                                   )"""

            
            #for i in range(delta.size(0) - 1, -1, -1):
            #delta_t, mask_t = delta[:, i], mask[:, i]
            delta_t, mask_t = delta[:], mask[:]
            advantage = torch.zeros(delta_t.size(0)).to(device)
            #print(delta_t.shape, advantage.shape, mask_t.shape)
            advantage_lst = []
            advantage = delta_t + (self.gamma * self.lambd * advantage) * mask_t
            advantage_lst.insert(0, advantage)

            advantage_lst = torch.cat(advantage_lst, dim=0).T
            # Get local advantage to train value function
            local_advantages = state_values + advantage_lst
            # Normalizing the advantage
            advantages = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-10)

        # Optimize policy for ppo epochs:
        epoch_surr_loss = 0
        for _ in range(self.K_epochs):
            indexes = np.random.permutation(actions.size(0))
            # Train PPO and icm
            for i in range(0, len(indexes), self.ppo_batch_size):
                batch_ind = indexes[i:i + self.ppo_batch_size]
                batch_curr_states = curr_states[batch_ind, :]
                batch_actions = actions[batch_ind]
                batch_mask = mask[batch_ind]
                batch_advantages = advantages[batch_ind]
                batch_local_advantages = local_advantages[batch_ind]
                batch_old_logprobs = old_logprobs[batch_ind]

                # Finding actions logprobs and states values
                batch_logprobs, batch_state_values, batch_dist_entropy = self.policy.evaluate(batch_curr_states,
                                                                                              batch_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(batch_logprobs - batch_old_logprobs.detach())

                # Apply leaner decay and multiply 16 times cause agents_batch is 16 long
                decay_epsilon = linear_decay_eps(self.timestep * 16)
                decay_beta = linear_decay_beta(self.timestep * 16)

                # Finding Surrogate Loss:
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - decay_epsilon, 1 + decay_epsilon) * batch_advantages
                #print(batch_state_values.shape,
                #      batch_local_advantages.detach().shape)
                loss = -torch.min(surr1, surr2) * batch_mask + \
                    0.5 * nn.MSELoss(reduction='none')(batch_state_values,
                                                       batch_local_advantages.detach()) * batch_mask - \
                    decay_beta * batch_dist_entropy * batch_mask
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                linear_decay_lr(self.optimizer, self.timestep * 16)

                epoch_surr_loss += loss.item()

        self._icm_update(self.icm_epochs, self.icm_batch_size, curr_states, next_states, actions, mask)
        """print('Lr',
                               self.optimizer.param_groups[0]['lr'],
                               self.timestep
                               )
        print('Surrogate_loss',
                               epoch_surr_loss / (self.ppo_epochs * (len(indexes) // self.ppo_batch_size + 1)),
                               self.timestep
                               )"""

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _icm_update(self, epochs, batch_size, curr_states, next_states, actions, mask):
        epoch_forw_loss = 0
        epoch_inv_loss = 0
        for _ in range(epochs):
            indexes = np.random.permutation(actions.size(0))
            for i in range(0, len(indexes), batch_size):
                batch_ind = indexes[i:i + batch_size]
                batch_curr_states = curr_states[batch_ind, :]
                batch_next_states = next_states[batch_ind, :]
                batch_actions = actions[batch_ind]
                batch_mask = mask[batch_ind]

                _, inv_loss, forw_loss = self.icm(batch_actions,
                                                  batch_curr_states,
                                                  batch_next_states,
                                                  batch_mask)
                epoch_forw_loss += forw_loss.item()
                epoch_inv_loss += inv_loss.item()
                unclip_intr_loss = 10 * (0.2 * forw_loss + 0.8 * inv_loss)

                # take gradient step
                self.optimizer_icm.zero_grad()
                unclip_intr_loss.backward()
                self.optimizer_icm.step()
                linear_decay_lr(self.optimizer_icm, self.timestep * 16)
        """print('Forward_loss',
                               epoch_forw_loss / (epochs * (len(indexes) // batch_size + 1)),
                               self.timestep
                               )
        print('Inv_loss',
                                epoch_inv_loss / (epochs * (len(indexes) // batch_size + 1)),
                                self.timestep
                                )"""
