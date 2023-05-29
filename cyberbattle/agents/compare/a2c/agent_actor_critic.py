import math
import random
from typing import NamedTuple, Optional, Tuple
import numpy as np
from numpy import ndarray
import logging

from cyberbattle._env import cyberbattle_env
from cyberbattle.agents.compare.a2c.agent_wrapper import EnvironmentBounds

import cyberbattle.agents.compare.a2c.agent_wrapper as w
from cyberbattle.agents.baseline.learner import Learner

import torch


class StateActionModel:
    """How the state is modelled in the enviroment"""

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(ep, [
            w.Feature_discovered_not_owned_nodes_sliding(ep),
            w.Feature_discovered_credential_count(ep)
        ])

        self.source_node_features = w.ConcatFeatures(ep, [
            w.Feature_active_node_properties(ep),
            w.Feature_success_actions_at_node(ep)
        ])

        self.target_node_features = w.ConcatFeatures(ep, [
            w.Feature_active_node_id(ep)
        ])

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection +
                                            self.source_node_features.feature_selection +
                                            self.target_node_features.feature_selection)

        self.action_space = w.AbstractAction(ep)

    def valid_actions(self, wrapped_env: w.AgentWrapper, observation):
        """returns a list of valid actions and the nodes they can be carried out from"""

        nodes_and_actions = []
        discovered_nodes = np.union1d(w.owned_nodes(observation), w.discovered_nodes_notowned(observation))

        for from_node in w.owned_nodes(observation):
            for local_action in range(self.action_space.n_local_actions):
                trial_action = self.action_space.abstract_to_gymaction(from_node, observation, local_action, None)
                if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                    nodes_and_actions.append((from_node, local_action, -1))

            for remote_action in range(self.action_space.n_local_actions, self.action_space.n_local_actions + self.action_space.n_remote_actions):
                for target_node in discovered_nodes:
                    if target_node != from_node:
                        trial_action = self.action_space.abstract_to_gymaction(from_node, observation, remote_action, target_node)
                        if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                            nodes_and_actions.append((from_node, remote_action, target_node))

            for connect_action in range(self.action_space.n_local_actions + self.action_space.n_remote_actions, self.action_space.n_actions):
                trial_action = self.action_space.abstract_to_gymaction(from_node, observation, connect_action, None)
                if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                    nodes_and_actions.append((from_node, connect_action, -1))

        return nodes_and_actions


class Memory:
    """The memory structure that stores the critic value function and the actors state action policy"""

    def __init__(self, ep: EnvironmentBounds, hash_size):
        self.hash_size = hash_size

        self.actor = torch.zeros([2, hash_size], dtype=torch.float64)

        self.critic = torch.zeros([2, hash_size], dtype=torch.float64)

    def state_action_index(self, state_space, abstract_action):
        """Turns a state action pair into an index for the actor tensor"""
        feature_vector = np.append(state_space, abstract_action)
        hash_number = abs(hash(str(feature_vector)))
        return hash_number % self.hash_size

    def state_index(self, state_space):
        """Turns the state into an index for the critic tensor"""
        hash_number = abs(hash(str(state_space)))
        return hash_number % self.hash_size


class ChosenActionMetadata(NamedTuple):
    """Metadata attached to every gym action"""

    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}, state={self.actor_state}]"


class ActorCriticPolicy(Learner):

    def __init__(self,
                 ep: EnvironmentBounds,
                 gamma: float,
                 λ: float,
                 learning_rate: float,
                 hash_size: int
                 ):

        self.n_local_actions = ep.local_attacks_count
        self.n_remote_actions = ep.remote_attacks_count
        self.model = StateActionModel(ep)
        self.gamma = gamma
        self.λ = λ
        self.learning_rate = learning_rate
        self.hash_size = hash_size

        self.memory = Memory(ep, hash_size=hash_size)

    def parameters_as_string(self):
        return f'γ={self.gamma}, lr={self.learning_rate}, λ={self.λ},\n' \
               f'hash_size={self.hash_size}'

    def all_parameters_as_string(self) -> str:
        model = self.model
        return f'{self.parameters_as_string()}\n' \
            f'dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, ' \
            f'Q={[f.name() for f in model.state_space.feature_selection]} ' \
            f"-> 'abstract_action'"

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray, target_features: Optional[ndarray]) -> ndarray:
        """Turns seperate state features into one vector"""
        if target_features is None:
            return np.concatenate((np.array(global_state, dtype=np.float32),
                                   np.array(actor_features, dtype=np.float32)))
        else:
            return np.concatenate((np.array(global_state, dtype=np.float32),
                                   np.array(actor_features, dtype=np.float32),
                                   np.array(target_features, dtype=np.float32)))

    def update_memory(self,
                      reward: float,
                      actor_state: ndarray,
                      abstract_action: int,
                      next_actor_state: Optional[ndarray]):
        """The actor's and critic's memories are updated with reward from the action just used"""

        # The temporal difference error, δ, is calculated then used to update the actor and critic
        current_state_index = self.memory.state_index(actor_state)
        if next_actor_state is None:
            δ = reward - self.memory.critic[0][current_state_index].item()
        else:
            next_state_index = self.memory.state_index(next_actor_state)
            δ = reward + (self.gamma * self.memory.critic[0][next_state_index].item()) - self.memory.critic[0][current_state_index].item()

        # Update the Actor
        current_state_action_index = self.memory.state_action_index(actor_state, abstract_action)

        self.memory.actor[1][current_state_action_index] += 1

        self.memory.actor[0][current_state_action_index] += self.learning_rate * δ * self.memory.actor[1][current_state_action_index].item()
        self.memory.actor[0][current_state_action_index] = round(self.memory.actor[0][current_state_action_index].item(), 5)
        self.memory.actor[0][current_state_action_index] = max(0, self.memory.actor[0][current_state_action_index].item())
        self.memory.actor[0][current_state_action_index] = min(100, self.memory.actor[0][current_state_action_index].item())

        non_zero_indicies = torch.argwhere(self.memory.actor[1]).numpy()
        for i in non_zero_indicies:
            self.memory.actor[1][i] = self.memory.actor[1][i].item() * self.gamma * self.λ

        # Update the Critic
        self.memory.critic[1][current_state_index] += 1

        non_zero_indicies_v = torch.argwhere(self.memory.critic[0]).numpy()
        non_zero_indicies_e = torch.argwhere(self.memory.critic[1]).numpy()
        non_zero_indicies = np.union1d(non_zero_indicies_v, non_zero_indicies_e)

        for i in non_zero_indicies:

            self.memory.critic[0][i] = self.memory.critic[0][i].item() + (self.learning_rate * δ * self.memory.critic[1][i].item())
            self.memory.critic[1][i] = self.memory.critic[1][i].item() * self.gamma * self.λ
            self.memory.critic[0][i] = max(0, self.memory.critic[0][i].item())

    def on_step(self, wrapped_env: w.AgentWrapper, reward: float, done: bool, action_metadata):

        if done:
            self.update_memory(reward,
                               actor_state=action_metadata.actor_state,
                               abstract_action=action_metadata.abstract_action,
                               next_actor_state=None
                               )
        else:
            self.update_memory(reward,
                               actor_state=action_metadata.actor_state,
                               abstract_action=action_metadata.abstract_action,
                               next_actor_state=wrapped_env.state
                               )

    def new_episode(self):
        torch.mul(self.memory.actor[1], 0)
        torch.mul(self.memory.critic[1], 0)

    def end_of_episode(self, i_episode, t):
        return None

    def end_of_iteration(self, t, done):
        return None

    def metadata_from_gymaction(self, wrapped_env: w.AgentWrapper, gym_action):
        """Takes in a gym action and returns it's metadata"""
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.model.source_node_features.get(wrapped_env.state, actor_node)
        abstract_action = self.model.action_space.abstract_from_gymaction(gym_action)

        if 'remote_vulnerability' in gym_action:
            target_node = self.model.target_node_features.get(wrapped_env.state, gym_action['remote_vulnerability'][1])
        else:
            target_node = None

        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features, target_node))

    def get_action(self, wrapped_env: w.AgentWrapper, observation, exploit):
        """Uses Gibbs Softmax distribution to select the next action to be used"""
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        valid_nodes_and_actions = self.model.valid_actions(wrapped_env, observation)
        if not valid_nodes_and_actions:
            gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
            metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
            return gym_action, metadata

        # The p_values are the estimated returns from the actor function of taking the action in the current state
        p_values = []
        for item in valid_nodes_and_actions:
            source_node_features = self.model.source_node_features.get(wrapped_env.state, item[0])

            if item[1] < self.n_local_actions or item[1] - self.n_local_actions > self.n_remote_actions:
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, None)
            else:
                target_node_features = self.model.target_node_features.get(wrapped_env.state, item[2])
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, target_node_features)

            action_state_index = self.memory.state_action_index(actor_state_vector, item[1])

            p_values.append(self.memory.actor[0][action_state_index].item())

        if exploit:
            indicies_of_chosen_actions = [i for i, x in enumerate(p_values) if x == max(p_values)]
            chosen_action_index = random.choice(indicies_of_chosen_actions)
            chosen_action = valid_nodes_and_actions[chosen_action_index]

        else:
            softmax_denominator = 0
            for p_value in p_values:
                softmax_denominator += math.exp(p_value)

            probabilities = []
            for p_value in p_values:
                probabilities.append(math.exp(p_value) / softmax_denominator)

            chosen_action = random.choices(valid_nodes_and_actions, weights=probabilities, k=1)[0]

        if chosen_action[1] < self.n_local_actions or chosen_action[1] - self.n_local_actions > self.n_remote_actions:
            gym_action = self.model.action_space.abstract_to_gymaction(chosen_action[0], observation, chosen_action[1], None)
        else:
            gym_action = self.model.action_space.abstract_to_gymaction(chosen_action[0], observation, chosen_action[1], chosen_action[2])

        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)

        return gym_action, metadata
