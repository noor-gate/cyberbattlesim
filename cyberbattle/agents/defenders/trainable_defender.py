import random
import numpy
import torch
from abc import abstractmethod
from cyberbattle.agents.baseline.agent_wrapper import AgentWrapper
from cyberbattle.simulation.model import Environment
from cyberbattle.simulation.actions import DefenderAgentActions


import logging


class TrainableDefender:
    """Define the step function for a defender agent.
    Gets called after each step executed by the attacker agent."""

    @abstractmethod
    def step(self, wrapped_env: AgentWrapper, actions: DefenderAgentActions):
        return None
