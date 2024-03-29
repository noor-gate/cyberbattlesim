#!/usr/bin/python3.9

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""CLI to run the baseline Deep Q-learning and Random agents
   on a sample CyberBattle gym environment and plot the respective
   cummulative rewards in the terminal.

Example usage:

    python -m run --training_episode_count 50  --iteration_count 9000 --rewardplot_width 80  --chain_size=20 --ownership_goal 1.0

"""
from typing import cast
import torch
import gym
import logging
import asciichartpy
import argparse
import sys
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle.agents import train
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.agent_ppo as ppo
import cyberbattle.agents.baseline.agent_tabularqlearning as tql
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.learner as learner
from cyberbattle._env.defender import ExternalRandomEvents, ScanAndReimageCompromisedMachines
from cyberbattle.agents.defenders.ppo_defender import PPODefender


parser = argparse.ArgumentParser(description='Run simulation with DQL baseline agent.')

parser.add_argument('--training_episode_count', default=50, type=int,
                    help='number of training epochs')

parser.add_argument('--eval_episode_count', default=5, type=int,
                    help='number of evaluation epochs')

parser.add_argument('--iteration_count', default=9000, type=int,
                    help='number of simulation iterations for each epoch')

parser.add_argument('--reward_goal', default=2180, type=int,
                    help='minimum target rewards to reach for the attacker to reach its goal')

parser.add_argument('--ownership_goal', default=1.0, type=float,
                    help='percentage of network nodes to own for the attacker to reach its goal')

parser.add_argument('--rewardplot_width', default=80, type=int,
                    help='width of the reward plot (values are averaged across iterations to fit in the desired width)')

parser.add_argument('--chain_size', default=4, type=int,
                    help='size of the chain of the CyberBattleChain sample environment')

parser.add_argument('--learner', default='dql', type=str)

parser.add_argument('--random_agent', dest='run_random_agent', action='store_true', help='run the random agent as a baseline for comparison')
parser.add_argument('--no-random_agent', dest='run_random_agent', action='store_false', help='do not run the random agent as a baseline for comparison')
parser.set_defaults(run_random_agent=True)

parser.add_argument('--transfer_eval', action='store_false')

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

cyberbattlechain = gym.make('CyberBattleChain-v0',
                            size=args.chain_size,
                            attacker_goal=cyberbattle_env.AttackerGoal(
                                own_atleast_percent=args.ownership_goal,
                                reward=args.reward_goal))

cyberbattlechain_defender = gym.make('CyberBattleChain-v0',
                                     size=10,
                                     attacker_goal=cyberbattle_env.AttackerGoal(
                                         own_atleast=0,
                                         own_atleast_percent=1.0
                                     ),
                                     defender_constraint=cyberbattle_env.DefenderConstraint(
                                         maintain_sla=0.80
                                     ),
                                     defender_agent=ScanAndReimageCompromisedMachines(
                                         probability=0.6,
                                         scan_capacity=2,
                                         scan_frequency=5))

cyberbattlechain_defender2 = gym.make('ActiveDirectory-v0',
                                      defender_constraint=cyberbattle_env.DefenderConstraint(
                                          maintain_sla=0.80
                                      ),
                                      defender_agent=ScanAndReimageCompromisedMachines(
                                          probability=0.6,
                                          scan_capacity=2,
                                          scan_frequency=5))


cyberbattlechain_defender_eval = gym.make('CyberBattleChain-v0',
                                          size=16,
                                          attacker_goal=cyberbattle_env.AttackerGoal(
                                              own_atleast=0,
                                              own_atleast_percent=1.0
                                          ),
                                          defender_constraint=cyberbattle_env.DefenderConstraint(
                                              maintain_sla=0.80
                                          ),
                                          defender_agent=ScanAndReimageCompromisedMachines(
                                              probability=0.6,
                                              scan_capacity=2,
                                              scan_frequency=5))


ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=cyberbattlechain.identifiers
)

all_runs = []

if args.learner == 'tql':
    all_runs.append(learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain_defender,
        environment_properties=ep,
        learner=tql.QTabularLearner(
            ep=ep,
            gamma=0.015,
            learning_rate=0.01,
            exploit_percentile=100),  # torch default is 1e-2
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="Tabular Q-Leaner"
    ))

trained_learner = None

if args.learner == 'ppo':
    all_runs.append(train.run(learner=ppo.PPOLearner(ep=ep, gamma=0.015),
                              defender=PPODefender(ep=ep, gamma=0.015),
                              env=cyberbattlechain,
                              ep=ep,
                              title="PPO"))

if args.learner == 'dql':
    # Run Deep Q-learning
    trained_learner = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain_defender2,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.01),  # torch default is 1e-2
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=True,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="DQL"
    )
    all_runs.append(trained_learner)
"""  all_runs.append(learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain_defender,
        environment_properties=ep,
        learner=trained_learner['learner'],  # torch default is 1e-2
        episode_count=args.eval_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=True,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="DQL"))"""

"""if args.transfer_eval:
    learner.transfer_learning_evaluation(
        environment_properties=ep,
        trained_learner=cast(learner.TrainedLearner, trained_learner),
        eval_env=cast(cyberbattle_env.CyberBattleEnv, cyberbattlechain_defender_eval),
        eval_epsilon=0.9,
        eval_episode_count=args.eval_episode_count,
        iteration_count=args.iteration_count
    )"""

if args.run_random_agent:
    random_run = learner.epsilon_greedy_search(
        cyberbattlechain,
        ep,
        learner=learner.RandomPolicy(),
        episode_count=args.eval_episode_count,
        iteration_count=args.iteration_count,
        epsilon=1.0,  # purely random
        render=False,
        verbosity=Verbosity.Quiet,
        title="Random search"
    )
    all_runs.append(random_run)

colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]

print("Episode duration")
print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 30, 'colors': colors}))

print("Cumulative rewards")
c = p.averaged_cummulative_rewards(all_runs, args.rewardplot_width)
print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))

print("Average reward: ", p.mean_reward(all_runs[0]))
print("Average episode length: ", p.average_episode_length(all_runs[0]))
print("Average direct exploit: ", p.episodes_direct_exploit_averaged(all_runs[0]))
