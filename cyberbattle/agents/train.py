import math
import sys

import torch
from cyberbattle._env.defender import DefenderAgent
from cyberbattle.agents import utils
from cyberbattle.agents.agent_ppo import PPOLearner
from cyberbattle.agents.agent_ppo_better import PPOLearnerBetter
from cyberbattle.agents.baseline.learner import Breakdown, Learner, Outcomes, PolicyStats, RandomPolicy, Stats, TrainedLearner
from cyberbattle.agents.ppo_defender import PPODefender
from cyberbattle.simulation.actions import DefenderAgentActions
from .baseline.plotting import PlotTraining
from .baseline.agent_wrapper import AgentWrapper, EnvironmentBounds, Verbosity, ActionTrackingStateAugmentation
import logging
import numpy as np
from cyberbattle._env import cyberbattle_env
from typing import Optional
import progressbar

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

TRAIN_EPISODE_COUNT = 1
ITERATION_COUNT = 1


def run(learner: Learner,
        env: cyberbattle_env.CyberBattleEnv,
        ep: EnvironmentBounds,
        title: str,
        defender: Optional[PPODefender] = None) -> TrainedLearner:
    if isinstance(learner, PPOLearnerBetter):
        return train_policy_curiosity(env, ep, learner, defender, title, render=False)
    elif isinstance(learner, PPOLearner):
        return train_policy(env, ep, learner, defender, title, render=False)
    elif isinstance(learner, RandomPolicy):
        return train_epsilon_greedy(env, ep, learner, title, epsilon=1, render=False)
    else:
        return train_epsilon_greedy(env, ep, learner, title, epsilon=0.9, render=False)


def train_policy(
        cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
        environment_properties: EnvironmentBounds,
        learner: PPOLearnerBetter,
        defender: Optional[PPODefender],
        title: str,
        verbosity: Verbosity = Verbosity.Quiet,
        render=True,
        render_last_episode_rewards_to: Optional[str] = None,
        plot_episodes_length=True) -> TrainedLearner:

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    defender_actuator = DefenderAgentActions(cyberbattle_gym_env.environment)

    time_step = 0
    update_timestep = 10

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    plot_title = f"{title} (epochs={TRAIN_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TRAIN_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TRAIN_EPISODE_COUNT} PPO "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        current_ep_reward = 0.0
        all_rewards = []
        all_availability = []
        learner.new_episode()

        stats = PolicyStats(reward=Breakdown(local=0, remote=0, connect=0),
                            noreward=Breakdown(local=0, remote=0, connect=0)
                            )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)

        for t in bar(range(1, 1 + ITERATION_COUNT)):

            # defender action
            defender_reward = 0
            if defender:
                defender_actuator.on_attacker_step_taken()
                defender_reward = defender.step(wrapped_env, observation, defender_actuator)

            # attacker action
            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
                action_metadata = learner.metadata_from_gymaction(wrapped_env, gym_action)
                abstract_action, _, _, actor_state = action_metadata

                with torch.no_grad():
                    log_prob = torch.tensor(0).to(device)
                    tensor_action = torch.tensor(abstract_action).to(device)
                    state_val = learner.policy_old.critic(torch.tensor(actor_state).to(device))
                    learner.buffer.states.append(torch.tensor(actor_state).to(device))
                    learner.buffer.actions.append(tensor_action)
                    learner.buffer.logprobs.append(log_prob)
                    learner.buffer.state_values.append(state_val)

            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            # if defender:
            #    print(f"defender_action={defender_action} attacker_action={gym_action} reward={reward}")

            # attacker reward
            learner.buffer.rewards.append(reward - defender_reward)
            learner.buffer.is_terminals.append(done)

            # defender reward is negative of attacker
            if defender:
                defender.buffer.rewards.append(defender_reward - reward)
                defender.buffer.is_terminals.append(done)

            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[outcome]['remote'] += 1
            else:
                stats[outcome]['connect'] += 1

            # if reward > 0:
                # print(gym_action)

            time_step += 1
            current_ep_reward += reward
            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            bar.update(t, reward=current_ep_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if time_step % update_timestep == 0:
                learner.update()

            if time_step % update_timestep == 0:
                if defender:
                    defender.update()

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{current_ep_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TRAIN_EPISODE_COUNT \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={TRAIN_EPISODE_COUNT} {loss_string}")

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TRAIN_EPISODE_COUNT
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

    wrapped_env.close()
    print("simulation ended")
    if plot_episodes_length:
        plottraining.plot_end()

    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        all_episodes_direct_exploit=all_episodes_direct_exploit,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=title
    )

def train_policy_curiosity(
        cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
        environment_properties: EnvironmentBounds,
        learner: PPOLearnerBetter,
        defender: Optional[PPODefender],
        title: str,
        verbosity: Verbosity = Verbosity.Quiet,
        render=True,
        render_last_episode_rewards_to: Optional[str] = None,
        plot_episodes_length=True) -> TrainedLearner:

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    defender_actuator = DefenderAgentActions(cyberbattle_gym_env.environment)

    time_step = 0
    update_timestep = 10

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    plot_title = f"{title} (epochs={TRAIN_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TRAIN_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TRAIN_EPISODE_COUNT} PPO with CURIOSITY"
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        current_ep_reward = 0.0
        all_rewards = []
        all_availability = []
        learner.new_episode()

        stats = PolicyStats(reward=Breakdown(local=0, remote=0, connect=0),
                            noreward=Breakdown(local=0, remote=0, connect=0)
                            )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)

        for t in bar(range(1, 1 + ITERATION_COUNT)):

            # defender action
            defender_reward = 0
            if defender:
                defender_actuator.on_attacker_step_taken()
                defender_reward = defender.step(wrapped_env, observation, defender_actuator)

            # attacker action
            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
                action_metadata = learner.metadata_from_gymaction(wrapped_env, gym_action)
                abstract_action, _, _, actor_state = action_metadata

                with torch.no_grad():
                    log_prob = torch.tensor(0).to(device)
                    tensor_action = torch.tensor(abstract_action).to(device)
                    state_val = learner.policy_old.critic(torch.tensor(actor_state).to(device))
                    learner.buffer.states.append(torch.tensor(actor_state).to(device))
                    learner.buffer.actions.append(tensor_action)
                    learner.buffer.logprobs.append(log_prob)
                    learner.buffer.state_values.append(state_val)

            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            # if defender:
            #    print(f"defender_action={defender_action} attacker_action={gym_action} reward={reward}")

            # attacker reward
            learner.buffer.rewards.append(reward - defender_reward)
            learner.buffer.is_terminals.append(done)

            # defender reward is negative of attacker
            if defender:
                defender.buffer.rewards.append(defender_reward - reward)
                defender.buffer.is_terminals.append(done)

            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[outcome]['remote'] += 1
            else:
                stats[outcome]['connect'] += 1

            # if reward > 0:
                # print(gym_action)

            time_step += 1
            current_ep_reward += reward
            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            bar.update(t, reward=current_ep_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if time_step % update_timestep == 0:
                learner.update(time_step)

            if time_step % update_timestep == 0:
                if defender:
                    defender.update()

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{current_ep_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TRAIN_EPISODE_COUNT \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={TRAIN_EPISODE_COUNT} {loss_string}")

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TRAIN_EPISODE_COUNT
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

    wrapped_env.close()
    print("simulation ended")
    if plot_episodes_length:
        plottraining.plot_end()

    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        all_episodes_direct_exploit=all_episodes_direct_exploit,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=title
    )


def train_epsilon_greedy(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    learner: Learner,
    title: str,
    epsilon,
    epsilon_minimum=0.0,
    epsilon_multdecay: Optional[float] = None,
    epsilon_exponential_decay: Optional[int] = None,
    render=True,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Quiet,
    plot_episodes_length=True
) -> TrainedLearner:
    """Epsilon greedy search for CyberBattle gym environments

    Parameters
    ==========

    - cyberbattle_gym_env -- the CyberBattle environment to train on

    - learner --- the policy learner/exploiter

    - episode_count -- Number of training episodes

    - iteration_count -- Maximum number of iterations in each episode

    - epsilon -- explore vs exploit
        - 0.0 to exploit the learnt policy only without exploration
        - 1.0 to explore purely randomly

    - epsilon_minimum -- epsilon decay clipped at this value.
    Setting this value too close to 0 may leed the search to get stuck.

    - epsilon_decay -- epsilon gets multiplied by this value after each episode

    - epsilon_exponential_decay - if set use exponential decay. The bigger the value
    is, the slower it takes to get from the initial `epsilon` to `epsilon_minimum`.

    - verbosity -- verbosity of the `print` logging

    - render -- render the environment interactively after each episode

    - render_last_episode_rewards_to -- render the environment to the specified file path
    with an index appended to it each time there is a positive reward
    for the last episode only

    - plot_episodes_length -- Plot the graph showing total number of steps by episode
    at th end of the search.

    Note on convergence
    ===================

    Setting 'minimum_espilon' to 0 with an exponential decay <1
    makes the learning converge quickly (loss function getting to 0),
    but that's just a forced convergence, however, since when
    epsilon approaches 0, only the q-values that were explored so
    far get updated and so only that subset of cells from
    the Q-matrix converges.

    """

    print(f"###### {title}\n"
          f"Learning with: episode_count={TRAIN_EPISODE_COUNT},"
          f"iteration_count={ITERATION_COUNT},"
          f"ϵ={epsilon},"
          f'ϵ_min={epsilon_minimum}, '
          + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '')
          + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') +
          f"{learner.parameters_as_string()})")

    initial_epsilon = epsilon

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    steps_done = 0
    plot_title = f"{title} (epochs={TRAIN_EPISODE_COUNT}, ϵ={initial_epsilon}, ϵ_min={epsilon_minimum}," \
        + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '') \
        + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') \
        + learner.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TRAIN_EPISODE_COUNT + 1):

        print(f"  ## Episode: {i_episode}/{TRAIN_EPISODE_COUNT} {title} "
              f"ϵ={epsilon:.4f}, "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []
        learner.new_episode()

        stats = Stats(exploit=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      explore=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      exploit_deflected_to_explore=0
                      )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)

        for t in bar(range(1, 1 + ITERATION_COUNT)):

            if epsilon_exponential_decay:
                epsilon = epsilon_minimum + math.exp(-1. * steps_done /
                                                     epsilon_exponential_decay) * (initial_epsilon - epsilon_minimum)

            steps_done += 1

            x = np.random.rand()
            if x <= epsilon:
                action_style, gym_action, action_metadata = learner.explore(wrapped_env)
            else:
                action_style, gym_action, action_metadata = learner.exploit(wrapped_env, observation)
                if not gym_action:
                    stats['exploit_deflected_to_explore'] += 1
                    _, gym_action, action_metadata = learner.explore(wrapped_env)

            # Take the step
            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            action_type = 'exploit' if action_style == 'exploit' else 'explore'
            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[action_type][outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[action_type][outcome]['remote'] += 1
            else:
                stats[action_type][outcome]['connect'] += 1

            learner.on_step(wrapped_env, observation, reward, done, info, action_metadata)
            assert np.shape(reward) == ()

            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            total_reward += reward
            bar.update(t, reward=total_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} {action_style} r={reward} cum_reward:{total_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TRAIN_EPISODE_COUNT \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            learner.end_of_iteration(t, done)

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={ITERATION_COUNT} {loss_string}")

        utils.print_stats_epsilon(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else ITERATION_COUNT
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

        if epsilon_multdecay:
            epsilon = max(epsilon_minimum, epsilon * epsilon_multdecay)

    wrapped_env.close()
    print("simulation ended")
    if plot_episodes_length:
        plottraining.plot_end()

    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        all_episodes_direct_exploit=all_episodes_direct_exploit,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=title
    )


def train_policy_curiosity(
        cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
        environment_properties: EnvironmentBounds,
        learner: PPOLearner,
        defender: Optional[PPODefender],
        title: str,
        verbosity: Verbosity = Verbosity.Quiet,
        render=True,
        render_last_episode_rewards_to: Optional[str] = None,
        plot_episodes_length=True) -> TrainedLearner:

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    defender_actuator = DefenderAgentActions(cyberbattle_gym_env.environment)

    time_step = 0
    update_timestep = 10

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    plot_title = f"{title} (epochs={TRAIN_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TRAIN_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TRAIN_EPISODE_COUNT} PPO "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        current_ep_reward = 0.0
        all_rewards = []
        all_availability = []
        learner.new_episode()

        stats = PolicyStats(reward=Breakdown(local=0, remote=0, connect=0),
                            noreward=Breakdown(local=0, remote=0, connect=0)
                            )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)

        for t in bar(range(1, 1 + ITERATION_COUNT)):

            # defender action
            defender_reward = 0
            if defender:
                defender_actuator.on_attacker_step_taken()
                defender_reward = defender.step(wrapped_env, observation, defender_actuator)

            # attacker action
            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
                action_metadata = learner.metadata_from_gymaction(wrapped_env, gym_action)
                abstract_action, _, _, actor_state = action_metadata

                with torch.no_grad():
                    log_prob = torch.tensor(0).to(device)
                    tensor_action = torch.tensor(abstract_action).to(device)
                    state_val = learner.policy_old.critic(torch.tensor(actor_state).to(device))
                    learner.buffer.states.append(torch.tensor(actor_state).to(device))
                    learner.buffer.actions.append(tensor_action)
                    learner.buffer.logprobs.append(log_prob)
                    learner.buffer.state_values.append(state_val)

            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            # if defender:
            #    print(f"defender_action={defender_action} attacker_action={gym_action} reward={reward}")

            # attacker reward
            learner.buffer.rewards.append(reward - defender_reward)
            learner.buffer.is_terminals.append(done)

            # defender reward is negative of attacker
            if defender:
                defender.buffer.rewards.append(defender_reward - reward)
                defender.buffer.is_terminals.append(done)

            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[outcome]['remote'] += 1
            else:
                stats[outcome]['connect'] += 1

            # if reward > 0:
                # print(gym_action)

            time_step += 1
            current_ep_reward += reward
            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            bar.update(t, reward=current_ep_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if time_step % update_timestep == 0:
                learner.update()

            if time_step % update_timestep == 0:
                if defender:
                    defender.update()

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{current_ep_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TRAIN_EPISODE_COUNT \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={TRAIN_EPISODE_COUNT} {loss_string}")

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TRAIN_EPISODE_COUNT
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

    wrapped_env.close()
    print("simulation ended")
    if plot_episodes_length:
        plottraining.plot_end()

    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        all_episodes_direct_exploit=all_episodes_direct_exploit,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=title
    )
