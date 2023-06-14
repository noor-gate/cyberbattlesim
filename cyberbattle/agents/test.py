import math
import sys

import torch
from cyberbattle.agents import utils
from cyberbattle.agents.agent_ppo import PPOLearner
from cyberbattle.agents.compare.a2c.agent_actor_critic import ActorCriticPolicy
from cyberbattle.agents.ppo_curiosity.agent_ppo_curiosity import PPOLearnerBetter
from cyberbattle.agents.baseline.learner import Breakdown, Learner, Outcomes, PolicyStats, RandomPolicy, Stats, TrainedLearner
from cyberbattle.agents.defenders.ppo_defender import PPODefender
from cyberbattle.agents.train import train_epsilon_greedy
from cyberbattle.simulation.actions import DefenderAgentActions
from .baseline.plotting import PlotTraining
from .baseline.agent_wrapper import AgentWrapper, EnvironmentBounds, Verbosity, ActionTrackingStateAugmentation
import logging
import numpy as np
from cyberbattle._env import cyberbattle_env
from typing import Optional
import progressbar

TEST_EPISODE_COUNT = 1
ITERATION_COUNT = 100


def run(learner: Learner,
        env: cyberbattle_env.CyberBattleEnv,
        ep: EnvironmentBounds,
        title: str,
        defender: Optional[PPODefender]) -> TrainedLearner:
    if isinstance(learner, PPOLearner):
        return test_policy(env, ep, learner, defender, title, render=False)
    elif isinstance(learner, PPOLearnerBetter):
        return test_policy_curiosity(env, ep, learner, defender, title, render=False)
    elif isinstance(learner, RandomPolicy):
        return train_epsilon_greedy(env, ep, learner, defender, title, epsilon=1, render=False)
    elif isinstance(learner, ActorCriticPolicy):
        return gibbs_softmax_search(
            env,
            ep,
            agent=learner,
            title="Actor-Critic five",
            episode_count=TEST_EPISODE_COUNT,
            iteration_count=ITERATION_COUNT,
            exploit=False,
            render=False,
            verbosity=Verbosity.Quiet,
        )
    else:
        return test_epsilon_greedy(env, ep, learner, title, render=False)


def test_random(gym_env: cyberbattle_env.CyberBattleEnv, title):
    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    for i_episode in range(TEST_EPISODE_COUNT):

        all_rewards = []
        all_availability = []
        observation = gym_env.reset()

        total_reward = 0

        for t in range(ITERATION_COUNT):
            action = gym_env.sample_valid_action()

            observation, reward, done, info = gym_env.step(action)

            total_reward += reward
            all_rewards.append(total_reward)
            all_availability.append(info['network_availability'])

            if reward > 0:
                print('####### rewarded action: {action}')
                print(f'total_reward={total_reward} reward={reward}')

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(gym_env.get_directly_exploited_nodes())

    gym_env.close()
    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        all_episodes_direct_exploit=all_episodes_direct_exploit,
        learner=RandomPolicy(),
        trained_on=gym_env.name,
        title=title
    )


def test_policy(
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

    plot_title = f"{title} (epochs={TEST_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TEST_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TEST_EPISODE_COUNT} PPO "
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
            if defender:
                defender_actuator.on_attacker_step_taken()

            # attacker action
            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
                
            observation, reward, done, info = wrapped_env.step(gym_action)     

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

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{current_ep_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TEST_EPISODE_COUNT \
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
            print(f"  Episode {i_episode} stopped at t={TEST_EPISODE_COUNT} {loss_string}")

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TEST_EPISODE_COUNT
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


def test_policy_curiosity(
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

    plot_title = f"{title} (epochs={TEST_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TEST_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TEST_EPISODE_COUNT} PPO "
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
            if defender:
                defender_actuator.on_attacker_step_taken()

            # attacker action
            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])

            observation, reward, done, info = wrapped_env.step(gym_action)

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


            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{current_ep_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == TEST_EPISODE_COUNT \
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
            print(f"  Episode {i_episode} stopped at t={TEST_EPISODE_COUNT} {loss_string}")

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TEST_EPISODE_COUNT
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

def test_policy_2(env: cyberbattle_env.CyberBattleEnv,
                ep: EnvironmentBounds,
                learner: PPOLearner,
                title: str,
                render=True,
                render_last_episode_rewards_to: Optional[str] = None,
                plot_episodes_length=True):
    wrapped_env = AgentWrapper(env,
                               ActionTrackingStateAugmentation(ep, env.reset()))

    time_step = 0

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    plot_title = f"{title} (epochs={TEST_EPISODE_COUNT}" \
        + learner.parameters_as_string() + ")"
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TEST_EPISODE_COUNT + 1):
        print(f"  ## Episode: {i_episode}/{TEST_EPISODE_COUNT} PPO "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        current_ep_reward = 0.0
        all_rewards = []
        all_availability = []

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

            gym_action, action_metadata = learner.select_action(wrapped_env, observation)

            if not gym_action:
                gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])

            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[outcome]['remote'] += 1
            else:
                stats[outcome]['connect'] += 1

            time_step += 1
            current_ep_reward += reward
            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            bar.update(t, reward=current_ep_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if i_episode == TEST_EPISODE_COUNT \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        utils.print_stats_policy(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else TEST_EPISODE_COUNT
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
        trained_on=env.name,
        title=title
    )


def test_epsilon_greedy(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    learner: Learner,
    title: str,
    render=True,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Quiet,
    plot_episodes_length=True
) -> TrainedLearner:

    print(f"###### {title}\n"
          f"Learning with: episode_count={TEST_EPISODE_COUNT},"
          f"iteration_count={ITERATION_COUNT}," +
          f"{learner.parameters_as_string()})")

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    steps_done = 0
    plot_title = f"{title}(epochs={TEST_EPISODE_COUNT}," \
        + learner.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, TEST_EPISODE_COUNT + 1):

        print(f"  ## Episode: {i_episode}/{TEST_EPISODE_COUNT} {title} "
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

            steps_done += 1

            action_style, gym_action, action_metadata = learner.exploit(wrapped_env, observation)
            if not gym_action:
                break

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

            if i_episode == TEST_EPISODE_COUNT \
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


def gibbs_softmax_search(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    agent: ActorCriticPolicy,
    title: str,
    episode_count: int,
    iteration_count: int,
    exploit: bool,
    render=True,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Normal,
    plot_episodes_length=True
) -> TrainedLearner:

    print(f"###### {title}\n"
          f"Learning with: episode_count={episode_count},"
          f"iteration_count={iteration_count}," +
          f"{agent.parameters_as_string()}")

    all_episodes_rewards = []
    all_episodes_availability = []
    all_episodes_direct_exploit = []

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    steps_done = 0
    plot_title = f"{title} (epochs={episode_count}"  \
        + agent.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, episode_count + 1):

        print(f"\n  ## Episode: {i_episode}/{episode_count} '{title}' "
              f"{agent.parameters_as_string()}")

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []
        #agent.new_episode()

        """stats = Stats(Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                               noreward=Breakdown(local=0, remote=0, connect=0))
                      )"""

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

        for t in bar(range(1, 1 + iteration_count)):

            steps_done += 1

            gym_action, action_metadata = agent.get_action(wrapped_env, observation, exploit)

            # Take the step
            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            outcome = 'reward' if reward > 0 else 'noreward'
            """if 'local_vulnerability' in gym_action:
                stats[outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[outcome]['remote'] += 1
            else:
                stats[outcome]['connect'] += 1"""

            #agent.on_step(wrapped_env, reward, done, action_metadata)
            assert np.shape(reward) == ()

            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            total_reward += reward
            bar.update(t, reward=total_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} r={reward} cum_reward:{total_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {agent.stateaction_as_string(action_metadata)}")

            if i_episode == episode_count \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            agent.end_of_iteration(t, done)

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = agent.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={iteration_count} {loss_string}")

        # print_stats(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)
        all_episodes_direct_exploit.append(cyberbattle_gym_env.get_directly_exploited_nodes())

        length = episode_ended_at if episode_ended_at else iteration_count
        agent.end_of_episode(i_episode=i_episode, t=length)
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
        learner=agent,
        trained_on=cyberbattle_gym_env.name,
        title=plot_title
    )
