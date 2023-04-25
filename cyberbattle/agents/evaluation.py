import gym
import numpy as np
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle._env.defender import ExternalRandomEvents, ScanAndReimageCompromisedMachines
from cyberbattle.agents import test, train
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_tabularqlearning as tql
import cyberbattle.agents.agent_ppo as ppo
import cyberbattle.agents.baseline.plotting as p

TRAINING_CHAIN_SIZE = 20
TEST_CHAIN_SIZE = 16
REWARD_GOAL = 2180
OWN_ATLEAST_PERCENT = 1.0
ngyms = 1

results = open("results.txt", "w")
results.close()


def eval_agent(training_envs, test_env, ep, network):
    agents = [(learner.RandomPolicy(), "Random"),
              (tql.QTabularLearner(ep=ep, gamma=0.015, learning_rate=0.01, exploit_percentile=100), "Tabular Q-Learning"),
              (dqla.DeepQLearnerPolicy(ep=ep, gamma=0.015, replay_memory_size=10000, target_update=10, batch_size=512, learning_rate=0.01), "Deep Q-Learning"),
              (ppo.PPOLearner(ep=ep, gamma=0.015), "PPO")]

    trained_agents = []

    results = open("results.txt", "a")
    results.write(f"**** {network} ****\n\n\n\n")
    results.write("---- TRAINING ----\n\n\n")

    all_runs = []

    for (agent, title) in agents:
        # Training loop for each agent
        rewards, lengths, direct_exploit_avg = [], [], []
        trained_agent = agent
        run = None
        for (i, env) in enumerate(training_envs):
            run = train.run(trained_agent, env, ep, f"{title}")
            rewards.append(p.mean_reward(run))
            lengths.append(p.average_episode_length(run))
            direct_exploit_avg.append(p.episodes_direct_exploit_averaged(run))
            if title != "Random":
                trained_agent = run["learner"]
        all_runs.append(run)
        trained_agents.append((trained_agent, title))

        results.write(title + "\n")
        results.write(f"Average training reward: {round(np.average(rewards), 2)}\n")
        results.write(f"Average training episode length: {round(np.average(lengths), 2)}\n")
        results.write(f"Average training direct exploit: {round(np.average(direct_exploit_avg), 2)}\n\n")

    p.plot_averaged_cummulative_rewards(f"{network} cumulative training rewards", all_runs)
    p.plot_episodes_length(f"{network} training episode lengths", all_runs)

    all_runs = []
    results.write("\n\n\n---- TESTING ----\n\n\n")

    for (agent, title) in trained_agents:
        if title == "Random":
            run = train.run(agent, test_env, ep, title)
        else:
            run = test.run(agent, test_env, ep, title)
        all_runs.append(run)

        results.write(run["title"] + "\n")
        results.write(f"Average test reward: {round(p.mean_reward(run), 2)}\n")
        results.write(f"Average test episode length: {round(p.average_episode_length(run), 2)}\n")
        results.write(f"Average test direct exploit: {round(p.episodes_direct_exploit_averaged(run), 2)}\n\n\n")

    p.plot_averaged_cummulative_rewards(f"{network} cumulative test rewards", all_runs)
    p.plot_episodes_length(f"{network} test episode lengths", all_runs)

    results.close()

# CYBER BATTLE CHAIN

#  WITH NO DEFENDER


# Creating environments
training_envs = [gym.make("CyberBattleChain-v0", size=((4 * (i + 2)))) for i in range(ngyms)]
test_env = gym.make("CyberBattleChain-v0", size=((4 * (ngyms + 2))))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=30,
    maximum_node_count=40,
    identifiers=training_envs[0].identifiers
)


eval_agent(training_envs, test_env, ep, "Chain without defender")

# WITH EXTERNAL RANDOM EVENTS DEFENDER

training_envs = [gym.make('CyberBattleChain-v0',
                          size=((4 * (i + 2))),
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ExternalRandomEvents()) for i in range(ngyms)]
test_env = gym.make("CyberBattleChain-v0",
                    size=((4 * (ngyms + 2))),
                    defender_constraint=cyberbattle_env.DefenderConstraint(
                        maintain_sla=0.80
                    ),
                    defender_agent=ExternalRandomEvents())

eval_agent(training_envs, test_env, ep, "Chain with external random events defender")


# WITH SCAN AND COMPROMISE DEFENDER

training_envs = [gym.make('CyberBattleChain-v0',
                          size=((4 * (i + 2))),
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ScanAndReimageCompromisedMachines(
                              probability=0.6,
                              scan_capacity=2,
                              scan_frequency=5)) for i in range(ngyms)]
test_env = gym.make("CyberBattleChain-v0",
                    size=((4 * (ngyms + 2))),
                    defender_constraint=cyberbattle_env.DefenderConstraint(
                        maintain_sla=0.80
                    ),
                    defender_agent=ScanAndReimageCompromisedMachines(
                        probability=0.6,
                        scan_capacity=2,
                        scan_frequency=5))

eval_agent(training_envs, test_env, ep, "Chain with scan and compromise defender")

# ACTIVE DIRECTORY

# WITHOUT DEFENDER

gymids = [f"ActiveDirectory-v{i}" for i in range(0, ngyms)]
training_envs = [gym.make(gymid) for gymid in gymids]
map(lambda g: g.seed(1), training_envs)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=30,
    maximum_total_credentials=50,
    identifiers=training_envs[0].identifiers
)

test_env = gym.make(f'ActiveDirectory-v{ngyms}')
test_env.seed(1)

eval_agent(training_envs, test_env, ep, "Active directory without defender")

# WITH EXTERNAL RANDOM EVENTS DEFENDER

gymids = [f"ActiveDirectory-v{i}" for i in range(0, ngyms)]
training_envs = [gym.make(gymid,
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ExternalRandomEvents()) for gymid in gymids]
map(lambda g: g.seed(1), training_envs)
test_env = gym.make(f"ActiveDirectory-v{ngyms}",
                    defender_constraint=cyberbattle_env.DefenderConstraint(
                        maintain_sla=0.80
                    ),
                    defender_agent=ExternalRandomEvents())
test_env.seed(1)

eval_agent(training_envs, test_env, ep, "Active directory with external random events defender")

# WITH DEFENDER

gymids = [f"ActiveDirectory-v{i}" for i in range(0, ngyms)]
training_envs = [gym.make(gymid,
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ScanAndReimageCompromisedMachines(
                              probability=0.6,
                              scan_capacity=2,
                              scan_frequency=5)) for gymid in gymids]
map(lambda g: g.seed(1), training_envs)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=30,
    maximum_total_credentials=50,
    identifiers=training_envs[0].identifiers
)

test_env = gym.make(f'ActiveDirectory-v{ngyms}',
                    defender_constraint=cyberbattle_env.DefenderConstraint(
                        maintain_sla=0.80
                    ),
                    defender_agent=ScanAndReimageCompromisedMachines(
                        probability=0.6,
                        scan_capacity=2,
                        scan_frequency=5))
test_env.seed(1)

eval_agent(training_envs, test_env, ep, "Active directory with scan and compromise defender")
