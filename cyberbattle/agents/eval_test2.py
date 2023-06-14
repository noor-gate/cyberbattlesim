import gym
import numpy as np
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle._env.defender import ExternalRandomEvents, ScanAndReimageCompromisedMachines
from cyberbattle.agents import test, train
from cyberbattle.agents.compare.a2c.agent_actor_critic import ActorCriticPolicy
from cyberbattle.agents.ppo_curiosity.agent_ppo_curiosity import PPOLearnerBetter
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_tabularqlearning as tql
import cyberbattle.agents.agent_ppo as ppo
import cyberbattle.agents.baseline.plotting as p
from cyberbattle.agents.defenders.ppo_defender import PPODefender
from cyberbattle.agents.compare.sarsa.agent_sarsa_lambda import SarsaLambdaPolicy

TRAINING_CHAIN_SIZE = 20
TEST_CHAIN_SIZE = 16
REWARD_GOAL = 2180
OWN_ATLEAST_PERCENT = 1.0
ngyms = 7

results = open("results.txt", "w")
results.close()


def eval_agent(training_envs, test_envs, ep, network, defender=None):
    agents = [(ActorCriticPolicy(ep, gamma=0.01, λ=0.1, learning_rate=0.1, hash_size=98689), "A2C"),
              (SarsaLambdaPolicy(ep, gamma=0.015), "SARSA"),
              (learner.RandomPolicy(), "Random"),
              (tql.QTabularLearner(ep=ep, gamma=0.015, learning_rate=0.01, exploit_percentile=100), "Tabular Q-Learning"),
              (dqla.DeepQLearnerPolicy(ep=ep, gamma=0.015, replay_memory_size=10000, target_update=10, batch_size=512, learning_rate=0.01), "Deep Q-Learning"),
              (ppo.PPOLearner(ep=ep, gamma=0.15), "PPO"),
              (PPOLearnerBetter(ep=ep, gamma=0.15), "PPO Curiosity"),
              ]

    trained_agents = []

    results = open("results_active_train.txt", "a")
    results.write(f"**** {network} ****\n\n\n\n")
    results.write("---- TRAINING ----\n\n\n")

    all_runs = []
    
    for (agent, title) in agents:
        # Training loop for each agent
        rewards, lengths, direct_exploit_avg = [], [], []
        trained_agent = agent
        run = None
        for (i, env) in enumerate(training_envs):
            if defender:
                defender.set_node_count(env)
            run = train.run(trained_agent, env, ep, f"{title}", defender)
            print(f"*** {network} {i} ***")
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
    """
    # p.plot_episodes_length(f"{network} training episode lengths", all_runs)
    env_rewards = {0: [], 1: [], 2: [], 3: [], 4: []}
    all_runs = []
    results.write("\n\n\n---- TESTING ----\n\n\n")
    all_rewards = []
    for (agent, title) in trained_agents:
        rewards, lengths, direct_exploit_avg = [], [], []
        trained_agent = agent
        run = None
        for (i, env) in enumerate(test_envs):
            run = test.run(trained_agent, env, ep, f"{title}", defender)

            env_rewards[i].append([sum(r) for r in run['all_episodes_rewards']])
            rewards.append(p.mean_reward(run))
            lengths.append(p.average_episode_length(run))
            direct_exploit_avg.append(p.episodes_direct_exploit_averaged(run))
            
        all_runs.append(run)
        all_rewards.append(rewards)
        results.write(f"Average test reward: {round(np.average(rewards), 2)}\n")
        results.write(f"Average test episode length: {round(np.average(lengths), 2)}\n")
        results.write(f"Average test direct exploit: {round(np.average(direct_exploit_avg), 2)}\n\n")

    print(env_rewards)
    #print(np.shape(env_rewards[0]))
    for i in env_rewards.keys():
        best_agents = []
        for j in range(len(env_rewards[i][0])):
            maximum_ep = 0
            best_agent = 0
            for k in range(len(env_rewards[i])):
                if env_rewards[i][k][j] > maximum_ep:
                    maximum_ep = env_rewards[i][k][j]
                    best_agent = k
            best_agents.append(best_agent)
        results.write(f"Env {i}: \n")
        for k in range(len(env_rewards[i])):
            results.write(f"{k} : {best_agents.count(k)}/{len(env_rewards[i][0])}\n")


    p.plot_averaged_cummulative_rewards(f"{network} cumulative test rewards", all_runs)
    # p.plot_episodes_length(f"{network} test episode lengths", all_runs)"""

    results.close()

"""
# CYBER BATTLE CHAIN

#  WITH NO DEFENDER

# Creating environments
training_envs = [gym.make("CyberBattleChain-v0", size=((4 * (i + 2)))) for i in range(ngyms)]
test_env = gym.make("CyberBattleChain-v0", size=((4 * (ngyms + 2))))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=training_envs[0].identifiers
)

eval_agent(training_envs, test_env, ep, "Chain without defender")
# eval_agent(training_envs, test_env, ep, "Chain without defender")

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


# WITH PPO DEFENDER

training_envs = [gym.make("CyberBattleChain-v0", size=((4 * (i + 2)))) for i in range(ngyms)]
test_env = gym.make("CyberBattleChain-v0", size=((4 * (ngyms + 2))))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=training_envs[0].identifiers
)

eval_agent(training_envs, test_env, ep, "Chain with PPO defender", defender=PPODefender(ep=ep, gamma=0.15, env=training_envs[0]))

"""


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


# PPO DEFENDER
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

eval_agent(training_envs, test_env, ep, "Active directory with PPO defender", defender=PPODefender(ep=ep, gamma=0.15, env=training_envs[-1]))
"""

# RANDOM NETWORKS
seeds = [2, 3, 4, 7, 8]
training_envs = [gym.make("CyberBattleRandom-v0", seed=i + 11) for i in range(4)]
test_envs = [gym.make("CyberBattleRandom-v0", seed=i) for i in seeds]
ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=30,
    maximum_total_credentials=50,
    identifiers=training_envs[0].identifiers
)

# NO DEFENDER
#eval_agent(training_envs, test_envs, ep, "Random without defender")

# SCAN AND COMPROMISE DEFENDER
training_envs = [gym.make("CyberBattleRandom-v0", seed=i + 15,
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ScanAndReimageCompromisedMachines(
                              probability=0.6,
                              scan_capacity=2,
                              scan_frequency=5)) for i in range(8)]
test_envs = [gym.make("CyberBattleRandom-v0", seed=i,
                      defender_constraint=cyberbattle_env.DefenderConstraint(
                          maintain_sla=0.80
                      ),
                      defender_agent=ScanAndReimageCompromisedMachines(
                          probability=0.6,
                          scan_capacity=2,
                          scan_frequency=5)) for i in seeds]

eval_agent(training_envs, test_envs, ep, "Random with scan and compromise defender")

# EXTERNAL RANDOM EVENTS DEFENDER

training_envs = [gym.make("CyberBattleRandom-v0", seed=i + 15,
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ExternalRandomEvents()) for i in range(8)]
test_envs = [gym.make("CyberBattleRandom-v0", seed=i,
                      defender_constraint=cyberbattle_env.DefenderConstraint(
                          maintain_sla=0.80
                      ),
                      defender_agent=ExternalRandomEvents()) for i in seeds]

eval_agent(training_envs, test_envs, ep, "Random with external random events defender")

# PPO DEFENDER

training_envs = [gym.make("CyberBattleRandom-v0", seed=i + 15) for i in range(8)]
test_envs = [gym.make("CyberBattleRandom-v0", seed=i) for i in seeds]

eval_agent(training_envs, test_envs, ep, "Random with PPO defender", defender=PPODefender(ep=ep, gamma=0.15, env=training_envs[0]))



# COMPARING DEFENDERS

training_envs = [gym.make('CyberBattleChain-v0',
                          size=((4 * (i + 2))),
                          defender_constraint=cyberbattle_env.DefenderConstraint(
                              maintain_sla=0.80
                          ),
                          defender_agent=ExternalRandomEvents()) for i in range(ngyms)]

test_env_1 = gym.make("CyberBattleChain-v0",
                      size=10,
                      defender_constraint=cyberbattle_env.DefenderConstraint(
                          maintain_sla=0.80
                      ))

test_env_2 = gym.make("CyberBattleChain-v0",
                      size=10,
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
    identifiers=test_env_1.identifiers
)

agent = ppo.PPOLearner(ep=ep, gamma=0.015)
all_runs = []
print("NO DEFENDER")
run = train.run(agent, test_env_1, ep, "No defender")
all_runs.append(run)
print("PPO DEFENDER")
agent = ppo.PPOLearner(ep=ep, gamma=0.15)
run = train.run(agent, test_env_1, ep, "PPO defender", defender=PPODefender(ep=ep, gamma=0.15, env=test_env_1))
all_runs.append(run)
print("SCAN AND COMPROMISE")
agent = ppo.PPOLearner(ep=ep, gamma=0.15)
run = train.run(agent, test_env_2, ep, "Scan and compromise")
all_runs.append(run)

p.plot_averaged_cummulative_rewards(f"Chain cumulative training rewards", all_runs)
"""