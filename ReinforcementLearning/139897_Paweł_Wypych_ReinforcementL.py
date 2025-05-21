import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

'''
Implementacje uczenia przez wzmacnianie za pomocą wybranej biblioteki (np. gymnasium)
na podstawie dowolnego problemu (proszę użyć środowiska dostępnego już w bibliotece, carpole, pendolium, mountaincar etc)
- prosze przygotować wizualizację nauki
- Proszę dokonać nauki modelu przez min. dwie metody; np Qlearning i SARSA oraz je porównać, jakie są różnice?
- Prosze dokonać testów dla różnych parametrów modeli oraz je porównać, jakie są różnica? (Krzywe uczenia, czas obliczeń, wpływ hiperparametrów:
    - Współczynnik uczenia (learning rate)
    - Współczynnik dyskontowania (discount factor)
    - Strategia eksploracji (ε-greedy, zmniejszanie ε w czasie)
'''

def create_bins(n_bins=10):
    # CartPole ma 4 zmienne ciągłe
    bins = [
        np.linspace(-2.4, 2.4, n_bins),        # Cart Position
        np.linspace(-5, 5, n_bins),            # Cart Velocity
        np.linspace(-.2095, .2095, n_bins),    # Pole Angle
        np.linspace(-4, 4, n_bins)             # Pole Velocity At Tip
    ]
    return bins


def discretize_state(state, bins):
    return tuple(
        min(len(b) - 1, max(0, int(np.digitize(s, b) - 1))) for s, b in zip(state, bins)
    )


def init_q_table():
    return defaultdict(lambda: np.zeros(2))  # 2 akcje: np. lewo/prawo


# strategia e-greedy
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(2)
    else:
        return np.argmax(q_table[state])


def q_learning(env, episodes=1000, alpha=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    bins = create_bins()
    q_table = init_q_table()
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0

        done = False
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, terminated, _ , _ = env.step(action)
            done = terminated # Pole Angle is greater than ±12°  or Cart Position is greater than ±2.4
            next_state_dis = discretize_state(next_state, bins)

            # Q-learning update
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state_dis]) - q_table[state][action])

            state = next_state_dis
            total_reward += reward


        rewards_per_episode.append(total_reward)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if ep%100==0:
            print(f'[Q-Learining alpha:{alpha} gamma:{gamma} decay:{epsilon_decay}  ] Episode: {ep} Epsilon: {epsilon} Mean Rewards: {mean_rewards:0.1f}')
        if mean_rewards > 1000:
            break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return rewards_per_episode, q_table


def sarsa(env, episodes=1000, alpha=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    bins = create_bins()
    q_table = init_q_table()
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        action = choose_action(state, q_table, epsilon)
        total_reward = 0

        done = False
        while not done:
            next_state, reward, terminated, _, _ = env.step(action)
            done = terminated   # Pole Angle is greater than ±12°  or Cart Position is greater than ±2.4
            next_state_dis = discretize_state(next_state, bins)
            next_action = choose_action(next_state_dis, q_table, epsilon)

            # SARSA update
            q_table[state][action] += alpha * (
                reward + gamma * q_table[next_state_dis][next_action] - q_table[state][action])

            state, action = next_state_dis, next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if ep%100==0:
            print(f'[SARSA  alpha:{alpha} gamma:{gamma} decay:{epsilon_decay}] Episode: {ep} Epsilon: {epsilon} Mean Rewards: {mean_rewards:0.1f}')
        if mean_rewards > 1000:
            break
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return rewards_per_episode, q_table


def plot_learning_curve(rewards_q, rewards_sarsa, label1="Q-learning", label2="SARSA"):
    plt.plot(rewards_q, label=label1)
    plt.plot(rewards_sarsa, label=label2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Porównanie Q-learning vs SARSA")
    plt.legend()
    plt.grid(True)
    plt.show()

def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

def run_experiment():
    env = gym.make("CartPole-v1")
    param_sets = [
        {"alpha": 0.4, "gamma": 0.99, "epsilon_decay": 0.999},
        {"alpha": 0.2, "gamma": 0.8, "epsilon_decay": 0.99},
        {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.999},
        {"alpha": 0.05, "gamma": 0.99, "epsilon_decay": 0.995},
        {"alpha": 0.2, "gamma": 0.95, "epsilon_decay": 0.999},]
    results = []
    episodes = 10000

    for i, params in enumerate(param_sets):
        print(f"\nTest {i+1} | alpha={params['alpha']} gamma={params['gamma']} decay={params['epsilon_decay']}")

        # Q-learning
        start = time.time()
        rewards_q, _ = q_learning(
            env,
            episodes=episodes,
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon_decay=params['epsilon_decay']
        )
        q_time = time.time() - start

        # SARSA
        start = time.time()
        rewards_sarsa, _ = sarsa(
            env,
            episodes=episodes,
            alpha=params['alpha'],
            gamma=params['gamma'],
            epsilon_decay=params['epsilon_decay']
        )
        sarsa_time = time.time() - start

        results.append({
            "params": params,
            "q_rewards": rewards_q,
            "sarsa_rewards": rewards_sarsa,
            "q_time": q_time,
            "sarsa_time": sarsa_time
        })



    env.close()
    os.makedirs("plots", exist_ok=True)
    print("\nPodsumowanie czasów uczenia:")
    for i, res in enumerate(results):
        p = res["params"]
        print(f"Test {i+1} | α={p['alpha']} γ={p['gamma']} decay={p['epsilon_decay']}: "
              f"Q={res['q_time']:.2f}s, SARSA={res['sarsa_time']:.2f}s")

        plt.figure(figsize=(10, 5))
        plt.plot(smooth(res['q_rewards']), label=f"Q-learning")
        plt.plot(smooth(res['sarsa_rewards']), label=f"SARSA")
        plt.title(f"Test {i+1}: α={p['alpha']} γ={p['gamma']} decay={p['epsilon_decay']}")
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Total Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"plots/Test_{i+1}_alfa={p['alpha']}_gamma={p['gamma']}_decay={p['epsilon_decay']}_q-time={res['q_time']:.2f}s_sarsa_time={res['sarsa_time']:.2f}s.png"
        plt.savefig(filename)
        plt.close("all")

if __name__ == "__main__":
    run_experiment()
