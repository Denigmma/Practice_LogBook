import json
import random
import numpy as np
from example import q_learning


# Функция для создания случайного МППР
def create_random_mdp(num_states=4, num_actions=2, num_outcomes=2):
    states = [f's{i}' for i in range(num_states)]
    actions = [f'a{i}' for i in range(num_actions)]

    transition_probs = {}
    reward_function = {}

    for state in states:
        transition_probs[state] = {}
        reward_function[state] = {}
        for action in actions:
            transition_probs[state][action] = {}
            reward_function[state][action] = {}

            outcomes = [f's{o}' for o in random.sample(range(num_states), num_outcomes)]
            probabilities = np.random.dirichlet(np.ones(num_outcomes))

            for outcome, prob in zip(outcomes, probabilities):
                transition_probs[state][action][outcome] = prob
                reward_function[state][action][outcome] = random.randint(-10, 10)

    return {'transition_probs': transition_probs, 'reward_function': reward_function}


# Функция для генерации эталонных значений Q-таблицы
def generate_optimal_q_values(mdp, gamma=0.9, max_steps=100):
    transition_probs = mdp['transition_probs']
    reward_function = mdp['reward_function']
    states = list(transition_probs.keys())
    actions = list(transition_probs[states[0]].keys())

    q_table = {state: {action: 0 for action in actions} for state in states}

    for _ in range(max_steps):
        for state in states:
            for action in actions:
                expected_value = 0
                for next_state, prob in transition_probs[state][action].items():
                    reward = reward_function[state][action][next_state]
                    max_next_q = max(q_table[next_state].values())
                    expected_value += prob * (reward + gamma * max_next_q)
                q_table[state][action] = expected_value

    return q_table


# Функция для сравнения двух Q-таблиц
def compare_q_tables(q_table1, q_table2, threshold=10):
    for state, actions in q_table1.items():
        for action, value1 in actions.items():
            value2 = q_table2[state][action]
            if abs(value1 - value2) > threshold:
                return True
    return False


# Функция для создания специально разработанного МППР с циклическими зависимостями
def create_special_mdp1():
    transition_probs = {
        's0': {'a0': {'s1': 0.8, 's2': 0.2}, 'a1': {'s3': 1.0}},
        's1': {'a0': {'s0': 0.5, 's4': 0.5}, 'a1': {'s2': 1.0}},
        's2': {'a0': {'s3': 0.6, 's4': 0.4}, 'a1': {'s0': 1.0}},
        's3': {'a0': {'s1': 0.3, 's2': 0.7}, 'a1': {'s4': 1.0}},
        's4': {'a0': {'s0': 0.9, 's1': 0.1}, 'a1': {'s2': 1.0}}
    }

    reward_function = {
        's0': {'a0': {'s1': 1, 's2': -1}, 'a1': {'s3': -2}},
        's1': {'a0': {'s0': 0, 's4': 2}, 'a1': {'s2': -1}},
        's2': {'a0': {'s3': 1, 's4': -1}, 'a1': {'s0': -2}},
        's3': {'a0': {'s1': -1, 's2': 2}, 'a1': {'s4': 1}},
        's4': {'a0': {'s0': 2, 's1': -1}, 'a1': {'s2': -2}}
    }

    return {'transition_probs': transition_probs, 'reward_function': reward_function}


# Функция для создания специально разработанного МППР с несбалансированными вероятностями
def create_special_mdp2():
    transition_probs = {
        's0': {'a0': {'s1': 0.99, 's2': 0.01}, 'a1': {'s3': 0.01, 's4': 0.99}},
        's1': {'a0': {'s0': 0.01, 's4': 0.99}, 'a1': {'s2': 0.99, 's3': 0.01}},
        's2': {'a0': {'s3': 0.99, 's4': 0.01}, 'a1': {'s0': 0.01, 's1': 0.99}},
        's3': {'a0': {'s1': 0.99, 's2': 0.01}, 'a1': {'s4': 0.99, 's0': 0.01}},
        's4': {'a0': {'s0': 0.99, 's1': 0.01}, 'a1': {'s2': 0.99, 's3': 0.01}}
    }

    reward_function = {
        's0': {'a0': {'s1': 5, 's2': -5}, 'a1': {'s3': -3, 's4': -6}},
        's1': {'a0': {'s0': -2, 's4': 6}, 'a1': {'s2': -3, 's3': 4}},
        's2': {'a0': {'s3': 3, 's4': -3}, 'a1': {'s0': -2, 's1': 3}},
        's3': {'a0': {'s1': -4, 's2': 4}, 'a1': {'s4': 3, 's0': -6}},
        's4': {'a0': {'s0': 6, 's1': -3}, 'a1': {'s2': -2, 's3': 3}}
    }

    return {'transition_probs': transition_probs, 'reward_function': reward_function}


# Функция для создания специально разработанного МППР с комбинированными структурами
def create_special_mdp3():
    transition_probs = {
        's0': {'a0': {'s1': 0.9, 's2': 0.1}, 'a1': {'s3': 0.8, 's4': 0.2}},
        's1': {'a0': {'s0': 0.6, 's4': 0.4}, 'a1': {'s2': 0.7, 's3': 0.3}},
        's2': {'a0': {'s3': 0.5, 's4': 0.5}, 'a1': {'s0': 0.3, 's1': 0.7}},
        's3': {'a0': {'s1': 0.4, 's2': 0.6}, 'a1': {'s4': 0.9, 's0': 0.1}},
        's4': {'a0': {'s0': 0.8, 's1': 0.2}, 'a1': {'s2': 0.6, 's3': 0.4}}
    }

    reward_function = {
        's0': {'a0': {'s1': 5, 's2': -5}, 'a1': {'s3': -3, 's4': -6}},
        's1': {'a0': {'s0': -2, 's4': 6}, 'a1': {'s2': -3, 's3': 4}},
        's2': {'a0': {'s3': 3, 's4': -3}, 'a1': {'s0': -2, 's1': 3}},
        's3': {'a0': {'s1': -4, 's2': 4}, 'a1': {'s4': 3, 's0': -6}},
        's4': {'a0': {'s0': 6, 's1': -3}, 'a1': {'s2': -2, 's3': 3}}
    }

    return {'transition_probs': transition_probs, 'reward_function': reward_function}


# Функция для создания специально разработанного МППР с циклом и несбалансированными вероятностями
def create_special_mdp4():
    transition_probs = {
        's0': {'a0': {'s1': 0.9, 's2': 0.1}, 'a1': {'s3': 0.8, 's4': 0.2}},
        's1': {'a0': {'s0': 0.6, 's4': 0.4}, 'a1': {'s2': 0.7, 's3': 0.3}},
        's2': {'a0': {'s3': 0.5, 's4': 0.5}, 'a1': {'s0': 0.3, 's1': 0.7}},
        's3': {'a0': {'s1': 0.4, 's2': 0.6}, 'a1': {'s4': 0.9, 's0': 0.1}},
        's4': {'a0': {'s0': 0.8, 's1': 0.2}, 'a1': {'s2': 0.6, 's3': 0.4}}
    }

    reward_function = {
        's0': {'a0': {'s1': 5, 's2': -5}, 'a1': {'s3': -3, 's4': -6}},
        's1': {'a0': {'s0': -2, 's4': 6}, 'a1': {'s2': -3, 's3': 4}},
        's2': {'a0': {'s3': 3, 's4': -3}, 'a1': {'s0': -2, 's1': 3}},
        's3': {'a0': {'s1': -4, 's2': 4}, 'a1': {'s4': 3, 's0': -6}},
        's4': {'a0': {'s0': 6, 's1': -3}, 'a1': {'s2': -2, 's3': 3}}
    }

    return {'transition_probs': transition_probs, 'reward_function': reward_function}


# Функция для генерации и тестирования МППР
def generate_and_test_mdps(num_mdps=10, threshold=10, episodes=10000, max_steps=100):
    problematic_mdps = []

    # Добавление специально разработанных МППР
    special_mdps = [
        create_special_mdp1(),
        create_special_mdp2(),
        create_special_mdp3(),
        create_special_mdp4()
    ]

    for special_mdp in special_mdps:
        optimal_q_values_special = generate_optimal_q_values(special_mdp)
        learned_q_values_special = q_learning(special_mdp, episodes=episodes, max_steps=max_steps)

        if compare_q_tables(optimal_q_values_special, learned_q_values_special, threshold):
            problematic_mdps.append(special_mdp)

    # Генерация случайных МППР
    for _ in range(num_mdps):
        mdp = create_random_mdp()
        optimal_q_values = generate_optimal_q_values(mdp)
        learned_q_values = q_learning(mdp, episodes=episodes, max_steps=max_steps)

        if compare_q_tables(optimal_q_values, learned_q_values, threshold):
            problematic_mdps.append(mdp)

    return problematic_mdps


problematic_mdps = generate_and_test_mdps()
with open('submit_up2.json', 'w') as f:
    json.dump(problematic_mdps, f, indent=4)

print(f"Сохранено {len(problematic_mdps)} проблемных МППР.")