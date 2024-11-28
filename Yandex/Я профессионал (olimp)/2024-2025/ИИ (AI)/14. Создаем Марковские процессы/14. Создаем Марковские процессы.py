import json
import random


def weighted_choice(choices, weights):
    total = sum(weights)
    cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]
    r = random.uniform(0, total)
    for i, cw in enumerate(cumulative_weights):
        if r < cw:
            return choices[i]


def q_learning(mdp, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, max_steps=100, init=10.0):
    transition_probs = mdp['transition_probs']
    reward_function = mdp['reward_function']

    q_table = {state: {action: init for action in actions} for state, actions in transition_probs.items()}
    states = list(transition_probs.keys())

    for episode in range(episodes):
        state = random.choice(states)
        steps = 0

        while steps < max_steps:
            steps += 1
            if random.random() < epsilon:
                action = random.choice(list(transition_probs[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_states = list(transition_probs[state][action].keys())
            probabilities = list(transition_probs[state][action].values())
            next_state = weighted_choice(next_states, weights=probabilities)

            if next_state in reward_function.get(state, {}).get(action, {}):
                reward = reward_function[state][action][next_state]
            else:
                reward = 0

            max_next_q = max(q_table[next_state].values()) if next_state in q_table else init
            q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])

            state = next_state

    return q_table


def compare_q_tables(estimated_q_table, reference_q_table, threshold=10):
    differences = []
    for state, actions in estimated_q_table.items():
        for action, q_value in actions.items():
            if state not in reference_q_table:
                reference_q_table[state] = {}
            if action not in reference_q_table[state]:
                reference_q_table[state][action] = 0

            if abs(q_value - reference_q_table[state][action]) > threshold:
                differences.append(state)
    return differences


def main():
    with open('submit.json', 'r') as f:
        mdps = json.load(f)

    incorrect_mdp_indices = []

    for idx, mdp in enumerate(mdps):
        estimated_q_table = q_learning(mdp)

        reference_q_table = {}
        for state, actions in mdp['transition_probs'].items():
            if state not in reference_q_table:
                reference_q_table[state] = {}
            for action in actions.keys():
                if action not in reference_q_table[state]:
                    reference_q_table[state][action] = 0

        diff = compare_q_tables(estimated_q_table, reference_q_table)
        if diff:
            incorrect_mdp_indices.append(idx)

    result = {"incorrect_mdp_indices": incorrect_mdp_indices}

    with open('submit_results.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
