import json
import random


def weighted_choice(choices, weights):
    # Рассчитать суммарный вес
    total = sum(weights)
    # Построить кумулятивные веса (накопленные суммы)
    cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]
    # Случайным образом выбрать число от 0 до total
    r = random.uniform(0, total)
    # Найти и вернуть соответствующий выбор
    for i, cw in enumerate(cumulative_weights):
        if r < cw:
            return choices[i]


def q_learning(mdp, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, max_steps=100, init=10.0):
    # Извлечь вероятности переходов и функцию награды из модели MDP
    transition_probs = mdp['transition_probs']
    reward_function = mdp['reward_function']

    # Инициализация Q-таблицы
    q_table = {state: {action: init for action in actions} for state, actions in transition_probs.items()}
    # Список всех состояний
    states = list(transition_probs.keys())

    for episode in range(episodes):
        # Начальное состояние выбирается случайно
        state = random.choice(states)
        steps = 0

        while steps < max_steps:
            steps += 1

            # Выбор действия: либо случайное действие (exploration), либо действие с максимальным Q-значением (exploitation)
            if random.random() < epsilon:
                action = random.choice(list(transition_probs[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)

            # Определить следующее состояние с учетом вероятностей переходов
            next_states = list(transition_probs[state][action].keys())
            probabilities = list(transition_probs[state][action].values())
            next_state = weighted_choice(next_states, weights=probabilities)

            # Получить награду за выбранное действие
            reward = reward_function[state][action][next_state]

            # Найти максимальное Q-значение для следующего состояния
            max_next_q = max(q_table[next_state].values()) if next_state in q_table else init
            # Обновить Q-значение текущего состояния и действия
            q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])

            # Перейти в следующее состояние
            state = next_state

    # Вернуть обновленную Q-таблицу
    return q_table


def main():
    with open('submit.json', 'r') as f:
        mdps = json.load(f)
    for mdp in mdps:
        print(q_learning(mdp))


if __name__ == '__main__':
    main()
