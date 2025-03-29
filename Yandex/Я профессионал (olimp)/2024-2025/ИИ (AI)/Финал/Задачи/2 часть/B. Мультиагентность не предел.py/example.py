import json
import numpy as np


def main():
    # значения полезностей, которые необходимо заполнить
    # индексы соответствуют наблюдениям, которые доступны в data.npz['submit']
    with open('submit.json', 'r') as f:
        submit = json.load(f)

    print(submit[:3])

    # загружаем данные
    data = np.load('data.npz', allow_pickle=True)

    # в списке submit находятся наблюдения каждого из агентов
    # для них нужно предсказать общую полезность и записать в submit
    submit_observations = data['submit'].tolist()
    print('idx:', submit_observations[0]['idx'], 'shape:', submit_observations[0]['observations'].shape)

    # в списке train находятся данные для обучения
    # которые представляют собой эпизоды взаимодействия агента со средой
    # т.е. списки observations, actions, rewards
    train = data['train'].tolist()
    print(train[0].keys())
    print('rewards:', train[0]['rewards'][:10])


if __name__ == '__main__':
    main()
