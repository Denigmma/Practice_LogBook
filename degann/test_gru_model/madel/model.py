import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Определение модели
model = Sequential()

# Добавляем слой GRU
model.add(GRU(units=64, return_sequences=True, input_shape=(L, 1)))

# Добавляем еще один слой GRU (при необходимости)
model.add(GRU(units=64, return_sequences=False))

# Добавляем полносвязный слой для предсказания результата
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer='adam', loss='mse')

# Просмотр архитектуры модели
model.summary()
