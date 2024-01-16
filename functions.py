from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Определение системы дифференциальных уравнений
def system(y, t, a, b, c, d):
	y1, y2 = y
	dy1_dt = a * y1 + b * y2
	dy2_dt = c * y1 + d * y2
	return [dy1_dt, dy2_dt]


# Функция для расчета MSE
def calculate_mse(y_true, y_pred):
	return np.mean((y_true - y_pred) ** 2)


# Определение модели NeuralODE для системы уравнений
def build_neural_ode_model_system():
	input_y = Input(shape=(2,))
	input_t = Input(shape=(1,))
	combined_input = Concatenate()([input_y, input_t])
	x = Dense(units=64, activation='relu')(combined_input)
	x = Dense(units=64, activation='relu')(x)
	x = Dense(units=64, activation='relu')(x)
	output_a = Dense(units=1, name='output_a')(x)
	output_b = Dense(units=1, name='output_b')(x)
	output_c = Dense(units=1, name='output_c')(x)
	output_d = Dense(units=1, name='output_d')(x)
	model = Model(inputs=[input_y, input_t], outputs=[output_a, output_b, output_c, output_d])
	return model


neural_ode_model_system = build_neural_ode_model_system()
neural_ode_model_system.compile(optimizer='adam', loss='mse')


# Функция для проведения одного эксперимента
def conduct_experiment(a, b, c, d, experiment_id):
	# Начальные условия для всех экспериментов
	y0 = [1.0, 2.0]
	time_points = np.linspace(0, 5, 100)

	# Решение системы уравнений и добавление шума к данным
	solution = odeint(system, y0, time_points, args=(a, b, c, d))
	noise = np.random.normal(0, 0.5, solution.shape)
	noisy_data = solution + noise

	# Подготовка данных для обучения
	X_train = [noisy_data, time_points.reshape(-1, 1)]
	y_train = [np.full((len(time_points), 1), a), np.full((len(time_points), 1), b), np.full((len(time_points), 1), c),
	           np.full((len(time_points), 1), d)]

	# Обучение модели
	neural_ode_model_system.fit(X_train, y_train, epochs=200, verbose=1)

	# Предсказание параметров
	predicted_params = neural_ode_model_system.predict(X_train)

	# Предсказанные параметры
	predicted_a, predicted_b, predicted_c, predicted_d = [p.flatten()[0] for p in predicted_params]

	# Вывод результатов
	print(f"Эксперимент {experiment_id}:")
	print("Истинные параметры:", [a, b, c, d])
	print("Предсказанные параметры (с шумом):", [predicted_a, predicted_b, predicted_c, predicted_d])

	# Предсказание параметров на основе истинных данных (без шума)
	predicted_params_clean = neural_ode_model_system.predict([solution, time_points.reshape(-1, 1)])
	predicted_a_clean, predicted_b_clean, predicted_c_clean, predicted_d_clean = [p.flatten()[0] for p in
	                                                                              predicted_params_clean]
	print("Предсказанные параметры (без шума):",
	      [predicted_a_clean, predicted_b_clean, predicted_c_clean, predicted_d_clean])

	# Создание предсказанного решения на основе предсказанных параметров
	predicted_solution = odeint(system, y0, time_points, args=(predicted_a, predicted_b, predicted_c, predicted_d))

	# Расчёт MSE для предсказанных параметров
	mse = calculate_mse(solution, predicted_solution)
	print("MSE:", mse)

	# Визуализация результатов
	plt.figure(figsize=(10, 8))

	# Подграфик для y1
	plt.subplot(2, 1, 1)
	plt.scatter(time_points, noisy_data[:, 0], label='Симулированные данные y1 с шумом', color='blue')
	plt.scatter(time_points, solution[:, 0], label='Входные данные y1', color='orange')
	plt.plot(time_points, predicted_solution[:, 0], label='Оптимизированные данные y1', color='red')
	plt.plot(time_points, solution[:, 0], label='Истинные данные y1', color='green')
	plt.xlabel('Время')
	plt.ylabel('y1')
	plt.title(f"Результаты эксперимента {experiment_id} для y1")
	plt.legend()

	# Подграфик для y2
	plt.subplot(2, 1, 2)
	plt.scatter(time_points, noisy_data[:, 1], label='Симулированные данные y2 с шумом', color='blue')
	plt.scatter(time_points, solution[:, 1], label='Входные данные y2', color='orange')
	plt.plot(time_points, predicted_solution[:, 1], label='Оптимизированные данные y2', color='red')
	plt.plot(time_points, solution[:, 1], label='Истинные данные y2', color='blue')
	plt.xlabel('Время')
	plt.ylabel('y2')
	plt.title(f"Результаты эксперимента {experiment_id} для y2")
	plt.legend()
	plt.tight_layout()
	plt.show()
