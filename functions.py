from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model


# Definition of a system of differential equations
def system(y, t, a, b, c, d):
    y1, y2 = y
    dy1_dt = a * y1 + b * y2
    dy2_dt = c * y1 + d * y2
    return [dy1_dt, dy2_dt]


# Function for calculating MSE
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Definition of the NeuralODE model for the system of equations
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


# Function for a single experiment
def conduct_experiment(a, b, c, d, experiment_id):
    # Initial conditions for all experiments
    y0 = [1.0, 2.0]
    time_points = np.linspace(0, 5, 100)

    # Solving the system of equations and adding noise to the data
    solution = odeint(system, y0, time_points, args=(a, b, c, d))
    noise = np.random.normal(0, 0.5, solution.shape)
    noisy_data = solution + noise

    # Preparing data for training
    x_train = [noisy_data, time_points.reshape(-1, 1)]
    y_train = [np.full((len(time_points), 1), a),
               np.full((len(time_points), 1), b),
               np.full((len(time_points), 1), c),
               np.full((len(time_points), 1), d)]

    # Model training
    neural_ode_model_system.fit(x_train, y_train, epochs=200, verbose=1)

    # Prediction of parameters
    predicted_params = neural_ode_model_system.predict(x_train)

    # Predicted parameters
    predicted_a, predicted_b, predicted_c, predicted_d = [p.flatten()[0] for p in predicted_params]

    # Results output
    print(f"Эксперимент {experiment_id}:")
    print("Истинные параметры:", [a, b, c, d])
    print("Предсказанные параметры (с шумом):", [predicted_a, predicted_b, predicted_c, predicted_d])

    # Prediction of parameters based on true data (without noise)
    predicted_params_clean = neural_ode_model_system.predict([solution, time_points.reshape(-1, 1)])
    predicted_a_clean, predicted_b_clean, predicted_c_clean, predicted_d_clean = [p.flatten()[0] for p in
                                                                                  predicted_params_clean]
    print("Предсказанные параметры (без шума):",
          [predicted_a_clean, predicted_b_clean, predicted_c_clean, predicted_d_clean])

    # Creating a predicted solution based on the predicted parameters
    predicted_solution = odeint(system, y0, time_points, args=(predicted_a, predicted_b, predicted_c, predicted_d))

    # MSE calculation for predicted parameters
    mse = calculate_mse(solution, predicted_solution)
    print("MSE:", mse)

    # Visualization of results
    plt.figure(figsize=(10, 8))

    # Subgraph for y1
    plt.subplot(2, 1, 1)
    plt.scatter(time_points, noisy_data[:, 0], label='Симулированные данные y1 с шумом', color='blue')
    plt.scatter(time_points, solution[:, 0], label='Входные данные y1', color='orange')
    plt.plot(time_points, predicted_solution[:, 0], label='Оптимизированные данные y1', color='red')
    plt.plot(time_points, solution[:, 0], label='Истинные данные y1', color='green')
    plt.xlabel('Время')
    plt.ylabel('y1')
    plt.title(f"Результаты эксперимента {experiment_id} для y1")
    plt.legend()

    # Subgraph for y2
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
