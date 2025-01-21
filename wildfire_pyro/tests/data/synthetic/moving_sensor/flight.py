import pandas as pd
import numpy as np
import math

# 2*pi corresponds to 24 hours
number_of_sensors = 100
number_of_measuments = 50
sensor_lifetime = (math.pi / 12, math.pi / 6)  # from 1h to 2h
times = (0, 8 * math.pi)  # 4 days

# sensor lifetime between 1h and 2h


def generate_sensor_data():
    lifetime = np.random.uniform(sensor_lifetime[0], sensor_lifetime[1])

    start_time = np.random.uniform(times[0], times[1] - lifetime)

    start_position = (np.random.uniform(0, 1), np.random.uniform(0, 1))
    end_position = (np.random.uniform(0, 1), np.random.uniform(0, 1))

    # A dataframe with data collected by the sensor
    df = pd.DataFrame({
        't': np.linspace(start_time, start_time + lifetime, number_of_measuments),
        # sensor moves uniformly form start to end position
        'lat': np.linspace(start_position[0], end_position[0], number_of_measuments),
        'lon': np.linspace(start_position[1], end_position[1], number_of_measuments),
    })

    # Calculate `y` column
    df['y'] = np.sin(df['t']) * df['lat'] + np.cos(df['t']) * df['lon']

    return df


# Generate data for all sensors
data = pd.concat(
    [generate_sensor_data() for _ in range(number_of_sensors)])

# Save data to a file
data.to_csv('flight.csv', index=False)

# Passado
training = data[data['t'] < 4 * math.pi]
validation = data[(data['t'] >= 4 * math.pi) & (data['t'] < 6 * math.pi)]

# Disponível no presente
testing = data[data['t'] >= 6 * math.pi]

training.to_csv('flight_train.csv', index=False)
validation.to_csv('flight_val.csv', index=False)
testing.to_csv('flight_test.csv', index=False)

# Onde vou calcular a métrica
evaluation_set = pd.DataFrame({
    't': times[1],
    'lat': np.linspace(0, 1, 50),
    'lon': np.linspace(0, 1, 50),
})
evaluation_set['y'] = np.sin(evaluation_set['t']) * evaluation_set['lat'] + \
    np.cos(evaluation_set['t']) * evaluation_set['lon']
