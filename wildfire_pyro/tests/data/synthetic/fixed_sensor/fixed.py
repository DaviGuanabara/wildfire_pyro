import pandas as pd
import numpy as np
import math

# 2*pi corresponds to 24 hours

number_of_sensors = 100
number_of_measuments = 50
sensor_lifetime = (math.pi/2, math.pi) # 6h to 12h
times = (0, 8 * math.pi) # 4 days of data

def generate_sensor_data():
    lifetime = np.random.uniform(sensor_lifetime[0], sensor_lifetime[1])

    start_time = np.random.uniform(times[0], times[1]-lifetime) # start at any time

    # A dataframe with data collected by the sensor
    df = pd.DataFrame({
        't': np.linspace(start_time, start_time+lifetime, number_of_measuments), # perform the same number of measurement uniformly during lifespan
        'lat': np.random.uniform(0, 1), # position fixed
        'lon': np.random.uniform(0, 1),
    })

    # Calculate `y` column
    df['y'] = np.sin(df['t']) * df['lat'] + np.cos(df['t']) * df['lon']

    return df

# Generate data for all sensors
data = pd.concat(
    [generate_sensor_data() for _ in range(number_of_sensors)])

# Save data to a file
data.to_csv('fixed.csv', index=False)

training = data[data['t'] < 4 * math.pi] # first 2 days
validation = data[(data['t'] >= 4 * math.pi) & (data['t'] < 6 * math.pi)] # 3rd day
testing = data[data['t'] >= 6 * math.pi] # 4th day


training.to_csv('fixed_train.csv', index=False)
validation.to_csv('fixed_val.csv', index=False)
testing.to_csv('fixed_test.csv', index=False)
