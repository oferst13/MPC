import numpy as np
import cfg
import pandas as pd


rain_event = '20-21'
suffix = '.csv'
num_forecast_files = cfg.forecast_files
swap_probs = [0.2, 0.3, 0.4]
vol_probs = [0.2, 0.3, 0.4]
for i in range(1, num_forecast_files + 1, 2):
    idx = str(i)
    input_filename = '-'.join([rain_event, idx])
    input_filename = ''.join([input_filename, suffix])
    print(input_filename)
    rain = np.genfromtxt(input_filename, delimiter=',')
    for timestep in range(len(rain)):
        rain[timestep] += rain[timestep] * np.random.uniform(-vol_probs[timestep // 6], vol_probs[timestep // 6])
        rain[timestep] = np.round_(rain[timestep], 2)
        if np.random.rand() < swap_probs[timestep // 6] and timestep > 0:
            try:
                rain[timestep], rain[timestep+1] = rain[timestep+1], rain[timestep]
                print(timestep, ' swap')
            except IndexError:
                print('last')
    output_filename = '-'.join([rain_event, idx])
    output_filename = 'swap'.join([output_filename, suffix])
    rain_output = pd.DataFrame(rain)
    rain_output.to_csv(output_filename, index=False, header=False)