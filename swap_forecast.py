import numpy as np
import cfg


rain_event = '09-10'
suffix = 'rand.csv'
num_forecast_files = cfg.forecast_files
for i in range(1, num_forecast_files+1, 2):
    idx = str(i)
    input_filename = '-'.join([rain_event, idx])
    input_filename =''.join([input_filename, suffix])
    print(input_filename)
    rain = np.genfromtxt(input_filename, delimiter=',')