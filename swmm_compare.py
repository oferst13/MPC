from datetime import datetime

#import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
#import benchmark as bm
import pyswmm
from scipy import integrate

#swmm_flow = np.zeros((4, bm.sim_len), dtype='float')
#tank_flow = np.zeros((3, bm.sim_len), dtype='float')
#overflows = source.overflows.copy()
#releases = source.releases_volume.copy()
with pyswmm.Simulation('clustered.inp') as sim:
    tank1_s = pyswmm.Nodes(sim)['tank1']
    tank2_s = pyswmm.Nodes(sim)['tank2']
    tank3_s = pyswmm.Nodes(sim)['tank3']
    tank4_s = pyswmm.Nodes(sim)['tank4']
    outfall_s = pyswmm.Nodes(sim)['outfall']
    node111_s = pyswmm.Nodes(sim)['111']
    node11_s = pyswmm.Nodes(sim)['11']
    node12_s = pyswmm.Nodes(sim)['12']
    node11_s = pyswmm.Nodes(sim)['1']
    node21_s = pyswmm.Nodes(sim)['21']
    node2_s = pyswmm.Nodes(sim)['2']
    sim.start_time = datetime(2021, 1, 1, 0, 0, 0)
    sim.end_time = datetime(2021, 1, 2)
    #sim.step_advance(cfg.dt)
    i = 0
    for step in sim:
        tank_flow[0, i] = 1000*(overflows[0, i] + releases[0, i]) / bm.dt
        tank1_s.generated_inflow(float(tank_flow[0, i]))
        tank_flow[1, i] = 1000 * (overflows[1, i] + releases[1, i]) / bm.dt
        tank2_s.generated_inflow(float(tank_flow[1, i]))
        tank_flow[2, i] = 1000 * (overflows[2, i] + releases[2, i]) / bm.dt
        tank3_s.generated_inflow(float(tank_flow[2, i]))
        '''
        swmm_flow[0, i] = j1.total_inflow
        swmm_flow[1, i] = j2.total_inflow
        swmm_flow[2, i] = j3.total_inflow
        swmm_flow[3, i] = out.total_inflow
        i += 1
        '''

plt.plot(source.hours[0:bm.zero_Q + 100], source.pipe_Q[2, :bm.zero_Q + 100, 1], label="kinemtic")
plt.plot(source.hours[0:bm.zero_Q + 100], 0.001*swmm_flow[3, :bm.zero_Q + 100], label="dynamic")
plt.legend()
plt.show()
swmm_sum = integrate.simps(swmm_flow*bm.dt/1000, bm.t[0:-1])
print('_')
