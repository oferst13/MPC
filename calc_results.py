import pandas as pd
from scipy import integrate
import copy
from tank import Tank
from pipe import Pipe
from node import Node
import cfg
import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
import pyswmm
from datetime import datetime, timedelta


class Scenario:
    def __init__(self):
        self.reset_scenario()

    def reset_scenario(self):
        self.last_outflow = 0
        self.max_flow = None
        self.last_Q = None
        self.release_intervals = None
        self.obj_Q = None
        self.fitness = None
        self.rw_supply = None
        self.outfall_flow = None
        self.swmm_flow = None
        self.available_water = None
        self.max_swmm_flow = None

    def calc_obj_Q(self):
        tot_out_vol = outfall.get_outflow_volume()
        if tot_out_vol > 0.0001:
            # self.obj_Q = tot_out_vol / (Tank.get_last_overflow() * cfg.dt)
            self.obj_Q = tot_out_vol / (outfall.get_zero_Q() * cfg.dt)
        else:
            self.obj_Q = 0.00001

    def set_fitness(self):
        to_min: float = 0
        for i in range(self.last_Q):
            to_min += abs(outfall.get_flow(i) - self.obj_Q)
        self.fitness = to_min

    def set_release_intervals(self):
        self.release_intervals = math.ceil(Tank.get_last_overflow() * (cfg.dt / cfg.release_dt))
        if swmm_optim:
            self.release_intervals = int(cfg.forecast_len * (cfg.dt / cfg.release_dt))

    def set_last_outflow(self):
        self.last_outflow = Tank.get_last_overflow()

    def set_max_flow(self):
        self.max_flow = outfall.get_max_Q()

    def set_max_swmm_flow(self):
        self.max_swmm_flow = np.max(self.swmm_flow)

    def set_last_Q(self):
        self.last_Q = outfall.get_zero_Q()

    def set_rw_supply(self):
        self.rw_supply = Tank.get_rw_supply_all()

    def set_outfall_flow(self):
        if outfall.lat_flows is None:
            self.outfall_flow = copy.copy(pipe6.outlet_Q)
        else:
            self.outfall_flow = pipe6.outlet_Q + np.pad(outfall.lat_flows,
                                                        (0, len(pipe6.outlet_Q) - len(outfall.lat_flows)), 'constant')

    def set_atts(self):
        self.set_last_outflow()
        self.set_max_flow()
        self.set_last_Q()
        self.set_release_intervals()
        self.calc_obj_Q()
        self.set_fitness()
        self.set_rw_supply()
        self.set_outfall_flow()
        self.set_available_water()

    def set_swmm_flow(self, flow):
        self.swmm_flow = flow
        self.set_max_swmm_flow()

    def set_available_water(self):
        self.available_water = self.rw_supply + Tank.get_tot_storage()


def set_forecast_idx(first, last, diff):
    return np.arange(first, last, diff)


def calc_fitness():
    to_min: float = 0
    for i in range(outfall.get_zero_Q()):
        to_min += abs(outfall.get_flow(i) - baseline.obj_Q)
    return to_min


def set_demands_per_dt():
    demands = np.array([])
    for demand in cfg.demands_3h:
        demands = np.append(demands, np.ones(int(cfg.demand_dt / cfg.dt)) * (demand * (cfg.dt / cfg.demand_dt)))
    demand_PD = demands * cfg.PD / 100
    return demand_PD


def set_rain_input(rainfile, rain_dt, duration):
    rain = np.zeros(int(duration / (rain_dt / cfg.dt)))
    rain_input = np.genfromtxt(rainfile, delimiter=',')
    rain[:len(rain_input)] = rain_input
    return rain


def rain_input_from_array(array, duration, idx=None, forecast_type='actual'):
    rain = np.zeros(int(duration / (cfg.rain_dt / cfg.dt)))
    if idx is not None:
        rain_input = array[idx]
    else:
        rain_input = array
    rain[:len(rain_input)] = rain_input
    return rain


def set_rain_filename(prefix, idx, is_forecast):
    idx_str = str(idx)
    cur_filename = '-'.join([prefix, idx_str])
    if is_forecast:
        cur_filename = 'swap.'.join([cur_filename, 'csv'])
    else:
        cur_filename = '.'.join([cur_filename, 'csv'])
    return cur_filename


def calc_mass_balance():
    mass_balance = 100 * (abs(outfall.get_outflow_volume() - Tank.get_cum_outflow())) / (Tank.get_cum_outflow())
    return mass_balance


def run_model(duration, rain, swmm_=False):
    for i in range(duration):
        if sum(rain[int(i // (cfg.rain_dt / cfg.dt)):-1]) + Tank.get_tot_storage() == 0:
            break  # this should break forecast run only!
        for tank in Tank.all_tanks:
            tank.tank_fill(i)
            tank.calc_release(i, baseline.last_outflow)
            tank.rw_use(i, event_start_idx)
        if (Pipe.get_tot_Q(i - 1) + Tank.get_tot_outflow(i)) < 1e-3:
            if swmm_ is False:
                continue
            try:
                if np.sum(lat_flows[:, i]) < 0.01:
                    continue
            except IndexError:
                continue
        for node in Node.all_nodes:
            node.handle_flow(i, swmm=swmm_)
            for pipe in node.giving_to:
                pipe.calc_q_outlet(i)


def dump_to_file(thingy, filename):
    with open(r'policies/' + filename, 'wb') as file:
        pickle.dump(thingy, file)


def unload_from_file(filename):
    with open(r'policies/' + filename, 'rb') as file:
        thingy = pickle.load(file)
    return thingy


def plot_compare(outflow1, outflow2, units, outflow3=None, legend1='Baseline', legend2='Controlled',
                 legend3='Tank Outflows'):
    plt.rc('font', size=11)
    if units == 'CMS':
        cutoff = 0.0005
    else:
        cutoff = 0.5
    last_q = np.max(np.nonzero(outflow1 > cutoff))
    plot_hours = np.ceil(last_q * cfg.dt / 3600)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
    fig.set_size_inches(7.5, 5)
    rain_hours = np.linspace(0, int(cfg.sim_days * 24), int(cfg.sim_days * 24 * 3600 / cfg.rain_dt) + 1,
                             dtype='longfloat')
    axs[0].bar(rain_hours[np.nonzero(rain_hours <= plot_hours)],
               act_rain[0:len(rain_hours[np.nonzero(rain_hours <= plot_hours)])],
               width=cfg.rain_dt / 3600,
               align='edge')
    axs[0].spines['bottom'].set_visible(False)
    # axs[0].axes.xaxis.set_visible(False)
    axs[0].tick_params(labelbottom=False)
    axs[0].set_xlim([0, plot_hours])
    # axs[0].set_ylim([0,5])
    axs[0].set_ylabel('Rain\n (mm/10-minutes)')
    axs[0].invert_yaxis()
    axs[0].grid(True)
    axs[1].plot(cfg.hours[np.nonzero(cfg.hours <= plot_hours)],
                outflow1[0:len(cfg.hours[np.nonzero(cfg.hours <= plot_hours)])], 'r-',
                label=legend1)
    axs[1].plot(cfg.hours[np.nonzero(cfg.hours <= plot_hours)],
                outflow2[0:len(cfg.hours[np.nonzero(cfg.hours <= plot_hours)])], 'b-',
                label=legend2)
    if outflow3:
        axs[1].plot(cfg.hours[np.nonzero(cfg.hours <= plot_hours)],
                    outflow3[0:len(cfg.hours[np.nonzero(cfg.hours <= plot_hours)])],
                    'g--', label=legend3)
    if units == 'CMS':
        axs[1].set_ylabel('Outfall Flow Rate ' + r'($\frac{m^3}{s}$)')
    else:
        axs[1].set_ylabel('Outfall Flow Rate (LPS)')
    axs[1].set_xlabel('t (hours)')
    axs[1].set_xlim([0, plot_hours])
    axs[1].set_ylim(bottom=0)
    axs[1].spines['top'].set_visible(False)
    axs[1].grid(True)
    fig.tight_layout(pad=0)
    plt.legend()
    plt.show()


def swmm_run(rain, hours, filename):
    outfall_s_flow = np.zeros(int(hours * 3600 / cfg.dt) + 1)
    with pyswmm.Simulation(filename) as sim:
        sim.step_advance(cfg.dt)
        outfall_s = pyswmm.Nodes(sim)['outfall']
        rg1 = pyswmm.RainGages(sim)['RG1']
        tank1_s = pyswmm.Nodes(sim)['tank1']
        tank2_s = pyswmm.Nodes(sim)['tank2']
        tank3_s = pyswmm.Nodes(sim)['tank3']
        tank4_s = pyswmm.Nodes(sim)['tank4']
        sim.start_time = datetime(2021, 1, 1, 0, 0, 0)
        sim.end_time = sim.start_time + timedelta(minutes=(hours * 60) + 1)
        tank_list = [tank1_s, tank2_s, tank3_s, tank4_s]
        i = 0
        for step in sim:
            for idx, tank_node in enumerate(tank_list):
                inflow = Tank.all_tanks[idx].get_outflow(i) * 1000
                tank_node.generated_inflow(float(inflow))
            rg1.total_precip = rain[int(i // (cfg.rain_dt / cfg.dt))] * 6
            outfall_s_flow[i] = outfall_s.total_inflow
            i += 1
            print(sim.current_time)
    return outfall_s_flow


def swmm_run_inflows(rain, hours, filename):
    with pyswmm.Simulation(filename) as sim:
        sim.step_advance(cfg.dt)
        outfall_s = pyswmm.Nodes(sim)['outfall']
        rg1 = pyswmm.RainGages(sim)['RG1']
        j111 = pyswmm.Nodes(sim)['111']
        j11 = pyswmm.Nodes(sim)['11']
        j12 = pyswmm.Nodes(sim)['12']
        j21 = pyswmm.Nodes(sim)['21']
        sim.start_time = datetime(2021, 1, 1, 0, 0, 0)
        sim.end_time = sim.start_time + timedelta(minutes=(hours * 60) + 1)
        node_list = [j111, j11, j12, j21, outfall_s]
        node_inflows = np.zeros((len(node_list), int(hours * 3600 / cfg.dt)))
        i = 0
        for step in sim:
            rg1.total_precip = rain[int(i // (cfg.rain_dt / cfg.dt))] * 6
            for idx, node in enumerate(node_list):
                node_inflows[idx, i] = node.lateral_inflow
            # outfall_s_flow[step] = outfall_s.total_inflow
            i += 1
            print(sim.current_time)
    return node_inflows


def swmm_compare(rain):  # has to be coded explicitly :(
    outfall_s_flow = np.zeros(len(pipe6.outlet_Q))
    with pyswmm.Simulation('clustered-no_roof.inp') as sim:
        tank1_s = pyswmm.Nodes(sim)['tank1']
        tank2_s = pyswmm.Nodes(sim)['tank2']
        tank3_s = pyswmm.Nodes(sim)['tank3']
        tank4_s = pyswmm.Nodes(sim)['tank4']
        outfall_s = pyswmm.Nodes(sim)['outfall']
        rg1 = pyswmm.RainGages(sim)['RG1']
        sim.start_time = datetime(2021, 1, 1, 0, 0, 0)
        sim.end_time = datetime(2021, 1, 1, 22)
        sim.step_advance(cfg.dt)
        i = 0
        tank_list = [tank1_s, tank2_s, tank3_s, tank4_s]
        for step in sim:
            for idx, tank_node in enumerate(tank_list):
                inflow = Tank.all_tanks[idx].get_outflow(i) * 1000
                # if inflow > 0:
                # print(inflow)
                tank_node.generated_inflow(float(inflow))
            # rg1.total_precip = rain[int(i // (cfg.rain_dt / cfg.dt))] * 6
            outfall_s_flow[i] = outfall_s.total_inflow
            if (i - 1) % cfg.dt == 0:
                print(sim.current_time)
            i += 1
    plt.plot(cfg.hours[:len(pipe6.outlet_Q)], pipe6.outlet_Q, label='kinematic')
    plt.plot(cfg.hours[:len(pipe6.outlet_Q)], 0.001 * outfall_s_flow, label='dynamic')
    plt.legend()
    plt.show()


def plot_release_policy(release_arr):
    plt.figure()
    plt.rc('font', size=11)
    plot_hours = np.ceil(baseline.last_Q * cfg.dt / 3600)
    t = np.arange(0, plot_hours + 1, cfg.control_interval / 3600)
    releases_2plot = np.c_[release_arr, np.zeros((len(Tank.all_tanks), int(len(t) - release_arr.shape[1])))]
    ls = ['-', '-.', ':', '--']
    cl = ['xkcd:dark sky blue', 'r', 'xkcd:goldenrod', 'xkcd:kiwi']
    for gg, graph in enumerate(releases_2plot):
        plt.step(t, graph * 10, cl[gg], where='post', label=f'Tank {gg + 1}', linewidth=4 - 0.7 * gg, linestyle=ls[gg])
    fig = plt.gcf()
    fig.set_size_inches(6, 3.75)
    fig.tight_layout(pad=1.5)
    plt.legend(loc='center right')
    plt.xlabel('t (hours)')
    plt.ylabel('Valve Opening %')
    plt.xlim([0, plot_hours])
    plt.show()


def plot_tank_storage():
    plt.figure()
    plt.rc('font', size=11)
    plot_hours = np.ceil(baseline.last_Q * cfg.dt / 3600)
    t = np.arange(plot_hours + 1)
    plot_storage = np.zeros((len(Tank.all_tanks), cfg.sim_len))
    for idx, tnk in enumerate(Tank.all_tanks):
        plot_storage[idx] = 100 * (tnk.all_storage / tnk.tank_size)
    ls = ['-', '-.', ':', '--']
    cl = ['xkcd:dark sky blue', 'r', 'xkcd:goldenrod', 'xkcd:kiwi']
    for gg, graph in enumerate(plot_storage):
        plt.plot(cfg.hours[np.nonzero(cfg.hours <= plot_hours)], graph[np.nonzero(cfg.hours <= plot_hours)],
                 cl[gg]
                 , label=f'Tank {gg + 1}', linewidth=4 - 0.7 * gg, linestyle=ls[gg])
    fig = plt.gcf()
    # plt.xticks(np.arange(0, plot_hours+1, 1.0))
    fig.set_size_inches(7.5, 5)
    fig.tight_layout(pad=1.5)
    plt.legend(loc='upper right')
    # , bbox_to_anchor=(0.6,0))
    plt.xlabel('t (hours)')
    plt.ylabel('Tank storage %')
    plt.xlim([0, plot_hours])
    plt.show()


def rain_compare():
    for i in range(1, cfg.forecast_files + 1, 2):
        idx = str(i)
        input_filename = '-'.join(['09-10', idx])
        input_real = ''.join([input_filename, '.csv'])
        input_fore = ''.join([input_filename, 'swap.csv'])
        rrain = np.genfromtxt(input_real, delimiter=',')
        fore_rain = np.genfromtxt(input_fore, delimiter=',')
        x = np.arange(len(rrain))
        plt.figure()
        bar1 = plt.bar(x - 0.25, rrain, 0.5)
        bar2 = plt.bar(x + 0.25, fore_rain, 0.5)
        plt.legend((bar1, bar2), ('real', 'forecast'))
        plt.show()


def set_swmm_file(time, state):
    state_key = (time > 0, state)
    filename = cfg.swmm_files[state_key]
    return filename


def set_no_rwh_scenario():
    Pipe.reset_pipe_all(cfg.sim_len, 'factory')
    Tank.reset_all(cfg.sim_len, 'factory')
    Tank.set_inflow_forecast_all(act_rain)
    for tt in Tank.all_tanks:
        tt.tank_size = tt.roof * 0.005
    run_model(cfg.sim_len, act_rain, swmm_optim)
    no_rwh = Scenario()
    no_rwh.set_atts()
    no_rwh.set_swmm_flow(swmm_run(act_rain, 21, 'clustered-no_roof.inp'))
    return no_rwh


def set_passive_scenario(array_like, valve_setting):
    Pipe.reset_pipe_all(cfg.sim_len, 'factory')
    Tank.reset_all(cfg.sim_len, 'factory')
    Tank.set_inflow_forecast_all(act_rain)
    passive_array = np.ones_like(array_like) * valve_setting
    Tank.set_releases_all(passive_array)
    run_model(cfg.sim_len, act_rain, swmm_optim)
    passive = Scenario()
    passive.set_atts()
    passive.set_swmm_flow(swmm_run(act_rain, 21, 'clustered-no_roof.inp'))
    return passive


num_forecasts = len(cfg.rain_array_stacked)
forecast_indices = set_forecast_idx(0, num_forecasts, int(cfg.sample_interval / cfg.forecast_interval))
tank1_dict = {'name': 'tank1', 'n_tanks': 30, 'init_storage': 0, 'roof': 9000, 'dwellers': 180}
tank2_dict = {'name': 'tank2', 'n_tanks': 35, 'init_storage': 0, 'roof': 10000, 'dwellers': 190}
tank3_dict = {'name': 'tank3', 'n_tanks': 25, 'init_storage': 0, 'roof': 8500, 'dwellers': 155}
tank4_dict = {'name': 'tank4', 'n_tanks': 50, 'init_storage': 0, 'roof': 14000, 'dwellers': 645}
# tank1 = Tank('tank1', 30, 0, 9000, 180)
# tank2 = Tank('tank2', 35, 0, 10000, 190)
# tank3 = Tank('tank3', 25, 0, 8500, 150)
# tank4 = Tank('tank4', 50, 0, 14000, 650)
tank1 = Tank(tank1_dict)
tank2 = Tank(tank2_dict)
tank3 = Tank(tank3_dict)
tank4 = Tank(tank4_dict)

outlet1 = Pipe('outlet1', 250, 0.4, 0.02)
outlet2 = Pipe('outlet2', 330, 0.4, 0.015)
outlet3 = Pipe('outlet3', 220, 0.4, 0.02)
outlet4 = Pipe('outlet4', 350, 0.4, 0.007)

pipe1 = Pipe('pipe1', 400, 0.4, 0.0063)
# pipe1 = Pipe('pipe1', 200, 0.4, 0.0125)
pipe2 = Pipe('pipe2', 500, 0.6, 0.002)
# pipe2 = Pipe('pipe2', 200, 0.6, 0.005)
pipe3 = Pipe('pipe3', 400, 0.4, 0.0013)
# pipe3 = Pipe('pipe3', 150, 0.4, 0.0033)
pipe4 = Pipe('pipe4', 400, 0.8, 0.0088)
# pipe4 = Pipe('pipe4', 200, 0.8, 0.0175)
pipe5 = Pipe('pipe5', 300, 0.4, 0.005)
# pipe5 = Pipe('pipe5', 200, 0.4, 0.02)
pipe6 = Pipe('pipe6', 200, 0.8, 0.01)
# pipe6 = Pipe('pipe6', 120, 0.8, 0.017)

tank1_out = Node('tank1_out', [tank1], [outlet1], tank_node=True)
tank2_out = Node('tank1_out', [tank2], [outlet2], tank_node=True)
tank3_out = Node('tank1_out', [tank3], [outlet3], tank_node=True)
tank4_out = Node('tank1_out', [tank4], [outlet4], tank_node=True)

node111 = Node('node111', [outlet1], [pipe1], lat_node=True)
node11 = Node('node11', [pipe1, outlet2], [pipe2], lat_node=True)
node12 = Node('node12', [outlet3], [pipe3], lat_node=True)
node1 = Node('node1', [pipe2, pipe3], [pipe4])
node21 = Node('node21', [outlet4], [pipe5], lat_node=True)
node2 = Node('node2', [pipe4, pipe5], [pipe6])
outfall = Node('outfall', [pipe6], lat_node=True)

demands_PD = set_demands_per_dt()
Tank.set_daily_demands_all(demands_PD)  # happens only once
swmm_optim = True
if swmm_optim is False:
    for node in Node.lat_nodes:
        node.lat_node = False
    Node.lat_nodes = []
real_time = 0

meta_path = 'rain_files/df_rain_files/df_events/meta/'
meta_file = 'all_metas.csv'
meta_df = pd.read_csv(meta_path + meta_file, index_col=False)
for file in cfg.files:
    event_df = pd.read_csv(file, index_col=False)
    rain_header = list(event_df)[1]
    rain_array = event_df[rain_header].to_numpy()
    sim_days = min(math.ceil(len(event_df) * cfg.rain_dt / (3600 * 24)) + 0.5,
                   round(len(event_df) * cfg.rain_dt / (3600 * 24)) + 1)
    sim_len = int(sim_days * 24 * 60 * 60 / cfg.dt)
    event_start_time = datetime.strptime(event_df['Time'][0].split(' ')[1], '%H:%M:%S').time()
    event_start_idx = int(event_start_time.hour * 60 + event_start_time.minute * (60 / cfg.dt)) - 1
    Pipe.reset_pipe_all(sim_len, 'factory')
    Tank.reset_all(sim_len, 'factory')
    baseline = Scenario()
    act_rain = rain_input_from_array(rain_array, sim_len)
    Tank.set_inflow_forecast_all(act_rain)
    lat_flows = swmm_run_inflows(act_rain, sim_days * 24, cfg.swmm_files[(False, 'sim')])
    Node.set_lat_flows_all(lat_flows)
    run_model(sim_len, act_rain, swmm_optim)
    baseline.set_atts()
    baseline.set_swmm_flow(swmm_run(act_rain, sim_days * 24, 'clustered-no_roof.inp'))
    Pipe.reset_pipe_all(sim_len, 'factory')
    Tank.reset_all(sim_len, 'factory')
    Tank.set_inflow_forecast_all(act_rain)
    event_dates = file.split('\\')[1].split('.')[0]
    arr = unload_from_file(event_dates + '-perfect')
    Tank.set_releases_all(arr)
    run_model(sim_len, act_rain, swmm_optim)
    optimized = Scenario()
    optimized.set_atts()
    optimized.set_swmm_flow(swmm_run(act_rain, sim_days * 24, 'clustered-no_roof.inp'))
    event_idx = meta_df.loc[meta_df['Dates'] == file.split('\\')[1].split('.')[0]].index[0]
    meta_df.at[event_idx, 'Baseline_max'] = baseline.max_swmm_flow
    meta_df.at[event_idx, 'Optimized_max'] = optimized.max_swmm_flow
    meta_df.at[event_idx, 'Flow_reduction'] = (baseline.max_swmm_flow - optimized.max_swmm_flow) * 100 / \
                                              baseline.max_swmm_flow
    meta_df.at[event_idx, 'Baseline_water'] = baseline.available_water
    meta_df.at[event_idx, 'Optimized_water'] = optimized.available_water
    meta_df.at[event_idx, 'Water_reduction'] = (baseline.available_water - optimized.available_water) * 100 / \
                                              baseline.available_water
print('end')
