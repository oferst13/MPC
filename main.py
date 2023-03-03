from scipy import integrate
import copy
from tank import Tank
from pipe import Pipe
from node import Node
import cfg
import numpy as np
import math
from timer import Timer
import pygad
import GA_params as ga
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
            self.obj_Q = tot_out_vol / (Tank.get_last_overflow() * cfg.dt)
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
            tank.rw_use(i)
        if (Pipe.get_tot_Q(i - 1) + Tank.get_tot_outflow(i)) < 1e-3:
            if swmm_ is False:
                continue
            try:
                if np.sum(lat_flows[:, i]) < 0.1:
                    continue
            except IndexError:
                continue
        for node in Node.all_nodes:
            node.handle_flow(i, swmm=swmm_)
            for pipe in node.giving_to:
                pipe.calc_q_outlet(i)


def fitness_func(release_vector, idx):
    for tank in Tank.all_tanks:
        tank.reset_tank(cfg.forecast_len, 'iter')
    for pipe in Pipe.all_pipes:
        pipe.reset_pipe(cfg.forecast_len, 'iter')
    release_array = np.reshape(release_vector, (len(Tank.all_tanks), baseline.release_intervals))
    Tank.set_releases_all(release_array)
    run_model(cfg.forecast_len, forecast_rain, swmm_optim)
    # print(f"Mass Balance Error: {calc_mass_balance():0.2f}%")
    fitness = 1.0 / calc_fitness()
    return float(fitness)


def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


def set_ga_instance():
    ga_inst = pygad.GA(num_generations=ga.num_generations,
                       initial_population=ga.pop_init(baseline.release_intervals, len(Tank.all_tanks)),
                       num_parents_mating=ga.set_parent_num(baseline.release_intervals, len(Tank.all_tanks)),
                       gene_space=ga.gene_space,
                       parent_selection_type=ga.parent_selection,
                       crossover_type=ga.crossover_type,
                       crossover_probability=ga.crossover_prob,
                       mutation_type=ga.mutation_type,
                       mutation_probability=ga.mutation_prob,
                       mutation_by_replacement=ga.mutation_by_replacement,
                       stop_criteria=ga.stop_criteria,
                       fitness_func=fitness_func,
                       on_generation=on_gen)
    return ga_inst


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
        sim.end_time = sim.start_time + timedelta(minutes=(hours*60)+1)
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
        sim.end_time = sim.start_time + timedelta(minutes=(hours*60)+1)
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
optimize = False

if optimize:
    for forecast_idx in forecast_indices:
        # forecast_file = set_rain_filename('20-21', forecast_idx, is_forecast=True)
        # forecast_rain = set_rain_input(forecast_file, cfg.rain_dt, cfg.forecast_len)
        forecast_rain = rain_input_from_array(cfg.rain_array_stacked, cfg.forecast_len, idx=forecast_idx)
        Tank.set_inflow_forecast_all(forecast_rain)  # happens once a forecast is made
        try:
            baseline
        except NameError:
            baseline = Scenario()
        else:
            baseline.reset_scenario()
        for tank in Tank.all_tanks:
            tank.reset_tank(cfg.forecast_len, 'cycle')
        for pipe in Pipe.all_pipes:
            pipe.reset_pipe(cfg.forecast_len, 'cycle')
        sim_state = 'sim'
        swmm_file = set_swmm_file(real_time, sim_state)
        lat_flows = swmm_run_inflows(forecast_rain, cfg.forecast_hr, swmm_file)
        Node.set_lat_flows_all(lat_flows)
        run_model(cfg.forecast_len, forecast_rain, swmm_optim)
        baseline.set_atts()
        zero_Q = outfall.get_zero_Q()
        last_overflow = Tank.get_last_overflow()
        obj_Q = (integrate.simps(baseline.outfall_flow, cfg.t[:len(baseline.outfall_flow)])) / cfg.forecast_len

        if baseline.obj_Q > 0.0001 and Tank.get_cum_overflow() > 0.5:
            ga_instance = set_ga_instance()
            ga_instance.run()
            best_solution = np.reshape(ga_instance.best_solution()[0],
                                       (len(Tank.all_tanks), baseline.release_intervals))
        else:
            best_solution = np.zeros((len(Tank.all_tanks), int(cfg.release_array)))
        if forecast_idx == forecast_indices[0]:
            best_solution_all = best_solution[:, 0:int(cfg.sample_interval / cfg.control_interval)]
        else:
            best_solution_all = np.concatenate(
                (best_solution_all, best_solution[:, 0:int(cfg.sample_interval / cfg.control_interval)]), axis=1)
        Tank.reset_all(cfg.sample_len, 'iter')
        Tank.set_releases_all(best_solution)
        Pipe.reset_pipe_all(cfg.sample_len, 'iter')
        #period_file = set_rain_filename('20-21', forecast_idx, is_forecast=False)
        #period_rain = set_rain_input(period_file, cfg.rain_dt, cfg.forecast_len)
        period_rain = rain_input_from_array(cfg.rain_array_stacked, cfg.forecast_len, idx=forecast_idx)
        Tank.set_inflow_forecast_all(period_rain)
        sim_state = 'real'
        swmm_file = set_swmm_file(real_time, sim_state)
        lat_flows = swmm_run_inflows(period_rain, cfg.sample_hr, swmm_file)
        run_model(cfg.sample_len, period_rain, swmm_optim)
        real_time += cfg.sample_len

    print(best_solution_all)

real_rain = True
if real_rain:
    Pipe.reset_pipe_all(cfg.sim_len, 'factory')
    Tank.reset_all(cfg.sim_len, 'factory')
    try:
        baseline
    except NameError:
        baseline = Scenario()
    else:
        baseline.reset_scenario()
    #act_rain = set_rain_input('09-10.csv', cfg.rain_dt, cfg.sim_len)
    act_rain = rain_input_from_array(cfg.rain_array, cfg.sim_len)
    Tank.set_inflow_forecast_all(act_rain)
    lat_flows = swmm_run_inflows(act_rain, cfg.sim_days*24, cfg.swmm_files[(False, 'sim')])
    Node.set_lat_flows_all(lat_flows)
    run_model(cfg.sim_len, act_rain, swmm_optim)
    baseline.set_atts()
    baseline.set_swmm_flow(swmm_run(act_rain, cfg.sim_days*24, 'clustered-no_roof.inp'))
    Pipe.reset_pipe_all(cfg.sim_len, 'factory')
    Tank.reset_all(cfg.sim_len, 'factory')
    Tank.set_inflow_forecast_all(act_rain)
    arr = unload_from_file('2002-12-09 - 2002-12-10-perfect')
    Tank.set_releases_all(arr)
    run_model(cfg.sim_len, act_rain, swmm_optim)
    print(f"Mass Balance Error: {calc_mass_balance():0.2f}%")
    optimized = Scenario()
    optimized.set_atts()
    optimized.set_swmm_flow(swmm_run(act_rain, cfg.sim_days*24, 'clustered-no_roof.inp'))
print('end')
