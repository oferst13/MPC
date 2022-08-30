from scipy import integrate
from tank import Tank
from pipe import Pipe
from node import Node
import cfg
import numpy as np
import math
from timer import Timer
import pygad
import GA_params as ga


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

    def calc_obj_Q(self):
        self.obj_Q = outfall.get_outflow_volume() / (Tank.get_last_overflow() * cfg.dt)

    def set_fitness(self):
        to_min: float = 0
        for i in range(self.last_Q):
            to_min += abs(outfall.get_flow(i) - self.obj_Q)
        self.fitness = to_min

    def set_release_intervals(self):
        self.release_intervals = math.ceil(Tank.get_last_overflow() * (cfg.dt / cfg.release_dt))

    def set_last_outflow(self):
        self.last_outflow = Tank.get_last_overflow()

    def set_max_flow(self):
        self.max_flow = outfall.get_max_Q()

    def set_last_Q(self):
        self.last_Q = outfall.get_zero_Q()


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


def set_forecast_filename(prefix, idx):
    idx_str = str(idx)
    cur_filename = '-'.join([prefix, idx_str])
    cur_filename = '.'.join([cur_filename, 'csv'])
    return cur_filename


def calc_mass_balance():
    mass_balance = 100 * (abs(outfall.get_outflow_volume() - Tank.get_cum_outflow())) / (Tank.get_cum_outflow())
    return mass_balance


def run_model(duration):
    for i in range(duration):
        if sum(forecast_rain[int(i // (cfg.rain_dt / cfg.dt)):-1]) + Tank.get_tot_storage() == 0:
            break  # this should break forecast run only!
        for tank in Tank.all_tanks:
            tank.tank_fill(i)
            tank.calc_release(i, baseline.last_outflow)
            tank.rw_use(i)
        if i < 1 or (Pipe.get_tot_Q(i - 1) + Tank.get_tot_outflow(i)) < 1e-3:
            continue
        for node in Node.all_nodes:
            node.handle_flow(i)
            for pipe in node.giving_to:
                pipe.calc_q_outlet(i)


def fitness_func(release_vector, idx):
    for tank in Tank.all_tanks:
        tank.reset_tank()
    for pipe in Pipe.all_pipes:
        pipe.reset_pipe()
    release_array = np.reshape(release_vector, (len(Tank.all_tanks), baseline.release_intervals))
    for num, tank in enumerate(Tank.all_tanks):
        tank.set_releases(release_array[num, :])
    run_model()
    print(f"Mass Balance Error: {calc_mass_balance():0.2f}%")
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

# runtime = Timer()
# runtime.start()

num_forecast_files = 26

tank1_dict = {'name': 'tank1', 'n_tanks': 30, 'init_storage': 0, 'roof': 9000, 'dwellers': 180}
tank2_dict = {'name': 'tank2', 'n_tanks': 35, 'init_storage': 0, 'roof': 10000, 'dwellers': 180}
tank3_dict = {'name': 'tank3', 'n_tanks': 25, 'init_storage': 0, 'roof': 8500, 'dwellers': 180}
tank4_dict = {'name': 'tank4', 'n_tanks': 50, 'init_storage': 0, 'roof': 14000, 'dwellers': 180}
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
pipe2 = Pipe('pipe2', 500, 0.6, 0.002)
pipe3 = Pipe('pipe3', 400, 0.4, 0.0013)
pipe4 = Pipe('pipe4', 400, 0.8, 0.0088)
pipe5 = Pipe('pipe5', 300, 0.4, 0.005)
pipe6 = Pipe('pipe6', 200, 0.8, 0.01)

tank1_out = Node('tank1_out', [tank1], [outlet1], tank_node=True)
tank2_out = Node('tank1_out', [tank2], [outlet2], tank_node=True)
tank3_out = Node('tank1_out', [tank3], [outlet3], tank_node=True)
tank4_out = Node('tank1_out', [tank4], [outlet4], tank_node=True)

node111 = Node('node111', [outlet1], [pipe1])
node11 = Node('node11', [pipe1, outlet2], [pipe2])
node12 = Node('node12', [outlet3], [pipe3])
node1 = Node('node1', [pipe2, pipe3], [pipe4])
node21 = Node('node21', [outlet4], [pipe5])
node2 = Node('node2', [pipe4, pipe5], [pipe6])
outfall = Node('outfall', [pipe6])

demands_PD = set_demands_per_dt()
Tank.set_daily_demands_all(demands_PD)  # happens only once

for forecast_idx in range(1, num_forecast_files + 1):
    # Create forecast - currently real rain only!
    forecast_file = set_forecast_filename('09-10', forecast_idx)
    forecast_rain = set_rain_input(forecast_file, cfg.rain_dt, cfg.forecast_len)
    Tank.set_inflow_forecast_all(forecast_rain)  # happens once a forecast is made
    baseline = Scenario()
    run_model(cfg.forecast_len)

    baseline.set_last_outflow()
    baseline.set_max_flow()
    baseline.set_last_Q()
    baseline.set_release_intervals()
    baseline.calc_obj_Q()
    baseline.set_fitness()
mass_balance_err = calc_mass_balance()
print(f"Mass Balance Error: {mass_balance_err:0.2f}%")
zero_Q = outfall.get_zero_Q()
last_overflow = Tank.get_last_overflow()
obj_Q = integrate.simps(pipe6.outlet_Q[:zero_Q], cfg.t[:zero_Q]) / (last_overflow)

optim = input('optimize? Y/N')
if optim == 'Y':
    ga_instance = set_ga_instance()
    ga_instance.run()
# runtime.stop()
print('d')
