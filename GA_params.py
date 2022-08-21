import numpy as np
import math

num_generations = 250
gene_space = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parent_selection = "rank"
crossover_type = "uniform"
crossover_prob = 0.4
mutation_prob = 0.07
mutation_by_replacement = True
mutation_type = "random"
stop_criteria = "saturate_80"


def pop_init(release_interval, n_tanks):
    num_genes = release_interval * n_tanks
    num_sols = math.ceil(num_genes * 1.3)
    sol_zero = np.zeros(num_genes)
    sol_one = np.ones(num_genes) * gene_space[0]
    sol_two = np.ones(num_genes) * gene_space[1]
    sol_three = np.ones(num_genes) * gene_space[2]
    pre_det = np.array([sol_zero,
                        sol_one,
                        sol_two,
                        sol_three])
    rest = np.random.randint(11, size=(num_sols - np.shape(pre_det)[0], num_genes))
    initial_pop = np.concatenate((pre_det, rest), axis=0)
    return initial_pop


def set_parent_num(release_hr, n_tanks):
    num_parents = math.ceil(release_hr * n_tanks * 2 * 0.06)
    return num_parents
