import copy

import alg_dmrsd
import statics
import random
from random import randrange
import xlsxwriter


def generate_traces(G):
    T = []
    for i in range(20):
        ti = [randrange(12)]
        ti_len = 2 + randrange(6)
        pt = -1
        ct = ti[0]
        while len(ti) < ti_len:
            cnl = [i for i in G[ct] if i != pt]
            if len(cnl) == 0:
                break
            random.shuffle(cnl)
            nt = cnl[0]
            ti.append(nt)
            pt = ct
            ct = nt
        T.append(ti)
    return T

def generate_activity_matrix(O, A, T):
    AM = []
    for t in T:
        row = [-1 for i in range(len(A) + 1)]
        AM.append(row)
    for i, row in enumerate(AM):
        AM[i][-1] = 0
        for j, col in enumerate(row[:-1]):
            if j in T[i]:
                AM[i][j] = 1
                if j in O:
                    AM[i][-1] = 1
            else:
                AM[i][j] = 0
    return AM

def write_data_to_excel(data):
    columns = [
        {'header': 'instance_number'},
        {'header': 'fan'},
        {'header': 'nodfap'},
        {'header': 'notg'},
        {'header': 'iteration'},
        {'header': 'max iteration'},
        {'header': 'oracle'},
        {'header': 'diagnosis'},
        {'header': 'revealed information'},
        {'header': 'precision'},
        {'header': 'recall'}
    ]
    # write the data to xlsx file
    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(data), len(columns) - 1, {'data': data, 'columns': columns})
    workbook.close()

def run_single_sandbox_experiment(instance_number):
    # get the sandbox inputs
    O, A, G, T, AM = statics.get_sandbox_inputs(instance_number)
    RD = []

    # the algorithm
    RD = alg_dmrsd.DMRSD(instance_number, 2, 0, 0, O, A, G, T, AM)

    print(RD)


def run_benchmark_experiment():
    pass


def run_random_experiments(instance_numbers,
                           faulty_agents_numbers,
                           number_of_different_faulty_agents_permutations,
                           number_of_trace_generations):
    results = []
    for instance_number in instance_numbers:
        _, A, G, _, _ = statics.get_sandbox_inputs(instance_number)
        for faulty_agents_number in faulty_agents_numbers:
            for nodfap in range(number_of_different_faulty_agents_permutations):
                A_copy = copy.deepcopy(A)
                random.shuffle(A_copy)
                O = A_copy[:faulty_agents_number]
                for notg in range(number_of_trace_generations):
                    T = generate_traces(G)
                    AM = generate_activity_matrix(O, A, T)
                    print(f'running instance with:')
                    print(f'        - instance number: {instance_number}')
                    print(f'        - faulty agents number: {faulty_agents_number}')
                    print(f'        - faulty agents permutation number: {nodfap}')
                    print(f'        - trace generation number: {notg}')
                    print(f'faulty agents: {O}')
                    print(f'traces:')
                    for t in T:
                        print(t)
                    print(f'activity matrix:')
                    for row in AM:
                        print(row)
                    print('')
                    result = alg_dmrsd.DMRSD(instance_number,
                                             faulty_agents_number,
                                             nodfap,
                                             notg,
                                             O, A, G, T, AM)
                    results += result
    write_data_to_excel(results)
    print(9)


if __name__ == '__main__':
    print('Hi, PyCharm')

    # 'sandbox', 'benchmark', 'random'
    experiment_type = 'random'

    if experiment_type == 'sandbox':
        run_single_sandbox_experiment(3)
    elif experiment_type == 'benchmark':
        run_benchmark_experiment()
    else:
        run_random_experiments([1, 2, 3], [2, 3, 4, 5], 3, 10)
        # run_random_experiments([1], [2], 3, 10)
    print('Bye, PyCharm')

