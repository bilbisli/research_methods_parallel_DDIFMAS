import random
from datetime import datetime

import matplotlib.pyplot as plt
import xlsxwriter
import networkx as nx
import algorithms
import numpy as np
from random import randrange

def create_random_graph(noa):
    # use erdos-renyi algorithm to generate random graph
    G = nx.erdos_renyi_graph(noa, 0.5, seed=123, directed=False)
    # make sure it is connected graph
    connected = nx.is_connected(G)
    # while the graph is not connected, generate a new one
    while not connected:
        G = nx.erdos_renyi_graph(noa, 0.5, seed=123, directed=False)
        connected = nx.is_connected(G)
    # plot the graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    return G

def choose_faulty_agents(noa, nof):
    # generate an ordered list of the agents (their numbers)
    F = [i for i in range(noa)]
    # shuffle the list
    random.shuffle(F)
    # choose the first number of faulty agents from the list to be faulty
    F = F[:nof]
    return F

def adjacency_matrix(A):
    # build an adjacency list for each node
    res_A = [[] for _ in A]
    # for each node (i) in the input matrix A
    for i, row in enumerate(A):
        # for each node (c) in that row
        for j, c in enumerate(row):
            # if the node (c) equals one, that means it is a neighbour of node (i)
            if c == 1:
                res_A[i].append(j)
    return res_A

def generate_traces(G, noa, nor):
    T = []
    # let networkx return the adjacency matrix A
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)
    A = adjacency_matrix(A)
    for i in range(nor):
        # choose a random starting node
        current_node = randrange(noa)
        # choose a random trace length
        trace_length = 2 + randrange(noa)
        # initialize the trace
        trace = [current_node]
        # while can advance (neighbours not visited previously)
        while len(trace) < trace_length:
            available_neighbours = [i for i in A[current_node] if i not in trace]
            if len(available_neighbours) == 0:
                break
            # choose random neighbour and advance there
            random.shuffle(available_neighbours)
            current_node = available_neighbours[0]
            trace.append(current_node)
        # append the resulting trace
        T.append(trace)
    return T

def no_failing_rows(S):
    for row in S:
        if row[-1] == 1:
            return False
    return True

def generate_spectrum(noa, nor, afp, F, T):
    # create an empty (with 2's) spectrum
    S = [[2 for _ in range(noa + 1)] for _ in range(nor)]

    # for each run (row) in the spectrum
    for i, run in enumerate(S):
        # initialize the error vector cell to 0
        run[-1] = 0
        # for each agent in that row set its involvement
        for j in range(len(run[:-1])):
            # if it was in the corresponding trace update its cell to 1
            if j in T[i]:
                run[j] = 1
            else:
                run[j] = 0
        # for each involved agent in that row, if it is faulty,
        # add its healthy probability to the probability of a trace
        # turning out ok
        trace_success_probability = 1.0
        for j in range(len(run[:-1])):
            if run[j] == 1 and j in F:
                trace_success_probability *= (1-afp)
        rnd = random.random()
        if rnd > trace_success_probability:
            run[-1] = 1
    return S

def write_data_to_excel(data):
    columns = [
        {'header': 'instance_number'},
        {'header': 'noa'},
        {'header': 'nof'},
        {'header': 'afp'},
        {'header': 'nor'},
        {'header': 'noi'},
        {'header': 'oracle'},
        {'header': 'spectrum'},
        {'header': 'diagnosis algorithm'},
        {'header': 'Missing Information (cells)'},
        {'header': 'Known Information Before (cells)'},
        {'header': 'Known Information Before % (cells)'},
        {'header': 'Known Information Before - Highest (cells)'},
        {'header': 'Known Information Before - Highest % (cells)'},
        {'header': 'diagnoses'},
        {'header': 'ranked diagnoses'},
        {'header': 'number of diagnoses'},
        {'header': 'Sent Information - Diagnosis'},
        {'header': 'Sent Information - Ranking'},
        {'header': 'Sent Information - Both'},
        {'header': 'Sent Information avg agent - Both'},
        {'header': 'Revealed Information (cells)'},
        {'header': 'Revealed Information % (cells)'},
        {'header': 'Revealed Information % of Missing (cells)'},
        {'header': 'Revealed Information - Sum'},
        {'header': 'Revealed Information - Mean'},
        {'header': 'Revealed Information - Last (cells)'},
        {'header': '---------- Revealed Information - Last % (cells)'},
        {'header': 'Known Information After (cells)'},
        {'header': 'Known Information After % (cells)'},
        {'header': 'Known Information After - Highest (cells)'},
        {'header': 'Known Information After - Highest % (cells)'},
        {'header': 'Critical Information Revealed (cells)'},
        {'header': 'Critical Information Revealed % (cells)'},
        {'header': 'Wasted Effort'},
        {'header': 'Wasted Effort Percent'},
        {'header': 'Useful Effort'},
        {'header': 'Useful Effort Percent'},
        {'header': 'Runtime'},
        {'header': 'Weighted Precision 10'},
        {'header': 'Weighted Precision 20'},
        {'header': 'Weighted Precision 30'},
        {'header': 'Weighted Precision 40'},
        {'header': 'Weighted Precision 50'},
        {'header': 'Weighted Precision 60'},
        {'header': 'Weighted Precision 70'},
        {'header': 'Weighted Precision 80'},
        {'header': 'Weighted Precision 90'},
        {'header': 'Weighted Precision 100'},
        {'header': 'Weighted Recall 10'},
        {'header': 'Weighted Recall 20'},
        {'header': 'Weighted Recall 30'},
        {'header': 'Weighted Recall 40'},
        {'header': 'Weighted Recall 50'},
        {'header': 'Weighted Recall 60'},
        {'header': 'Weighted Recall 70'},
        {'header': 'Weighted Recall 80'},
        {'header': 'Weighted Recall 90'},
        {'header': 'Weighted Recall 100'}
    ]
    # write the data to xlsx file
    workbook = xlsxwriter.Workbook('results.xlsx')
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(data), len(columns) - 1, {'data': data, 'columns': columns})
    workbook.close()

def run_random_experiments(number_of_agents, number_of_faulty, agent_fault_probabilities, number_of_runs, number_of_instances, s_time):
    t_report = []
    results = []
    noa_l = len(number_of_agents)
    nof_l = len(number_of_faulty)
    afp_l = len(agent_fault_probabilities)
    nor_l = len(number_of_runs)
    noi_l = number_of_instances
    total_instances = noa_l * nof_l * afp_l * nor_l * noi_l
    for noa_i, noa in enumerate(number_of_agents):
        G = create_random_graph(noa)
        result_rows = 0
        for nof_i, nof in enumerate(number_of_faulty):
            F = choose_faulty_agents(noa, nof)
            F.sort()
            for afp_i, afp in enumerate(agent_fault_probabilities):
                for nor_i, nor in enumerate(number_of_runs):
                    for inum in range(number_of_instances):
                        instance_num = noa_i * (nof_l * afp_l * nor_l * noi_l) + nof_i * (afp_l * nor_l * noi_l) + afp_i * (nor_l * noi_l) + nor_i * noi_l + inum + 1
                        T = generate_traces(G, noa, nor)
                        S = generate_spectrum(noa, nor, afp, F, T)
                        while no_failing_rows(S):
                            T = generate_traces(G, noa, nor)
                            S = generate_spectrum(noa, nor, afp, F, T)
                        print(f'running instance {instance_num}/{total_instances} ({inum+1}/{number_of_instances}) with:')
                        print(f'        - number of agents: {noa} ({noa_i+1}/{noa_l})')
                        print(f'        - number of faulty agents: {nof} ({nof_i+1}/{nof_l})')
                        print(f'        - agent fault probability: {afp} ({afp_i+1}/{afp_l})')
                        print(f'        - number of runs: {nor} ({nor_i + 1}/{nor_l})')
                        try:
                            result_coef = algorithms.COEF(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            result_dcoefI1D4R2 = algorithms.DCOEF_I1D4R2(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            result_mrsd = algorithms.MRSD(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            result_dmrsdI1D1R1 = algorithms.DMRSD_I1D1R1(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            result_dmrsdI1D2R1 = algorithms.DMRSD_I1D2R1(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            result_dmrsdI1D3R1 = algorithms.DMRSD_I1D3R1(instance_num, noa, nof, afp, nor, inum + 1, F, S)
                            results += result_coef
                            result_rows += 1
                            results += result_dcoefI1D4R2
                            result_rows += 1
                            results += result_mrsd
                            result_rows += 1
                            results += result_dmrsdI1D1R1
                            result_rows += 1
                            results += result_dmrsdI1D2R1
                            result_rows += 1
                            results += result_dmrsdI1D3R1
                            result_rows += 1
                        except Exception as e:
                            print(f'\n\n\n#####################################################################')
                            print(f'#####################################################################')
                            print(f'#####################################################################')
                            print(f'########################### Exception! ##############################')
                            print(type(e))
                            print(e.args)
                            print(e)
                            print(f'instance_num: {instance_num}\n')
                            print(f'noa: {noa}\n')
                            print(f'nof: {nof}\n')
                            print(f'afp: {afp}\n')
                            print(f'nor: {nor}\n')
                            print(f'inum + 1: {inum + 1}\n')
                            print(f'F: {F}\n')
                            Sstring = ',\r\n'.join(list(map(lambda arr: str(arr), S)))
                            print(f'S:\n{Sstring}\n')
                            raise
        e_time = datetime.now()
        d = e_time - s_time
        t_report.append(f'number of agents {noa}: {result_rows} rows, {d}')
        s_time = datetime.now()
    write_data_to_excel(results)
    print(9)
    return t_report


if __name__ == '__main__':
    print('Hi, DDIFMAS pipeline!')

    start_time = datetime.now()

    # number_of_agents_list = [6, 7, 8, 9, 10, 11, 12, 13]
    # number_of_faulty_list = [1, 2, 3, 4, 5]
    # agent_fault_probabilities_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # number_of_runs_list = [10, 20, 30, 40, 50]
    # number_of_instances_list = 30

    number_of_agents_list = [8]
    number_of_faulty_list = [5]
    agent_fault_probabilities_list = [0.9]
    number_of_runs_list = [50]
    number_of_instances_list = 30

    # run_random_experiments([5, 6, 7, 8, 9], [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], 10)
    time_report = run_random_experiments(number_of_agents_list, number_of_faulty_list, agent_fault_probabilities_list,
                                         number_of_runs_list, number_of_instances_list, start_time)

    end_time = datetime.now()
    delta = end_time - start_time
    print(',\r\n'.join(list(map(lambda arr: str(arr), time_report))))
    print(f'time to finish: {delta}')

    print('Bye, DDIFMAS Pipeline!')
