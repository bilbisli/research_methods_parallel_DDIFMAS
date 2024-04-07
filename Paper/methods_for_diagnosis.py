import copy
import multiprocessing
import random
import time

import functions


def diagnosis_0(spectrum, local_spectra):
    """
    the traditional hitting set algorithm
    :param local_spectra: for calculating information sent
    :param spectrum: the local spectra of the agents
    :return: a set of diagnoses
    """
    info_sent_diagnosis = 0
    for ls in local_spectra:
        for row in ls:
            for c in row:
                if c == 0 or c == 1:
                    info_sent_diagnosis += 1

    # calculate conflicts
    conflicts = []
    for i, row in enumerate(spectrum):
        if row[-1] == 1:
            conf = [j for j, a in enumerate(row[:-1]) if a == 1]
            conflicts.append(conf)

    # compute diagnoses
    diagnoses = functions.conflict_directed_search(conflicts)

    # sort diagnoses
    for d in diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(diagnoses)
    return diagnoses_sorted, info_sent_diagnosis


def refine_revealed_information_table_D1(revealed_information_table):
    return revealed_information_table


def calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra, missing_information_cells):
    # for each revealed row it goes like this:
    # if it is not in the local spectrum,
    # then the revealed information is as the
    # length of the row.
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent = 0, 0.0, []
    revealed_information_percent_per_agent = []
    for ai, atab in enumerate(revealed_information_tables):
        revealed_information_a = 0
        for conf in atab:
            if conf not in local_spectra[ai]:
                revealed_information_a += len(conf)
        revealed_information_sum += revealed_information_a
        revealed_information_per_agent.append(revealed_information_a)
        revealed_information_percent_per_agent.append(revealed_information_a * 1.0 / missing_information_cells[ai]
                                                      if missing_information_cells[ai] != 0 else 0.0)
    revealed_information_mean = revealed_information_sum * 1.0 / len(revealed_information_tables)
    return revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_per_agent[-1], revealed_information_percent_per_agent, \
        revealed_information_percent_per_agent[-1]


def diagnosis_1(local_spectra, missing_information_cells):
    """
    go over the agents, each agent computes the diagnoses it can
    with the information it has and then passes it to the next agent
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    for a in range(number_of_agents):
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, a in enumerate(row[:-1]) if a == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # selecting which diagnoses to pass on
        passed_diagnoses = diagnoses
        # print(f'{a}     sent diagnoses: {passed_diagnoses}')

    # sort diagnoses
    for d in passed_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


# def parallel_worker(a, local_spectra, passed_diagnoses, revealed_information_tables, missing_information_cells):
#     local_spectrum = local_spectra[a]
#
#     # reveal information given the diagnoses from the previous agents
#     revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
#     refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
#     revealed_information_tables[a] = refined_revealed_information_table
#
#     # calculate conflicts
#     conflicts = []
#     for i, row in enumerate(local_spectrum):
#         if row[-1] == 1:
#             conf = [j for j, a in enumerate(row[:-1]) if a == 1]
#             conflicts.append(conf)
#
#     # calculate local diagnoses
#     local_diagnoses = functions.conflict_directed_search(conflicts)
#
#     # join the previous diagnoses to create a conflict set of diagnoses
#     conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]
#
#     # create united conflict set by labeling every diagnosis to a number
#     labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
#         conflicts_of_diagnoses)
#
#     # filter out empty conflicts
#     labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)
#
#     # calculate raw united diagnoses
#     diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)
#
#     # translate back the the united diagnoses
#     diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)
#
#     # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
#     diagnoses = functions.refine_diagnoses(diagnoses_translated)
#
#     return diagnoses
#
#
# def parallel_diagnosis_1(local_spectra, missing_information_cells):
#     information_sent = 0
#     revealed_information_tables = multiprocessing.Manager().list()
#     diagnoses_per_agent = []
#     passed_diagnoses = []
#     number_of_agents = len(local_spectra)
#
#     while True:
#         pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#         results = []
#
#         for a in range(number_of_agents // 2):
#             results.append(pool.apply_async(parallel_worker, (a, local_spectra, passed_diagnoses,
#                                                               revealed_information_tables,
#                                                               missing_information_cells)))
#
#         pool.close()
#         pool.join()
#
#         # Retrieve results
#         for result in results:
#             diagnoses = result.get()
#             diagnoses_per_agent.append(copy.deepcopy(diagnoses))
#             passed_diagnoses.extend(diagnoses)
#
#         # Check if there are remaining agents
#         if number_of_agents % 2 == 1:
#             a = number_of_agents // 2
#             diagnoses = parallel_worker(a, local_spectra, passed_diagnoses,
#                                         revealed_information_tables, missing_information_cells)
#             diagnoses_per_agent.append(copy.deepcopy(diagnoses))
#             passed_diagnoses.extend(diagnoses)
#
#         # Check if there are no more agents left
#         if a == number_of_agents - 1:
#             break
#
#     # Sort diagnoses
#     for d in passed_diagnoses:
#         d.sort()
#     diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)
#
#     # Calculate sum, mean, and last agent revealed information based on the tables
#     revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
#         revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
#         = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
#                                                     missing_information_cells)
#
#     return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
#         revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
#         revealed_information_percent_last

################3

def get_neighbour_of_agent(agent_index: int, agent_indices_stack: list or tuple) -> int:
    if agent_index == agent_indices_stack[-1]:
        return 0
    return agent_indices_stack[agent_indices_stack.index(agent_index) + 1]


def semi_parallel_diagnosis_1(local_spectra, missing_information_cells):
    # calculate all diagnosis for each agent using their local spectra information
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    # added for simulating parallelism
    # previous_step_diagnoses = []
    current_cumulative_max_time = 0
    current_step_times = []
    final_diagnoses = []
    revealed_information_array = []

    number_of_steps = 0
    for a in range(number_of_agents):
        # start timing the step
        current_step_start_time = time.time()
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, a in enumerate(row[:-1]) if a == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        # we will use this to access the diagnoses of the agents in the next step
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # for first step - no diagnoses are passed in order to simulate the first step of the parallel algorithm
        # this is different from the original diagnosis_1
        passed_diagnoses = []

        # print(f'{a}     sent diagnoses: {passed_diagnoses}')
        # end timing the step
        current_step_end_time = time.time()
        current_step_times.append(current_step_end_time - current_step_start_time)
    current_cumulative_max_time += max(current_step_times)
    number_of_steps += 1
    agent_indices_stack = list(range(number_of_agents))
    # until there is only one diagnosis left, combine the diagnoses of the agents
    while agent_indices_stack:
        print('step:', number_of_steps)
        print('agent_indices_stack:', agent_indices_stack)
        agent_indices_stack_length = len(agent_indices_stack)
        current_step_times = []
        print('agents processed in current step:', agent_indices_stack[::2])
        # print('diagnoses_per_agent:', diagnoses_per_agent)
        print('len(diagnoses_per_agent):', len(diagnoses_per_agent))
        agent_indices_stack_copy = copy.deepcopy(agent_indices_stack)

        for agent_index in agent_indices_stack_copy[::2]:
            # start timing the step
            current_step_start_time = time.time()

            # if there is no neighbor it means that the agent is the last one and there are odd agents
            # in this case, the agent will not participate in the current step
            agent_neighbour = get_neighbour_of_agent(agent_index, agent_indices_stack)
            print('agent_neighbour:', agent_neighbour)
            if agent_neighbour:
                agent_indices_stack.remove(agent_neighbour)
                neighbour_diagnoses = diagnoses_per_agent[agent_neighbour]
                passed_diagnoses = neighbour_diagnoses
            # if there is no neighbour and the first agent is the only one left then the loop is broken
            # and the final diagnoses are returned
            elif agent_index == 0:
                agent_indices_stack.remove(agent_index)
                final_diagnoses = diagnoses_per_agent[agent_index]
                print('final_diagnoses for agent:', agent_index, final_diagnoses)
                break
            else:
                break
            print('agent_index:', agent_index)
            agent_diagnoses = diagnoses_per_agent[agent_index]
            # there is no local diagnoses after the first step
            # this local_diagnoses is actually the diagnoses made by the agent in the previous step
            local_diagnoses = agent_diagnoses
            # print(f'{a}   passed diagnoses: {passed_diagnoses}')
            information_sent += len(passed_diagnoses)
            # no local spectrum is used after the first step
            # local_spectrum = local_spectra[a]

            # reveal information given the diagnoses from the previous agents
            revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
            refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
            revealed_information_tables[agent_index] = refined_revealed_information_table

            # there is no local diagnoses after the first step
            # # calculate conflicts
            # conflicts = []
            # for i, row in enumerate(local_spectrum):
            #     if row[-1] == 1:
            #         conf = [j for j, a in enumerate(row[:-1]) if a == 1]
            #         conflicts.append(conf)
            #
            # # calculate local diagnoses
            # local_diagnoses = functions.conflict_directed_search(conflicts)
            # # print(f'{a}    local diagnoses: {local_diagnoses}')

            # join the previous diagnoses to create a conflict set of diagnoses
            conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

            # create united conflict set by labeling every diagnosis to a number
            labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
                conflicts_of_diagnoses)

            # filter out empty conflicts
            labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

            # calculate raw united diagnoses
            diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

            # translate back the the united diagnoses
            diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

            # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
            diagnoses = functions.refine_diagnoses(diagnoses_translated)
            # keep using the diagnoses_per_agent to access the diagnoses of the agents in the next step
            diagnoses_per_agent[agent_index] = copy.deepcopy(diagnoses)
            # print(f'{a} combined diagnoses: {diagnoses}')

            # for first step - no diagnoses are passed in order to simulate the first step of the parallel algorithm
            # this is different from the original diagnosis_1
            # passed_diagnoses = []

            # print(f'{a}     sent diagnoses: {passed_diagnoses}')
            # end timing the step
            current_step_end_time = time.time()
            current_step_times.append(current_step_end_time - current_step_start_time)
        # condition for the case that there is only one agent left and the inner loop was broken
        # before step completion which means that the final diagnoses were already calculated
        # at the end of the previous step and the previous iteration was the last step
        if current_step_times:
            current_cumulative_max_time += max(current_step_times)
            number_of_steps += 1
    # Sort diagnoses
    for d in final_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(final_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
                                                    missing_information_cells)
    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last, current_cumulative_max_time, number_of_steps


################3


def parallel_agent_worker(a, local_spectra, i, passed_diagnoses_to_next_agents, number_of_agents):
    local_spectrum = local_spectra[a]

    # get the diagnoses passed to the agent
    passed_diagnoses = passed_diagnoses_to_next_agents[i] if passed_diagnoses_to_next_agents else []

    # reveal information given the diagnoses from the previous agents
    revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
    refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)

    # calculate conflicts
    conflicts = []
    for i, row in enumerate(local_spectrum):
        if row[-1] == 1:
            conf = [j for j, a in enumerate(row[:-1]) if a == 1]
            conflicts.append(conf)

    # calculate local diagnoses
    local_diagnoses = functions.conflict_directed_search(conflicts)
    # print(f'{a}    local diagnoses: {local_diagnoses}')

    # join the previous diagnoses to create a conflict set of diagnoses
    conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

    # create united conflict set by labeling every diagnosis to a number
    labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
        conflicts_of_diagnoses)

    # filter out empty conflicts
    labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

    # calculate raw united diagnoses
    diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

    # translate back the united diagnoses
    diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

    # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
    diagnoses = functions.refine_diagnoses(diagnoses_translated)

    return diagnoses, refined_revealed_information_table


def parallel_diagnosis_111(local_spectra, missing_information_cells):
    """
    1. go over half of the agents, each of them computes the diagnoses it can
        with the information it has in parallel
    2. the diagnoses are passed on
    3. the already diagnosed agents are removed
    4. if there are agents left return to 1
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    print('local_spectra length:', len(local_spectra))
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    current_number_of_agents = number_of_agents
    passed_diagnoses_to_next_agents = []

    agent_indices = list(range(number_of_agents))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    final_diagnoses = []
    while agent_indices:
        current_number_of_agents //= 2
        current_agent_indices = agent_indices[:current_number_of_agents]
        agent_indices = agent_indices[current_number_of_agents:]
        if len(agent_indices) == 1 and len(current_agent_indices) == 0:
            current_agent_indices = agent_indices
            agent_indices = []
        print('current_number_of_agents:', current_number_of_agents)
        print('current_agent_indices:', current_agent_indices)
        print('agent_indices:', agent_indices)
        print('passed_diagnoses_to_next_agents:', passed_diagnoses_to_next_agents)
        input()

        agent_parameters_generator = ((a, local_spectra, i, passed_diagnoses_to_next_agents, number_of_agents)
                                      for i, a in enumerate(current_agent_indices))
        current_diagnoses_per_agent = pool.starmap_async(parallel_agent_worker,
                                                         agent_parameters_generator)
        passed_diagnoses_to_next_agents = []
        for diagnoses, refined_revealed_information_table in current_diagnoses_per_agent.get():
            passed_diagnoses_to_next_agents.append(diagnoses)
            revealed_information_tables.append(refined_revealed_information_table)
            information_sent += len(diagnoses)
            diagnoses_per_agent.append(copy.deepcopy(diagnoses))
            final_diagnoses = diagnoses

        print('indices done:', number_of_agents - len(agent_indices))
    print('revealed_information_tables:', revealed_information_tables)
    print('len of revealed_information_tables:', len(revealed_information_tables))
    for d in final_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(final_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def parallel_diagnosis_11(local_spectra, missing_information_cells):
    """
    1. go over half of the agents, each of them computes the diagnoses it can
        with the information it has in parallel
    2. the diagnoses are passed on
    3. the already diagnosed agents are removed
    4. if there are agents left return to 1
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    number_of_agents = len(local_spectra)
    current_number_of_agents = number_of_agents
    passed_diagnoses = []
    agent_indices = list(range(number_of_agents))
    passed_diagnoses_to_next_agents = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    while agent_indices:
        revealed_information_tables_to_agent = []

        current_agents_results = []
        current_number_of_agents = current_number_of_agents // 2
        current_agent_indices = agent_indices[:current_number_of_agents]
        agent_indices = agent_indices[current_number_of_agents:]
        for i, a in enumerate(current_agent_indices):
            passed_diagnoses_to_agent = passed_diagnoses_to_next_agents[i] if passed_diagnoses_to_next_agents else []
            results = pool.apply_async(parallel_agent_worker, (a, local_spectra, passed_diagnoses_to_agent,
                                                               revealed_information_tables))
            current_agents_results.append(results)

        # for a in range(number_of_agents):
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, a in enumerate(row[:-1]) if a == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # selecting which diagnoses to pass on
        passed_diagnoses = diagnoses
        # print(f'{a}     sent diagnoses: {passed_diagnoses}')

    # sort diagnoses
    for d in passed_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def parallel_diagnosis_1(local_spectra, missing_information_cells):
    """
    1. go over half of the agents, each of them computes the diagnoses it can
        with the information it has in parallel
    2. the diagnoses are passed on
    3. the already diagnosed agents are removed
    4. if there are agents left return to 1
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    print('local_spectra length:', len(local_spectra))
    print('local_spectra:', local_spectra)
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    for a in range(number_of_agents):
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D1(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, a in enumerate(row[:-1]) if a == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # selecting which diagnoses to pass on
        passed_diagnoses = diagnoses
        print(f'{a}     sent diagnoses: {passed_diagnoses}')

    # sort diagnoses
    for d in passed_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def pass_lowest_cardinality_D2(diagnoses):
    if len(diagnoses) == 0:
        return []
    lowest_cardinality = min([len(d) for d in diagnoses])
    lowest_cardinality_diagnoses = [copy.deepcopy(d) for d in diagnoses if len(d) == lowest_cardinality]
    return lowest_cardinality_diagnoses


def refine_revealed_information_table_D2(revealed_information_table):
    refined_revealed_information_table = []
    for rit in revealed_information_table:
        rrit = []
        for d in rit:
            if d == 1:
                rrit.append(d)
            else:
                rrit.append(2)
        refined_revealed_information_table.append(rrit)
    return refined_revealed_information_table


def ones_not_subset_in_spectra(conf, local_spectrum):
    conf_ones_indices = [ci for ci, c in enumerate(conf) if c == 1]
    conf_ones_indices_set = set(conf_ones_indices)
    for row in local_spectrum:
        row_ones_indices = [ci for ci, c in enumerate(row) if c == 1]
        row_ones_indices_set = set(row_ones_indices)
        if conf_ones_indices_set.issubset(row_ones_indices_set):
            return False
    return True


def calculate_revealed_information_metrics_D2(revealed_information_tables, local_spectra, missing_information_cells):
    # for each revealed row it goes like this:
    # for the ones that are in the tables, if there is no row
    # in the local spectrum that contains the ones,
    # then the revealed information is as the
    # numbers of ones in the row.
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent = 0, 0.0, []
    revealed_information_percent_per_agent = []

    for ai, atab in enumerate(revealed_information_tables):
        revealed_information_a = 0
        for conf in atab:
            if ones_not_subset_in_spectra(conf, local_spectra[ai]):
                revealed_information_a += len([c for c in conf if c == 1])
        revealed_information_sum += revealed_information_a
        revealed_information_per_agent.append(revealed_information_a)
        revealed_information_percent_per_agent.append(revealed_information_a * 1.0 / missing_information_cells[ai]
                                                      if missing_information_cells[ai] != 0 else 0.0)
    revealed_information_mean = revealed_information_sum * 1.0 / len(revealed_information_tables)
    return revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_per_agent[-1], revealed_information_percent_per_agent, \
        revealed_information_percent_per_agent[-1]


def diagnosis_2(local_spectra, missing_information_cells):
    """
    go over the agents, each agent computes the diagnoses it can
    with the information it has and then passes only the lowest
    cardinality diagnoses to the next agent
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    for a in range(number_of_agents):
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D2(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, aj in enumerate(row[:-1]) if aj == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # selecting which diagnoses to pass on
        passed_diagnoses = pass_lowest_cardinality_D2(diagnoses)
        # print(f'{a}     sent diagnoses: {passed_diagnoses}')

    # sort diagnoses
    for d in passed_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D2(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def pass_one_of_the_lowest_cardinality_D3(diagnoses):
    if len(diagnoses) == 0:
        return []
    lowest_cardinality = min([len(d) for d in diagnoses])
    lowest_cardinality_diagnoses = [copy.deepcopy(d) for d in diagnoses if len(d) == lowest_cardinality]
    random.shuffle(lowest_cardinality_diagnoses)
    one_of_the_lowest_cardinality_diagnoses = lowest_cardinality_diagnoses[:1]
    return one_of_the_lowest_cardinality_diagnoses


def diagnosis_3(local_spectra, missing_information_cells):
    """
    go over the agents, each agent computes the diagnoses it can
    with the information it has and then passes only one of the lowest
    cardinality diagnoses to the next agent (this is decided randomally)
    :param local_spectra: the local spectra of the agents
    :param missing_information_cells: for metric gathering purposes
    :return: a set of diagnoses
    """
    information_sent = 0
    revealed_information_tables = []
    diagnoses_per_agent = []
    passed_diagnoses = []
    number_of_agents = len(local_spectra)
    for a in range(number_of_agents):
        # print(f'{a}   passed diagnoses: {passed_diagnoses}')
        information_sent += len(passed_diagnoses)
        local_spectrum = local_spectra[a]

        # reveal information given the diagnoses from the previous agents
        revealed_information_table = functions.reveal_information(passed_diagnoses, number_of_agents)
        refined_revealed_information_table = refine_revealed_information_table_D2(revealed_information_table)
        revealed_information_tables.append(refined_revealed_information_table)

        # calculate conflicts
        conflicts = []
        for i, row in enumerate(local_spectrum):
            if row[-1] == 1:
                conf = [j for j, aj in enumerate(row[:-1]) if aj == 1]
                conflicts.append(conf)

        # calculate local diagnoses
        local_diagnoses = functions.conflict_directed_search(conflicts)
        # print(f'{a}    local diagnoses: {local_diagnoses}')

        # join the previous diagnoses to create a conflict set of diagnoses
        conflicts_of_diagnoses = [passed_diagnoses, local_diagnoses]

        # create united conflict set by labeling every diagnosis to a number
        labelled_conflicts_of_diagnoses, d_diag_to_num, d_num_to_diag = functions.label_diagnosis_sets(
            conflicts_of_diagnoses)

        # filter out empty conflicts
        labelled_conflicts_of_diagnoses = functions.filter_empty(labelled_conflicts_of_diagnoses)

        # calculate raw united diagnoses
        diagnoses_raw = functions.conflict_directed_search(labelled_conflicts_of_diagnoses)

        # translate back the the united diagnoses
        diagnoses_translated = functions.labels_to_diagnoses(diagnoses_raw, d_num_to_diag)

        # refining diagnoses by unifying them, removing duplicates, and removing supersets, and storing them
        diagnoses = functions.refine_diagnoses(diagnoses_translated)
        diagnoses_per_agent.append(copy.deepcopy(diagnoses))
        # print(f'{a} combined diagnoses: {diagnoses}')

        # selecting which diagnoses to pass on
        passed_diagnoses = pass_one_of_the_lowest_cardinality_D3(diagnoses)
        # print(f'{a}     sent diagnoses: {passed_diagnoses}')

    # sort diagnoses
    for d in passed_diagnoses:
        d.sort()
    diagnoses_sorted = functions.sort_diagnoses_by_cardinality(passed_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_D2(revealed_information_tables, local_spectra,
                                                    missing_information_cells)

    return diagnoses_sorted, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def diagnosis_4(spectra):
    """
    because this is a single fault diagnosis,
    the result is |A| single agent diagnoses,
    one for each agent
    :param spectra: the spectra
    :return: list of single diagnoses
    """
    return [[j] for j in range(len(spectra))], 0


def diagnosis_coef_0(S, local_spectra):
    """
    because this is a single fault diagnosis,
    the result is |A| single agent diagnoses,
    one for each agent
    :param local_spectra: for calculating information sent
    :param S: the spectrum
    :return: list of single diagnoses
    """
    info_sent_diagnosis = 0
    for ls in local_spectra:
        for row in ls:
            for c in row:
                if c == 0 or c == 1:
                    info_sent_diagnosis += 1
    return [[j] for j in range(len(S[0][:-1]))], info_sent_diagnosis
