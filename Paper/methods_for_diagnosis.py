import copy
import random

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
        = calculate_revealed_information_metrics_D1(revealed_information_tables, local_spectra, missing_information_cells)

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
        = calculate_revealed_information_metrics_D2(revealed_information_tables, local_spectra, missing_information_cells)

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
        = calculate_revealed_information_metrics_D2(revealed_information_tables, local_spectra, missing_information_cells)

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
