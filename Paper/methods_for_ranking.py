import copy

import functions
import sympy

# def ranking_0(spectrum, diagnoses):
#     """
#     ranks the diagnoses. for each diagnosis, the diagnoser
#     computes a corresponding estimation function
#     and then maximizes it
#     :param spectrum: the spectrum
#     :param diagnoses: the diagnosis list
#     :return: ranked diagnosis list
#     """
#     ranked_diagnoses = []
#     for diagnosis in diagnoses:
#         print(f'ranking diagnosis: {diagnosis}')
#
#         # divide the spectrum to activity matrix and error vector
#         activity_matrix = [row[:-1] for row in spectrum]
#         error_vector = [row[-1] for row in spectrum]
#
#         # calculate the probability of the diagnosis
#         likelihood, H = functions.calculate_e_dk(diagnosis, activity_matrix, error_vector)
#
#         # save the result
#         ranked_diagnoses.append([diagnosis, likelihood, H])
#         print(f'finished ranking diagnosis: {diagnosis}, rank: [{diagnosis},{likelihood}]')
#
#     # normalize the diagnosis probabilities
#     normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
#     return normalized_diagnoses

def estimation_and_derivative_functions(diagnosis, spectrum):
    # declare variables
    h = []
    for hj in range(len(spectrum[0][:-1])):
        h.append(sympy.symbols(f'h{hj}'))
    ef, DF = functions.estimation_and_derivative_functions(h, spectrum, diagnosis)
    return ef, DF, h

def eval_grad_R0(diagnosis, H, DF):
    Gradients = {}
    for a in diagnosis:
        gradient_value = functions.substitute_and_eval(H, DF[a])
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients

def ranking_0(spectrum, diagnoses, step):
    """
    ranks the diagnoses. for each diagnosis, the diagnoser
    computes a corresponding estimation function
    and then maximizes it
    :param spectrum: the spectrum
    :param diagnoses: the diagnosis list
    :param step: the gradient step
    :return: ranked diagnosis list
    """
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to zero
        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        # while true
        #   calculate P_n(d, H, M)
        #   if condition is is reached, abort
        #   calculate gradients
        #   update H
        # print(f'diagnosis {diagnosis} ranking...')

        # initialize H values of the agents involved in the diagnosis to 0.5
        H = {}
        for a in diagnosis:
            H[f'h{a}'] = 0.5

        # initialize an epsilon value for the stop condition: |P_n(d, H, LS) - P_{n-1}(d, H, LS)| < epsilon
        epsilon = 0.0005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [[0.0, {}]]

        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        ef, DF, h = estimation_and_derivative_functions(diagnosis, spectrum)

        # while true
        while True:
            # calculate P_n(d, H, S)
            P = functions.substitute_and_eval(H, ef)
            P = float(P)
            P_arr.append([P, copy.deepcopy(H)])
            # if condition is reached, abort
            if abs(P_arr[-1][0] - P_arr[-2][0]) < epsilon:
                likelihood = P_arr[-1][0]
                H = P_arr[-1][1]
                break
            if P_arr[-1][0] > 1.0:
                likelihood = P_arr[-2][0]
                H = P_arr[-2][1]
                break
            # calculate gradients
            Gradients = eval_grad_R0(diagnosis, H, DF)
            # update H
            number_of_agents = len(spectrum[0][:-1])
            H, _ = update_h(H, Gradients, step, number_of_agents)
            # print(P_arr)
            # print(H)

        ranked_diagnoses.append([diagnosis, likelihood, H])
        # print(f'diagnosis {diagnosis} rank: [{diagnosis},{likelihood}]')

    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses

def local_estimation_and_derivative_functions(diagnosis, local_spectra):
    LF = []
    # declare variables
    h = []
    for hj in range(len(local_spectra)):
        h.append(sympy.symbols(f'h{hj}'))
    r = []
    for ri in range(len(local_spectra[0])):
        r.append(sympy.symbols(f'r{ri}'))
    for a, lsa in enumerate(local_spectra):
        local_table, gpef, lef, gpdf, ldf = functions.local_estimation_and_derivative_functions_for_agent(h, r, a, lsa, diagnosis)
        LF.append([local_table, gpef, lef, gpdf, ldf])
    return LF, h, r

def eval_P(H, LF):
    # information sent during the evaluation of P
    information_sent_eval_P = 0
    # first P calculation
    P = functions.substitute_and_eval(H, LF[0][2])
    # rest P calculations
    for a in list(range(len(LF)))[1:]:
        information_sent_eval_P += 1
        extended_P = functions.extend_P(P, a, LF[a][0])
        P = functions.substitute_and_eval(H, extended_P)
    return P, information_sent_eval_P

def eval_grad(diagnosis, H, P, LF):
    information_sent_eval_grad = 0
    Gradients = {}
    for a in diagnosis:
        information_sent_eval_grad += 1
        rs_function = P / LF[a][2]
        rs_value = functions.substitute_and_eval(H, rs_function)
        gradient_function = rs_value*LF[a][4]
        gradient_value = functions.substitute_and_eval(H, gradient_function)
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients, information_sent_eval_grad

def update_h(H, Gradients, step, number_of_agents):
    information_sent_update_h = 0
    for key in Gradients.keys():
        information_sent_update_h += number_of_agents - 1
        if H[key] + step * Gradients[key] > 1.0:
            H[key] = 1.0
        elif H[key] + step * Gradients[key] < 0.0:
            H[key] = 0.0
        else:
            H[key] = H[key] + step * Gradients[key]
    return H, information_sent_update_h

def ranking_1(local_spectra, diagnoses, step):
    """
    ranks the diagnoses. for each diagnosis, the agents
    compute a corresponding partial estimation function
    and then pass numeric results following a certain
    order, until the global function is maximized
    :param local_spectra: the local spectra of each agent
    :param diagnoses: the diagnosis list
    :param step: the gradient step
    :return: ranked diagnosis list
    """
    information_sent = 0
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to zero
        # create symbolic local estimation function (LE) for each of the agents
        # create symbolic local derivative function (LD) for each of the agents
        # while true
        #   calculate P_n(d, H, M)
        #   if condition is is reached, abort
        #   calculate gradients
        #   update H
        # print(f'diagnosis {diagnosis} ranking...')

        # initialize H values of the agents involved in the diagnosis to 0.5
        H = {}
        for a in diagnosis:
            H[f'h{a}'] = 0.5

        # initialize an epsilon value for the stop condition: |P_n(d, H, LS) - P_{n-1}(d, H, LS)| < epsilon
        epsilon = 0.0005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [0.0]

        # create symbolic local estimation function (LE) for each of the agents
        # create symbolic local derivative function (LD) for each of the agents
        LF, h, r = local_estimation_and_derivative_functions(diagnosis, local_spectra)

        # while true
        while True:
            # calculate P_n(d, H, LS) and record the sent information
            P, information_sent_eval_P = eval_P(H, LF)
            information_sent += information_sent_eval_P
            P = float(P)
            P_arr.append(P)
            # if condition is is reached, abort
            if abs(P_arr[-1] - P_arr[-2]) < epsilon:
                likelihood = P_arr[-1]
                break
            if P_arr[-1] > 1.0:
                likelihood = P_arr[-2]
                break
            # calculate gradients
            Gradients, information_sent_eval_grad = eval_grad(diagnosis, H, P_arr[-1], LF)
            information_sent += information_sent_eval_grad
            # update H
            number_of_agents = len(local_spectra)
            H, information_sent_update_h = update_h(H, Gradients, step, number_of_agents)
            information_sent += information_sent_update_h
            # print(P_arr)
            # print(H)

        ranked_diagnoses.append([diagnosis, likelihood, H])
        # print(f'diagnosis {diagnosis} rank: [{diagnosis},{likelihood}]')
    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses, information_sent


def request_data_R2(requesting_agent, helping_agent, helping_spectrum):
    new_not_in_fault, new_not_in_ok = 0, 0

    for row in helping_spectrum:
        if 2 not in row:  # helping agent can help here
            if row[requesting_agent] == 1:  # requesting agent doesnt need help here
                continue
            else:
                if 1 in row[:helping_agent]:  # some previous agent (in the row before helping agent) already helped
                    continue
                else:  # no other previous agent has seen this row, helping agent needs to help
                    if row[-1] == 1:
                        new_not_in_fault += 1
                    else:
                        new_not_in_ok += 1

    return new_not_in_fault, new_not_in_ok

def reveal_information_for_single(num_of_agents, new_not_in_fault, new_not_in_ok, j, i):
    revealed_information_table = []
    for k in range(new_not_in_fault):
        row = [2 for ag in range(num_of_agents+1)]
        row[j] = 0
        row[i] = 1
        row[-1] = 1
        revealed_information_table.append(row)

    for k in range(new_not_in_ok):
        row = [2 for ag in range(num_of_agents+1)]
        row[j] = 0
        row[i] = 1
        row[-1] = 0
        revealed_information_table.append(row)
    return revealed_information_table

def calculate_revealed_information_metrics_R2(revealed_information_tables, missing_information_cells):
    # for each revealed row it goes like this:
    # each row contributes 3 revealed information
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent = 0, 0.0, []
    revealed_information_percent_per_agent = []

    for ai, atab in enumerate(revealed_information_tables):
        revealed_information_a = 0
        for conf in atab:
            revealed_information_a += 3
        revealed_information_sum += revealed_information_a
        revealed_information_per_agent.append(revealed_information_a)
        revealed_information_percent_per_agent.append(revealed_information_a * 1.0 / missing_information_cells[ai]
                                                      if missing_information_cells[ai] != 0 else 0.0)
    revealed_information_mean = revealed_information_sum * 1.0 / len(revealed_information_tables)
    return revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_per_agent[-1], revealed_information_percent_per_agent, \
        revealed_information_percent_per_agent[-1]

def ranking_2(local_spectra, diagnoses, missing_information_cells):
    """
    ranks the diagnoses. for each diagnosis, which is essentially
    a single fault of one of the agents, the corresponding agent
    ranks it. to achieve this the said agents request information
    from other agents
    :param local_spectra: the local spectra of each agent
    :param diagnoses: the diagnosis list
    :param missing_information_cells: for metric gathering purposes
    :return: ranked diagnosis list
    """
    information_sent = 0
    revealed_information_tables = []
    ranked_diagnoses = []
    num_of_agents = len(local_spectra)
    for j in range(num_of_agents):
        spectrum_j = local_spectra[j]

        in_fault, in_ok, not_in_fault, not_in_ok = 0, 0, 0, 0
        for row in spectrum_j:
            if row[j] == 1:
                if row[-1] == 1:
                    in_fault += 1
                else:
                    in_ok += 1

        revealed_information_table = []
        for i in range(num_of_agents):
            if i != j:
                information_sent += 1  # the agent sends a request - it is worth 1 unit
                new_not_in_fault, new_not_in_ok = request_data_R2(j, i, local_spectra[i])
                information_sent += 2  # the agent receives two number - those are 2 units
                revealed_information_table_from_i = reveal_information_for_single(num_of_agents, new_not_in_fault, new_not_in_ok, j, i)
                revealed_information_table += revealed_information_table_from_i
                not_in_fault += new_not_in_fault
                not_in_ok += new_not_in_ok
        revealed_information_tables.append(revealed_information_table)

        likelihood = functions.single_fault_ochiai(in_fault, in_ok, not_in_fault, not_in_ok)

        ranked_diagnoses.append([diagnoses[j], likelihood, {}])

    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)

    # calculate sum, mean, and last agent revealed information based on the tables
    revealed_information_sum, revealed_information_mean, revealed_information_per_agent, \
        revealed_information_last, revealed_information_percent_per_agent, revealed_information_percent_last \
        = calculate_revealed_information_metrics_R2(revealed_information_tables, missing_information_cells)

    return normalized_diagnoses, information_sent, revealed_information_sum, revealed_information_mean, \
        revealed_information_per_agent, revealed_information_last, revealed_information_percent_per_agent, \
        revealed_information_percent_last


def ranking_coef_0(S, diagnoses):
    ranked_diagnoses = []
    num_of_agents = len(diagnoses)
    for j in range(num_of_agents):
        in_fault, in_ok, not_in_fault, not_in_ok = 0, 0, 0, 0
        for row in S:
            if row[j] == 1:
                if row[-1] == 1:
                    in_fault += 1
                else:
                    in_ok += 1
            else:
                if row[-1] == 1:
                    not_in_fault += 1
                else:
                    not_in_ok += 1
        likelihood = functions.single_fault_ochiai(in_fault, in_ok, not_in_fault, not_in_ok)
        ranked_diagnoses.append([diagnoses[j], likelihood, {}])

    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)

    return normalized_diagnoses
