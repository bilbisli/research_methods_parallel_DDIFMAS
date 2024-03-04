import copy
# from typing import List, Callable
# from scipy.optimize import minimize
# import numpy as np
import math

import sympy


def conflict_directed_search(conflicts):
    if len(conflicts) == 0:
        return []
    diagnoses = []
    new_diagnoses = [[conflicts[0][i]] for i in range(len(conflicts[0]))]
    for conflict in conflicts[1:]:
        diagnoses = new_diagnoses
        new_diagnoses = []
        while len(diagnoses) != 0:
            diagnosis = diagnoses.pop(0)
            intsec = list(set(diagnosis) & set(conflict))
            if len(intsec) == 0:
                new_diags = [diagnosis + [c] for c in conflict]

                def filter_supersets(new_diag):
                    for d in diagnoses + new_diagnoses:
                        if set(d) <= set(new_diag):
                            return False
                    return True

                filtered_new_diags = list(filter(filter_supersets, new_diags))
                new_diagnoses += filtered_new_diags
            else:
                new_diagnoses.append(diagnosis)
    diagnoses = new_diagnoses
    return diagnoses

def label_diagnosis_sets(local_diagnosis_sets):
    d_diag_to_num = {}
    d_num_to_diag = {}
    global_conflict_set = [[-1 for _ in a] for a in local_diagnosis_sets]

    # give every unique local diagnosis a unique number
    unique_num = 0
    for j, a in enumerate(local_diagnosis_sets):
        for di, d in enumerate(a):
            d.sort()
            if f'{d}' not in d_diag_to_num.keys():
                d_diag_to_num[f'{d}'] = unique_num
                d_num_to_diag[f'{unique_num}'] = d
                unique_num += 1

    # for every diagnosis in the local diagnosis sets
    # insert its number to the global conflict set
    for j, a in enumerate(local_diagnosis_sets):
        for di, d in enumerate(a):
            global_conflict_set[j][di] = d_diag_to_num[f'{d}']

    return global_conflict_set, d_diag_to_num, d_num_to_diag

def refine_diagnoses(global_diagnoses):
    # for each global diagnosis, merge the different
    # local diagnoses to a unified set
    merged = [[] for _ in global_diagnoses]
    for gdi, gd in enumerate(global_diagnoses):
        m = []
        for ldi, ld in enumerate(gd):
            for j, a in enumerate(ld):
                if a not in m:
                    m.append(a)
        merged[gdi] = m

    # remove duplicate united sets and supersets
    for d in merged:
        d.sort()
    merged.sort(key=len)
    no_dups = []
    for d in merged:
        if d not in no_dups:
            no_dups.append(d)
    no_supersets = []
    for d in no_dups:
        d_set = set(d)
        d_is_superset = False
        for ns in no_supersets:
            ns_set = set(ns)
            if ns_set.issubset(d_set):
                d_is_superset = True
        if not d_is_superset:
            no_supersets.append(d)

    return no_supersets

def filter_empty(conflicts):
    filtered = []
    for conf in conflicts:
        if len(conf) > 0:
            filtered.append(copy.deepcopy(conf))
    return filtered

def labels_to_diagnoses(diagnosis_labels, d_num_to_diag):
    # translate back diagnosis numbers to local diagnosis sets
    diagnoses = [[[] for _ in gdn] for gdn in diagnosis_labels]
    for gdi, gdn in enumerate(diagnosis_labels):
        for ni, n in enumerate(gdn):
            diagnoses[gdi][ni] = d_num_to_diag[f'{n}']
    return diagnoses

def sort_diagnoses_by_cardinality(diagnoses):
    dic = {}
    for d in diagnoses:
        dic[f'{d}'] = '+' + '_'.join(list(map(lambda item: str(item), d)))
    max_len = 0
    for key in dic.keys():
        max_len = max(max_len, len(dic[key]))
    max_len += 1
    for key in dic.keys():
        dic[key] = '+' * (max_len - len(dic[key])) + dic[key]
    string_arr = []
    for key in dic.keys():
        string_arr.append(dic[key])
    sorted_string_arr = sorted(string_arr)
    sorted_diagnoses = []
    for ssa in sorted_string_arr:
        res = ssa.replace('+', "")
        splitted = res.split('_')
        diag = [int(item) for item in splitted]
        sorted_diagnoses.append(diag)
    return sorted_diagnoses

# def calculate_e_dk(dk: List[int], activity_matrix: List[List[int]], error_vector: List[int]):
#     funcArr = ['(-1)']
#     objective: Callable[[List[float]], float] = None
#     active_vars = [False] * len(activity_matrix[0])
#
#     # get the active vars in this diagnosis
#     for i, e in enumerate(error_vector):
#         for j, c in enumerate(activity_matrix[i]):
#             if activity_matrix[i][j] == 1 and j in dk:
#                 active_vars[j] = True
#
#     # re-labeling variables to conform to scipy's requirements
#     index_rv = 0
#     renamed_vars = {}
#     for i, av in enumerate(active_vars):
#         if av:
#             renamed_vars[str(i)] = index_rv
#             index_rv += 1
#
#     # building the target function as a string
#     for i, e in enumerate(error_vector):
#         fa = "1*"
#         for j, c in enumerate(activity_matrix[i]):
#             if activity_matrix[i][j] == 1 and j in dk:
#                 fa = fa + f"x[{renamed_vars[str(j)]}]*"
#         fa = fa[:-1]
#         if error_vector[i] == 1:
#             fa = "*(1-" + fa + ")"
#         else:
#             fa = "*(" + fa + ")"
#         funcArr.append(fa)
#
#     # using dynamic programming to initialize the target function
#     func = ""
#     for fa in funcArr:
#         func = func + fa
#     objective = eval(f'lambda x: {func}')
#
#     # building bounds over the variables
#     # and the initial health vector
#     b = (0.0, 1.0)
#     initial_h = 0.5
#     bnds = []
#     h0 = []
#     for av in active_vars:
#         if av:
#             bnds.append(b)
#             h0.append(initial_h)
#
#     # solving the minimization problem
#     h0 = np.array(h0)
#     sol = minimize(objective, h0, method="L-BFGS-B", bounds=bnds, tol=1e-3, options={'maxiter': 100})
#
#     # extracting H's and P
#     H = {}
#     for h in renamed_vars:
#         H[f'h{h}'] = float(sol.x[renamed_vars[h]])
#     P = -sol.fun
#
#     return P, H

def estimation_and_derivative_functions(h, spectrum, diagnosis):
    # estimation function
    ef = sympy.Integer(1)
    for row in spectrum:
        row_comp = sympy.Integer(1)
        for fa in diagnosis:
            if row[fa] == 1:
                # row_comp += f'(h{fa})'
                row_comp = row_comp * h[fa]
        if row[-1] == 1:
            # row_comp = '(1 - ' + row_comp + ')'
            row_comp = 1 - row_comp
        ef = ef * row_comp

    # derivative functions
    DF = []
    for hi, hvar in enumerate(h):
        df = sympy.diff(ef, hvar)
        DF.append(df)
    return ef, DF

def local_estimation_and_derivative_functions_for_agent(h, r, a, lsa, diagnosis):
    # populate the local estimation function table
    local_table = []
    for ri, row in enumerate(lsa):
        le_tab_row = []
        if 2 in row:    # the case where the row is no visible
            row_comp = r[ri]
            else_having = []
        else:           # the case where the row is visible
            row_comp = sympy.Integer(1)
            for fa in diagnosis:
                if row[fa] == 1:
                    # row_comp += f'(h{fa})'
                    row_comp = row_comp * h[fa]
            if row[-1] == 1:
                # row_comp = '(1 - ' + row_comp + ')'
                row_comp = 1 - row_comp
            else_having = []
            for c in range(len(row[:-1])):
                if c != a:
                    if row[c] == 1:
                        else_having.append(c)
        le_tab_row.append(row_comp)
        le_tab_row.append(else_having)
        local_table.append(le_tab_row)

    # global partial estimation function
    gpef = sympy.Integer(1)
    for row in local_table:
        gpef = gpef * row[0]

    # local estimation function
    lef = sympy.Integer(1)
    for row in local_table:
        if row[0] not in r:
            lef = lef * row[0]

    # global partial derivative function
    gpdf = sympy.diff(gpef, h[a])

    # local derivative function
    ldf = sympy.diff(lef, h[a])

    return local_table, gpef, lef, gpdf, ldf

def substitute_and_eval(H, lf):
    free_syms = lf.free_symbols
    substitution = {}
    for fs in free_syms:
        substitution[fs] = H[fs.name]
    P = lf.subs(substitution)
    return P

def extend_P(P, a, local_table):
    new_P = P
    for row in local_table:
        if not isinstance(row[0], int):
            if not row[0].is_symbol or row[0].name[0] == 'h':
                if len(row[1]) == 0:
                    new_P = new_P * row[0]
                else:
                    min_agent = min(row[1])
                    if min_agent > a:
                        new_P = new_P * row[0]
    return new_P

def normalize_diagnoses(ranked_diagnoses):
    normalized_diagnoses = []
    probabilities_sum = 0.0
    for diagnosis in ranked_diagnoses:
        normalized_diagnoses.append(copy.deepcopy(diagnosis))
        probabilities_sum += diagnosis[1]
    for diagnosis in normalized_diagnoses:
        diagnosis[1] = diagnosis[1] / probabilities_sum
    return normalized_diagnoses

def calculate_wasted_effort(number_of_agents, oracle, diagnoses):
    healthy_agents = [a for a in list(range(number_of_agents)) if a not in oracle]
    agents_examined = []
    faulty_agents_examined = []
    healthy_agents_examined = []
    for diagnosis in diagnoses:
        if len(faulty_agents_examined) == len(oracle):
            break
        d_agents = diagnosis[0]
        for a in d_agents:
            if a not in agents_examined:
                agents_examined.append(a)
                if a in oracle:
                    faulty_agents_examined.append(a)
                else:
                    healthy_agents_examined.append(a)
    wasted_effort = len(healthy_agents_examined)
    wasted_effort_percent = wasted_effort * 1.0 / len(healthy_agents)
    useful_effort = len(faulty_agents_examined)
    useful_effort_percent = useful_effort * 1.0 / len(oracle)
    return wasted_effort, wasted_effort_percent, useful_effort, useful_effort_percent

def calculate_weighted_precision_and_recall(number_of_agents, oracle, diagnoses):
    top_k_precision_accums = [0 for _ in diagnoses]
    top_k_recall_accums = [0 for _ in diagnoses]
    healthy_agents = [a for a in list(range(number_of_agents)) if a not in oracle]
    for k in range(len(diagnoses)):
        top_k_diagnoses = [copy.deepcopy(diagnosis) for diagnosis in diagnoses[:k+1]]
        top_k_diagnoses_probability_sum = sum([d[1] for d in top_k_diagnoses])
        for tkd in top_k_diagnoses:
            tkd[1] = tkd[1] / top_k_diagnoses_probability_sum
        precision_accum = 0
        recall_accum = 0
        for diagnosis in top_k_diagnoses:
            diagnosed_agents = diagnosis[0]
            diagnosis_probability = diagnosis[1]
            diagnosed_healthy_agents = [a for a in list(range(number_of_agents)) if a not in diagnosed_agents]
            precision, recall = precision_recall_for_diagnosis(diagnosed_agents, diagnosed_healthy_agents, oracle, healthy_agents)
            if precision != "undef":
                precision_accum = precision_accum + precision * diagnosis_probability
            if recall != "undef":
                recall_accum = recall_accum + recall * diagnosis_probability
        top_k_precision_accums[k], top_k_recall_accums[k] = precision_accum, recall_accum
    return top_k_precision_accums, top_k_recall_accums

def precision_recall_for_diagnosis(diagnosed_agents, diagnosed_healthy_agents, oracle, healthy_agents):
    fp = len([a for a in diagnosed_agents if a not in oracle])
    fn = len([a for a in diagnosed_healthy_agents if a not in healthy_agents])
    tp = len([a for a in diagnosed_agents if a in oracle])
    tn = len([a for a in diagnosed_healthy_agents if a in healthy_agents])
    if (tp + fp) == 0:
        precision = "undef"
    else:
        precision = (tp + 0.0) / float(tp + fp)
    if (tp + fn) == 0:
        recall = "undef"
    else:
        recall = (tp + 0.0) / float(tp + fn)
    return precision, recall

def reveal_information(diagnoses, number_of_agents):
    # conflict-diagnosis duality:
    # Stern, Roni Tzvi, et al.
    # "Exploring the duality in conflict-directed model-based diagnosis."
    # Twenty-Sixth AAAI Conference on Artificial Intelligence. 2012.
    revealed_information_table = []
    diags = copy.deepcopy(diagnoses)
    conflicts = conflict_directed_search(diags)
    for conf in conflicts:
        row = [0 for _ in range(number_of_agents+1)]
        for c in conf:
            row[c] = 1
        row[-1] = 1
        revealed_information_table.append(row)
    return revealed_information_table


def single_fault_ochiai(n11, n10, n01, n00):
    """
    ochiai
                                     n11
                result = -----------------------------
                             ______________________
                           \/ (n11+n10) * (n11+n01)

    """
    # soj = ((dm[0][j] * 1.0) / math.sqrt((dm[0][j] + dm[1][j]) * (dm[0][j] + dm[2][j]))) if dm[0][j] != 0 else 0
    result = n11 * 1.0 / math.sqrt((n11+n10) * (n11+n01)) if (n11+n10) * (n11+n01) != 0 else 0.0
    return result
