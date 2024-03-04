import copy
from random import shuffle
from agent import Agent
from operator import itemgetter

############################################ Support functions ########################################
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
############################################ Support functions ########################################

########################################## First level functions ######################################
def create_agents(A, G, T, AM):
    agents = []
    for a in A:
        # building the spectrum
        spectrum = [[2 for i in range(len(A)+1)] for _ in range(len(AM))]
        for i, r in enumerate(spectrum):
            # his entire column
            spectrum[i][a] = AM[i][a]
            # the entire vector e
            spectrum[i][-1] = AM[i][-1]
            # only the cells of the neighbours for rows belonging to messages that a forwarded
            trace = T[i]
            for tci, tc in enumerate(trace):
                if tc == a:
                    if tci == 0:
                        spectrum[i][trace[tci + 1]] = AM[i][trace[tci + 1]]
                    elif tci == len(trace) - 1:
                        spectrum[i][trace[tci - 1]] = AM[i][trace[tci - 1]]
                    else:
                        spectrum[i][trace[tci + 1]] = AM[i][trace[tci + 1]]
                        spectrum[i][trace[tci - 1]] = AM[i][trace[tci - 1]]
        agent = Agent(a, G[a], spectrum)
        agents.append(agent)
    return agents

def no_agent_covers_all_runs(agents, C):
    for c in C:
        if len(agents[c].m4_runs_coverage) == len(agents[c].m3_spectrum):
            return False
    return True


def cover_runs(agents, C):
    for c in C:
        lam = agents[c].m3_spectrum
        agents[c].m4_runs_coverage = [i for i, run in enumerate(lam) if (run[-1] == 0 or (run[-1] == 1 and 1 in run[:-1]))]


def elect_candidates_for_resumption(agents, C):
    new_C = []
    max_cov = max([len(a.m4_runs_coverage) for a in agents])
    for c in C:
        if len(agents[c].m4_runs_coverage) == max_cov:
            new_C.append(c)
    return new_C


def filter_candidates_for_resumption(agents, C):
    new_C = []
    for c in C:
        c_ag = agents[c]
        c_ag.m6_current_helper_num = -1
        c_ag.m7_current_helper_strength = -1
        needed_help_amount = len(c_ag.m3_spectrum) - len(c_ag.m4_runs_coverage)
        for a in agents:
            if a.m1_num != c:
                # print(f'agent c - {c} asking help from agent a - {a.m1_num} with {needed_help_amount} runs')
                help_strength = max(len(a.m4_runs_coverage) - sum(a.m5_helped_with[c]), 0)
                # print(f'agent a - {a.m1_num} can help agent c - {c} with {help_strength} amount')
                if help_strength > c_ag.m7_current_helper_strength:
                    # print(f'============== agent a - {a.m1_num} is chosen (for now) with strength {help_strength}')
                    c_ag.m7_current_helper_strength = help_strength
                    c_ag.m6_current_helper_num = a.m1_num
    max_contrib = max([agents[c].m7_current_helper_strength for c in C])
    for c in C:
        if agents[c].m7_current_helper_strength >= max_contrib:
            new_C.append(c)
    return new_C


def update_projection(agents, C, revi):
    for c in C:
        c_ag = agents[c]
        needed_runs = [i for i, run in enumerate(c_ag.m3_spectrum) if i not in c_ag.m4_runs_coverage]
        hn = c_ag.m6_current_helper_num
        # print(f'agent c - {c} is asking the helper with num - {hn} for runs {needed_runs}')
        helper_ag = agents[hn]
        for i, r in enumerate(helper_ag.m3_spectrum):
            if i in needed_runs:
                # print(f'agent c - {c} is asking the helper with num - {hn} for run {i}')
                helper_ag = agents[hn]
                # print(f'helper agent - {hn} records that agent c - {c} has 0 in his column for run {i}')
                helper_ag.m3_spectrum[i][c] = 0
                revi += 1
                # print(f'helper agent - {hn} helps agents c - {c} by revealing his')
                helper_ag.m5_helped_with[c][i] = 1
                c_ag.m3_spectrum[i][hn] = helper_ag.m3_spectrum[i][hn]
                revi += 1
            else:
                # print(f'agent c - {c} does not need help with run {i} - helper agent - {hn} marks this run as helped')
                helper_ag.m5_helped_with[c][i] = 1
    return revi

def rank_diagnoses(agents, diagnoses_c):
    ranked_diagnoses = [[item, -1] for item in diagnoses_c]
    for d, diag in enumerate(diagnoses_c):
        rank = 1.0
        for c in diag:
            rank += agents[c].m9_apriori
        ranked_diagnoses[d][1] = rank

    # normalize the values
    sum_ranks = 0.0
    normalized_ranked_diagnoses = [[item, -1] for item in diagnoses_c]
    for i, rd in enumerate(ranked_diagnoses):
        sum_ranks += rd[1]
    for i, rd in enumerate(ranked_diagnoses):
        normalized_ranked_diagnoses[i][1] = rd[1] / sum_ranks

    # sort the diagnoses
    normalized_ranked_diagnoses = sorted(normalized_ranked_diagnoses, key=itemgetter(1), reverse=True)
    return normalized_ranked_diagnoses

def compute_diagnoses(agents, C):
    diagnoses = {}
    for c in C:
        lam = agents[c].m3_spectrum
        conflicts = []
        for i, row in enumerate(lam):
            # option #1 - every line generates conflicts - means that there can also be empty conflicts
            # if row[-1] == 1:
            #     conf = [j for j, comp in enumerate(row[:-1]) if comp == 1]
            #     conflicts.append(conf)
            # option #2 - only lines that generate non empty conflicts are accepted
            if row[-1] == 1 and 1 in row[:-1]:
                conf = [j for j, comp in enumerate(row[:-1]) if comp == 1]
                conflicts.append(conf)
        diagnoses_c = conflict_directed_search(conflicts)
        ranked_diagnoses = rank_diagnoses(agents, diagnoses_c)
        diagnoses[f'{c}'] = ranked_diagnoses
    return diagnoses

def compute_centralized_diagnosis(AM):
    conflicts = []
    for i, row in enumerate(AM):
        if row[-1] == 1 and 1 in row[:-1]:
            conf = [j for j, comp in enumerate(row[:-1]) if comp == 1]
            conflicts.append(conf)
    diagnoses = conflict_directed_search(conflicts)
    return diagnoses
########################################## First level functions ######################################

###################################### Metric calculation functions ###################################
def precision_recall_for_diagnosis(faulty_agents, dg, pr, healthy_agents):
    fp = len([i1 for i1 in dg if i1 in healthy_agents])
    fn = len([i1 for i1 in faulty_agents if i1 not in dg])
    tp = len([i1 for i1 in dg if i1 in faulty_agents])
    tn = len([i1 for i1 in healthy_agents if i1 not in dg])
    if ((tp + fp) == 0):
        precision = "undef"
    else:
        precision = (tp + 0.0) / float(tp + fp)
        a = precision
        precision = precision * float(pr)
    if ((tp + fn) == 0):
        recall = "undef"
    else:
        recall = (tp + 0.0) / float(tp + fn)
        recall = recall * float(pr)
    return precision, recall

def compute_precision_recall(O, agents, C, diagnoses):
    precisions = {}
    recalls = {}
    healthy_agents = [i for i in range(len(agents)) if i not in O]
    for c in C:
        precision_c = 0.0
        recall_c = 0.0
        # diagnoses_c = agents[c].m8_diagnoses
        diagnoses_c = diagnoses[f'{c}']
        # print(f'agent {c} diagnoses: {diagnoses_c}')
        for d in diagnoses_c:
            dg = d[0]
            pr = d[1]
            prec, rec = precision_recall_for_diagnosis(O, dg, pr, healthy_agents)
            if (rec != "undef"):
                recall_c = recall_c + rec
            if (prec != "undef"):
                precision_c = precision_c + prec
        precisions[f'{c}'] = precision_c
        recalls[f'{c}'] = recall_c

    return precisions, recalls

def combine_diagnoses_and_normalize(diagnoses):
    combined_diagnoses = []

    # combine diagnoses
    for key in diagnoses.keys():
        for d in diagnoses[key]:
            combined_diagnoses.append(d)

    # fuse same diagnoses and average their probability
    combined_diagnoses2 = []
    for d in combined_diagnoses:
        if d[0] not in combined_diagnoses2:
            combined_diagnoses2.append(d[0])
            combined_diagnoses2.append(d[1])
            combined_diagnoses2.append(1)
        else:
            idx = combined_diagnoses2.index(d[0])
            combined_diagnoses2[idx+1] += d[1]
            combined_diagnoses2[idx+2] += 1
    combined_diagnoses3 = []
    for i in range(0, len(combined_diagnoses2), 3):
        combined_diagnoses3.append([combined_diagnoses2[i], combined_diagnoses2[i+1] / combined_diagnoses2[i+2]])

    # normalize the diagnoses
    sum_ranks = 0
    for d in combined_diagnoses3:
        sum_ranks += d[1]

    for d in combined_diagnoses3:
        d[1] = d[1] / sum_ranks

    # sort the diagnoses
    combined_diagnoses3 = sorted(combined_diagnoses3, key=itemgetter(1), reverse=True)

    return combined_diagnoses3

# def compute_precision_recall_single(O, agents, combined_diagnoses):
#     precision = 0.0
#     recall = 0.0
#     healthy_agents = [i for i in range(len(agents)) if i not in O]
#     for d in combined_diagnoses:
#         dg = d[0]
#         pr = d[1]
#         prec, rec = precision_recall_for_diagnosis(O, dg, pr, healthy_agents)
#         if (rec != "undef"):
#             recall = recall + rec
#         if (prec != "undef"):
#             precision = precision + prec
#     return precision, recall

def calc_precision_recall_single(O, agents, combined_diagnoses):
    top_k_precision_accums = [0 for _ in combined_diagnoses]
    top_k_recall_accums = [0 for _ in combined_diagnoses]
    validAgents = [i for i in range(len(agents)) if i not in O]
    for k in range(len(combined_diagnoses)):
        top_k_diagnoses = combined_diagnoses[:k+1]
        top_k_diagnoses_sum = sum([d1[1] for d1 in top_k_diagnoses])
        for tkd in top_k_diagnoses:
            tkd[1] = tkd[1] / top_k_diagnoses_sum
        precision_accum = 0
        recall_accum = 0
        for d in top_k_diagnoses:
            dg = d[0]
            pr = d[1]
            precision, recall = precision_recall_for_diagnosis(O, dg, pr, validAgents)
            if (recall != "undef"):
                recall_accum = recall_accum + recall
            if (precision != "undef"):
                precision_accum = precision_accum + precision
        top_k_precision_accums[k], top_k_recall_accums[k] = precision_accum, recall_accum
    return top_k_precision_accums, top_k_recall_accums

###################################### Metric calculation functions ###################################

################################################ Algorithm ############################################
def DMRSD(instance_number,
          faulty_agents_number,
          nodfap,
          notg,
          O, A, G, T, AM):
    # showing the centralized hitting set
    centralized_diagnosis = compute_centralized_diagnosis(AM)
    print(f'centralized diagnosis: {centralized_diagnosis}')
    print('')

    # the final result lines to be returned
    result = []

    # builds the agent including projection
    agents = create_agents(A, G, T, AM)

    # the candidate list C is a sublist of agents that indicates which agents
    # are still working on diagnosing the system
    C = copy.deepcopy(A)

    # revealed information is 0 at this point
    revealed_information = 0

    # while there is not a single agent that covers all the bad runs
    iteration = 0

    # here the agents in C will compute the first diagnoses (a hitting set
    # per agent). at teh end of every iteration the agents will compute
    # again a hitting set
    print(f'iteration: {iteration}')
    diagnoses = compute_diagnoses(agents, C)
    print(diagnoses)
    combined_diagnoses = combine_diagnoses_and_normalize(diagnoses)
    print(f'combined_diagnoses: {combined_diagnoses}')
    precision_single, recall_single = calc_precision_recall_single(O, agents, combined_diagnoses)
    precision = precision_single[-1] if len(precision_single) > 0 else precision_single
    recall = recall_single[-1] if len(recall_single) > 0 else recall_single
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print('')

    # adding the pre iterations to the result lines
    result.append([
        instance_number,
        faulty_agents_number,
        nodfap,
        notg,
        iteration,
        -1,
        str(O),
        '\r\n'.join(list(map(lambda dic: str(dic), combined_diagnoses))),
        revealed_information,
        precision,
        recall
    ])

    while no_agent_covers_all_runs(agents, C):
        iteration += 1
        print(f'iteration: {iteration}')
        revi = 0

        # each agent tries to cover as much runs as he can, the runs coverage is a list
        # of numbers, each number in the list indicates which runs are covered by the agent
        cover_runs(agents, C)

        # candidates for resuming of the diagnosis process are elected based
        # on their coverage counts and needed counts
        C = elect_candidates_for_resumption(agents, C)

        # candidates are filtered by which of them can use help to cover as much as possible
        # runs by picking only help from one other agent
        # each agent in C asks all the other agents - in how many runs they can help him
        # important note - the agent does not disclose in which specific runs he needs help
        # important note - the other agents do not answer in which runs they can hep him
        C = filter_candidates_for_resumption(agents, C)

        # local activity matrices for candidates in C are updated
        revi = update_projection(agents, C, revi)
        revealed_information += revi
        print(f'revealed information iteration {iteration}: {revi}, total until now: {revealed_information}')

        # computes the updated diagnoses
        diagnoses = compute_diagnoses(agents, C)
        print(f'diagnoses: {diagnoses}')

        # computes and prints the precision and recall for every diagnosis of every agent
        precisions, recalls = compute_precision_recall(O, agents, C, diagnoses)
        print(f'precisions: {precisions}')
        print(f'recalls: {recalls}')

        # combine the diagnoses of the agents to a single list and then calculate for them weighted precision recall
        combined_diagnoses = combine_diagnoses_and_normalize(diagnoses)
        print(f'combined_diagnoses: {combined_diagnoses}')
        precision_single, recall_single = calc_precision_recall_single(O, agents, combined_diagnoses)
        precision = precision_single[-1] if len(precision_single) > 0 else precision_single
        recall = recall_single[-1] if len(recall_single) > 0 else recall_single
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print('')

        # adding to the result line
        result.append([
            instance_number,
            faulty_agents_number,
            nodfap,
            notg,
            iteration,
            -1,
            str(O),
            '\r\n'.join(list(map(lambda dic: str(dic), combined_diagnoses))),
            revealed_information,
            precision,
            recall
        ])
    for rl in result:
        rl[5] = iteration
    print(9)

    return result
