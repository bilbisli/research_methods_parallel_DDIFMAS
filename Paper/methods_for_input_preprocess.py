def input_preprocess_1(noa, S):
    """
    this method takes the spectrum and divides it between the agents
    each agent get only the rows in which it participates (i.e., the
    rows for which it has 1 in its column)
    :param noa: number of agents
    :param S: the complete spectrum
    :return: a list of local spectra, one for each agent
    """
    # initialize for each agent an empty local spectrum
    local_spectra = [[[2 for _ in run] for run in S] for _ in range(noa)]

    # for each agent, populate the relevant rows in his local spectrum
    for a in range(noa):
        for i, row in enumerate(S):
            if row[a] == 1:
                for j in range(len(row)):
                    local_spectra[a][i][j] = S[i][j]

    # calculate for each spectrum the number of missing rows
    missing_information_cells = []
    for ls in local_spectra:
        missing_cells = 0
        for row in ls:
            if 2 in row:
                missing_cells += len(row)
        missing_information_cells.append(missing_cells)

    return local_spectra, missing_information_cells
