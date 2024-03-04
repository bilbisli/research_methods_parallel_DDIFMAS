class Agent(object):

    def __init__(self, num, neighbours, spectrum):
        # agent number identifier - the same as in the spectra column
        self.m1_num = num

        # the identifiers of the agent's neighbours
        self.m2_neighbours = neighbours

        # the agent local activity matrix or spectrum
        self.m3_spectrum = spectrum

        # the runs that this agent already covers
        self.m4_runs_coverage = []

        # a list of all the agents and runs that this agent helped them with
        self.m5_helped_with = [[0 for _ in range(len(spectrum))] for _ in range(len(spectrum[0][:-1]))]

        # the identifier of the current other agent that is about to help this one
        self.m6_current_helper_num = -1

        # the amount of runs that the current helper agent can hep this one with
        self.m7_current_helper_strength = -1

        # the diagnoses that this agent can generate given its spectra
        self.m8_diagnoses = []

        # a priori probability - in how many runs did it participate
        self.m9_apriori = len([i for i, row in enumerate(spectrum) if row[num] == 1]) / len(spectrum)



