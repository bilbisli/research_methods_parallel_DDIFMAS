import methods_for_diagnosis, methods_for_input_preprocess, algorithms

instance_num = 5
noa = 7
nof = 2
afp = 0.5
nor = 10
inum = 4
G = []
F = [0, 1]
T = []
S = [
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 1, 0, 0]
]

# result_mrsd = algorithms.MRSD(instance_num, noa, nof, afp, nor, inum + 1, G, F, T, S)
result_dmrsdI1D1R1 = algorithms.DMRSD_I1D2R1(instance_num, noa, nof, afp, nor, inum + 1, G, F, T, S)

print(9)
