import numpy as np


def convert_1D_to_nD_index(index_1D, max_ranks, cellular_automata_dimensions):
    returning_nD_index = []
    n = len(cellular_automata_dimensions)
    ca = cellular_automata_dimensions[0]
    for j in range(1, n):
        ca *= cellular_automata_dimensions[j]
    for i in range(1, n - 1):
        ca /= cellular_automata_dimensions[n - i]
        returning_nD_index.append(int(index_1D / ca))
        index_1D = int(index_1D % ca)
    returning_nD_index.append(int(index_1D % cellular_automata_dimensions[1]))
    returning_nD_index.append(int(index_1D / cellular_automata_dimensions[1]))
    returning_nD_index.reverse()
    return returning_nD_index


def convert_nD_to_1D_index(index_nD, max_ranks, cellular_automata_dimensions):
    returning_1D_index = 0
    n = len(cellular_automata_dimensions)
    ca = cellular_automata_dimensions[0]
    for i in range(2, n):
        ca *= cellular_automata_dimensions[i - 1]
        returning_1D_index += index_nD[-(n - i)] * ca
    returning_1D_index += index_nD[-n] * cellular_automata_dimensions[1]
    returning_1D_index += index_nD[-(n - 1)]
    return returning_1D_index


def get_1D_neighbours(index_1D, max_ranks, cellular_automata_dimensions):
    returning_1D_neighbours = []
    index_nD = convert_1D_to_nD_index(index_1D, max_ranks, cellular_automata_dimensions)
    for idx in range(len(cellular_automata_dimensions)):
        new_nD = index_nD.copy()
        new_nD[idx] += 1
        if new_nD[idx] >= cellular_automata_dimensions[idx]:
            new_nD[idx] = 0
        else:
            new_1D = convert_nD_to_1D_index(new_nD, max_ranks, cellular_automata_dimensions)
            if new_1D >= max_ranks:
                new_nD[idx] = 0
        new_1D = convert_nD_to_1D_index(new_nD, max_ranks, cellular_automata_dimensions)
        returning_1D_neighbours.append(new_1D)
    return returning_1D_neighbours


def get_linear_ranking_selection_probabilities(population_count, selection_pressure=1.5):
    returning_probabilities = []
    s = selection_pressure
    mu = population_count
    for i in range(population_count - 1, -1, -1):
        returning_probabilities.append((2 - s) / mu + (2 * i * (s - 1)) / (mu * (mu - 1)))
    return np.array(returning_probabilities)


def get_cellular_automata_linear_selection_neighbour_1D_index(index_1D, max_ranks, cellular_automata_dimensions, grid_mse_1D_list):
    oneD_neighbours = get_1D_neighbours(index_1D, max_ranks, cellular_automata_dimensions)
    mu = len(oneD_neighbours)
    s = 1.5  # 1 < s <= 2: 1 means NO pressure at all, 2 means worst aint's selected
    idx = np.random.choice(mu, 1, p=get_linear_ranking_selection_probabilities(mu, s))
    neighborhood_mse = []
    for oneD_index in oneD_neighbours:
        neighborhood_mse.append(grid_mse_1D_list[oneD_index])
    neighborhood_mse.sort()
    selected_neighborhood_mse = neighborhood_mse[idx[0]]
    return grid_mse_1D_list.index(selected_neighborhood_mse)


if __name__ == '__main__':
    agentsBuffer_mse = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
    index_1D = 0
    max_ranks = len(agentsBuffer_mse)
    CA_dimensions = [5, 3, 3]
    lr_selected_1D_agent = get_cellular_automata_linear_selection_neighbour_1D_index(index_1D, max_ranks, CA_dimensions, agentsBuffer_mse)

    print("CA dimensions: {}".format(CA_dimensions))
    print("index_1D: {}".format(index_1D))
    print("max_ranks: {}".format(max_ranks))
    print("agentsBuffer: {}".format(agentsBuffer_mse))
    print("Linearly selected 1D agent neighbour: {}".format(lr_selected_1D_agent))
