import numpy as np


class CellularAutomataIndexing:

    def __init__(self):
        pass
        # rank, size, cellular_automata_dimensions
        # index_nD,
        # population_count, selection_pressure=1.5):
        # grid_mse_rank_list

    def convert_1D_to_nD_index(self, rank, size, cellular_automata_dimensions):
        returning_nD_index = []
        n = len(cellular_automata_dimensions)
        ca = cellular_automata_dimensions[0]
        for j in range(1, n):
            ca *= cellular_automata_dimensions[j]
        for i in range(1, n - 1):
            ca /= cellular_automata_dimensions[n - i]
            returning_nD_index.append(int(rank / ca))
            rank = int(rank % ca)
        returning_nD_index.append(int(rank % cellular_automata_dimensions[1]))
        returning_nD_index.append(int(rank / cellular_automata_dimensions[1]))
        returning_nD_index.reverse()
        return returning_nD_index

    def convert_nD_to_1D_index(self, index_nD, size, cellular_automata_dimensions):
        returning_1D_index = 0
        n = len(cellular_automata_dimensions)
        ca = cellular_automata_dimensions[0]
        for i in range(2, n):
            ca *= cellular_automata_dimensions[i - 1]
            returning_1D_index += index_nD[-(n - i)] * ca
        returning_1D_index += index_nD[-n] * cellular_automata_dimensions[1]
        returning_1D_index += index_nD[-(n - 1)]
        return returning_1D_index

    def get_1D_neighbours(self, rank, size, cellular_automata_dimensions):
        returning_1D_neighbours = []
        index_nD = self.convert_1D_to_nD_index(rank, size, cellular_automata_dimensions)
        for idx in range(len(cellular_automata_dimensions)):
            new_nD = index_nD.copy()
            new_nD[idx] += 1
            if new_nD[idx] >= cellular_automata_dimensions[idx]:
                new_nD[idx] = 0
            else:
                new_1D = self.convert_nD_to_1D_index(new_nD, size, cellular_automata_dimensions)
                if new_1D >= size:
                    new_nD[idx] = 0
            new_1D = self.convert_nD_to_1D_index(new_nD, size, cellular_automata_dimensions)
            returning_1D_neighbours.append(new_1D)
        return returning_1D_neighbours

    def get_linear_ranking_selection_probabilities(self, population_count, selection_pressure=1.5):
        returning_probabilities = []
        s = selection_pressure
        mu = population_count
        for i in range(population_count - 1, -1, -1):
            returning_probabilities.append((2 - s) / mu + (2 * i * (s - 1)) / (mu * (mu - 1)))
        return np.array(returning_probabilities)

    def get_cellular_automata_linear_selection_neighbour_1D_index(self, rank, size, cellular_automata_dimensions, grid_mse_rank_list):

        if len(cellular_automata_dimensions) > 1:
            oneD_neighbours = self.get_1D_neighbours(rank, size, cellular_automata_dimensions)
            print("== oneD_neighbours: {}".format(oneD_neighbours))  # TODO: test
            mu = len(oneD_neighbours)
            s = 1.5  # 1 < s <= 2: 1 means NO pressure at all, 2 means worst aint's selected
            idx = np.random.choice(mu, 1, p=self.get_linear_ranking_selection_probabilities(mu, s))
            neighborhood_mse = []
            for oneD_index in oneD_neighbours:
                neighborhood_mse.append(grid_mse_rank_list[oneD_index])
            neighborhood_mse.sort()
            selected_neighborhood_mse = neighborhood_mse[idx[0]]
            return grid_mse_rank_list.index(selected_neighborhood_mse)
        else:
            next_index = rank + 1
            if next_index > size - 1:
                next_index = 0
            return next_index
