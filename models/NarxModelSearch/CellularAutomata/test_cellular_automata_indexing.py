import pytest
from cellular_automata_indexing import CellularAutomataIndexing


@pytest.fixture
def cellular_automata_indexing():
    return CellularAutomataIndexing()


@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [3, 3, 3], [1, 3, 9], [101, 103, 109], 0),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [3, 3, 3], [2, 4, 10], [102, 104, 110], 1),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [3, 3, 3], [11, 15, 8], [111, 115, 108], 17)
    ])
def test_3D_indexing(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list,
                     mse_list, current_mpi_rank):
    mse_index = cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
        current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)
    assert mse_index in lr_selected_rank_list
    assert agents_buffer_mse[mse_index] in mse_list


@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [3, 3, 3], [11, 15, 8], [111, 115, 108], 18)
    ])
def test_3D_indexing_out_of_bounds(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions,
                                   lr_selected_rank_list, mse_list, current_mpi_rank):
    with pytest.raises(Exception):
        cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
            current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)


@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [12, 3], [11, 8], [111, 108], 18)
    ])
def test_2D_indexing_out_of_bounds(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions,
                                   lr_selected_rank_list, mse_list, current_mpi_rank):
    with pytest.raises(Exception):
        cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
            current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)



@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [12], [11], [111], 18)
    ])
def test_1D_indexing_out_of_bounds(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions,
                                   lr_selected_rank_list, mse_list, current_mpi_rank):
    with pytest.raises(Exception):
        cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
            current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)


@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [3, 12], [103, 112], 15),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [1, 4], [101, 104], 0),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [10, 13], [110, 113], 9),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [14, 17], [114, 117], 13),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [0, 17], [100, 117], 16),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [5, 4], [1, 16], [101, 116], 17)
    ])
def test_2D_indexing(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list,
                     mse_list, current_mpi_rank):
    mse_index = cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
        current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)
    assert mse_index in lr_selected_rank_list
    assert agents_buffer_mse[mse_index] in mse_list


@pytest.mark.parametrize(
    'agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list, mse_list, current_mpi_rank', [
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [22], [0], [100], 17),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [22], [1], [101], 0),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [22], [17], [117], 16),
        ([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
         18, [22], [13], [113], 12)
    ])
def test_1D_indexing(cellular_automata_indexing, agents_buffer_mse, mpi_size, ca_dimensions, lr_selected_rank_list,
                     mse_list, current_mpi_rank):
    mse_index = cellular_automata_indexing.get_cellular_automata_linear_selection_neighbour_1D_index(
        current_mpi_rank, mpi_size, ca_dimensions, agents_buffer_mse)
    assert mse_index in lr_selected_rank_list
    assert agents_buffer_mse[mse_index] in mse_list
