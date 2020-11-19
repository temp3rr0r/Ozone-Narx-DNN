from __future__ import print_function
import os
import pandas as pd
from base.ModelSearch import random_model_search, \
    differential_evolution_model_search, basin_hopping_model_search, \
    simplicial_homology_global_optimization_model_search, local_model_search, \
    particle_swarm_optimization_model_search, bounds, get_random_model, \
    dual_annealing_model_search, bayesian_optimization_model_search, genetic_algorithm_model_search
import time
from mpi4py import MPI
import json
from CellularAutomata.cellular_automata_indexing import CellularAutomataIndexing

print("--- Usage:\n\tmpiexec -n <integer: process count> python mpiNeuroevolutionIslands.py")


def estimate_total_message_count(islands_in, size_in, data_manipulation_in):
    """
    Tries to estimate the total MPI messages needed for the current Island Transpeciation configuration.
    :param islands_in: Vector of strings, indicating each island type.
    :param size_in: Total island count.
    :param data_manipulation_in: Data object, containing information on islands: agents & iterations.
    :return: The estimated total message count for all the islands.
    """
    total_message_count = 0
    iterations = data_manipulation_in["iterations"]
    pso_message_count = (iterations + 1) * data_manipulation_in["agents"]
    rand_message_count = iterations
    bo_message_count = iterations
    ga_message_count = (iterations + 1) * data_manipulation_in["agents"]
    de_message_count = (  # (data_manipulation["iterations"] + 1)
            2 * data_manipulation_in["agents"] * len(bounds))
    bh_message_count = iterations
    sg_message_count = iterations
    da_message_count = iterations

    for i in range(1, size_in):
        if islands_in[i] == "pso":
            total_message_count += pso_message_count
        elif islands_in[i] == "de":
            total_message_count += de_message_count
        elif islands_in[i] == "rand":
            total_message_count += rand_message_count
        elif islands_in[i] == "bo":
            total_message_count += bo_message_count
        elif islands_in[i] == "ga":
            total_message_count += ga_message_count
        elif islands_in[i] == "bh":
            total_message_count += bh_message_count
        elif islands_in[i] == "sg":
            total_message_count += sg_message_count
        elif islands_in[i] == "da":
            total_message_count += da_message_count
        elif islands_in[i] == "ls":
            total_message_count += rand_message_count

    return int(total_message_count)


with open('settings/data_manipulation.json') as f:  # Read the settings json file
    data_manipulation = json.load(f)
modelLabel = data_manipulation["modelLabel"]

# Read last best model parameters for local search
if os.path.exists("foundModels/best_model_parameters.pkl"):
    best_model_parameters_df = pd.read_pickle("foundModels/best_model_parameters.pkl")
    data_manipulation["best_model_parameters"] = best_model_parameters_df["best_model_parameters"]
    print("data_manipulation['best_model_parameters']: {}".format(data_manipulation["best_model_parameters"]))

# First island in vector is not considered
islands = ['ls'] * 7  # Local search islands
# islands = ['pso', 'ga', 'bo', 'de', 'rand'] * 7  # TODO: test/debug DA islands
# islands = ['rand'] * 20
# islands = ['bo'] * 7  # TODO: test/debug DA islands
# islands = ['pso', 'ga', 'bo', 'de', 'rand', 'ls'] * 7  # TODO: LS island =OUT=> global islands

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

if rank == 0:  # Master Node

    swappedAgent = -1  # Rand init buffer agent
    startTime = time.time()
    totalSecondsWork = 0
    mean_mse_threshold = data_manipulation["min_mse_threshold"]  # TODO:
    max_evaluations_threshold = data_manipulation["iterations"]  # TODO: Max 350 global and 150 local fitness evaluations

    for worker in range(1, size):  # Init workers
        initDataToWorkers = {"command": "init", "island": islands[worker]}
        comm.send(initDataToWorkers, dest=worker, tag=0)
        print("--- Rank {}. Sending data: {} to {}...".format(rank, initDataToWorkers, worker))

    swapCounter = 0
    agentBuffer = get_random_model()
    agentsBuffer = [get_random_model() for i in range(size - 1)]  # Storage for all island agents
    agentsMse = [mean_mse_threshold] * (size - 1)  # Store all island agents mse

    cellularAutomataIndexing = CellularAutomataIndexing()  # 1D <-> nD Cellular Automata grid indexing

    overallMinMse = 10e4
    evaluations = 0
    bestIsland = ""

    totalMessageCount = estimate_total_message_count(islands, size, data_manipulation)
    print("--- Expecting {} total messages...".format(totalMessageCount))

    for messageId in range(totalMessageCount):
        swapCounter += 1

        # Worker to master

        req = comm.irecv(tag=1)
        data_worker_to_master = req.wait()

        totalSecondsWork += data_worker_to_master["worked"]
        print("mean_mse: {} ({}: {})".format(data_worker_to_master["mean_mse"], data_worker_to_master["island"],
                                             data_worker_to_master["iteration"]))

        agentsBuffer[data_worker_to_master["rank"] - 1] = data_worker_to_master["agent"]
        agentsMse[data_worker_to_master["rank"] - 1] = data_worker_to_master["mean_mse"]

        evaluations += 1
        if data_worker_to_master["mean_mse"] < overallMinMse:
            overallMinMse = data_worker_to_master["mean_mse"]
            bestIsland = data_worker_to_master["island"]
            if data_manipulation["sendBestAgentFromBuffer"]:
                agentBuffer = data_worker_to_master["agent"]  # Send the best agent received so far

            print("--- New overall min MSE: {} ({}: {}) (overall: {})".format(
                overallMinMse, data_worker_to_master["island"], data_worker_to_master["iteration"], evaluations))

        # if data_worker_to_master["mean_mse"] <= mean_mse_threshold:  # Stop condition if mean_mse <= threshold
        #     print("Abort: mean_mse = {} less than {} threshold".format(data_worker_to_master["mean_mse"], mean_mse_threshold))
        #     # TODO: store file on abort
        #     comm.Abort()
        if evaluations >= max_evaluations_threshold:  # TODO: stop condition if too many evaluations
            print("Abort: evaluations = {} more than maximum {}".format(evaluations, max_evaluations_threshold))
            # TODO: store file on abort
            comm.Abort()

        # Master to worker
        agent_to_send = 0  # Default self for 1 island
        current_rank = data_worker_to_master["rank"]
        if size > 2:  # 2+ islands
            # Pick best agent from nD grid neighbors
            agent_to_send = cellularAutomataIndexing.get_cellular_automata_linear_selection_neighbour_1D_index(
                current_rank - 1, size - 1, data_manipulation["cellular_automata_dimensions"], agentsMse)

        dataMasterToWorker = {"swapAgent": True, "agent": agentsBuffer[agent_to_send],
                              "mean_mse": agentsMse[agent_to_send],
                              "iteration": data_worker_to_master["iteration"], "fromRank": agent_to_send + 1}
        comm.send(dataMasterToWorker, dest=data_worker_to_master["rank"], tag=2)

    endTime = time.time()
    print("--- Overall min MSE (total evals: {}): {} ({})".format(evaluations, overallMinMse, bestIsland))
    print("--- Total work: %d secs in %.2f secs, speedup: %.2f / %d" % (
        totalSecondsWork, round(endTime - startTime, 2),
        totalSecondsWork / round(endTime - startTime, 2), size - 1))

else:  # Worker Node

    print("waiting({})...".format(rank))

    initData = comm.recv(source=0, tag=0)  # Block wait the init command by the master
    if initData["command"] == "init":

        print("working({})...".format(rank))
        print("working({})...".format(rank))
        island = initData["island"]  # Get the island type from the master
        print("--- Rank {}. Data Received: {}!".format(rank, initData))
        print("--- Island: {}".format(island))

        data_manipulation["rank"] = rank
        data_manipulation["island"] = island
        data_manipulation["comm"] = comm

        # TODO: add/test (single or multi-agent) optimizers:
        # TODO: - Reinforcement Learning for continuous + discrete spaces
        # TODO: - XGBoost
        # TODO: - Ant Colony Optimization (layer types only or bounded numerical if possible)
        # TODO: - Inductive Learning Programming (Known ts-DL layers/techniques (legends) =(progol)=>
        # TODO:     ML learned rules =(prolog)=> candidate layers
        # TODO: - Differentiable optimizers (convex solvers, other gradient solvers)
        # TODO: - RBF (if ez to implement) optimizers
        # TODO: - Memetic (?) algorithms
        # TODO: - Tabu search (?)
        # TODO: - Tree-structured Parzen Estimators (TPE)
        # TODO: Global + Local search islands: IN THE SAME "island" LEVEL (sg, bh & da do it already anyway, LS between)

        if island == 'rand':
            random_model_search(data_manipulation)
        elif island == 'bo':
            bayesian_optimization_model_search(data_manipulation)
        elif island == 'ga':
            genetic_algorithm_model_search(data_manipulation)
        elif island == 'pso':
            particle_swarm_optimization_model_search(data_manipulation)
        elif island == 'de':
            differential_evolution_model_search(data_manipulation)
        elif island == 'bh':
            basin_hopping_model_search(data_manipulation)
        elif island == 'da':
            dual_annealing_model_search(data_manipulation)
        elif island == 'sg':
            simplicial_homology_global_optimization_model_search(data_manipulation)
        elif island == 'ls':
            local_model_search(data_manipulation)

        print("--- Done({})!".format(island))
