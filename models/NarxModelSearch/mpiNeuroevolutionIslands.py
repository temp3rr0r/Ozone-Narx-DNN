from __future__ import print_function
from ModelSearch import random_model_search, \
    differential_evolution_model_search, basin_hopping_model_search, simple_homology_global_optimization_model_search, \
    particle_swarm_optimization_model_search, bounds, get_random_model, dual_annealing_model_search
import time
from mpi4py import MPI
import json


def get_total_message_count(islands_in, size_in, data_manipulation_in):

    total_message_count = 0
    iterations = data_manipulation_in["iterations"]
    pso_message_count = (iterations + 1) * data_manipulation_in["agents"]
    rand_message_count = iterations
    bh_message_count = iterations  # TODO: bh
    de_message_count = (  # (data_manipulation["iterations"] + 1)
            2 * data_manipulation_in["agents"] * len(bounds))

    for i in range(1, size_in):
        if islands_in[i] == "pso":
            total_message_count += pso_message_count
        elif islands_in[i] == "de":
            total_message_count += de_message_count
        elif islands_in[i] == "rand":
            total_message_count += rand_message_count
        elif islands_in[i] == "bh":
            total_message_count += bh_message_count

    return int(total_message_count)


with open('settings/data_manipulation.json') as f:
    data_manipulation = json.load(f)
modelLabel = data_manipulation["modelLabel"]

# islands = ['bh', 'pso', 'de', 'rand']
# islands = ['rand', 'pso', 'de', 'rand', 'pso', 'de', 'pso', ] * 4
# islands = ['de', 'de', 'de', 'rand', 'de', 'pso', 'de'] * 4
# islands = ['', 'pso', 'pso', 'rand', 'de', 'de'] * 4
islands = ['rand', 'pso', 'de', 'pso', 'de'] * 4
# islands = ['rand', 'pso', 'de', 'da', 'sg'] * 4
# islands = ['rand'] * 32
# islands = ['pso'] * 32
# islands = ['da'] * 32
# islands = ['sg', 'da', 'bh'] * 32
# islands = ['pso', 'de'] * 32

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

if rank == 0:  # Master Node

    swappedAgent = -1  # Rand init buffer agent
    startTime = time.time()
    totalSecondsWork = 0
    mean_mse_threshold = 3000.0

    for worker in range(1, size):  # Init workers
        initDataToWorkers = {"command": "init", "island": islands[worker % 3]}
        comm.send(initDataToWorkers, dest=worker, tag=0)
        print("--- Rank {}. Sending data: {} to {}...".format(rank, initDataToWorkers, worker))

    swapCounter = 0
    agentBuffer = get_random_model()
    agentsBuffer = [get_random_model()] * (size - 1)  # Store all island agents
    agentsMse = [mean_mse_threshold] * (size - 1)  # Store all island agents mse

    overallMinMse = 10e4
    evaluations = 0
    bestIsland = ""

    totalMessageCount = get_total_message_count(islands, size, data_manipulation)
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
        # if dataWorkerToMaster["mean_mse"] <= mean_mse_threshold:  # TODO: stop condition if mean_mse <= threshold
            # print("Abort: mean_mse = {} less than ".format(dataWorkerToMaster["mean_mse"]))
            # comm.Abort()  # TODO: block for func call sync

        # Master to worker
        agent_to_send = 0  # Default self for 1 island
        current_rank = data_worker_to_master["rank"]
        if size > 2:  # 2+ islands
            agent_to_send = current_rank - 2  # Get the best from the previous island
            if agent_to_send < 0:  # If first island, get last island from buffer
                agent_to_send = size - 2

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
        island = initData["island"]  # Get the island type from the master
        print("--- Rank {}. Data Received: {}!".format(rank, initData))
        print("--- Island: {}".format(island))

        data_manipulation["rank"] = rank
        data_manipulation["island"] = island
        data_manipulation["comm"] = comm

        # TODO: add/test (single or multi-agent) optimizers:
        # TODO: - Reinforcement Learning
        # TODO: - Bayesian Optimization (no derivatives needed)
        # TODO: - (traditional) Genetic Algorithms
        # TODO: - XGBoost
        # TODO: - Ant Colony Optimization (layer types only or bounded numerical if possible)
        # TODO: - Inductive Learning Programming (Known ts-DL layers/techniques (legends) =(progol)=>
        # TODO:     ML learned rules =(prolog)=> candidate layers
        # TODO: - Differentiable optimizers (convex solvers, other gradient solvers)
        # TODO: - RBF (if ez to implement) optimizers
        # TODO: - Memetic (?) algorithms
        # TODO: - Tabu search (?)
        # TODO: - Tree-structured Parzen Estimators (TPE)

        if island == 'rand':
            random_model_search(data_manipulation)
        elif island == 'pso':
            particle_swarm_optimization_model_search(data_manipulation)
        elif island == 'de':
            differential_evolution_model_search(data_manipulation)
        elif island == 'bh':
            basin_hopping_model_search(data_manipulation)
        elif island == 'da':
            dual_annealing_model_search(data_manipulation)
        elif island == 'sg':
            simple_homology_global_optimization_model_search(data_manipulation)

        print("--- Done({})!".format(island))
