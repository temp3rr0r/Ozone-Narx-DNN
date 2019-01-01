from __future__ import print_function
from ModelSearch import random_model_search, \
    differential_evolution_model_search, basin_hopping_model_search, \
    particle_swarm_optimization_model_search, bounds, get_random_model
import time
from mpi4py import MPI
import json

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # These lines should be called asap, after the os import
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only by default
# os.environ["PATH"] += os.pathsep + 'C:/Users/temp3rr0r/Anaconda3/Library/bin/graphviz'
# os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/pkgs/graphviz-2.38.0-h6538335_1009/Library/bin/graphviz'

with open('settings/data_manipulation.json') as f:
    data_manipulation = json.load(f)
modelLabel = data_manipulation["modelLabel"]

# islands = ['bh', 'pso', 'de', 'rand']
islands = ['rand', 'pso', 'de', 'rand', 'pso', 'de', 'pso'] * 4
# islands = ['de', 'de', 'de', 'rand', 'de', 'pso', 'de'] * 4
# islands = ['', 'pso', 'pso', 'rand', 'de', 'de'] * 4
# islands = ['rand', 'pso', 'pso', 'de', 'rand', 'de'] * 4
# islands = ['rand'] * 32
# islands = ['bh'] * 32
# islands = ['pso'] * 32
# islands = ['de'] * 32
# islands = ['pso', 'de'] * 32


def get_total_message_count(islands_in, size_in, data_manipulation_in):

    # TODO: should all have close to equal iterations. rand most importantly
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
    overallMinMse = 10e4  # TODO: formalize it
    evaluations = 0
    bestIsland = ""

    totalMessageCount = get_total_message_count(islands, size, data_manipulation)
    print("--- Expecting {} total messages...".format(totalMessageCount))

    for messageId in range(totalMessageCount):  # TODO 1000-1200 bh iters
        swapCounter += 1

        # Worker to master

        req = comm.irecv(tag=1)  # TODO: test sync
        data_worker_to_master = req.wait()
        # dataWorkerToMaster = comm.recv(tag=1)

        # print("--- Rank {}. Data Received: {} from {}!".format(rank, dataWorkerToMaster, worker))
        totalSecondsWork += data_worker_to_master["worked"]
        print("mean_mse: {} ({}: {})".format(data_worker_to_master["mean_mse"], data_worker_to_master["island"],
                                             data_worker_to_master["iteration"]))
        evaluations += 1
        if data_worker_to_master["mean_mse"] < overallMinMse:
            overallMinMse = data_worker_to_master["mean_mse"]
            bestIsland = data_worker_to_master["island"]
            if data_manipulation["sendBestAgentFromBuffer"]:
                agentBuffer = data_worker_to_master["agent"]  # TODO: Send the best agent received so far
            print("--- New overall min MSE: {} ({}: {}) (overall: {})".format(
                overallMinMse, data_worker_to_master["island"], data_worker_to_master["iteration"], evaluations))
        # if dataWorkerToMaster["mean_mse"] <= mean_mse_threshold:  # TODO: stop condition if mean_mse <= threshold
            # print("Abort: mean_mse = {} less than ".format(dataWorkerToMaster["mean_mse"]))
            # comm.Abort()  # TODO: block for func call sync

        # Master to worker

        dataMasterToWorker = {"swapAgent": False, "agent": None}
        if swapCounter > data_manipulation["swapEvery"]:
            print("========= Swapping...")
            swapCounter = 0
            dataMasterToWorker["swapAgent"] = True
            dataMasterToWorker["agent"] = agentBuffer
            agentBuffer = data_worker_to_master["agent"]
        comm.send(dataMasterToWorker, dest=data_worker_to_master["rank"], tag=2)  # TODO: test send async
        # req = comm.isend(dataMasterToWorker, dest=dataWorkerToMaster["rank"], tag=2)  # TODO: test send async
        # req.wait()

    endTime = time.time()
    print("--- Overall min MSE (total evals: {}): {} ({})".format(evaluations, overallMinMse, bestIsland))
    print("--- Total work: %d secs in %.2f secs, speedup: %.2f / %d" % (
        totalSecondsWork, round(endTime - startTime, 2),
        totalSecondsWork / round(endTime - startTime, 2), size - 1))
    # comm.Disconnect()

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

        if island == 'rand':
            random_model_search(data_manipulation)
        elif island == 'pso':
            particle_swarm_optimization_model_search(data_manipulation)
        elif island == 'de':
            differential_evolution_model_search(data_manipulation)
        elif island == 'bh':
            basin_hopping_model_search(data_manipulation)

        print("--- Done({})!".format(island))
