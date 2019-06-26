from __future__ import print_function
import os
import random
import pandas as pd
from GlobalOptimizationAlgorithms.DifferentialEvolution import differential_evolution
from GlobalOptimizationAlgorithms.SimplicialHomologyGlobalOptimization import shgo
from GlobalOptimizationAlgorithms.DualAnnealing import dual_annealing
from GlobalOptimizationAlgorithms.BasinHopping import basinhopping
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from base import NeuroevolutionModelTraining as baseMpi
from GlobalOptimizationAlgorithms.pyswarm.pso import pso
from base.ModelRequester import train_model_requester_rabbit_mq
import numpy as np
from base.bounds import bounds, ub, lb
from scipy.optimize import minimize


def local_exposer_train_model_requester_rabbit_mq(x):
    """
    Encapsulates train_model_requester_rabbit_mq, in order to expose the MSE only of the function return.
    :param x: Full model parameters
    :return: Float scale MSE.
    """
    return train_model_requester_rabbit_mq(x)[0]  # Should return MSE only


def local_model_search(data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'ls'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation

    # Read last best model parameters for local search
    if os.path.exists("foundModels/best_model_parameters.pkl"):
        best_model_parameters_df = pd.read_pickle("foundModels/best_model_parameters.pkl")
        data_manipulation["best_model_parameters"] = best_model_parameters_df["best_model_parameters"]
        x0 = np.array(data_manipulation["best_model_parameters"])[0]
        print("data_manipulation['best_model_parameters']: {}".format(data_manipulation["best_model_parameters"]))
        print("x0: {}".format(x0))
    else:
        x0 = np.array(get_random_model())

    bounds = [(low, high) for low, high in zip(lb, ub)]  # rewrite the bounds in the way required by L-BFGS-B

    # Bounded local optimizers
    local_search_method = "L-BFGS-B"
    options = dict(maxfun=iterations)
    if data_manipulation["rank"] % 10 == 1:
        local_search_method = "L-BFGS-B"
        options = dict(maxfun=iterations)
    elif data_manipulation["rank"] % 10 == 2:
        local_search_method = "SLSQP"
        options = dict(maxiter=iterations)
    elif data_manipulation["rank"] % 10 == 3:
        local_search_method = "TNC"
        options = dict(maxiter=iterations)
    elif data_manipulation["rank"] % 10 == 4:
        local_search_method = "trust-constr"
        options = dict(maxiter=iterations)

    print("\n=== {}\n".format(local_search_method))

    res = minimize(
        x0=x0,
        fun=local_exposer_train_model_requester_rabbit_mq,
        # fun=baseMpi.ackley,  # for testing
        method=local_search_method,
        bounds=bounds,
        options=options
    )

    print(res)


def basin_hopping_model_search(data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'bh'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation
    x0 = np.array(get_random_model())
    bounds = [(low, high) for low, high in zip(lb, ub)]  # rewrite the bounds in the way required by L-BFGS-B

    minimizer_kwargs = dict(
        # method="TNC",  # TODO: TNC and L-BFGS-B only for constraint bounded local optimization?
        method="L-BFGS-B", jac=True,  # TODO: normalize data
        bounds=bounds,  # TODO: check minimisers: https://docs.scipy.org/doc/scipy/reference/optimize.html
    )  # TODO: test method = "SLSQP". TODO: test Jacobian methods from minimizers

    res = basinhopping(
        # baseMpi.train_model,
        train_model_requester_rabbit_mq,
        # baseMpi.trainModelTester,  # TODO: call fast dummy func
        x0, minimizer_kwargs=minimizer_kwargs, niter=iterations, data_manipulation=data_manipulation)
    print(res)


def simplicial_homology_global_optimization_model_search(data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'sg'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation

    # scipy.optimize.shgo(func, bounds, args=(), constraints=None, n=100, iters=1, callback=None, minimizer_kwargs=None,
    #                     options=None, sampling_method='simplicial')[source]
    xopt1 = shgo(
        # baseMpi.trainModelTester,
        train_model_requester_rabbit_mq,
        # baseMpi.train_model,
        bounds,
        n=agents,
        iters=iterations,
        sampling_method='sobol',
        # sampling_method='simplicial',
        data_manipulation=data_manipulation)  # TODO: test other SHGO params

    print_optimum(xopt1, xopt1)


def dual_annealing_model_search(data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'da'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation
    x0 = np.array(get_random_model())

    # Default values
    initial_temp = 5230.0
    visit = 2.62
    accept = -5.0
    restart_temp_ratio = 2e-05
    if data_manipulation["rank"] % 10 == 1:
        initial_temp = 5230.0
        visit = 2.62
        accept = -5.0
        restart_temp_ratio = 2e-05
    elif data_manipulation["rank"] % 10 == 2:
        initial_temp *= 0.75
        visit *= 0.75
        accept *= 0.75
        restart_temp_ratio *= 0.75
    elif data_manipulation["rank"] % 10 == 3:
        initial_temp *= 0.25
        visit *= 0.25
        accept *= 0.25
        restart_temp_ratio *= 0.25
    elif data_manipulation["rank"] % 10 == 4:
        initial_temp *= 0.95
        visit *= 0.95
        accept *= 0.95
        restart_temp_ratio *= 0.75
    elif data_manipulation["rank"] % 10 == 5:
        initial_temp *= 0.05
        visit *= 0.05
        accept *= 0.05
        restart_temp_ratio *= 0.05
    elif data_manipulation["rank"] % 10 == 6:
        initial_temp *= 0.85
        visit *= 0.85
        accept *= 0.85
        restart_temp_ratio *= 0.85
    elif data_manipulation["rank"] % 10 == 7:
        initial_temp *= 0.15
        visit *= 0.15
        accept *= 0.15
        restart_temp_ratio *= 0.15
    elif data_manipulation["rank"] % 10 == 8:  # Max values
        initial_temp = 5.e4
        visit = 3
        accept = -5.0
        restart_temp_ratio = 0.99
    elif data_manipulation["rank"] % 10 == 9:  # Min values
        initial_temp = 0.02
        visit = 0.01
        accept = -1e4
        restart_temp_ratio = 0.01

    xopt1 = dual_annealing(
        # baseMpi.trainModelTester,
        train_model_requester_rabbit_mq,
        # baseMpi.train_model,
        bounds,
        initial_temp=initial_temp,
        visit=visit,
        accept=accept,
        restart_temp_ratio=restart_temp_ratio,
        maxiter=iterations,
        no_local_search=True,
        # polish=polish
        x0=x0,
        data_manipulation=data_manipulation)  # TODO: test other DA params

    print_optimum(xopt1, xopt1)


def differential_evolution_model_search(data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'de'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation

    polish = False
    strategy = "best1bin"  # Dispatch of mutation strategy (Binomial or Exponential)
    if data_manipulation["rank"] % 10 == 1:
        strategy = "best1bin"
    elif data_manipulation["rank"] % 10 == 2:
        strategy = "best1exp"
    elif data_manipulation["rank"] % 10 == 3:
        strategy = "rand2exp"
    elif data_manipulation["rank"] % 10 == 4:
        strategy = "best2bin"
    elif data_manipulation["rank"] % 10 == 5:
        strategy = "rand2bin"
    elif data_manipulation["rank"] % 10 == 6:
        strategy = "randtobest1exp"
    elif data_manipulation["rank"] % 10 == 7:
        strategy = "randtobest1bin"
    elif data_manipulation["rank"] % 10 == 8:
        strategy = "rand1bin"
    elif data_manipulation["rank"] % 10 == 9:
        strategy = "best2exp"
    print("--- Using strategy: {}".format(strategy))

    xopt1 = differential_evolution(
        # baseMpi.trainModelTester,
        train_model_requester_rabbit_mq,
        # baseMpi.train_model,
        bounds, popsize=agents, maxiter=iterations,
        polish=polish, strategy=strategy, data_manipulation=data_manipulation)

    print_optimum(xopt1, xopt1)


def particle_swarm_optimization_model_search(data_manipulation=None, iterations=100):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'pso'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation

    omega = 0.5  # Particle velocity
    phip = 0.5  # Search away from particle's best known position (scaling factor)
    phig = 0.5  # Search away swarm's best known position (scaling factor)
    if data_manipulation["rank"] % 10 == 1:
        omega = 0.5
        phip = 0.5
        phig = 0.5
    elif data_manipulation["rank"] % 10 == 2:
        omega = 0.75
        phip = 0.75
        phig = 0.75
    elif data_manipulation["rank"] % 10 == 3:
        omega = 0.25
        phip = 0.25
        phig = 0.25
    elif data_manipulation["rank"] % 10 == 4:
        omega = 0.95
        phip = 0.95
        phig = 0.95
    elif data_manipulation["rank"] % 10 == 5:
        omega = 0.05
        phip = 0.05
        phig = 0.05
    elif data_manipulation["rank"] % 10 == 6:
        omega = 0.85
        phip = 0.85
        phig = 0.85
    elif data_manipulation["rank"] % 10 == 7:
        omega = 0.15
        phip = 0.15
        phig = 0.15
    elif data_manipulation["rank"] % 10 == 8:
        omega = 0.99
        phip = 0.99
        phig = 0.99
    elif data_manipulation["rank"] % 10 == 9:
        omega = 0.01
        phip = 0.01
        phig = 0.01

    xopt1, fopt1 = pso(
        # baseMpi.trainModelTester,
        # baseMpi.train_model,
        train_model_requester_rabbit_mq,
        lb, ub, maxiter=iterations, swarmsize=agents, omega=omega, phip=phip, debug=True,
        phig=phig, rank=data_manipulation["rank"], data_manipulation=data_manipulation)
    print_optimum(xopt1, fopt1)


def list_to_bayesian_optimization_pbounds_dictionary(list_values, dictionary_keys):
    returning_dictionary = {}
    print("dictionary_indices: {}".format(dictionary_keys))
    list_index = 0
    for dictionary_index in dictionary_keys:
        returning_dictionary[dictionary_index] = list_values[list_index]
        list_index += 1
    return returning_dictionary


def bayesian_optimization_model_search(data_manipulation=None, iterations=100):

    # TODO: bayesian optimization init
    pbounds = {}
    pbound_idx = 0
    for bound in bounds:
        pbound_idx = pbound_idx + 1
        pbounds["x{}".format(pbound_idx)] = bound
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)  # TODO: change those?

    iterations = data_manipulation["iterations"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'bo'
    baseMpi.train_model.folds = data_manipulation["folds"]

    min_mean_mse = 3000.0
    max_mean_mse = -1
    best_rand_agent = None
    worst_rand_agent = None
    swap = False
    suggestion = False
    k = data_manipulation["swapEvery"]
    for i in range(iterations):
        data_manipulation["iteration"] = i
        baseMpi.train_model.data_manipulation = data_manipulation

        # x = np.array(get_random_model())
        # TODO: get next list of params
        if suggestion:  # TODO: received candidate model, suggested it in bayesian optimization
            # next_list = [21.0, 440, 0.0, 477.94477243642075, 64.0, 64.0, 0.01, 0.04621180938412835,
            #              0.048467420303749884,
            #              0.01, 0.04996100829587216, 0.25, 0.01, 1.0, 0.01, 1.0, 1.0, 0.45479139193597523, 0.0, 0.0, 1.0,
            #              5.0, 5.0, 5.0,
            #              9.0, 9.0, 0.0]
            # received_agent = data_master_to_worker["agent"]
            # worst_rand_agent = received_agent
            next_list = worst_rand_agent  # TODO: the received data_master_to_worker["agent"]

            next_point = list_to_bayesian_optimization_pbounds_dictionary(next_list, pbounds.keys())
            suggestion = False
        else:
            next_point = optimizer.suggest(utility)
        x = np.array(list(next_point.values()))

        mean_mse, data_worker_to_master = train_model_requester_rabbit_mq(x)  # TODO: test with ackley
        # TODO: register sample & result
        target = mean_mse
        optimizer.register(params=next_point, target=-target)  # TODO: negative: default bo tries to maximize

        if mean_mse < min_mean_mse:  # Update best found agent
            best_rand_agent = x
            min_mean_mse = mean_mse
            print("=== Bayesian Optimization island {}, new min_mean_mse: {}, {}".format(data_worker_to_master["rank"], min_mean_mse,
                                                                        best_rand_agent))
        if mean_mse > max_mean_mse:
            worst_rand_agent = x
            max_mean_mse = mean_mse
            print("=== Bayesian Optimization island {}, new max_mean_mse: {}, {}".format(data_worker_to_master["rank"], max_mean_mse,
                                                                        worst_rand_agent))
        # Always send the best agent back
        # Worker to master
        data_worker_to_master["mean_mse"] = min_mean_mse
        data_worker_to_master["agent"] = best_rand_agent
        comm = data_manipulation["comm"]
        req = comm.isend(data_worker_to_master, dest=0, tag=1)  # Send data async to master
        req.wait()
        # Master to worker
        data_master_to_worker = comm.recv(source=0, tag=2)  # Receive data sync (blocking) from master
        # Replace worst agent
        if i % k == 0 and i > 0:  # Send back found agent
            swap = True
        if swap and data_master_to_worker["iteration"] >= (int(i / k) * k):
            print("========= Swapping (ranks: from-{}-to-{})... (iteration: {}, every: {}, otherIteration: {})".format(
                data_master_to_worker["fromRank"], data_worker_to_master["rank"], i, k,
                data_master_to_worker["iteration"]))
            received_agent = data_master_to_worker["agent"]
            worst_rand_agent = received_agent
            swap = False
            suggestion = True

    print("=== Bayesian Optimization island {}, max Mse: {}, min Mse: {}, {}, {}"
          .format(data_worker_to_master["rank"], max_mean_mse, min_mean_mse, worst_rand_agent, best_rand_agent))

    print(optimizer.max)  # TODO: check


def black_box_function_ga(individual):

    # x = np.array(get_random_model()) # TODO:
    x = individual.copy()
    # print("individual ", individual)
    for idx in range(len(x)):  # TODO: what if it goes in [-inf, 0) or (1, +inf]?
        x[idx] = x[idx] * (bounds[idx][1] - bounds[idx][0]) + bounds[idx][0]
    x = np.array(x)
    # print("un-normalized x ", x)

    mean_mse, data_worker_to_master = train_model_requester_rabbit_mq(x)
    black_box_function_ga.data["evaluation"] += 1
    evaluation = black_box_function_ga.data["evaluation"]
    if evaluation % black_box_function_ga.k == 0:
        print("--- swap evaluation: {}".format(evaluation))

    return (mean_mse,)

def genetic_algorithm_model_search(data_manipulation=None, iterations=100):
    from deap import algorithms, base, creator, tools

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'ga'
    baseMpi.train_model.folds = data_manipulation["folds"]
    min_mean_mse = 3000.0
    max_mean_mse = -1
    best_rand_agent = None
    worst_rand_agent = None
    swap = False
    k = data_manipulation["swapEvery"]

    # TODO: init GA
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
                     n=len(bounds))  # TODO: param count
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", black_box_function_ga)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    black_box_function_ga.pop = toolbox.population(n=agents)
    black_box_function_ga.data = {"evaluation": 0}
    black_box_function_ga.k = 5
    ngen, cxpb, mutpb = iterations, 0.5, 0.2

    for i in range(iterations):
        data_manipulation["iteration"] = i
        baseMpi.train_model.data_manipulation = data_manipulation

        # TODO: GA
        print("=== Generation: {}".format(i))
        black_box_function_ga.pop = toolbox.select(black_box_function_ga.pop, k=len(black_box_function_ga.pop))
        black_box_function_ga.pop = algorithms.varAnd(black_box_function_ga.pop, toolbox, cxpb, mutpb)
        invalids = [ind for ind in black_box_function_ga.pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        # if mean_mse < min_mean_mse:  # Update best found agent
        #     best_rand_agent = x
        #     min_mean_mse = mean_mse
        #     print("=== Genetic Algorithm island {}, new min_mean_mse: {}, {}".format(data_worker_to_master["rank"], min_mean_mse,
        #                                                                 best_rand_agent))
        # if mean_mse > max_mean_mse:
        #     worst_rand_agent = x
        #     max_mean_mse = mean_mse
        #     print("=== Genetic Algorithm island {}, new max_mean_mse: {}, {}".format(data_worker_to_master["rank"], max_mean_mse,
        #                                                                 worst_rand_agent))
        # # Always send the best agent back
        # # Worker to master
        # data_worker_to_master["mean_mse"] = min_mean_mse
        # data_worker_to_master["agent"] = best_rand_agent
        # comm = data_manipulation["comm"]
        # req = comm.isend(data_worker_to_master, dest=0, tag=1)  # Send data async to master
        # req.wait()
        # # Master to worker
        # data_master_to_worker = comm.recv(source=0, tag=2)  # Receive data sync (blocking) from master
        # # Replace worst agent
        # if i % k == 0 and i > 0:  # Send back found agent
        #     swap = True
        # if swap and data_master_to_worker["iteration"] >= (int(i / k) * k):
        #     print("========= Swapping (ranks: from-{}-to-{})... (iteration: {}, every: {}, otherIteration: {})".format(
        #         data_master_to_worker["fromRank"], data_worker_to_master["rank"], i, k,
        #         data_master_to_worker["iteration"]))
        #     received_agent = data_master_to_worker["agent"]
        #     worst_rand_agent = received_agent
        #     swap = False

    # print("=== Genetic Algorithm island {}, max Mse: {}, min Mse: {}, {}, {}"
    #       .format(data_worker_to_master["rank"], max_mean_mse, min_mean_mse, worst_rand_agent, best_rand_agent))

    print()
    print(tools.selBest(black_box_function_ga.pop, k=1))

def random_model_search(data_manipulation=None, iterations=100):

    iterations = data_manipulation["iterations"]
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'rand'
    baseMpi.train_model.folds = data_manipulation["folds"]

    min_mean_mse = 3000.0
    max_mean_mse = -1
    best_rand_agent = None
    worst_rand_agent = None
    swap = False
    k = data_manipulation["swapEvery"]
    for i in range(iterations):
        data_manipulation["iteration"] = i
        baseMpi.train_model.data_manipulation = data_manipulation
        x = np.array(get_random_model())
        mean_mse, data_worker_to_master = train_model_requester_rabbit_mq(x)
        if mean_mse < min_mean_mse:  # Update best found agent
            best_rand_agent = x
            min_mean_mse = mean_mse
            print("=== Rand island {}, new min_mean_mse: {}, {}".format(data_worker_to_master["rank"], min_mean_mse,
                                                                        best_rand_agent))
        if mean_mse > max_mean_mse:
            worst_rand_agent = x
            max_mean_mse = mean_mse
            print("=== Rand island {}, new max_mean_mse: {}, {}".format(data_worker_to_master["rank"], max_mean_mse,
                                                                        worst_rand_agent))
        # Always send the best agent back
        # Worker to master
        data_worker_to_master["mean_mse"] = min_mean_mse
        data_worker_to_master["agent"] = best_rand_agent
        comm = data_manipulation["comm"]
        req = comm.isend(data_worker_to_master, dest=0, tag=1)  # Send data async to master
        req.wait()
        # Master to worker
        data_master_to_worker = comm.recv(source=0, tag=2)  # Receive data sync (blocking) from master
        # Replace worst agent
        if i % k == 0 and i > 0:  # Send back found agent
            swap = True
        if swap and data_master_to_worker["iteration"] >= (int(i / k) * k):
            print("========= Swapping (ranks: from-{}-to-{})... (iteration: {}, every: {}, otherIteration: {})".format(
                data_master_to_worker["fromRank"], data_worker_to_master["rank"], i, k,
                data_master_to_worker["iteration"]))
            received_agent = data_master_to_worker["agent"]
            worst_rand_agent = received_agent
            swap = False

    print("=== Rand island {}, max Mse: {}, min Mse: {}, {}, {}"
          .format(data_worker_to_master["rank"], max_mean_mse, min_mean_mse, worst_rand_agent, best_rand_agent))


def get_random_model():
    return [random.randint(lb[0], ub[0]),  # batch_size
             random.randint(lb[1], ub[1]), random.randint(lb[2], ub[2]),  # epoch_size, optimizer
             random.randint(lb[3], ub[3]), random.randint(lb[4], ub[4]), random.randint(lb[5], ub[5]),  # units
             random.uniform(lb[6], ub[6]), random.uniform(lb[7], ub[7]), random.uniform(lb[8], ub[8]),  # dropout
             random.uniform(lb[9], ub[9]), random.uniform(lb[10], ub[10]), random.uniform(lb[11], ub[11]),  # recurrent_dropout
             random.uniform(lb[12], ub[12]), random.uniform(lb[13], ub[13]), random.uniform(lb[14], ub[14]),  # gaussian noise std
             random.randint(lb[15], ub[15]), random.randint(lb[16], ub[16]), random.randint(lb[17], ub[17]),  # gaussian_noise
             random.randint(lb[18], ub[18]), random.randint(lb[19], ub[19]), random.randint(lb[20], ub[20]),  # batch normalization
             random.randint(lb[21], ub[21]), random.randint(lb[22], ub[22]), random.randint(lb[23], ub[23]),  # base layer types
             random.randint(lb[24], ub[24]), random.randint(lb[25], ub[25]), random.randint(lb[26], ub[26])]  # layer initializers, normal/uniform he/lecun


def print_optimum(xopt1, fopt1):
    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
