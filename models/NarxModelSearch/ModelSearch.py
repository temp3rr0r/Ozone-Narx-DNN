from __future__ import print_function
import random
from EvolutionaryAlgorithms.DifferentialEvolution import differential_evolution
import BaseNarxModelMpi as baseMpi
from EvolutionaryAlgorithms.pyswarm.pso import pso
from scipy.optimize import basinhopping
import numpy as np

# Model Search Space bounds
# TODO: Add weights initializer search: https://keras.io/initializers/
bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)  # TODO: reduced batch size to try avoiding OOM
          (350, 600), (0, 4),  # , 5)    # epoch_size, optimizer
          # (1023, 1024), (1023, 1024), (1023, 1024),  # TODO: 1024, 1024, 1024  # units
          (64, 512), (64, 512), (64, 512),
          # (32, 512), (32, 196), (32, 384),
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # dropout
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # recurrent_dropout
          (0.01, 1), (0.01, 1), (0.01, 1),  # gaussian noise std
          (0, 1), (0, 1), (0, 1),  # gaussian_noise
          (0, 1), (0, 1), (0, 1)]  # batch normalization

# Lower Bounds
lb = [bounds[0][0],  # batch_size
      bounds[1][0], bounds[2][0],  # epoch_size, optimizer
      bounds[3][0], bounds[4][0], bounds[5][0],  # units
      bounds[6][0], bounds[7][0], bounds[8][0],  # dropout
      bounds[9][0], bounds[10][0], bounds[11][0],  # recurrent_dropout
      bounds[12][0], bounds[13][0], bounds[14][0],  # gaussian noise std
      bounds[15][0], bounds[16][0], bounds[17][0],  # gaussian_noise
      bounds[18][0], bounds[19][0], bounds[20][0]]  # batch normalization

# Upper Bounds
ub = [bounds[0][1],  # batch_size
      bounds[1][1], bounds[2][1],  # epoch_size, optimizer
      bounds[3][1], bounds[4][1], bounds[5][1],  # units
      bounds[6][1], bounds[7][1], bounds[8][1],  # dropout
      bounds[9][1], bounds[10][1], bounds[11][1],  # recurrent_dropout
      bounds[12][1], bounds[13][1], bounds[14][1],  # gaussian noise std
      bounds[15][1], bounds[16][1], bounds[17][1],  # gaussian_noise
      bounds[18][1], bounds[19][1], bounds[20][1]]  # batch normalization


def basin_hopping_model_search_mpi(x_data, y_data, data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    args = (x_data, y_data)

    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'bh'
    baseMpi.train_model.folds = data_manipulation["folds"]
    baseMpi.train_model.data_manipulation = data_manipulation
    bounds = [(low, high) for low, high in zip(lb, ub)]  # rewrite the bounds in the way required by L-BFGS-B

    # TODO: normalize data
    data_manipulation["bounds"] = bounds
    bounds = np.array([(0, 1)] * len(lb))
    x0 = np.array([random.uniform(0, 1)] * len(lb))

    # TODO: check minimisers: https://docs.scipy.org/doc/scipy/reference/optimize.html
    minimizer_kwargs = dict(method="TNC", bounds=bounds, args=args)  # TODO: test method = "SLSQP". TODO: test Jacobian methods from minimizers
    # minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, args=y_test)  # use method L-BFGS-B because the problem is smooth and bounded
    # minimizer_kwargs = dict(method="SLSQP", bounds=bounds, args=y_test)

    #def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None,
    #                  callback=None, interval=50, disp=False, niter_success=None,seed=None):

    res = basinhopping(
        baseMpi.train_model,
        # baseMpi.trainModelTester,  # TODO: call fast dummy func
        x0, minimizer_kwargs=minimizer_kwargs, niter=iterations)
    print(res)


def differential_evolution_model_search_mpi(x_data, y_data, data_manipulation=None):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    args = (x_data, y_data)
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
        # baseMpi.trainModelTester,  # TODO: call fast dummy func
        baseMpi.train_model_requester_rabbit_mq,
        # baseMpi.train_model,
        bounds, args=args, popsize=agents, maxiter=iterations,
        polish=polish, strategy=strategy)  # TODO: test DE params
    print_optimum(xopt1, xopt1)


def particle_swarm_optimization_model_search_mpi(x_data, y_data, data_manipulation=None, iterations=100):

    iterations = data_manipulation["iterations"]
    agents = data_manipulation["agents"]
    args = (x_data, y_data)
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
        # baseMpi.trainModelTester,   # TODO: call fast dummy func
        # baseMpi.train_model,
        baseMpi.train_model_requester_rabbit_mq,
        lb, ub, maxiter=iterations, swarmsize=agents, omega=omega, phip=phip, debug=True,
        phig=phig, args=args, rank=data_manipulation["rank"], storeCheckpoints=data_manipulation["storeCheckpoints"])
    print_optimum(xopt1, fopt1)


def random_model_search_mpi(x_data, y_data, data_manipulation=None, iterations=100):

    iterations = data_manipulation["iterations"]
    args = (x_data, y_data)
    baseMpi.train_model.counter = 0  # Function call counter
    baseMpi.train_model.label = 'rand'
    baseMpi.train_model.folds = data_manipulation["folds"]
    for i in range(iterations):
        data_manipulation["iteration"] = i
        baseMpi.train_model.data_manipulation = data_manipulation
        # baseMpi.trainModelTester(np.array(getRandomModel()), *args)  # TODO: call fast dummy func
        # baseMpi.train_model(np.array(get_random_model()), *args)  # TODO: store rand agent to future island migration
        baseMpi.train_model_requester_rabbit_mq(np.array(get_random_model()), *args)  # TODO: rabbit Mq worker


def get_random_model():
    return [random.randint(lb[0], ub[0]),  # batch_size
             random.randint(lb[1], ub[1]), random.randint(lb[2], ub[2]),  # epoch_size, optimizer
             random.randint(lb[3], ub[3]), random.randint(lb[4], ub[4]), random.randint(lb[5], ub[5]),  # units
             random.uniform(lb[6], ub[6]), random.uniform(lb[7], ub[7]), random.uniform(lb[8], ub[8]),  # dropout
             random.uniform(lb[9], ub[9]), random.uniform(lb[10], ub[10]), random.uniform(lb[11], ub[11]),  # recurrent_dropout
             random.uniform(lb[12], ub[12]), random.uniform(lb[13], ub[13]), random.uniform(lb[14], ub[14]),  # gaussian noise std
             random.randint(lb[15], ub[15]), random.randint(lb[16], ub[16]), random.randint(lb[17], ub[17]), # gaussian_noise
             random.randint(lb[18], ub[18]), random.randint(lb[19], ub[19]), random.randint(lb[20], ub[20])]


def print_optimum(xopt1, fopt1):
    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
