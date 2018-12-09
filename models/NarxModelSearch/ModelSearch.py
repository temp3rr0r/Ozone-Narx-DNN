from __future__ import print_function
import random
# from scipy.optimize import differential_evolution
from EvolutionaryAlgorithms.DifferentialEvolution import differential_evolution
from BaseNarxModel import trainModel
import BaseNarxModelMpi as baseMpi
from pyswarm.pso import pso
from scipy.optimize import basinhopping
import numpy as np

# Model Search Space bounds
bounds = [(7, 365),  # batch_size (~ #days: week, month, year)
          (150, 500), (0, 3),  # , 5)    # epoch_size, optimizer
          # (1023, 1024), (1023, 1024), (1023, 1024),  # TODO: 1024, 1024, 1024  # units
          (32, 1024), (32, 256), (32, 512),
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


def differentialEvolutionModelSearch(x_data, y_data, dataManipulation=None):

    args = (x_data, y_data)
    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'de'
    xopt1, fopt1 = differential_evolution(trainModel, bounds, args=args)  # TODO: test DE params
    printOptimum(xopt1, fopt1)

def trainModel2(x, *args):
    trainModel.counter += 1
    modelLabel = trainModel.label
    dataManipulation = trainModel.dataManipulation
    x_data, y_data = args
    return 540.2 * x[0]

def trainModel3(x, *args):
# def trainModel3(x, *args):
    trainModel.counter += 1
    modelLabel = trainModel.label
    dataManipulation = trainModel.dataManipulation
    x_data, y_data = args

    # particleInject = particleEject + 0.2
    particleInject = {"swapAgent" : False}
    if np.random.randint(0, 10) > 2:
        particleInject["swapAgent"] = True
    particleInject["agent"] = np.zeros_like(x) + 0.1

    return 540.2 * x[0], particleInject

def particleSwarmOptimizationModelSearch(x_data, y_data, dataManipulation=None):

    args = (x_data, y_data)
    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'pso'
    # xopt1, fopt1 = pso(trainModel, lb, ub, args=args, particle_output=True)  # TODO: test return particle
    xopt1, fopt1 = pso(trainModel3, lb, ub, args=args)
    # TODO: test larger swarm, more iterations
    # pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
    #     swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,
    #     minfunc=1e-8, debug=False)
    printOptimum(xopt1, fopt1)


def basinHoppingpModelSearch(x_data, y_data, dataManipulation=None):

    args = (x_data, y_data)
    x0 = getRandomModel()
    bounds = [(low, high) for low, high in zip(lb, ub)]  # rewrite the bounds in the way required by L-BFGS-B

    # TODO: check minimisers: https://docs.scipy.org/doc/scipy/reference/optimize.html
    minimizer_kwargs = dict(method="TNC", bounds=bounds, args=args)  # TODO: test method = "SLSQP". TODO: test Jacobian methods from minimizers
    # minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, args=y_test)  # use method L-BFGS-B because the problem is smooth and bounded
    # minimizer_kwargs = dict(method="SLSQP", bounds=bounds, args=y_test)

    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'bh'
    res = basinhopping(trainModel, x0, minimizer_kwargs=minimizer_kwargs)
    print(res)


def basinHoppingpModelSearchMpi(x_data, y_data, dataManipulation=None):

    iterations = dataManipulation["iterations"]
    args = (x_data, y_data)
    x0 = getRandomModel()
    bounds = [(low, high) for low, high in zip(lb, ub)]  # rewrite the bounds in the way required by L-BFGS-B

    # TODO: check minimisers: https://docs.scipy.org/doc/scipy/reference/optimize.html
    minimizer_kwargs = dict(method="TNC", bounds=bounds, args=args)  # TODO: test method = "SLSQP". TODO: test Jacobian methods from minimizers
    # minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, args=y_test)  # use method L-BFGS-B because the problem is smooth and bounded
    # minimizer_kwargs = dict(method="SLSQP", bounds=bounds, args=y_test)

    baseMpi.trainModel.counter = 0  # Function call counter
    baseMpi.trainModel.label = 'bh'
    baseMpi.trainModel.folds = dataManipulation["folds"]
    baseMpi.trainModel.dataManipulation = dataManipulation

    # def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
    #                  minimizer_kwargs=None, take_step=None, accept_test=None,
    #                  callback=None, interval=50, disp=False, niter_success=None,
    #                  seed=None):

    res = basinhopping(baseMpi.trainModel3, x0, minimizer_kwargs=minimizer_kwargs, niter=iterations)
    print(res)

def randomModelSearch(x_data, y_data, dataManipulation=None, iterations=100):

    args = (x_data, y_data)
    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'rand'
    for i in range(iterations):
        trainModel(np.array(getRandomModel()), *args)

def differentialEvolutionModelSearchMpi(x_data, y_data, dataManipulation=None):

    iterations = dataManipulation["iterations"]
    agents = dataManipulation["agents"]
    args = (x_data, y_data)
    baseMpi.trainModel.counter = 0  # Function call counter
    baseMpi.trainModel.label = 'de'
    baseMpi.trainModel.folds = dataManipulation["folds"]
    baseMpi.trainModel.dataManipulation = dataManipulation
    # xopt1, fopt1 = differential_evolution(baseMpi.trainModel, bounds, args=args,  #  TODO: call fast dummy func
    polish = False
    xopt1 = differential_evolution(baseMpi.trainModel3, bounds, args=args,
                                          popsize=agents, maxiter=iterations, polish=polish)  # TODO: test DE params
    printOptimum(xopt1, xopt1)

def particleSwarmOptimizationModelSearchMpi(x_data, y_data, dataManipulation=None, iterations=100):

    iterations = dataManipulation["iterations"]
    agents = dataManipulation["agents"]
    args = (x_data, y_data)
    baseMpi.trainModel.counter = 0  # Function call counter
    baseMpi.trainModel.label = 'pso'
    baseMpi.trainModel.folds = dataManipulation["folds"]
    baseMpi.trainModel.dataManipulation = dataManipulation
    xopt1, fopt1 = pso(baseMpi.trainModel2, lb, ub, maxiter=iterations,  # TODO: call fast dummy func
                       swarmsize=agents, args=args)  # TODO: test other than default params
    print("==========Agents: {}".format(agents))
    printOptimum(xopt1, fopt1)

def randomModelSearchMpi(x_data, y_data, dataManipulation=None, iterations=100):

    iterations = dataManipulation["iterations"]
    args = (x_data, y_data)
    baseMpi.trainModel.counter = 0  # Function call counter
    baseMpi.trainModel.label = 'rand'
    baseMpi.trainModel.folds = dataManipulation["folds"]
    for i in range(iterations):
        dataManipulation["iteration"] = i
        baseMpi.trainModel.dataManipulation = dataManipulation
        # baseMpi.trainModel(np.array(getRandomModel()), *args)  # TODO: call fast dummy func
        baseMpi.trainModel2(np.array(getRandomModel()), *args)


def getRandomModel():
    return [random.randint(lb[0], ub[0]),  # batch_size
             random.randint(lb[1], ub[1]), random.randint(lb[2], ub[2]),  # epoch_size, optimizer
             random.randint(lb[3], ub[3]), random.randint(lb[4], ub[4]), random.randint(lb[5], ub[5]),  # units
             random.uniform(lb[6], ub[6]), random.uniform(lb[7], ub[7]), random.uniform(lb[8], ub[8]),  # dropout
             random.uniform(lb[9], ub[9]), random.uniform(lb[10], ub[10]), random.uniform(lb[11], ub[11]),  # recurrent_dropout
             random.uniform(lb[12], ub[12]), random.uniform(lb[13], ub[13]), random.uniform(lb[14], ub[14]),  # gaussian noise std
             random.randint(lb[15], ub[15]), random.randint(lb[16], ub[16]), random.randint(lb[17], ub[17]), # gaussian_noise
             random.randint(lb[18], ub[18]), random.randint(lb[19], ub[19]), random.randint(lb[20], ub[20])]


def printOptimum(xopt1, fopt1):
    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
