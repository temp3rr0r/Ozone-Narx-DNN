from __future__ import print_function
import random
from scipy.optimize import differential_evolution
from BaseNarxModel import trainModel
from pyswarm.pso import pso
from scipy.optimize import basinhopping

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


def particleSwarmOptimizationModelSearch(x_data, y_data, dataManipulation=None):

    args = (x_data, y_data)
    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'pso'
    xopt1, fopt1 = pso(trainModel, lb, ub, args=args)
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


def randomModelSearch(x_data, y_data, dataManipulation=None, iterations=100):

    args = (x_data, y_data)
    trainModel.counter = 0  # Function call counter
    trainModel.dataManipulation = dataManipulation
    trainModel.label = 'rand'
    for i in range(iterations):
        trainModel(getRandomModel(), *args)


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
