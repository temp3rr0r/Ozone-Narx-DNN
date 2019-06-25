from bayes_opt import BayesianOptimization
import math

def black_box_function2(x1, x2, x3):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

def black_box_function(x1, x2, x3):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """

    return math.sin(x2 - 400) ** 2

# Bounded region of parameter space
bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)
          (350, 600), (0, 4),  # , 5)    # epoch_size, optimizer
          # (64, 512), (64, 512), (64, 512),
          # (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # dropout
          # (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # recurrent_dropout
          # (0.01, 1), (0.01, 1), (0.01, 1),  # gaussian noise std
          # (0, 1), (0, 1), (0, 1),  # gaussian_noise
          # (0, 1), (0, 1), (0, 1),  # batch normalization
          # (0, 5), (0, 5), (0, 5),  # base layer types (plain/bidirectional: LSTM, GRU, Simple RNN)
          # (0, 9), (0, 9), (0, 9)
          ]  # layer initializers, normal/uniform he/lecun,...

pbounds = {}
k = 0
for bound in bounds:
    k = k + 1
    pbounds["x{}".format(k)] = bound

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)
from bayes_opt import UtilityFunction
# from GlobalOptimizationAlgorithms.BayesianOptimization.bayes_opt import UtilityFunction

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)
target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)
optimizer.register(
    params=next_point_to_probe,
    target=target,
)

# TODO: try to alter "next"/empty queue/reset queue AND send/receive neighbour
for i in range(10):
    if i == 200:
        next_point = {"x1": 31, "x2": 600, "x3": 2}
        print("suggestion: {}".format(next_point))
    else:
        next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)

    optimizer.register(params=next_point, target=target)
    print(target, next_point)

print()
print(optimizer.max)

print("min (7, 350, 1)? {}".format(black_box_function(7, 350, 1)))
print("min (31, 600, 2)? {}".format(black_box_function(31, 600, 2)))

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
