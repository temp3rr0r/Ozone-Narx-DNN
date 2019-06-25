
# from bayes_opt import BayesianOptimization
from base.bounds import bounds
from GlobalOptimizationAlgorithms.BayesianOptimization.bayes_opt import BayesianOptimization
from base.NeuroevolutionModelTraining import ackley

def black_box_function2(x1, x2, x3):
    x = [x1, x2, x3]
    return - ackley(x)

def black_box_function(x1, x2, x3):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x1 ** 2 - (x2 - 1) ** 2 + 1


# if __name__== "__main__":

# Bounded region of parameter space
# pbounds = {'x': (2, 4), 'y': (-3, 3)}


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
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=1,
    n_iter=10,
)

print(optimizer.max)

print("min (31, 550, 50)? {}".format(black_box_function(31, 350, 0)))

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
