from base.bounds import bounds
from base.NeuroevolutionModelTraining import ackley
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

def black_box_function(x1, x2, x3):
    return - ackley([x1, x2, x3])

pbounds = {}
k = 0
for bound in bounds:
    k = k + 1
    if k > 3:
        break
    pbounds["x{}".format(k)] = bound

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
# TODO: try to alter "next"/empty queue/reset queue AND send/receive neighbour
for i in range(10):
    if i == 2:
        next_point = {"x1": 21.9, "x2": 440.5, "x3": 3.8}
        print("== suggestion: {}".format(next_point))
    else:
        next_point = optimizer.suggest(utility)
    target = black_box_function(**next_point)

    optimizer.register(params=next_point, target=target)
    print(target, next_point)

print()
print(optimizer.max)

print('suggestion ("x1": 21.9, "x2": 440.5, "x3": 3.8)? {}'.format(black_box_function(29.9, 440.5, 3.8)))

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
