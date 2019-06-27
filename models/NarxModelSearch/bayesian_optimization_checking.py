from base.bounds import bounds
from base.NeuroevolutionModelTraining import ackley
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

def black_box_function(x):
    print("x: {}".format(x))
    return 33, - ackley(x)

def list_to_pbounds_dictionary(list_values, dictionary_indices):
    returning_dictionary = {}

    print("dictionary_indices: {}".format(dictionary_indices))

    list_index = 0
    for dictionary_index in dictionary_indices:
        returning_dictionary[dictionary_index] = list_values[list_index]
        list_index += 1
    return returning_dictionary


for rank in range(1, 14):
    print("== Rank:", rank)

    pbounds = {}
    pbound_idx = 0
    init_var_name = 'a'
    for bound in bounds:
        var_name = chr(ord(init_var_name) + pbound_idx)
        pbounds["{}".format(var_name)] = bound
        pbound_idx = pbound_idx + 1

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=rank,
    )

    if rank % 10 == 1:  # TODO: Acquisition Function "Upper Confidence Bound". Exploitation.
        utility = UtilityFunction(kind="ucb", kappa=0.1, xi=0.0)
    elif rank % 10 == 2:  # TODO: Acquisition Function "Upper Confidence Bound". Exploration.
        utility = UtilityFunction(kind="ucb", kappa=10, xi=0.0)
    elif rank % 10 == 3:  # TODO: Acquisition Function "Expected Improvement". Exploitation.
        utility = UtilityFunction(kind="ei", kappa=0.0, xi=1e-4)
    elif rank % 10 == 4:  # TODO: Acquisition Function "Expected Improvement". Exploration.
        utility = UtilityFunction(kind="ei", kappa=0.0,  xi=1e-1)
    elif rank % 10 == 5:  # TODO: Acquisition Function "Probability of Improvement". Exploitation.
        utility = UtilityFunction(kind="poi", kappa=0.0,  xi=1e-4)
    elif rank % 10 == 6:  # TODO: Acquisition Function "Probability of Improvement". Exploration.
        utility = UtilityFunction(kind="poi", kappa=0.0,  xi=1e-1)
    elif rank % 10 == 7:  # TODO: Acquisition Function "Upper Confidence Bound".
        utility = UtilityFunction(kind="ucb", kappa=5.0, xi=0.1)
    elif rank % 10 == 8:  # TODO: Acquisition Function "Upper Confidence Bound".
        utility = UtilityFunction(kind="ei", kappa=2.5, xi=1e-4)
    elif rank % 10 == 9:  # TODO: Acquisition Function "Upper Confidence Bound".
        utility = UtilityFunction(kind="poi", kappa=2.5, xi=1e-4)
    else:  # TODO: Acquisition Function "Upper Confidence Bound". Default.
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)


    # TODO: try to alter "next"/empty queue/reset queue AND send/receive neighbour
    for i in range(2):
        # if i == 2 or i == 3:
        #     next_list = [21.0, 440, 0.0, 477.94477243642075, 64.0, 64.0, 0.01, 0.04621180938412835, 0.048467420303749884,
        #      0.01, 0.04996100829587216, 0.25, 0.01, 1.0, 0.01, 1.0, 1.0, 0.45479139193597523, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0,
        #      9.0, 9.0, 0.0]
        #     next_point = list_to_pbounds_dictionary(next_list, pbounds.keys())
        #     print("== suggestion: {}".format(next_point))
        # else:
        #     next_point = optimizer.suggest(utility)
        #     print("next_point: {}".format(next_point))
        #     print("To list: {}".format(next_point.values()))
        next_point = optimizer.suggest(utility)
        print("next_point: {}".format(next_point))
        print("To list: {}".format(next_point.values()))
        nothing, target = black_box_function(list(next_point.values()))

        try:
            optimizer.register(params=next_point, target=-target)  # TODO: negative: default bo tries to maximize
        except KeyError as ke:
            print("=== KeyError Exception: {}. Continuing...".format(str(ke)))


        print(target, next_point)

    print()
    print(optimizer.max)

    # print('suggestion ("x1": 21.9, "x2": 440.5, "x3": 3.8)? {}'.format(black_box_function(29.9, 440.5, 3.8)))

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
