from base.bounds import bounds
from base.NeuroevolutionModelTraining import ackley
import numpy as np
import random
from deap import algorithms, base, creator, tools

def black_box_function(x):
    print("x: {}".format(x))
    # return 33, ackley(x)
    return 33, x[1] ** 2

def black_box_function_ga(individual):
    if island == "ga" or island == "sg":  # TODO: rescale-encapsulate to NAS Ozone DNN space
        x = individual.copy()
        # print("individual ", individual)
        for idx in range(len(x)):  # TODO: what if it goes in [-inf, 0) or (1, +inf]?
            x[idx] = x[idx] * (bounds[idx][1] - bounds[idx][0]) + bounds[idx][0]
        x = np.array(x)
        # print("un-normalized x ", x)
    # return (individual[1] ** 2,)
    # return (x[0] ** 2,)

    nothing, target = black_box_function(x)
    black_box_function_ga.data["iter"] += 1

    k = 5
    iteration = black_box_function_ga.data["iter"]
    best = tools.selBest(black_box_function_ga.pop, k=1)
    # TODO: island communication IO
    # print("best: {}".format(best))
    if iteration % k == 0:
        print("--- swap iteration: {}".format(iteration))

        # print("len(pop)", len(black_box_function_ga.pop))
        # print("pop[0]", black_box_function_ga.pop[0])
        # print("pop", pop)
        # print("-- Changing in generation modulo k = 5 an individual")
        # black_box_function_ga.pop[0][0] = 0.5
        # black_box_function_ga.pop[0][1] = 0.5

        # TODO: Island swap worst individual
        worst = tools.selWorst(black_box_function_ga.pop, k=1)
        worst_index = black_box_function_ga.pop.index(worst[0])
        # print("worst (index: {}): {}".format(worst_index, worst))
        print("worst (index: {}): {}".format(worst_index, black_box_function_ga.pop[worst_index]))
        black_box_function_ga.pop[worst_index][0] = 0.5
        black_box_function_ga.pop[worst_index][1] = 0.5
        black_box_function_ga.pop[worst_index][2] = 0.5
        print("previous worst (index: {}): {}".format(worst_index, black_box_function_ga.pop[worst_index]))

    return (target,)


black_box_function_ga.data = {"iter": 0}

island = "ga"
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(bounds))  # TODO: param count
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", black_box_function_ga)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    black_box_function_ga.pop = toolbox.population(n=20)

    # TODO: find all EA variations -> ~10 GA island versions
    # TODO: Gray/white box:
    ngen, cxpb, mutpb = 4, 0.5, 0.2
    fitnesses = toolbox.map(toolbox.evaluate, black_box_function_ga.pop)
    for ind, fit in zip(black_box_function_ga.pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ngen):
        # TODO: individual attribs should be float in [0, 1]
        print("=== Generation: {}".format(g))

        black_box_function_ga.pop = toolbox.select(black_box_function_ga.pop, k=len(black_box_function_ga.pop))
        black_box_function_ga.pop = algorithms.varAnd(black_box_function_ga.pop, toolbox, cxpb, mutpb)

        invalids = [ind for ind in black_box_function_ga.pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

    print()
    worst = tools.selWorst(black_box_function_ga.pop, k=1)
    worst_value = black_box_function_ga(worst[0])
    print("Worst: {} = {}".format(worst, worst_value))

    print()
    best = tools.selBest(black_box_function_ga.pop, k=1)
    best_value = black_box_function_ga(best[0])
    print("Best: {} = {}".format(best, best_value))

    x = best[0].copy()
    # print("individual ", individual)
    for i in range(len(x)):
        x[i] = x[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    x = np.array(x)
    print("un-normalized best: ", x)


    print('black_box_function_ga.data["iter"]', black_box_function_ga.data["iter"])