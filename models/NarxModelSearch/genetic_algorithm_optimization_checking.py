from base.bounds import bounds
from base.NeuroevolutionModelTraining import ackley
import numpy as np
import random
from deap import algorithms, base, creator, tools

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
    return (ackley(x),)


island = "ga"
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)  # TODO: param count
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", black_box_function_ga)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    pop = toolbox.population(n=10)

    # TODO: find all EA variations -> ~10 GA island versions
    # TODO: Gray/white box:
    ngen, cxpb, mutpb = 20, 0.5, 0.2
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ngen):
        # TODO: island communication IO
        # TODO: individual attribs should be float in [0, 1]

        pop = toolbox.select(pop, k=len(pop))

        pop = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # TODO: Island swap specific individuals
        print("len(pop)", len(pop))
        print("pop[0]", pop[0])
        # print("pop", pop)
        if g == 3:
            print("-- Changing in generation 12 an individual")
            pop[0][0] = 0.5
            pop[0][1] = 0.5
            print("-- pop[0]", pop[0])
            print("-- ", type(pop[0]))

        invalids = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

    print()
    best = tools.selBest(pop, k=1)
    best_value = black_box_function_ga(best[0])
    print("Best: {} = {}".format(best, best_value))

    x = best[0].copy()
    # print("individual ", individual)
    for i in range(len(x)):
        x[i] = x[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    x = np.array(x)
    print("un-normalized best: ", x)