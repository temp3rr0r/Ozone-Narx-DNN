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
    black_box_function_ga.data["evaluation"] += 1

    k = 5
    evaluation = black_box_function_ga.data["evaluation"]
    best = tools.selBest(black_box_function_ga.pop, k=1)
    # TODO: island communication IO
    # print("best: {}".format(best))
    if evaluation % black_box_function_ga.k == 0:
        print("--- swap evaluation: {}".format(evaluation))

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


black_box_function_ga.data = {"evaluation": 0}

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
black_box_function_ga.pop = toolbox.population(n=4)
black_box_function_ga.k = 5
ngen, cxpb, mutpb = 4, 0.5, 0.2

rank = 3

for g in range(ngen):
    # TODO: individual attribs should be float in [0, 1]
    print("=== Generation: {}".format(g))

    # TODO: EA algorithms: https://deap.readthedocs.io/en/master/api/algo.html#complete-algorithms
    if rank == 1:
        # TODO: eaSimple
        # cxpb: Probability of mating 2 individuals
        # mutpb: Probability of mutating an individual
        # varAnd: Crossover AND mutation
        # evaluate(population)
        # for g in range(ngen):
        #     population = select(population, len(population))
        #     offspring = varAnd(population, toolbox, cxpb, mutpb)
        #     evaluate(offspring)
        #     population = offspring

        black_box_function_ga.pop = toolbox.select(black_box_function_ga.pop, k=len(black_box_function_ga.pop))
        black_box_function_ga.pop = algorithms.varAnd(black_box_function_ga.pop, toolbox, cxpb, mutpb)
        invalids = [ind for ind in black_box_function_ga.pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
    elif rank == 2:
        # TODO: eaMuPlusLambda
        # cxpb: Probability of mating 2 individuals
        # mutpb: Probability of mutating an individual
        # varOr: Crossover AND mutation: crossover, mutation or reproduction
        # mu: The number of individuals to select for the next generation.
        # lambda_: The number of children to produce at each generation.
        # evaluate(population)
        # for g in range(ngen):
        #     offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        #     evaluate(offspring)
        #     population = select(population + offspring, mu)
        mu = int(len(black_box_function_ga.pop) * 1.5)
        lambda_ = int(len(black_box_function_ga.pop) / 2)

        old_population = black_box_function_ga.pop

        black_box_function_ga.pop = algorithms.varOr(black_box_function_ga.pop, toolbox, lambda_, cxpb, mutpb)
        invalids = [ind for ind in black_box_function_ga.pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        black_box_function_ga.pop = toolbox.select(black_box_function_ga.pop + old_population, k=mu)
    elif rank == 3:
        # TODO: eaMuCommaLambda
        # cxpb: Probability of mating 2 individuals
        # mutpb: Probability of mutating an individual
        # varOr: Crossover AND mutation: crossover, mutation or reproduction
        # mu: The number of individuals to select for the next generation.
        # lambda_: The number of children to produce at each generation.
        # evaluate(population)
        # for g in range(ngen):
        #     offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        #     evaluate(offspring)
        #     population = select(offspring, mu)
        mu = int(len(black_box_function_ga.pop) * 1.5)
        lambda_ = int(len(black_box_function_ga.pop) / 2)

        old_population = black_box_function_ga.pop

        black_box_function_ga.pop = algorithms.varOr(black_box_function_ga.pop, toolbox, lambda_, cxpb, mutpb)
        invalids = [ind for ind in black_box_function_ga.pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        black_box_function_ga.pop = toolbox.select(black_box_function_ga.pop, k=mu)
    else:
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


print('black_box_function_ga.data["evaluation"]', black_box_function_ga.data["evaluation"])