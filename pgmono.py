import operator
import math
import random
import csv
import itertools
import numpy

from datetime import datetime
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#with open("train.csv") as dataset:
#    reader = csv.reader(dataset)
#    train = list(list((True if elem == '1' else False) for elem in row) for row in reader)
random.seed(datetime.now())

i = 0
train = list()
while i < 250:
    line = list()
    line.append(bool(random.getrandbits(1)))
    line.append(bool(random.getrandbits(1)))
    line.append(bool(random.getrandbits(1)))
    train.append(line)
    i += 1
print train

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(bool, 2), bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval(individual):
    func = toolbox.compile(expr=individual)
    class1 = 0
    class2 = 0
    total = 0
    for line in train:
        result = func(line[0],line[1])
        if line[2] == result:
            class1 += 1
        total +=1
    return  class1,

toolbox.register("evaluate", eval)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(5)
    cxpb = 0.5
    mutpb = 0.3
    ngen = 50
    finalpop = algorithms.eaSimple(pop, toolbox, cxpb, mutpb,ngen, halloffame=hof)

    for ind in hof:
        print ind, ind.fitness.values

    return 0

if __name__ == "__main__":
    main()
