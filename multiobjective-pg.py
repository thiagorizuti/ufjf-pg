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


random.seed(datetime.now())

with open("ecoli3.csv") as dataset:
    reader = csv.reader(dataset)
    dataset = list(list(float(elem) for elem in row) for row in reader)
class0 = []
class1 = []
train = []
test = []
for line in dataset:
    if line[7] == 0:
        class0.append(line)
    else:
        class1.append(line)

random.shuffle(class0)
random.shuffle(class1)

l0 = len(class0)
l1 = len(class1)

for i in range(int(l0*0.1)):
    x = class0.pop(0)
    test.append(x)
for i in range(int(l1*0.1)):
    x = class1.pop(0)
    test.append(x)
random.shuffle(test)
train = class0 + class1
random.shuffle(train)



def safeDiv(a, b):
    if b == 0:
        return 0
    return a / b

pset = gp.PrimitiveSetTyped("main", [float,float,float,float,float,float,float],bool)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(operator.ne, [float, float], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(safeDiv, [float,float], float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval(individual):
    func = toolbox.compile(expr=individual)
    fit0 = 0
    fit1 = 0
    total0 = 0
    total1 = 0
    for line in train:
        result = func(line[0],line[1],line[2],line[3],
        line[4],line[5],line[6])
        if line[7] == 0:
            total0 += 1
            if result:
                fit0 += 1
        if line[7] == 1:
            total1 += 1
            if result:
                fit1 += 1
    return  float(fit0)/total0, float(fit1)/total1

toolbox.register("evaluate", eval)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def accuracy(individual, dataset):
    func = toolbox.compile(expr=individual)
    acc = 0    
    acc0 = 0
    acc1 = 0
    total = 0
    total0 = 0
    total1 = 0
    for line in dataset:
        result = func(line[0],line[1],line[2],line[3],
        line[4],line[5],line[6])
        if line[7] == 0:
            total +=1
            total0 += 1
            if result:
                acc += 1
                acc0 += 1
        if line[7] == 1:
            total +=1
            total1 += 1
            if result:
                acc += 1
                acc1 += 1
    return  float(acc)/total, float(acc0)/total0, float(acc1)/total1

def main():

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    cxpb = 0.5
    mutpb = 0.3
    ngen = 50
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb,ngen, halloffame=hof)

    for ind in hof:
        ind1 = ind
        print accuracy(ind,test)
       
    return 0


if __name__ == "__main__":
    main()
