import numpy as np
import os
import pandas as pd
import pickle
from gaft import GAEngine
from gaft.components import Population, BinaryIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

import dir_manage
from S_model import network

net_pickle = "four_large.pkl"
inputs_file = "four_large_input.csv"
target_file = "four_large_target.csv"
inputs = pd.read_csv(inputs_file)
target = pd.read_csv(target_file)
with open(os.path.join(dir_manage.cell_data_path, net_pickle), "rb") as f:
    net_information = pickle.load(f)
link_number = len(net_information["ordinary_cell"].keys())
range_of_indv = [(0, 5) for i in range(6*link_number)]

indv_template = BinaryIndividual(ranges=range_of_indv)
population = Population(indv_template=indv_template, size=100)
population.init()

selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population, selection, crossover, mutation)

@engine.fitness_register
@engine.minimize
def fitness(indv):

    global inputs, target, link_number

    net = network(net_pickle)
    x, = indv.solution
    net.split_rate = x[:3*link_number]
    net.staturated_flow = x[3*link_number:]

    return net.calculate_loss(inputs, target)
