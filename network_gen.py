import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from random import randint


def generate_users(nb_of_users):
    data = open("list_of_names.txt")
    data2 = [x.strip() for x in data]
    sex = ["Male", "Female", "Genderqueer/Non-Binary", "NaN"]
    age_dict = {}
    sex_dict = {}
    daily_use_dict = {}
    nb_of_connection_dict = {}
    name_list = [random.choice(data2) for i in range(nb_of_users)]
    for name in name_list:
        age_dict[name] = randint(10, 100)
        sex_dict[name] = random.choice(sex)
        daily_use_dict[name] = randint(0,10)

    return name_list, age_dict, sex_dict, daily_use_dict


def network_generation(nb_of_nodes, nb_of_edges):
    name_list, age_dict, sex_dict, daily_use_dict = generate_users(nb_of_nodes)

    my_network = nx.MultiGraph()
    my_network.add_nodes_from(name_list)
    nx.set_node_attributes(my_network, age_dict, 'age')
    nx.set_node_attributes(my_network, sex_dict, 'sex')
    nx.set_node_attributes(my_network, daily_use_dict, 'daily_use')
    pos = {}
    left_part = []
    right_part = []

    for index, node in enumerate(my_network.nodes):
        if my_network.nodes[node]['sex'] == "Male" or my_network.nodes[node]['sex'] == "Female":
            left_part.append(node)
            pos.update({node :(1,index)})
        else:
            right_part.append(node)
            pos.update({node : (2, index)})

    while(my_network.number_of_edges() < nb_of_edges):
        e1 = random.choice(left_part)
        e2 = random.choice(right_part)
        if my_network.has_edge(e1,e2) != True:
            my_network.add_edge(e1,e2)

    edge_attr_dict = {}

    for edge in my_network.edges:
        edge_attr_dict.update({edge : {}})

    nx.draw(my_network, with_labels=True, pos = pos)
    plt.show()

network_generation(20,20)