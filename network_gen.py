import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from random import randint

# Function that generate User's info in dict
def generate_users(nb_of_users):
    data = open("list_of_names.txt")
    data2 = [x.strip() for x in data]
    sex = ["Male", "Female", "Genderqueer/Non-Binary", "NaN"]
    age_dict = {}
    sex_dict = {}
    daily_use_dict = {}
    name_list = [random.choice(data2) for i in range(nb_of_users)]
    for name in name_list:
        age_dict[name] = randint(10, 100)
        sex_dict[name] = random.choice(sex)
        daily_use_dict[name] = randint(0,10)
    return name_list, age_dict, sex_dict, daily_use_dict


def node_generation(nb_of_nodes):
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
            left_part.append(node)                              # Position of node on the graph (bipartite)
            pos.update({node :(1,index)})
        else:
            right_part.append(node)                             # Position of node on the graph (bipartite)
            pos.update({node : (2, index)})

    return my_network


def edge_generation(my_network, nb_of_edges):
    left_part = []
    right_part = []
    for index, node in enumerate(my_network.nodes):
        if my_network.nodes[node]['sex'] == "Male" or my_network.nodes[node]['sex'] == "Female":
            left_part.append(node)
        else:
            right_part.append(node)

    while (my_network.number_of_edges() < nb_of_edges):
        e1 = random.choice(left_part)
        e2 = random.choice(right_part)
        if my_network.has_edge(e1, e2) != True:
            my_network.add_edge(e1, e2)

    edge_attr_dict = {}
    for edge in my_network.edges:
        edge_attr_dict.update({edge: {}})

    return my_network

if __name__ == "__main__":
    my_network = node_generation(20)
    my_network = edge_generation(my_network, nb_of_edges=10)

    name1 = list(my_network.nodes)[0]
    name2 = list(my_network.nodes)[1]
    print(name1, my_network.nodes[name1])
    print(name2, my_network.nodes[name2], '\n')

    my_network.add_edge(name1, name2, weight=4.7, lenght=3200, type='bute')

    print(my_network.get_edge_data(name1, name2))


    # nx.draw(my_network, with_labels=True)
    # plt.show()