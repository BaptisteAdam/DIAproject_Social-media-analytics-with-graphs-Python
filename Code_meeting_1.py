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
    my_network = nx.MultiGraph()            ## Multigraph = Graph with multiple features per edges
    my_network.add_nodes_from(name_list)
    nx.set_node_attributes(my_network, age_dict, 'age')
    nx.set_node_attributes(my_network, sex_dict, 'sex')
    nx.set_node_attributes(my_network, daily_use_dict, 'daily_use')
    return my_network



def edge_generation(my_network, nb_of_edges):
    """
    left_part = []
    right_part = []

    for index, node in enumerate(my_network.nodes):
        if my_network.nodes[node]['sex'] == "Male" or my_network.nodes[node]['sex'] == "Female":
            left_part.append(node)
        else:
            right_part.append(node)
    """
    while (my_network.number_of_edges() < nb_of_edges):
        e1 = random.choice(list(my_network.nodes))
        e2 = random.choice(list(my_network.nodes))

        if my_network.has_edge(e1, e2) != True:
            n1 = my_network.nodes[e1]
            n2 = my_network.nodes[e2]

            delta_age = abs(n1["age"] - n2["age"])
            delta_daily_use = abs(n1["daily_use"] - n2["daily_use"])
            my_network.add_edge(e1, e2, delta_age = delta_age, delta_daily_use = delta_daily_use)
    return my_network

# function that return the probability of an edge activation
def prob_edge_activation(delta_age, delta_daily_use):
    age = 1 - delta_age / 100
    daily_use = 1 - delta_daily_use/10
    return age * 0.5 + daily_use * 0.5



def graph_generation(nb_nodes, nb_of_edges):
    my_network = node_generation(nb_nodes)
    my_network = edge_generation(my_network, nb_of_edges)
    return my_network


# function that position A,B on the left and the others on the right
def position_nodes(my_network):
    pos = {}
    for index, node in enumerate(my_network.nodes):
        if my_network.nodes[node]['sex'] == "Male" or my_network.nodes[node]['sex'] == "Female":
            pos.update({node :(1,index)})
        else:
            pos.update({node : (2, index)})
    return pos



if __name__ == "__main__":
    my_network = graph_generation(10,30)
    #pos = position_nodes(my_network)
    pos = position_nodes(my_network)
    print(my_network.nodes(data = True))

    nx.draw(my_network, with_labels=True)
    plt.show()


