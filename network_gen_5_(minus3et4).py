import random
import math
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --------------------------------- #
#      Generation of the graph      #
# --------------------------------- #

# Function that generate User's info in dict
def generate_users(nb_of_users):
    data = open("list_of_names.txt")
    data2 = [x.strip() for x in data]
    sex = ["Male", "Female", "Genderqueer/Non-Binary"]
    color_sex = {"Male":"blue", "Female":"red", "Genderqueer/Non-Binary":"green"}

    age_dict = {}
    sex_dict = {}
    daily_use_dict = {}
    color_map = []

    name_list = [random.choice(data2) for i in range(nb_of_users)]
    for name in name_list:
        age_dict[name] = randint(10, 100)
        sex_dict[name] = random.choice(sex)
        color_map.append(color_sex[sex_dict[name]])
        daily_use_dict[name] = randint(0, 10)
    return name_list, age_dict, sex_dict, daily_use_dict, color_map

# Function that generates the nodes
def node_generation(nb_of_nodes):
    name_list, age_dict, sex_dict, daily_use_dict, color_map = generate_users(nb_of_nodes)
    my_network = nx.MultiGraph()            ## Multigraph = Graph with multiple features per edges
    my_network.add_nodes_from(name_list)
    nx.set_node_attributes(my_network, age_dict, 'age')
    nx.set_node_attributes(my_network, sex_dict, 'sex')
    nx.set_node_attributes(my_network, daily_use_dict, 'daily_use')
    return my_network, color_map

# Function that addd the nodes corresponding to the special feature D (here, they are "NaN" users)
def add_feature_D(my_network, color_map, nb):
    #generate NaN users
    data = open("list_of_names.txt")
    data2 = [x.strip() for x in data]

    age_dict = {}
    sex_dict = {}
    daily_use_dict = {}

    name_list = [random.choice(data2) for i in range(nb)]
    for name in name_list:
        age_dict[name] = randint(10, 100)
        sex_dict[name] = "NaN"
        color_map.append("Grey")
        daily_use_dict[name] = randint(0, 10)

    #add NaN users to the graph
    my_network.add_nodes_from(name_list)
    nx.set_node_attributes(my_network, age_dict, 'age')
    nx.set_node_attributes(my_network, sex_dict, 'sex')
    nx.set_node_attributes(my_network, daily_use_dict, 'daily_use')
    return my_network, color_map

# Function that generates edges between the existing nodes
def edge_generation(my_network, nb_of_edges):
    while my_network.number_of_edges() < nb_of_edges:
        e1 = random.choice(list(my_network.nodes))
        e2 = random.choice(list(my_network.nodes))

        if my_network.has_edge(e1, e2) != True and e1 != e2:
            n1 = my_network.nodes[e1]
            n2 = my_network.nodes[e2]

            delta_age = abs(n1["age"] - n2["age"])
            delta_daily_use = abs(n1["daily_use"] - n2["daily_use"])

            my_network.add_edge(e1, e2, delta_age=delta_age, delta_daily_use=delta_daily_use)
    return my_network

# Function that position A,B on the left and the others on the right
def position_nodes(my_network):
    pos = {}
    for index, node in enumerate(my_network.nodes):
        if my_network.nodes[node]['sex'] == "Male" or my_network.nodes[node]['sex'] == "Female":
            pos.update({node :(1, index)})
        else:
            pos.update({node : (2, index)})
    return pos

# Function that generates the nodes than the edges to make the whole graph
def graph_generation(nb_nodes, nb_of_edges):
    n_nodes = nb_nodes*3//4
    my_network, color_map = node_generation(n_nodes)
    my_network = edge_generation(my_network, nb_of_edges)
    my_network, color_map = add_feature_D(my_network, color_map, nb_nodes-n_nodes)
    return my_network, color_map


# --------------------------------- #
#        Tools for the graph        #
# --------------------------------- #

# Function that return the probability of an edge activation
def prob_edge_activation(my_network, node1, node2):
    edge = my_network.get_edge_data(node1, node2)[0]
    delta_age = edge['delta_age']
    delta_daily_use = edge['delta_daily_use']

    age = 1 - delta_age / 100
    daily_use = 1 - delta_daily_use/10
    return age * 0.5 + daily_use * 0.5

# Function that draws the graph, can choose if bipartite or not
def draw_graph(my_network, color_map, bipartite=False):
    if bipartite:
        pos = position_nodes(my_network)
        nx.draw(my_network, with_labels=True, node_color=color_map, pos=pos)
    else:
        nx.draw(my_network, with_labels=True, node_color=color_map)
    plt.show()

# --------------------------------- #
#             Question 2            #
# --------------------------------- #
# Monte-Carlo algorithm
def MC_Sampling(my_network, nb_seeds, message_type): 
    # Initialisation
    copy_network = my_network.copy()
    state_dict = {name:'waiting' for name in copy_network.nodes}
    nx.set_node_attributes(copy_network, state_dict, 'state')

    epsilon = 0.1
    delta = 0.1
    R = round(1/(epsilon**2)*math.log10(nb_seeds)*math.log10(1/delta)) #number of repetitions
    list_of_seeds = []
    value_seed = []

    # compute nb_seeds times
    for s in range(nb_seeds):
        computed_influence = {}
        # compute influence of each potential seed
        for node in copy_network.nodes:
            if copy_network.nodes[node]['state'] == "seed":
                continue           
            cumulated_influence = 0
            for i in range(R):
                cumulated_influence += compute_influence(copy_network, node, message_type, list_of_seeds)
            avg_influence = cumulated_influence/R
            computed_influence[node] = avg_influence
        # look for the node with the best influence
        seed = max(computed_influence, key=computed_influence.get)
        list_of_seeds.append(seed)
        copy_network.nodes[seed]['state'] = "seed"
        value_seed.append(computed_influence[seed])
    print("MC sampling done for {0} seeds and message type {1}".format(nb_seeds, message_type))
    return list_of_seeds, round(sum(value_seed), 2)

# Function that compute the influance of the possible seed, given already choosen seeds
def compute_influence(my_network, node, message_type, list_of_seeds, first=True):
    #reinitialize the states of the node for the seed simulation to come
    if first:
        for nod in my_network.nodes:
            if my_network.nodes[nod]['state'] != "seed":
                my_network.nodes[nod]['state'] = "waiting"

        #do the already choosen seeds simulation
        influence_seed(my_network, list_of_seeds)
     
    # actual seed simulation
    Z = 0
    activated_neigh = []
    exploration_nodes = nx.neighbors(my_network, node)

    for neigh in exploration_nodes:
        neighbor = my_network.nodes[neigh]
        if neighbor['state'] != "seed" and neighbor['state'] != "activated":
            prob = prob_edge_activation(my_network, node, neigh)
            if random.random() >= 1-prob:
                my_network.nodes[neigh]['state'] = "activated"
                activated_neigh.append(neigh)
                if my_network.nodes[neigh]["sex"] == message_type:
                    Z += 1
    if len(activated_neigh) == 0:
        return 0
    return Z + sum([compute_influence(my_network, neigh, message_type, list_of_seeds,  first=False) for neigh in activated_neigh])

# Function that compute the influence of the already choosen seeds
def influence_seed(my_network, list_of_seeds):
    activated_neigh = []
    for seed in list_of_seeds:
        exploration_nodes = nx.neighbors(my_network, seed)
        for neigh in exploration_nodes:
            neighbor = my_network.nodes[neigh]
            if neighbor['state'] != "seed" and neighbor['state'] != "activated":
                prob = prob_edge_activation(my_network, seed, neigh)
                if random.random() >= 1-prob:
                    my_network.nodes[neigh]['state'] = "activated"
                    activated_neigh.append(neigh)
    if len(activated_neigh) == 0:
        return 0
    return influence_seed(my_network, activated_neigh)

# --------------------------------- #
#             Plot graphs           #
# --------------------------------- #
# Function that plots the approximation error of the activation probabilites of every nodes          
def approx_err_plot(nb_nodes, nb_seeds):
    R = 50
    delta = np.array([0.1, 0.05, 0.01])
    
    fig, axs = plt.subplots(3)
    j = 0
    for dval in delta :
        x_axis = np.linspace(1,R, R)
        y_axis = np.sqrt((1/x_axis)*math.log10(nb_seeds)*math.log10(1/dval)) 
        axs[j].plot(x_axis, y_axis)
        axs[j].set_title('delta='+str(dval) )
        axs[j].set(xlabel='R, number of repetition' , ylabel='Approximation error')
        j += 1
    
    fig.suptitle("Approximation error of the activation probability")
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

# Function that plots the influence spread relative to the number of seed 
def influence_spread_plot(my_network, nb_seed_max):
    nb_nodes = my_network.number_of_nodes()
    y_axis_greedy = np.full(nb_seed_max-1, (1-1/np.exp(1))*nb_nodes*3//4)

    x_axis = np.linspace(2, nb_seed_max, nb_seed_max-1)
    y_axis = []
    for nb_seeds in range(2, nb_seed_max+1):
        result_MC = MC_Sampling(my_network, nb_seeds, "Male")
        print(result_MC)
        y_axis.append(result_MC[1] )

    plt.plot(x_axis, y_axis)
    plt.plot(x_axis, y_axis_greedy)
    plt.title("influence spread  with respect to the number of seeds")
    plt.xlabel("Number of seeds")
    plt.ylabel("Influance spread")
    plt.show()


# --------------------------------- #
#             Question 5            #
# --------------------------------- # 
# Function that optimizes the budget to give to every features 
def best_alloc_MI(my_network, cumu_budget):
    # compute all the results for each individual budgets
    possible_values = np.linspace(2, cumu_budget-4, cumu_budget-5)
    messages = ('Male', 'Female', 'Genderqueer/Non-Binary')
    IM_results = np.zeros((len(possible_values), len(messages))).tolist()
 
    for i, mess in enumerate(messages):
        for j, val in enumerate(possible_values):
            result = MC_Sampling(my_network, int(val), mess)
            IM_results[j][i] = result[1]
 
    budget_combi = budget_combination(cumu_budget)
    best_result = 0
    best_combi = ()
    
    # look for the best combination
    for A, B, C in budget_combi:
        indexA = list(possible_values).index(A)
        indexB = list(possible_values).index(B)
        indexC = list(possible_values).index(C)
        if IM_results[indexA][0] + IM_results[indexB][1] + IM_results[indexC][2] > best_result:
            best_result = IM_results[indexA][0] + IM_results[indexB][1] + IM_results[indexC][2]
            best_combi = (int(A), int(B), int(C))
 
    return best_combi

# Function that returns all the possible budget combinations
def budget_combination(budget):
    # minimum = 2 seeds for a feature
    # => maximum = budget - 2*minimum = budget - 4 seeds per feature   
    cumu_budget_list = np.linspace(2, budget-4, budget-5)
    list_alloc = []
    for i in cumu_budget_list:
        for j in cumu_budget_list:
            for y in cumu_budget_list:
                if i + j + y == budget:
                    list_alloc.append((i ,j ,y))
    return list_alloc

# --------------------------------- #
#         Debug (temporary)         #
# --------------------------------- #
# Function that returns length of the two longuest name of the network (+1, for the dash)
def longest_edge_name(my_network):
    network = list(my_network.nodes)
    maxi = len(network[0])
    max2 = len(network[1])

    for name in network[2:]:
        if len(name) > maxi:
            maxi = len(name)
        elif len(name) > max2:
            max2 = len(name)

    return maxi+max2+1

# Function that prints the information of the network
def print_info(my_network):
    print("\nnodes :")
    for node in my_network.nodes.data():
        print(node)

    print("\nedges :")
    l_e_n = longest_edge_name(my_network)
    for edge in my_network.edges:
        egde_name = str(edge[0])+"-"+str(edge[1])
        while len(egde_name) < l_e_n:
            egde_name += ' '
        print(egde_name+" :", my_network.get_edge_data(edge[0], edge[1])[0])

# --------------------------------- #
#               MAIN                #
# --------------------------------- #
if __name__ == "__main__":
    my_network, color_map = graph_generation(40, 60)

    # print_info(my_network)
    # print(MC_Sampling(my_network, 5, "Male"))

    # approx_err_plot(100, 3)
    # influence_spread_plot(my_network, 10) 

    # draw_graph(my_network, color_map, bipartite=False)

    print("best allocation of the budget : {}".format(best_alloc_MI(my_network, 8)))
