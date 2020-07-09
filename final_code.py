import random
import math
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import hungarian_algorithm as h


# --------------------------------- #
#      Generation of the graph      #
# --------------------------------- #

# Function that generate User's info in dict
def generate_users(nb_of_users):
    data = open("list_of_names.txt")
    data2 = [x.strip() for x in data]
    sex = ["Male", "Female", "Genderqueer/Non-Binary"]
    color_sex = {"Male":"#21B0F4", "Female":"#E55739", "Genderqueer/Non-Binary":"#0ADD92"}

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
def prob_edge_activation(my_network, node1, node2, randomized = False):
    if randomized == True:
        return random.random()**5

    
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
        nx.draw(my_network, with_labels=False, node_color=color_map, pos=pos)
    else:
        nx.draw(my_network, with_labels=True, node_color=color_map,node_size = 1000, edge_color ="#6D6D6D")
    plt.show()

def MC_Sampling(my_network, nb_seeds, message_type, randomized = False): 
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
                cumulated_influence += compute_influence(copy_network, node, message_type, list_of_seeds, randomized = randomized)
            avg_influence = cumulated_influence/R
            computed_influence[node] = avg_influence
        # look for the node with the best influence
        seed = max(computed_influence, key=computed_influence.get)
        list_of_seeds.append(seed)
        copy_network.nodes[seed]['state'] = "seed"
        value_seed.append(computed_influence[seed])
        print("MC's best \t\t: ", seed, "\tExpected reward of :", computed_influence[seed])

    return list_of_seeds, round(sum(value_seed), 2)


# Function that compute the influance of the possible seed, given already choosen seeds
def compute_influence(my_network, node, message_type, list_of_seeds, first=True, randomized = False):
    #reinitialize the states of the node for the seed simulation to come
    if first:
        for nod in my_network.nodes:
            if my_network.nodes[nod]['state'] != "seed":
                my_network.nodes[nod]['state'] = "waiting"
    
    # Compute 1 step of the cascade for each seed   
    activated_neigh = []
    activated_neigh_by_seeds = []
    for seed in list_of_seeds :
        exploration_nodes = nx.neighbors(my_network, seed) 
        for neigh in exploration_nodes:
            neighbor = my_network.nodes[neigh]
            if neighbor['state'] != "seed" and neighbor['state'] != "activated":
                prob = prob_edge_activation(my_network, seed, neigh, randomized = randomized)
                if random.random() >= 1-prob:
                    my_network.nodes[neigh]['state'] = "activated"
                    activated_neigh_by_seeds.append(neigh)

    # actual seed simulation
    Z = 0
    exploration_nodes = nx.neighbors(my_network, node)

    for neigh in exploration_nodes:
        neighbor = my_network.nodes[neigh]
        if neighbor['state'] != "seed" and neighbor['state'] != "activated":
            prob = prob_edge_activation(my_network, node, neigh, randomized=randomized)
            if random.random() >= 1-prob:
                my_network.nodes[neigh]['state'] = "activated"
                activated_neigh.append(neigh)
                if my_network.nodes[neigh]["sex"] == message_type:
                    Z += 1
    if len(activated_neigh) == 0:
        return 0
    return Z + sum([compute_influence(my_network, neigh, message_type, activated_neigh_by_seeds,  first=False ,randomized=randomized) for neigh in activated_neigh])



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

# -------------------------------  #
#           Question 3             #        
#        Multi-armed-bandit        #
# -------------------------------- #

def compute_upper_bound(nb_sex_type, t, nb_pull):
    return math.sqrt((2*nb_sex_type**2*math.log(t))/(nb_pull**2))

def count_sex_type(my_network, sex):
    sex_type = 0
    for node in my_network.nodes:
        if my_network.nodes[node]["sex"] == sex:
            sex_type += 1
    return sex_type

def UCB1_multi_seed(my_network, T, budget, message, show_cumulated_regret = False, show_expected_reward=False, randomized = False):
    seeds = []
    plt.style.use("fivethirtyeight")
    plt.figure()

    for i in range(budget):
        seed = UCB1_seed(my_network, T, message,i+1, seeds, show_expected_reward = show_expected_reward, randomized = randomized)
        seeds.append(seed)
    if show_cumulated_regret == True:
        plt.title("Cumulative regret for each seed")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend(loc="best")
        plt.show()
    
    return seeds


def UCB1_seed(my_network, T, message,iteration, seeds = [], show_expected_reward = False, randomized = False):
    nb_sex_type = count_sex_type(my_network, message)
    copy_network = my_network.copy()
    state_dict = {name:'waiting' for name in copy_network.nodes}
    nx.set_node_attributes(copy_network, state_dict, 'state')

    nb_nodes = copy_network.number_of_nodes()
    t = 0

    expected_reward = np.zeros(nb_nodes)
    nb_pull = np.zeros(nb_nodes)
    upper_bound = np.zeros(nb_nodes)

    collected_reward = [0]

    # Dictionary that gives an integer id to each node
    id_nodes = {}
    k=0
    for node in copy_network.nodes:
        id_nodes[node] = k
        k += 1

    for seed in seeds:
        copy_network.nodes[seed]['state'] = "seed"

    # First pull for every arm
    for node in copy_network.nodes:
        copy_2 = copy_network.copy()
        if copy_2.nodes[node]["state"] == "seed":
            expected_reward[id_nodes[node]] = 0
            upper_bound[id_nodes[node]] = 0
            continue
        t += 1
        id = id_nodes[node]
        nb_pull[id] += 1
        collected_reward.append(collected_reward[-1] + compute_influence(copy_2, node, message, seeds, randomized = randomized))

        expected_reward[id] = compute_influence(copy_2, node, message, seeds)
        upper_bound[id] = expected_reward[id] + compute_upper_bound(nb_sex_type, t, nb_pull[id])

    # Pull the best upper bound until t=T
    while t <= T:
        for node in my_network.nodes:
            if copy_network.nodes[node]['state'] != "seed":
                copy_network.nodes[node]['state'] = "waiting"

        if copy_network.nodes[node]["state"] == "seed":
            expected_reward[id_nodes[node]] = 0
            upper_bound[id_nodes[node]] = 0
            continue
        t+=1
        pulled_arm = np.argmax(upper_bound)
        node = [name for name, i in id_nodes.items() if i == pulled_arm][0]
        nb_pull[pulled_arm] += 1
        expected_reward[pulled_arm] = (expected_reward[pulled_arm]*(nb_pull[pulled_arm]-1) + compute_influence(copy_network, node, message, seeds, randomized =randomized) )/nb_pull[pulled_arm]
        collected_reward.append(collected_reward[-1] + compute_influence(copy_network, node, message, seeds, randomized = randomized))
        upper_bound[pulled_arm] = expected_reward[pulled_arm] + compute_upper_bound(nb_sex_type, t, nb_pull[pulled_arm])

    seed = [name for name, i in id_nodes.items() if i == np.argmax(expected_reward)][0]
    print("UCB1's best \t\t: ", seed, "\tExpected reward of : ", round(max(expected_reward),3))

    # Plot of the cumulative regret
    best = [t*max(expected_reward)-collected_reward[t] for t in range(T)]
    time = np.linspace(0,T,T)

    # Plot of the rewards per arm
    if show_expected_reward == True:
        plt.figure()
        x = [0]
        y = []
        y2 = []
        for node in copy_network.nodes:
            x.append(id_nodes[node])
            y.append(expected_reward[id_nodes[node]])
            y2.append(upper_bound[id_nodes[node]])
        plt.xlabel("Nodes")
        plt.ylabel("Reward")
        plt.title("Expected Reward and Upper Bound for each node (22 Male in the network)")
        plt.scatter(x[1::],y,c="r", label="Expected Reward")
        plt.scatter(x[1::],y2,c="g", label="Upper Bound")
        plt.legend(loc = "best")
        plt.show()
    
    plt.plot(time,best, label ="Seed n°{}".format(iteration) )
    return seed




# -------------------------------  #
#           Question 4             #        
#         Matching Problem         #
# -------------------------------- #

#Function returning all the nodes participating to the matching problem
def side_nodes(my_network, budget, message, randomized = False):

    copy_with_states = my_network.copy() #copy of my_network with feature 'state' added
    copy_2 = my_network.copy() #copy of my_network used in MC_Sampling

    state_dict = {name:'waiting' for name in copy_with_states.nodes}
    nx.set_node_attributes(copy_with_states, state_dict, 'state')

    seeds = MC_Sampling(copy_2, budget, message, randomized = randomized)[0] #compute the best nodes to use as seeds according to the network and the budget
    influence_seed(copy_with_states, seeds) #simulate a cascade on copy_1 starting from the previously determined seeds
    sidenodes = []
    for nod in copy_with_states.nodes:
        if copy_with_states.nodes[nod]['state'] == "seed" or copy_with_states.nodes[nod]['state'] == "activated" and copy_with_states.nodes[nod]['sex'] == message :
                n1 = my_network.nodes[nod]
                sidenodes.append((nod,n1))
    return sidenodes

#Function returning the probability that 2 nodes will accept a matching
def similarity(nodeL, nodeR):
    age_prob = 1 - abs(nodeL[2]-nodeR[2])/100
    daily_use_prob = 1 - abs(nodeL[3] - nodeR[3])/10
    return age_prob * 0.5 + daily_use_prob * 0.5

#Function returning the adjency matrix corresponding to the nodes which are in the matching problem
def adjency_matrix(my_network, list_nodes, weights):

    copy_nx = my_network.copy() #copy of my network

    list_left_nodes = []
    list_right_nodes = []

    #filtering the list of nodes participating in the matching to their corresponding sides
    for node, values in list_nodes :
        if values['sex'] == 'Male' or values['sex'] == 'Female' :
            list_left_nodes.append([node, values['sex'] , values['age'],values['daily_use']])
        elif values['sex'] == 'Genderqueer/Non-Binary':
            list_right_nodes.append([node, values['sex'] , values['age'], values['daily_use']])
    
    for node in copy_nx.nodes:
        if len(list_right_nodes) == len(list_left_nodes):
            break
        if copy_nx.nodes[node]['sex'] == "NaN":
            list_right_nodes.append([node, copy_nx.nodes[node]['sex'] , copy_nx.nodes[node]['age'], copy_nx.nodes[node]['daily_use']])

    adjency_matrix = np.zeros((len(list_left_nodes), len(list_right_nodes)))

    #filling the adjency matrix
    for i, nodeL in enumerate(list_left_nodes) :
        for j, nodeR in enumerate(list_right_nodes) :
            cte = weights.at[nodeL[1], nodeR[1]] #computes the cte relative to each pair of special feature.
            matching_pb = similarity(nodeL, nodeR)
            adjency_matrix[i, j] = int(cte*matching_pb*100) #the value of the edge between the two nodes is added to the adjency matrix
    return adjency_matrix

# Compute the cascade of the alreatdy chosen seeds (Not used anymore)

def influence_seed(my_network, list_of_seeds):
    activated_neigh = []
    for seed in list_of_seeds:
        exploration_nodes = nx.neighbors(my_network, seed)
        for neigh in exploration_nodes:
            neighbor = my_network.nodes[neigh]
            if neighbor['state'] != "seed" and neighbor['state'] != "activated":
                prob = prob_edge_activation(my_network, seed, neigh, True)
                if random.random() >= 1-prob:
                    my_network.nodes[neigh]['state'] = "activated"
                    activated_neigh.append(neigh)
    if len(activated_neigh) == 0:
        return 0
    return influence_seed(my_network, activated_neigh)


# --------------------------------- #
#             Question 5            #
# --------------------------------- # 
# Function that optimizes the budget to give to every features 
def best_alloc_MI(my_network, cumu_budget, randomized = False):
    # compute all the results for each individual budgets
    possible_values = np.linspace(2, cumu_budget-4, cumu_budget-5)
    messages = ('Male', 'Female', 'Genderqueer/Non-Binary')
    IM_results = np.zeros((len(possible_values), len(messages))).tolist()
 
    for i, mess in enumerate(messages):
        for j, val in enumerate(possible_values):
            result = MC_Sampling(my_network, int(val), mess, randomized = randomized)
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
    # Values for the creation of the network
    NB_NODES = 100
    NB_EDGES = 300
    NB_SEEDS = 6

    # Creating the network
    my_network, color_map = graph_generation(NB_NODES, NB_EDGES)
    # draw_graph(my_network, color_map, bipartite=False)

    question_1 = True
    question_2 = True
    question_3 = True
    question_4 = True
    question_5 = True

    if question_1:
        my_network2, color_map2 = graph_generation(50, 100)
        print_info(my_network)
        draw_graph(my_network2, color_map2, bipartite=False)
        print("\n")



    if question_2:
        MC_Sampling(my_network, NB_SEEDS, "Male", randomized=True)
        approx_err_plot(NB_NODES, NB_SEEDS)
        print("\n")


    if question_3:
        UCB1_multi_seed(my_network, 5000,NB_SEEDS,"Male", show_cumulated_regret=True, randomized = True)
        print("\n")




    # Defining the weights of each couple of message type
    w = {
        'Genderqueer/Non-Binary' : [3, 4],
        'NaN' : [2, 5], 
    }
    weights = pd.DataFrame(w, columns = ['Genderqueer/Non-Binary', "NaN"], index = ['Male', "Female"])



    if question_4:
        nodes_matching_pb_A = side_nodes(my_network, NB_SEEDS, "Male", randomized = True)
        matrix = adjency_matrix(my_network, nodes_matching_pb_A, weights)
        print("Adgency matrix :\n", matrix)
        print("\n\n")

        # matching
        res = h.hungarian_algorithm(-matrix)
        print("\n Optimal Matching : \n", res[1], "\n Values : ", -np.sum(res[0]))




    if question_5:
        #Find the best budget allocation for each features
        budgets_split = best_alloc_MI(my_network, NB_SEEDS, randomized=True)
        print("best allocation of the budget : {}\n".format(budgets_split))

        # Nodes participating in the matching problem
        nodes_matching_pb_A = side_nodes(my_network, budgets_split[0], "Male")
        nodes_matching_pb_B = side_nodes(my_network, budgets_split[1], "Female")
        nodes_matching_pb_C = side_nodes(my_network, budgets_split[2], "Genderqueer/Non-Binary")
        nodes_matching_pb = []
        nodes_matching_pb.extend(nodes_matching_pb_A)
        nodes_matching_pb.extend(nodes_matching_pb_B)
        nodes_matching_pb.extend(nodes_matching_pb_C)

        print("nodes for the matching :\n", nodes_matching_pb)

        # create the adjency matrix
        matrix = adjency_matrix(my_network, nodes_matching_pb, weights)
        print("Adgency matrix :\n", matrix)
        print("\n\n")

        # matching
        res = h.hungarian_algorithm(-matrix)
        print("\n Optimal Matching : \n", res[1], "\n Values : ", -np.sum(res[0]))
