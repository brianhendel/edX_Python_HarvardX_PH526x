import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

bernoulli.rvs(p=0.2)

N = 20
p = 0.2

#create empty graph
#add N nodes
#loop over all pairs of nodes
    #add an edge with prob p
def er_graph(N,p):
    """Generate an ER graph"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G

def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()), histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")
    
    
a1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
a2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

G1 = nx.to_networkx_graph(a1)
G2 = nx.to_networkx_graph(a2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    print("Average degree: %.2f" % np.mean(list(G.degree().values())))

basic_net_stats(G1)
basic_net_stats(G2)
plot_degree_distribution(G1)
plot_degree_distribution(G2)

gen = nx.connected_component_subgraphs(G1)
g = gen.__next__()










"""
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from(["u","v"])
G.add_edge(1,2)
G.add_edge("u","v")
G.add_edges_from([(1,3),(1,4),(1,5),(1,6), ("u", "w")])
G.remove_node(2)
G.remove_nodes_from([4,5])
G.remove_edge(1,3)
G.remove_edges_from([(1,2),("u","v")])
G.number_of_nodes()
G.number_of_edges()

G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
G.degree()
G.degree()[33]
"""




