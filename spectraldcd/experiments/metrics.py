import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
#import matlab.engine
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from community import community_louvain
from cdlib import FuzzyNodeClustering
import cdlib
from scipy.special import xlogy
from collections import Counter, defaultdict
#from geodesicdcd.experiments.TreeNode_class import TreeNode
from sklearn.metrics import adjusted_mutual_info_score
#import geodesicdcd.experiments.elementcentric_similarity_overlapping as elm
#import geodesicdcd.experiments.elementcentric_similarity_hierarchical as elh



def xlnx(x):
    """Returns x*log(x) for x > 0 or returns 0 otherwise."""
    if x <= 0.:
        return 0.
    return x*np.log(x)

def flattenator(newick):
    """Takes a hierarchical partition represented by nested lists and return a list of all its elements.
    
    Example
    >>> hp = [[3, 4, 5, 6], [[0], [1, 2]], [[7], [8, 9]]]
    >>> sorted(flattenator(hp))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    for e in newick:
        if isinstance(e,list):
            for ee in flattenator(e):
                yield ee
        else:
            yield e

def mean_arit(x,y):
    return .5*(x+y)

def HH(hp):
    """Returns the hierarchical entropy of a hierarchical partition.
    
    Note: this is not the most efficient implementation."""
    return HMI(hp,hp)[1]

def HMI(Ut,Us):
    """
    Computes the hierarchical mutual information between two hierarchical partitions.
    
    Returns
    n_ts,HMI(Ut,Us) : where n_ts is the number of common elements between the hierarchical partitions Ut and Us.
    
    NOTE: We label by u,v the children of t,s respectively.
    
    Examples
    >>>"""
    if not Ut or not Us:  # Check if either Ut or Us is empty
        return 0., 0. # this should be a trivial case, not sure if it's necessary
    
    if isinstance(Ut[0],list):
        if isinstance(Us[0],list):
            # Ut and Us are both internal nodes since they contain other lists.
            n_ts=0.
            H_uv=0.
            H_us=0.
            H_tv=0.
            mean_I_ts=0.0
            n_tv=defaultdict(float)            
            for Uu in Ut:
                n_us=0.
                for v,Uv in enumerate(Us):
                    n_uv,I_uv=HMI(Uu,Uv)
                    n_ts+=n_uv
                    n_tv[v]+=n_uv
                    n_us+=n_uv                    
                    H_uv+=xlnx(n_uv)
                    mean_I_ts+=n_uv*I_uv
                H_us+=xlnx(n_us)
            for _n_tv in n_tv.values():
                H_tv+=xlnx(_n_tv)
            if n_ts>0.:
                local_I_ts=np.log(n_ts)-(H_us+H_tv-H_uv)/n_ts
                mean_I_ts=mean_I_ts/n_ts
                I_ts=local_I_ts+mean_I_ts
                #print("... Ut =",Ut,"Us =",Us,"n_ts =",n_ts,"I_ts =",I_ts,"local_I_ts =",local_I_ts,"mean_I_ts =",mean_I_ts)
                return n_ts,I_ts
            else:
                #print("... Ut =",Ut,"Us =",Us,"n_ts =",0.0,"I_ts =",0.0)
                return 0.,0.
        else:
            # Ut is internal node and Us is leaf
            return len(set(flattenator(Ut))&set(Us)),0.
    else:
        if isinstance(Us,list):
            # Ut is leaf and Us internal node
            return len(set(flattenator(Us))&set(Ut)),0.          
        else:
            # Both Ut and Us are leaves
            return len(set(Ut)&set(Us)),0.

def NHMI(hp1,hp2,generalized_mean=mean_arit):
    """Returns the normalized hierarchical mutual information.
    
    By default, it uses the arithmetic mean for normalization. However, another generalized mean can be provided if desired."""
    gm = generalized_mean(HH(hp1),HH(hp2))
    if gm > 0.:
        return HMI(hp1,hp2)[1]/gm
    return 0.


    

def compute_and_plot_hnmis(predicted_labels, true_labels, graphs=None, figure=None, plot_label=None, plot=True):
        
    onmis = []
    # compute nmi for each pair of predicted and true labels
    for i in range(len(predicted_labels)):        
        hnmi = NHMI(predicted_labels[i], true_labels[i])
        onmis.append(hnmi)
        
    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)   
        plt.plot(range(1, len(predicted_labels) + 1), onmis, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Overlapping Normalized Mutual Information')
        plt.title('ONMI vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()    
        # show the plot if we created a new figure
        if figure is None:
            plt.show()

    return onmis


def compute_and_plot_hecs(predicted_labels, true_labels, graphs=None, figure=None, plot_label=None, plot=True):
    print("compute_and_plot_hecs is deprecated here")
    hecs = []
    for i in range(len(predicted_labels)):
        hec = elh.compare_tree_lists_ecs(predicted_labels[i], true_labels[i])
        hecs.append(hec)
        
    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)   
        plt.plot(range(1, len(predicted_labels) + 1), hecs, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Hierarchical Elementwise Similarity')
        plt.title('HEC vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()    
        # show the plot if we created a new figure
        if figure is None:
            plt.show()
            
    return hecs

def labels_to_partition(labels): # converts a list of community labels into a partition dictionary acceptable for modularity calculation
    return {node: community for node, community in enumerate(labels)}

def compute_and_plot_modularities(predicted_partitions, adjacency_matrices, true_partitions=None, figure=None, plot_label=None, plot=True):
    """
    Computes the modularity of given predicted (and optionally true) partitionings of a network over time and plots them.
    
    Parameters:
    predicted_partitions (list): List of predicted community structures over time, each a list of labels.
    adjacency_matrices (list): List of adjacency matrices of the network over time.
    true_partitions (list, optional): List of true community structures over time, each a list of labels.
    figure (matplotlib.figure.Figure, optional): An existing figure to plot on.
    plot_label (str, optional): Label for the plot series.
    
    Returns:
    tuple: A tuple containing lists of predicted modularities, and true modularities if provided.
    """
    predicted_modularities = []
    true_modularities = [] if true_partitions is not None else None
    
    # comput modularity for each time point
    for i, predicted_labels in enumerate(predicted_partitions):
        G = nx.from_numpy_array(adjacency_matrices[i])
        
        predicted_partition = labels_to_partition(predicted_labels)
        predicted_modularity = community_louvain.modularity(predicted_partition, G)
        predicted_modularities.append(predicted_modularity)
        
        if true_partitions is not None:
            true_labels = true_partitions[i]
            true_partition = labels_to_partition(true_labels)
            true_modularity = community_louvain.modularity(true_partition, G)
            true_modularities.append(true_modularity)
        
    # plotting
    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)  # use existing figure

        time_points = range(1, len(predicted_partitions) + 1)
        plt.plot(time_points, predicted_modularities, marker='o', linestyle='-', label=f'Predicted {plot_label}')

        if true_partitions is not None:
            plt.plot(time_points, true_modularities, marker='x', linestyle='-', label=f'True {plot_label}')

        plt.xlabel('Time Point')
        plt.ylabel('Modularity')
        plt.title('Modularity over Time')
        plt.grid(True)
        plt.legend()

        # sbow the plot if a new figure was created
        if figure is None:
            plt.show()
    
    return (predicted_modularities, true_modularities) if true_partitions is not None else predicted_modularities


def createFuzzyClusteringObject(labels, graph):
    """Given a 'labels' data structure of shape (num_nodes, num_communities) which assigns a community membership
    probability vector to each node, produce a cdlib FuzzyNodeClustering object."""
    
    num_nodes, num_communities = np.shape(labels)
    
    # Create the node_allocation dictionary
    node_allocation = {node_idx: {comm_idx: labels[node_idx, comm_idx] for comm_idx in range(num_communities)}
                       for node_idx in range(num_nodes)}
    
    # Create the communities list
    communities = [[] for _ in range(num_communities)]
    for node_idx in range(num_nodes):
        for comm_idx in range(num_communities):
            if labels[node_idx, comm_idx] > 0:  # optional threshold to include a node in a community
                communities[comm_idx].append(node_idx)
    
    # Create a simple graph for these nodes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))  # Adding nodes, assuming no specific edges are required for this example
    
    # Instantiate the FuzzyNodeClustering object
    fuzzy_clusters = FuzzyNodeClustering(
        communities=communities,
        node_allocation=node_allocation,
        graph=graph,
        overlap=True
    )
    
    return fuzzy_clusters




def h(w, n):
    """Compute the entropy term."""
    return -xlogy(w, w/n)

def compute_confusion_matrix(X_i, Y_j):
    """Compute the confusion matrix for two clusters."""
    a = np.sum((X_i == 0) & (Y_j == 0))
    b = np.sum((X_i == 0) & (Y_j == 1))
    c = np.sum((X_i == 1) & (Y_j == 0))
    d = np.sum((X_i == 1) & (Y_j == 1))
    n = len(X_i)
    return a, b, c, d, n

def H_star(X_i, Y_j):
    """Compute H*(X_i|Y_j) as defined in equation (2) of the paper CITE."""
    a, b, c, d, n = compute_confusion_matrix(X_i, Y_j)
    
    H_X_given_Y = (h(a, n) + h(b, n) + h(c, n) + h(d, n) - 
                   h(b + d, n) - h(a + c, n))
    
    if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
        return H_X_given_Y
    else:
        return h(c + d, n) + h(a + b, n)

def H_cover(X):
    """Compute the entropy of a cover."""
    n = X.shape[0]
    return np.sum([h(np.sum(X[:, i]), n) + h(n - np.sum(X[:, i]), n) 
                   for i in range(X.shape[1])])

def I_covers(X, Y):
    """Compute the mutual information between two covers."""
    n, K_X = X.shape
    K_Y = Y.shape[1]
    
    H_X_given_Y = np.sum([np.min([H_star(X[:, i], Y[:, j]) 
                                  for j in range(K_Y)]) 
                          for i in range(K_X)])
    
    H_Y_given_X = np.sum([np.min([H_star(Y[:, j], X[:, i]) 
                                  for i in range(K_X)]) 
                          for j in range(K_Y)])
    
    H_X = H_cover(X)
    H_Y = H_cover(Y)
    
    I_XY = 0.5 * (H_X - H_X_given_Y + H_Y - H_Y_given_X)
    
    return I_XY, H_X, H_Y

def NMI_max(X, Y):
    """Compute the normalized mutual information using max normalization."""
    I_XY, H_X, H_Y = I_covers(X, Y)
    return I_XY / max(H_X, H_Y)

def compute_and_plot_overlapping_elementwise_similarities(predicted_labels, true_labels, graphs=None, threshold=0.2, figure=None, plot_label=None, plot=False):
    print("compute_and_plot_overlapping_elementwise_similarities is deprecated here")
    oews = []
    for i in range(len(predicted_labels)):
        thresholded_predicted_labels = predicted_labels[i] >= threshold
        thresholded_true_labels = true_labels[i] >= threshold
        
       
        
        
        overlappingeltwisesim = elm.compare_community_matrices(thresholded_predicted_labels, thresholded_true_labels)
        oews.append(overlappingeltwisesim)
    
    
    #print(oews)

    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)   
        plt.plot(range(1, len(predicted_labels) + 1), oews, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Overlapping Elementwise Similarity')
        plt.title('Elemntwise sima aasdadf vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()    
        # show the plot if we created a new figure
        if figure is None:
            plt.show()
            
    return oews

def compute_and_plot_overlapping_nmis(predicted_labels, true_labels, graphs=None, threshold=0.2, figure=None, plot_label=None, plot=True):
    
    # the threshold is important; this metric operates only on binary memberships and the threshold turns probabilistic memberships into binary ones!
    
    onmis = []
    

    # compute nmi for each pair of predicted and true labels
    for i in range(len(predicted_labels)):
        #pred_label_obj = createFuzzyClusteringObject(predicted_labels[i], graphs[i])
        #true_label_obj = createFuzzyClusteringObject(true_labels[i],  graphs[i])
        
        #onmi = cdlib.evaluation.overlapping_normalized_mutual_information_MGH(pred_label_obj, true_label_obj)
        print(np.shape(predicted_labels[i]))
        print(np.shape(true_labels[i]))
        onmi = NMI_max(predicted_labels[i] >= threshold, true_labels[i] >= threshold)
        
        onmis.append(onmi)
        print(onmi)
        
        

    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)   
        plt.plot(range(1, len(predicted_labels) + 1), onmis, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Overlapping Normalized Mutual Information')
        plt.title('ONMI vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()    
        # show the plot if we created a new figure
        if figure is None:
            plt.show()

    return onmis

def compute_and_plot_nmis(predicted_labels, true_labels, figure=None, plot_label=None, plot=True):
    """
    co pares predicted labels to true labels using NMI and plots the results.
    
    Parameters:
    predicted_labels (list): List of predicted label vectors from spectral clustering or other methods.
    true_labels (list): List of true label vectors.
    figure (matplotlib.figure.Figure, optional): An existing figure to plot on.
    
    Returns:
    nmis (list): A list of Normalized Mutual Information values for each time point.
    """

    
    nmis = []
    
    # compute nmi for each pair of predicted and true labels
    for i, labels in enumerate(predicted_labels):
        nmi = normalized_mutual_info_score(true_labels[i], labels)
        nmis.append(nmi)
        
    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)  

        plt.plot(range(1, len(predicted_labels) + 1), nmis, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Normalized Mutual Information')
        plt.title('NMI vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()

        # show the plot if we created a new figure
        if figure is None:
            plt.show()
    
    return nmis


def compute_and_plot_amis(predicted_labels, true_labels, figure=None, plot_label=None, plot=True):
    """
    Compares predicted labels to true labels using AMI and plots the results.
    
    Parameters:
    predicted_labels (list): List of predicted label vectors from spectral clustering or other methods.
    true_labels (list): List of true label vectors.
    figure (matplotlib.figure.Figure, optional): An existing figure to plot on.
    plot_label (str, optional): Label for the plot legend.
    plot (bool): Whether to plot the results or not.
    
    Returns:
    amis (list): A list of Adjusted Mutual Information values for each time point.
    """
    
    #print("compute_and_plot_amis")
    
    amis = []
    
    # compute ami for each pair of predicted and true labels
    for i, labels in enumerate(predicted_labels):
        ami = adjusted_mutual_info_score(true_labels[i], labels, average_method='max')
        amis.append(ami)
        
    if plot:
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)  
        plt.plot(range(1, len(predicted_labels) + 1), amis, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Adjusted Mutual Information')
        plt.title('AMI vs Time')
        plt.grid(True)
        if plot_label is not None:
            plt.legend()
        # show the plot if we created a new figure
        if figure is None:
            plt.show()
    
    return amis


def compute_and_plot_aris(predicted_labels, true_labels, use_matlab_version=False, figure=None, plot_label=None, plot=True):
    """
    Compares predicted labels to true labels using ARI and plots the results.
    
    Parameters:
    predicted_labels (list): List of predicted label vectors from spectral clustering.
    true_labels (list): List of true label vectors.
    figure (matplotlib.figure.Figure, optional): An existing figure to plot on.
    
    Returns:
    aris (list): A list of Adjusted Rand Index values for each time point.
    """

    if use_matlab_version:
        raise ValueError("'MATLAB versions' of functions are not supported anymore.")
    aris = []

    if not use_matlab_version:
        for i, labels in enumerate(predicted_labels):
            # Compute ARI
            ari = adjusted_rand_score(true_labels[i], labels)
            aris.append(ari)

    #elif use_matlab_version:
    #    # Start the MATLAB engine
    #    eng = matlab.engine.start_matlab()
    #    print("Matlab engine started.")
    #    # Compute ARIs using MATLAB
    #    for i, labels in enumerate(predicted_labels):
    #        aris.append(eng.rand_index(true_labels[i], labels, 'adjusted'))
    #    print("ARIs computed.")
    #    # Quit the MATLAB engine
    #    eng.quit()
    #    print("Matlab engine stopped.")
        
    if plot:
        # Plotting
        if figure is None:
            plt.figure(figsize=(10, 5))
        else:
            plt.figure(figure.number)  # Use the existing figure

        plt.plot(range(1, len(predicted_labels) + 1), aris, marker='o', linestyle='-', label=plot_label)
        plt.xlabel('Time Point')
        plt.ylabel('Adjusted Rand Index')
        plt.title('ARI vs Time')
        plt.grid(True)

        # Show the plot if we created a new figure
        if figure is None:
            plt.show()
    
    return aris


def plot_community_sizes_over_time(labels_all, num_communities):
    # the kth entry of labels_all is a list of labels for each time point in the kth simulation
    # Initialize a list to store community sizes for each community over time for all simulations
    all_communities_sizes = np.zeros((num_communities, len(labels_all[0])))  # assuming all labels have the same length
    
    for k, labels in enumerate(labels_all):  # iterate over each simulation
        community_sizes = []
        for t in range(len(labels)):  # iterate over each time point
            # cou t the # of members in each community at time t
            community_sizes_at_t = [np.sum(np.array(labels[t]) == i) for i in range(1, num_communities+1)]
            community_sizes.append(community_sizes_at_t)
        
        community_sizes = np.array(community_sizes)  
        if k == 0: 
            all_communities_sizes = np.zeros_like(community_sizes)

        # Add this simulation's community sizes to the total (for averaging later, if needed)
        all_communities_sizes += community_sizes

        # plot each community's size over time (within this simulation)
        for i in range(num_communities):
            plt.plot(community_sizes[:, i], label=f'Simulation {k+1}, Community {i+1}', linestyle='--', marker='o')

    plt.xlabel('Time')
    plt.ylabel('Community Size')
    plt.title('Community Sizes Over Time for Each Simulation')
    plt.ylim(0, 100)  
    plt.legend()
    plt.show()

def plot_numerical_ranks_over_time(adjacency_matrices_all):
    def numerical_rank(matrix, percentage=0.5):
        u, s, vh = np.linalg.svd(matrix)
        norm_s = np.sum(s)  
        cumulative_s = np.cumsum(s) / norm_s
        rank = np.argmax(cumulative_s >= percentage) + 1  # Index of the first singular value making cumulative >= percentage
        # what fraction of the singular values are nonzero?
        num_nonzero_svs = np.sum(s > 1e-10) / len(s)
        #print(f"Fraction of nonzero singular values: {num_nonzero_svs}")

        return rank

    # # of simulations and time points (assuming all simulations have the same length)
    num_simulations = len(adjacency_matrices_all)
    num_time_points = len(adjacency_matrices_all[0])

    # Plotting setup
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)  

    for k in range(num_simulations): 
        ranks_adjacency = [numerical_rank(A) for A in adjacency_matrices_all[k]]
        plt.plot(ranks_adjacency, label=f'Simulation {k+1}', linestyle='--', marker='o')
        #dge.plot_singular_values(adjacency_matrices_all[k]) # screes

    plt.title('Numerical Rank Over Time for Adjacency Matrices')
    plt.xlabel('Time')
    plt.ylabel('Numerical Rank')
    plt.legend()

    plt.subplot(1, 2, 2) 
    for k in range(num_simulations):  
        identity_minus_laplacians = [np.eye(A.shape[0]) - nx.normalized_laplacian_matrix(nx.from_numpy_array(A)).toarray() for A in adjacency_matrices_all[k]]
        ranks_laplacian = [numerical_rank(L) for L in identity_minus_laplacians]
        plt.plot(ranks_laplacian, label=f'Simulation {k+1}', linestyle='--', marker='o')

    plt.title('Numerical Rank Over Time for Identity Minus Laplacians')
    plt.xlabel('Time')
    plt.ylabel('Numerical Rank')
    plt.legend()

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    pass