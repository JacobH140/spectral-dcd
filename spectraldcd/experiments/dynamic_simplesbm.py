import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from numpy.random import SeedSequence
import os
import pickle
from .dyn_graph_benchmarks import adj_list_to_tn_obj 

def sparse_sbm_dynamic_model_2(N=150, k=3, pin=[0.3, 0.3, 0.2], pout=0.15, p_switch=0.15, T=10, Totalsims=2, base_seed=None, community_probs=None, verbose=False, save_path=None):
    
    print("WARNING: sparsify_sbm_dynamic_model_2 is deprecated, use sbm_dynamic_model_2 with try_sparse=True instead.")

    
    ss = SeedSequence(base_seed)
    child_seeds = ss.spawn(Totalsims)
    True_labels_all = np.zeros((Totalsims, T, int(N)), dtype=int)
    Adjacency_all = []
    for sims in range(Totalsims):
        simulation_adj = []
        rng = np.random.default_rng(child_seeds[sims])
        labels = rng.integers(1, k+1, size=N) if not community_probs else rng.choice(range(1, k+1), size=N, p=community_probs)
        True_labels_all[sims, 0, :] = labels
        
        # Keep track of nodes that have already switched
        switched_nodes = set()
        
        for t in range(T):
            G = sp.lil_matrix((N, N))
            if t > 0:
                for i in range(1, k+1):
                    indices = np.where(labels == i)[0]
                    # Only consider nodes that haven't switched yet
                    available_indices = [idx for idx in indices if idx not in switched_nodes]
                    if available_indices:
                        num_switch = int(np.ceil(len(available_indices) * p_switch))
                        if num_switch > 0:
                            changing_members = rng.choice(available_indices, size=num_switch, replace=False)
                            new_labels = rng.integers(1, k+1, size=num_switch)
                            for cm, nl in zip(changing_members, new_labels):
                                if nl != i:  # Only switch if the new label is different
                                    labels[cm] = nl
                                    switched_nodes.add(cm)
            True_labels_all[sims, t, :] = labels
            clusters = {i: np.where(labels == i)[0] for i in range(1, k+1)}
            for i in range(1, k+1):
                cluster_indices = clusters[i]
                if len(cluster_indices) > 0:
                    Gk = rng.binomial(1, pin[i-1], (len(cluster_indices), len(cluster_indices)))
                    G[np.ix_(cluster_indices, cluster_indices)] = sp.csr_matrix(Gk)
            
            # Add inter-community edges
            for i in range(1, k+1):
                for j in range(i+1, k+1):
                    ci, cj = clusters[i], clusters[j]
                    if len(ci) > 0 and len(cj) > 0:
                        Gij = rng.binomial(1, pout, (len(ci), len(cj)))
                        G[np.ix_(ci, cj)] = sp.csr_matrix(Gij)
                        G[np.ix_(cj, ci)] = sp.csr_matrix(Gij.T)
            
            G = G + G.T
            G.setdiag(0)
            G = G.tocsr()
            simulation_adj.append(G)
        Adjacency_all.append(simulation_adj)
        if verbose:
            print(f'Simulation {sims+1} completed at all time steps.')
    
    if save_path:
        if not save_path.endswith('.npz'):
            save_path += '.npz'
        base_path = os.path.splitext(save_path)[0]
        
        # Save True_labels_all and shape
        np.savez(save_path, True_labels_all=True_labels_all, shape=(N, N))
        
        # Save each sparse matrix separately
        for sim in range(Totalsims):
            for t in range(T):
                sp.save_npz(f"{base_path}_adj_sim{sim}_t{t}.npz", Adjacency_all[sim][t])
    
    return Adjacency_all, True_labels_all

def load_sparse_sbm_data(save_path):
    if not save_path.endswith('.npz'):
        save_path += '.npz'
    base_path = os.path.splitext(save_path)[0]
    
    with np.load(save_path, allow_pickle=True) as data:
        True_labels_all = data['True_labels_all']
        shape = tuple(data['shape'])
    
    Totalsims, T, _ = True_labels_all.shape
    Adjacency_all = []
    
    for sim in range(Totalsims):
        simulation_adj = []
        for t in range(T):
            adj = sp.load_npz(f"{base_path}_adj_sim{sim}_t{t}.npz")
            simulation_adj.append(adj)
        Adjacency_all.append(simulation_adj)
    
    return Adjacency_all, True_labels_all


def sparsify_all(adjacency_matrices_all):
    all_adj_matrices_sparse = []
    for adj_matrices_list in adjacency_matrices_all:
        adj_matrices_list_sparse = [csr_matrix(A) for A in adj_matrices_list]
        all_adj_matrices_sparse.append(adj_matrices_list_sparse)
    return all_adj_matrices_sparse


def sbm_dynamic_model_2(N=150, k=3, pin=[0.3, 0.3, 0.2], pout=0.15, p_switch=0.15, T=10, Totalsims=2, base_seed=None, community_probs=None, verbose=False, try_sparse=False, save_path=None):
    """
    Simulate dynamic stochastic block model(s) over T time steps.

    Parameters:
    - N (int): Number of nodes in the network.
    - k (int): Number of communities.
    - pin (list of float): Intra-community connection probabilities, one for each of the k communities (often chosen to be constant) 
    - pout (float): Inter-community connection probability.
    - p_switch (float): Probability of a node switching communities at each time step.
    - T (int): Number of time steps, per simulation
    - Totalsims (int): Number of total simulations to perform.
    - base_seed (int, optional): Seed for the random number generator to ensure reproducibility.
    - community_probs (list of float, optional): Initial probability distribution over communities for node assignment.
      Usually set to None, in which case communities are assigned uniformly at random.

    Returns:
    - Adjacency_all (numpy.ndarray):array of shape (Totalsims, T, N, N) containing adjacency matrices for each simulation and time step.
    - True_labels_all (numpy.ndarray): array of shape (Totalsims, T, N) containing the true labels of nodes for each simulation and time step.

    Example Usage:
    >>> A, L = sbm_dynamic_model_2(N=100, k=3, pin=[0.25, 0.35, 0.4], pout=0.05, p_switch=0.1, T=5, Totalsims=1, base_seed=42)
    >>> print(A.shape, L.shape)  # Expected output: (1, 5, 100, 100) (1, 5, 100)

    """
    
    
    ss = SeedSequence(base_seed)
    child_seeds = ss.spawn(Totalsims)  # separate seeds for each simulation

    if not try_sparse:
        Adjacency_all = np.zeros((Totalsims, T, N, N))
    else:
        Adjacency_all = [[np.zeros((N, N)) for _ in range(T)] for _ in range(Totalsims)]
    True_labels_all = np.zeros((Totalsims, T, N))  # store the true labels for all simulations and all time steps

    for sims in range(Totalsims):
        rng = np.random.default_rng(child_seeds[sims])

        # data generation for t = 0
        G = rng.binomial(1, pout, (N, N))
        G = np.diag(np.diag(G)) + np.tril(G, -1) + np.tril(G, -1).T
        if not community_probs:
            labels = rng.integers(1, k+1, size=N)
        else:
            labels = rng.choice(range(1, k+1), size=N, p=community_probs)
        True_labels_all[sims, 0, :] = labels  # Store true labels for the first time step
        clusters = {i: np.where(labels == i)[0] for i in range(1, k+1)}

        # graph generation for t = 0
        for i in range(1, k+1):
            Gk = rng.binomial(1, pin[i-1], (len(clusters[i]), len(clusters[i])))
            Gk = np.diag(np.diag(Gk)) + np.tril(Gk, -1) + np.tril(Gk, -1).T
            G[np.ix_(clusters[i], clusters[i])] = Gk

        # iterate 
        for t in range(T):
            print("making graph for time step ", t)
            if t > 0:
                # data generation for t > 1
                G = rng.binomial(1, pout, (N, N))
                G = np.diag(np.diag(G)) + np.tril(G, -1) + np.tril(G, -1).T
                for i in range(1, k+1):
                    clusters[i] = np.where(labels == i)[0]
                    if clusters[i].size > 0: 
                        permute_indices = rng.permutation(len(clusters[i]))
                        changing_members = permute_indices[:int(np.ceil(len(clusters[i]) * p_switch))]
                        if changing_members.size > 0:
                            z = rng.binomial(1, p_switch, len(changing_members))
                            labels_temp = rng.integers(1, k+1, size=np.sum(z))
                            labels[changing_members[z == 1]] = labels_temp

                for i in range(1, k+1):
                    clusters[i] = np.where(labels == i)[0]
                    Gk = rng.binomial(1, pin[i-1], (len(clusters[i]), len(clusters[i])))
                    Gk = np.diag(np.diag(Gk)) + np.tril(Gk, -1) + np.tril(Gk, -1).T
                    G[np.ix_(clusters[i], clusters[i])] = Gk

            if try_sparse:
                Adjacency_all[sims][t] = csr_matrix(G)
            else:
                Adjacency_all[sims, t, :, :] = G
            True_labels_all[sims, t, :] = labels  # store true labels for this time step
        if verbose:
            print(f'Simulation {sims+1} completed')
    
    if verbose:
        # print community sizes for each simulation at each time step
        for sim in range(Totalsims):
            print(f"Simulation {sim+1}:")
            for t in range(T):
                print(f"Time step {t}:")
                for i in range(1, k+1):
                    community_size = np.sum(True_labels_all[sim, t, :] == i)
                    print(f"Community {i}: {community_size} nodes")
                print()
            print()


    sparse_adjacency_all = sparsify_all(Adjacency_all)
    #sparse_adjacency_all = Adjacency_all #?
    if save_path is not None:
        dg = [adj_list_to_tn_obj(sparse_adjacency, range(len(sparse_adjacency)), sparse_input=try_sparse)  for sparse_adjacency in sparse_adjacency_all]
        
        
        with open(save_path, 'wb') as f:
            pickle.dump((sparse_adjacency_all, True_labels_all, dg), f)


    return Adjacency_all, True_labels_all



if __name__ == "__main__":
    n = 4  # Number of nodes
    k = 3    # Number of communities
    T = 20  # Number of timestamps
    p_in = [0.3, 0.3, 0.2]  # Probability of edge within the same community
    p_out = 0.15  # Probability of edge between different communities
    p_switch = 0.1  # Probability of switching communities
    seed = 1  # Seed for reproducibility
    totalsims = 1  # Total number of simulations

    # see how the two models give different outputs
    #A1, L1 = sbm_dynamic_model_2(n, k, p_in, p_out, p_switch, T, totalsims, base_seed=seed)
    #A2, L2 = sbm_dynamic_model_2(n, k, p_in, p_out, p_switch, T, totalsims, base_seed=seed)
#
    #print(A1[0, 0, :, :])
    #print(A2[0, 0, :, :])

