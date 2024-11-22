import tnetwork as tn
import networkx as nx
import time


def adj_list_to_tn_obj(adj_matrices, timestamps=None, sparse_input=False):
    """
    Create a dynamic network from a list of adjacency matrices.
    
    :param adj_matrices: List of numpy arrays, where each array is an adjacency matrix.
    :param timestamps: List of timestamps corresponding to each adjacency matrix. If None, use integer indices.
    :return: A DynamicGraph object.
    """
    dg = tn.DynGraphSN() # snapshot representation of dynamic graph

    if timestamps is None:
        timestamps = range(len(adj_matrices))
    
    for adj_matrix, timestamp in zip(adj_matrices, timestamps):
        start = time.time()
        if sparse_input:
            G = nx.from_scipy_sparse_array(adj_matrix)
        else:
            G = nx.from_numpy_array(adj_matrix)
        dg.add_snapshot(t=timestamp, graphSN=G)
    return dg


    

def apply_tnetwork_alg(adj_matrix_list, alg, generated_network_SN=None, **kwargs):
    """
    Apply a dynamic community detection algorithm from tnetwork to a dynamic graph. Return both the DynCommunitiesSN object and the labels_pred 2d list
    """

    timestamps = range(len(adj_matrix_list))
    if generated_network_SN == None:
        generated_network_SN = adj_list_to_tn_obj(adj_matrix_list, timestamps)
    
    start = time.time()
    print("starting algorithm timer...")
    communitiesSN = alg(generated_network_SN, **kwargs)
    time_elapsed = time.time() - start
    print(f"Time elapsed: {time_elapsed:.2f} seconds")
    return communitiesSN, convert_to_label_lists(communitiesSN, timestamps)

def convert_to_label_lists(dyn_communities, timestamps):
    labels_pred_all = []
    for t in range(len(timestamps)):
        affiliations_t = dyn_communities.snapshot_affiliations(t=t)
        affiliations_t_ordered_by_node = {key: affiliations_t[key] for key in sorted(affiliations_t)}
        labels_pred = [list(affiliations_t_ordered_by_node[node])[0] for node in affiliations_t_ordered_by_node]
        labels_pred_all.append(labels_pred)

    return labels_pred_all

if __name__ =="__main__":
    pass

    
