import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy import sparse
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import sinm, cosm

from abc import ABC, abstractmethod

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


from sklearn.cluster import KMeans
import time

#import teneto
#from teneto.utils import process_input



def is_symmetric(matrix, tol=1e-8):
    if issparse(matrix):
        return (matrix != matrix.T).nnz == 0
    return np.allclose(matrix, matrix.T, atol=tol)

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from math import pi

import numpy as np
from scipy.sparse import issparse

def is_symmetric(Xi):
    if issparse(Xi):
        return (Xi - Xi.transpose()).nnz == 0
    else:
        return np.allclose(Xi, Xi.T)

def thetamajorizer(H, Y, X):
    T = len(X)
    k = H.shape[1]
    
    r = np.empty((T, k), dtype=np.float64)
    phi = np.empty((T, k), dtype=np.float64)
    bias = np.empty((T, k), dtype=np.float64)
    
    symmetric_flags = [is_symmetric(Xi) for Xi in X]
    
    if not H.flags['C_CONTIGUOUS']:
        H = np.ascontiguousarray(H)
    if not Y.flags['C_CONTIGUOUS']:
        Y = np.ascontiguousarray(Y)
    
    for ii in range(T):
        Xi = X[ii]
        symmetric = symmetric_flags[ii]
        
        if symmetric:
            XiT_H = Xi @ H  
            XiT_Y = Xi @ Y 
        else:
            if issparse(Xi):
                XiT = Xi.transpose().tocsr() 
            else:
                XiT = Xi.T  
            XiT_H = XiT @ H  
            XiT_Y = XiT @ Y 
        

        a = np.sum(np.abs(XiT_H)**2, axis=0)  
        

        b = np.real(np.sum(XiT_Y.conj() * XiT_H, axis=0))
        
        c = np.sum(np.abs(XiT_Y)**2, axis=0) 
        
        a_minus_c_over_2 = (a - c) / 2.0
        two_b = 2.0 * b
        
        np.sqrt(a_minus_c_over_2**2 + b**2, out=r[ii, :])
        
        np.arctan2(two_b, a - c, out=phi[ii, :])
        
        bias[ii, :] = (a + c) / 2.0
    
    return r, phi, bias

def esttheta(H, Y, X, t, niter=5, tH=0, Theta_init=None):
    r, phi, bias = thetamajorizer(H, Y, X)
    
    t = np.asarray(t, dtype=np.float64)[:, np.newaxis] - tH 
    tt = 2.0 * t  
    
    n2tr = -tt * r  
    L = tt * n2tr  
    
    if Theta_init is None:
        Theta = np.ones(H.shape[1], dtype=np.float64)  # Shape: (k,)
    else:
        Theta = Theta_init.astype(np.float64).copy()  # Shape: (k,)
    
    for _ in range(niter):
        arg = tt * Theta  
        arg -= phi 
        
        gradf = n2tr * np.sin(arg)  
        
        denom = (arg + np.pi) % (2 * np.pi) - np.pi  
        
        with np.errstate(divide='ignore', invalid='ignore'):
            curvf = np.divide(tt * gradf, denom, out=np.zeros_like(tt * gradf), where=denom!=0)
            curvf += L * (denom == 0)
        
        gradf_sum = np.sum(gradf, axis=0)  
        curvf_sum = np.sum(curvf, axis=0) 
        step = np.divide(gradf_sum, curvf_sum, out=np.zeros_like(gradf_sum), where=curvf_sum!=0)  # Shape: (k,)
        
        Theta -= step  
    
    return Theta




def estpoint_tangent(H, Y, Theta, X, t, tH=0):
    M_AB = 0.0

    for ti, Xi in zip(t, X):
        arg = Theta * (ti - tH)  # (k,)
        cos_arg = np.cos(arg)  # (k,)
        sin_arg = np.sin(arg)  # (k,)
        Ui = H * cos_arg + Y * sin_arg  

        if is_symmetric(Xi):
            G = Xi @ Ui  # (n_i, k)
            XG = G  # for symmetric matrices Xi @ G.T = G
        else:
            G = Ui.T @ Xi  # (k, n_i)
            XG = Xi @ G.T  # (d, k)

        XGcos = XG * cos_arg  
        XGsin = XG * sin_arg 

        M_AB_i = np.concatenate((XGcos, XGsin), axis=1)  #(d, 2k)
        M_AB += M_AB_i

    return M_AB

#def convert_to_snapshots(tnet, snapshot_resolution):
#    """
#    Converts various types of temporal network inputs into a list of sparse adjacency matrices.
#    Parameters:
#    tnet: array, dict, or TemporalNetwork - Input temporal network.
#    snapshot_resolution (int): Duration of each snapshot in the network's time unit.
#    Returns:
#    List of csr_matrix: Each matrix represents a network snapshot.
#    """
#    # Standardize input to TemporalNetwork format
#    tnet = process_input(tnet, ['C', 'G', 'TN'], 'TN')
#    # Calculate number of snapshots
#    T = int(np.ceil((tnet.timelabels[-1] - tnet.timelabels[0]) / snapshot_resolution))
#    
#    # Create empty list for snapshots
#    sparse_matrices = [csr_matrix((tnet.N, tnet.N)) for _ in range(T)]
#    # Fill sparse matrices
#    for _, row in tnet.network.iterrows():
#        i, j, t = int(row['i']), int(row['j']), row['t']
#        snapshot_index = int((t - tnet.timelabels[0]) / snapshot_resolution)
#        if snapshot_index < T:
#            sparse_matrices[snapshot_index][i, j] = row.get('weight', 1)
#    return sparse_matrices

#def teneto_dcd(tnet, snapshot_resolution, ke, negativeedge='anticommunity', kc='auto', stable_communities=False, mode='simple-nsc', fit_eigenvector_embeddings=False, smoothing_filter=None, smoothing_parameter=None):
#    """
#    Performs dynamic community detection on a temporal network using spectral geodesic smoothing.
#    
#    Parameters:
#    tnet : array, dict, TemporalNetwork
#        Input network
#    snapshot_resolution : int
#        Duration of each snapshot in the network's time unit
#    ke : int
#        Number of eigenvectors to use
#    negativeedge : str, optional
#        How to handle negative edges (default is 'anticommunity')
#    kc : str or int, optional
#        Number of communities (default is 'auto')
#    stable_communities : bool, optional
#        Whether to enforce stable communities across time (default is False)
#    mode : str, optional
#        Mode of spectral geodesic smoothing (default is 'simple-nsc')
#    fit_eigenvector_embeddings : bool, optional
#        Whether to fit eigenvector embeddings (default is False)
#    smoothing_filter : str or None, optional
#        Type of smoothing filter to apply (default is None)
#    smoothing_parameter : float or None, optional
#        Parameter for smoothing filter (default is None)
#    
#    Returns:
#    numpy.ndarray
#        2D array of community assignments (node, time)
#    """
#    # Convert input to list of snapshot matrices
#    matrix_list = convert_to_snapshots(tnet, snapshot_resolution)
#    
#    # Perform spectral geodesic smoothing
#    community_vectors, _ = spectral_geodesic_smoothing(
#        matrix_list, len(matrix_list), tnet.N, ke, kc, stable_communities, 
#        mode, fit_eigenvector_embeddings, smoothing_filter, smoothing_parameter
#    )
#    
#    # Convert list of community vectors to 2D array (node, time)
#    communities = np.array(community_vectors).T
#    
#    return communities
    

def sfit_point_tangent_geodesic(data, k, max_iter, tol=1e-5, rel_tol=1e-2):
    X, t = data  # X is list of matrices, t is list of times
    init_start = time.time()
    # Extract M1 and MT
    M1 = X[0]
    MT = X[-1]

    # are M1 and MT symmetric?
    M1_symmetric = is_symmetric(M1)
    MT_symmetric = is_symmetric(MT)
    
    print("M1 symmetric:", M1_symmetric)
    print("MT symmetric:", MT_symmetric)

    # rank-k truncated SVD of M1
    if M1_symmetric:
        if issparse(M1):
            S1, U1 = eigsh(M1, k=k, which='LM')
        else:
            S1, U1 = np.linalg.eigh(M1)
            idx = np.argsort(-S1)[:k]
            U1 = U1[:, idx]
            S1 = S1[idx]
    else:
        U1, S1, _ = svds(M1, k=k)
        idx = np.argsort(-S1)
        U1 = U1[:, idx]
        S1 = S1[idx]
    H1 = U1  # (d, k)

    # ..... and for MT also
    if MT_symmetric:
        if issparse(MT):
            ST, UT = eigsh(MT, k=k, which='LM')
        else:
            ST, UT = np.linalg.eigh(MT)
            idx = np.argsort(-ST)[:k]
            UT = UT[:, idx]
            ST = ST[idx]
    else:
        UT, ST, _ = svds(MT, k=k)
        idx = np.argsort(-ST)
        UT = UT[:, idx]
        ST = ST[idx]
    H_T = UT  # (d, k)

    H1T_H_T = H1.T @ H_T  # (k, k)

    Z, S, Q_T = np.linalg.svd(H1T_H_T, full_matrices=False)
    Q = Q_T.T  # (k, k)
    
    
    if np.any((S < -1 - 1e-6) | (S > 1 + 1e-6)): # clip S 
        print("Warning: S has entries more than 1e-6 outside the range [-1, 1]")
    S = np.clip(S, -1, 1)

    H_TQ = H_T @ Q  # (d, k)

    # orthogonal component
    H1_H1T_HTQ = H1 @ (H1.T @ H_TQ)  # (d, k)
    Orth_comp = H_TQ - H1_H1T_HTQ  # (d, k)

    # SVD of orthocomplement
    F, D, G_T = np.linalg.svd(Orth_comp, full_matrices=False)

    # direction Y = F G_T
    Y = F @ G_T  # (d, k)



    # Theta init (see paper)
    Theta = np.arccos(S)  # (k,)

    
    
    #print("init time:", time.time() - init_start)
    
    #print("Starting iterations")
    iter_start = time.time()
    point_tangent_times = []
    theta_times = []
    conv_check_times = []

    # init H = H1
    H = H1.copy()

    # set the initial values for convergence checking
    H_old = H.copy()
    Y_old = Y.copy()
    Theta_old = Theta.copy()

    # fix tH=0, at least until there seems to be a reason to not do so
    tH = 0 

    for itr in range(max_iter):        
        if itr == max_iter - 1:
            print("Warning: max iterations reached")
        
        #print(f"Iteration {itr}")
        
        # P-Update, P = [H, Y]
        pt_time = time.time()
        M_AB = estpoint_tangent(H, Y, Theta, X, t, tH)
        try: 
            U, _, Vh = np.linalg.svd(M_AB, full_matrices=False)
        except np.linalg.LinAlgError as e:
            print(f"SVD failed for M_AB: {str(e)}")
            print(f"Matrix shape: {M_AB.shape}")
            #print(f"Matrix rank: {np.linalg.matrix_rank(M_AB)}")
            #print(f"Matrix condition number: {np.linalg.cond(M_AB)}")
            print(f"nans, infs? {np.any(np.isnan(M_AB))} {np.any(np.isinf(M_AB))}")
            print("Trying to add a small amount of noise to the matrix")
            M_AB += 1e-6 * np.random.randn(*M_AB.shape)
            U, _, Vh = np.linalg.svd(M_AB, full_matrices=False)
        C = U @ Vh  # (d, 2k)
        H = C[:, :k]
        Y = C[:, k:]
        point_tangent_times.append(time.time() - pt_time)

        # Theta-Update
        theta_time = time.time()
        Theta = esttheta(H, Y, X, t, niter=5, tH=tH, Theta_init=Theta)
        theta_times.append(time.time() - theta_time)

        # Check convergence?
        conv_check_time = time.time()
        delta_H = np.linalg.norm(H - H_old)
        delta_Y = np.linalg.norm(Y - Y_old)
        delta_Theta = np.linalg.norm(Theta - Theta_old)


        rel_delta_H = delta_H / (np.linalg.norm(H) + 1e-15)
        rel_delta_Y = delta_Y / (np.linalg.norm(Y) + 1e-15)
        rel_delta_Theta = delta_Theta / (np.linalg.norm(Theta) + 1e-15)

        if (delta_H < tol and delta_Y < tol and delta_Theta < tol) or (rel_delta_H < rel_tol and rel_delta_Y < rel_tol and rel_delta_Theta < rel_tol):
            print(f"Converged in {itr} iterations")
            break
            
        # update the old values
        H_old = H.copy()
        Y_old = Y.copy()
        Theta_old = Theta.copy()
        
        conv_check_times.append(time.time() - conv_check_time)
    
    #print("point tangent time", np.sum(np.array(point_tangent_times)))
    #print("theta time", np.sum(np.array(theta_times)))
    #print("conv check time", np.sum(np.array(conv_check_times)))
    return H, Y, Theta



class SpectralGeodesicSmoother(ABC):
    def __init__(self, *args, d, T, sadj_list, ke='auto', kc_list='auto', stable_communities=False, fit_eigenvector_embeddings=False, which_eig='smallest', t=None, benefit_fn=None, smoothing_filter=None, smoothing_parameter=None, max_iter=1000):
        if len(args) > 0:
            raise ValueError("This class does not accept positional arguments")

        if fit_eigenvector_embeddings:
            print("Warning: fit_eigenvector_embeddings is not yet implemented (need to change geodesic init for it)")

        self.d = d
        self.max_iter = max_iter
        self.sadj_list = sadj_list
        self.ke = ke if ke != 'auto' else self.choose_ke()
        if isinstance(kc_list, int):
            kc_list = [kc_list]*T
        self.kc_list = kc_list
        self.stable_communities = stable_communities
        self.fit_eigenvector_embeddings = fit_eigenvector_embeddings
        self.which_eig = which_eig
        self.t = t if t is not None else np.linspace(0.0, 1.0, len(sadj_list))
        
        self.benefit_fn = benefit_fn
        self.smoothing_filter = smoothing_filter
        self.smoothing_parameter = smoothing_parameter
        self.T = T
        
        if self.stable_communities and kc_list == 'auto':
            self.kc_list = [ke]*T
        
    
    def benefit_fn_broadcast(self, sadj_list, labels_list):
        return [self.benefit_fn(sadj, labels) for sadj, labels in zip(sadj_list, labels_list)]
    
    
    @abstractmethod
    def make_clustering_matrix(self, sadj):
        pass
    
    def choose_ke(self):
        raise NotImplementedError("must implement choose_ke")
    
    def make_clustering_matrices(self):
        self.clustering_matrices = [self.make_clustering_matrix(sadj) for sadj in self.sadj_list]
    
    def make_modeled_clustering_matrices(self):
        # this is permitted to be overwritten by subclasses, but the following default implementation usually works out fine
        assert self.clustering_matrices is not None, "You must first run make_clustering_matrices"
        self.modeled_clustering_matrices = []
        for R in self.clustering_matrices:
            frobenius_norm = sparse.linalg.norm(R, 'fro')
            n,m= R.shape
            I = sparse.eye(n, m, format=R.format)
            if self.which_eig == 'smallest':
                self.modeled_clustering_matrices.append(frobenius_norm*I - R)
            elif self.which_eig == 'largest':
                self.modeled_clustering_matrices.append(frobenius_norm*I + R)
            elif self.which_eig == 'svd':
                self.modeled_clustering_matrices.append(R)
            else:
                raise ValueError("Invalid value for which_eig")
            

    # this never overwritten by subclasses
    def get_geodesic_embeddings(self):
        modeled_clustering_matrices = self.modeled_clustering_matrices 
        if self.fit_eigenvector_embeddings:
            self.Xs = [
                eigsh(sparse.csr_matrix(matrix), k=self.ke, which='LM', return_eigenvectors=True)[1]
                for matrix in modeled_clustering_matrices
            ]
        else:
            self.Xs = modeled_clustering_matrices
            
        H, Y, Theta = sfit_point_tangent_geodesic((self.Xs, self.t), k=self.ke, max_iter=self.max_iter)
        self.Us = [H @ cosm(np.diag(Theta)*self.t[i]) + Y @ sinm(np.diag(Theta)*self.t[i]) for i in range(self.T)]
    
    # this overwritten sometimes by subclasses
    def clustering_Euclidean(self):
        T = len(self.Us)
        #print("kc_list:", self.kc_list)

        if all(k == self.kc_list[0] for k in self.kc_list):  # Constant kc_list
            print("Using constant kc_list")
            k_val = self.kc_list[0]
            kmeans = KMeans(n_clusters=k_val, n_init=10)
            labels = [kmeans.fit_predict(U_i) for U_i in self.Us]
            return labels
        elif all(isinstance(k, int) for k in self.kc_list):  # Non-constant, provided kc_list
            labels = []
            for U_i, k_val in zip(self.Us, self.kc_list):
                kmeans = KMeans(n_clusters=k_val, n_init=10) 
                labels.append(kmeans.fit_predict(U_i))
            return labels
        else:  # Auto kc_list
            print("TODO: auto determine kmin (defaults to 2 right now)")
            kmin, kmax = 2, self.ke # TODO
            k_vals = range(kmin, kmax + 1)
            benefit_vs_time_and_k = np.zeros((len(k_vals), T)) - np.inf
            labels_by_k = {k: [] for k in k_vals}

            start = time.time()
            for i, U_i in enumerate(self.Us):
                for j, k_val in enumerate(k_vals):
                    kmeans = KMeans(n_clusters=k_val, n_init=10)
                    labels_i = kmeans.fit_predict(U_i)
                    labels_by_k[k_val].append(labels_i)
                    benefit = self.benefit_fn_broadcast([self.sadj_list[i]], [labels_i])[0]
                    benefit_vs_time_and_k[j, i] = benefit


            if self.smoothing_filter == 'median':
                kernel_size = self.smoothing_parameter
                smoothed_benefit = np.array([medfilt(row, kernel_size) for row in benefit_vs_time_and_k])
            elif self.smoothing_filter == 'gaussian':
                sigma = self.smoothing_parameter
                smoothed_benefit = np.array([gaussian_filter1d(row, sigma) for row in benefit_vs_time_and_k])
            else:
                smoothed_benefit = benefit_vs_time_and_k

            k_maximizing_benefit = [k_vals[l] for l in np.argmax(smoothed_benefit, axis=0)]

            best_labels_all = [labels_by_k[k][i] for i, k in enumerate(k_maximizing_benefit)]

            return best_labels_all
        
    
    # this is never overwritten by subclasses 
    def run_dcd(self):
        self.run_geo_embeddings()
        time_start = time.time()
        assignments = self.clustering_Euclidean()
        return assignments 
    
    def run_geo_embeddings(self):
        time_start = time.time()
        self.make_clustering_matrices()
        self.make_modeled_clustering_matrices() 
        time_start = time.time()
        self.get_geodesic_embeddings()
        print(f"Time to get geodesic embeddings: {time.time() - time_start}")
        

def spectral_geodesic_smoothing(sadj_list, T, num_nodes, ke, kc='auto', stable_communities=False, mode='simple-nsc', fit_eigenvector_embeddings=False, smoothing_filter=None, smoothing_parameter=None, **mode_kwargs):
    #print("spectral_geodesic_smoothing")
    T = len(sadj_list)
    if not isinstance(kc, list) and kc != 'auto':
        try:
            kc = [kc] * T
        except TypeError:
            raise ValueError("kc must be a list, a natural number, or the default 'auto'")

    types = ['simple', 'signed', 'directed', 'overlapping', 'multiview', 'cocommunity', 'hierarchical']
    algorithms = {
        'simple': ['nsc', 'smm', 'bhc'],
        'signed': ['srsc', 'gmsc', 'spmsc'],
        'overlapping': ['osc', 'csc'],
        'directed': ['ddsc', 'bsc', 'rwsc'],
        'multiview': ['gmsc', 'pmlc'],
        'cocommunity': ['scc'],
        'hierarchical': ['hsc']
    }

    class_mapping = {
        f"{type}-{alg}": f"{alg.upper()}"
        for type in types
        for alg in algorithms[type]
    }


    if mode not in class_mapping:
        valid_modes = ', '.join(class_mapping.keys())
        raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")


    class_name = class_mapping[mode]
    smoother_class = globals().get(class_name)

    if smoother_class is None:
        raise NotImplementedError(f"The algorithm class for mode '{mode}' is not implemented yet.")

    smoother = smoother_class(T=T, d=num_nodes, sadj_list=sadj_list, ke=ke, kc_list=kc,
                              stable_communities=stable_communities,
                              fit_eigenvector_embeddings=fit_eigenvector_embeddings, 
                              smoothing_filter=smoothing_filter,
                              smoothing_parameter=smoothing_parameter,
                              **mode_kwargs)  

    assignments = smoother.run_dcd()
    
    return assignments, smoother.Us


## Begin subclass implementations
class Simple(SpectralGeodesicSmoother):
    @staticmethod
    def calculate_modularity(adj_matrix: csr_matrix, communities: list) -> float:
        """
        Calculate the modularity of a network given its adjacency matrix and community assignments.

        :param adj_matrix: Sparse adjacency matrix of the network (scipy.sparse.csr_matrix)
        :param communities: List of community assignments for each node
        :return: Modularity value
        """
        if not isinstance(adj_matrix, csr_matrix):
            raise ValueError("adj_matrix must be a scipy.sparse.csr_matrix")

        if len(communities) != adj_matrix.shape[0]:
            raise ValueError("Number of community assignments must match number of nodes")

        n_edges = adj_matrix.sum() / 2
        n_nodes = adj_matrix.shape[0]

        modularity = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if communities[i] == communities[j]:
                    a_ij = adj_matrix[i, j]
                    k_i = adj_matrix.getrow(i).sum()
                    k_j = adj_matrix.getrow(j).sum()
                    expected = (k_i * k_j) / (2 * n_edges)
                    modularity += a_ij - expected

        modularity /= (2 * n_edges)
        return modularity
    
    @staticmethod
    def calculate_modularity_vectorized(adj_matrix: csr_matrix, communities: list) -> float:
        """
        Calculate the modularity of a network given its adjacency matrix and community assignments
        using a fully vectorized approach.

        :param adj_matrix: Sparse adjacency matrix of the network (scipy.sparse.csr_matrix)
        :param communities: List of community assignments for each node
        :return: Modularity value
        """
        if not isinstance(adj_matrix, csr_matrix):
            raise ValueError("adj_matrix must be a scipy.sparse.csr_matrix")

        n_nodes = adj_matrix.shape[0]

        if len(communities) != n_nodes:
            raise ValueError("Number of community assignments must match number of nodes")

        m = adj_matrix.sum() / 2
        if m == 0:
            raise ValueError("The network has no edges.")

        communities = np.array(communities)
        unique_communities, inverse_indices = np.unique(communities, return_inverse=True)
        n_communities = unique_communities.size

        degrees = np.array(adj_matrix.sum(axis=1)).flatten()

        degree_sum_per_community = np.bincount(inverse_indices, weights=degrees)

        community_matrix = csr_matrix(
            (np.ones(n_nodes), (inverse_indices, np.arange(n_nodes))),
            shape=(n_communities, n_nodes)
        )

        connections_within = community_matrix.dot(adj_matrix).dot(community_matrix.transpose())

        internal_edges_per_community = connections_within.diagonal() / 2

        modularity = (internal_edges_per_community.sum() / m) - np.sum((degree_sum_per_community / (2 * m)) ** 2)

        return modularity
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit_fn = self.calculate_modularity_vectorized
    

class Signed(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def power_mean_laplacian(self, A_pos, A_neg, p, epsilon=1e-6):
        pass
    
    def create_clustering_matrix(self, A_pos, A_neg, p, epsilon=1e-6):
        pass

class Directed(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Overlapping(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Multiview(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Cocommunity(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Hierarchical(SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

## end subclass implementations

## Begin subsubclass implementations

class NSC(Simple):
    def make_clustering_matrix(self, sadj):
        pass # we override the full make_modeled_clustering_matrices method using normalized_signless_laplacian (recall that normalized signless laplacian SVD is equivalent to spectral clustering with normalized graph laplacian)
    

    
    @staticmethod
    def normalized_signless_laplacian(A):
        degrees = A.sum(axis=1).A1 

        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = 1.0 / np.sqrt(degrees)
        D_inv_sqrt[np.isinf(D_inv_sqrt) | np.isnan(D_inv_sqrt)] = 0
        D_inv_sqrt_matrix = sparse.diags(D_inv_sqrt)
        D_plus_A = sparse.diags(degrees) + A
        L_signless = D_inv_sqrt_matrix @ D_plus_A @ D_inv_sqrt_matrix
        return L_signless
    
    def make_modeled_clustering_matrices(self):
        self.modeled_clustering_matrices =  [NSC.normalized_signless_laplacian(sadj) for sadj in self.sadj_list]
        
  


class SMM(Simple): 
    def __init__(self, *args, **kwargs):
        kwargs['which_eig'] = 'largest' 
        super().__init__(*args, **kwargs)
      
    def make_clustering_matrix(self, A): 
        degrees = A.sum(axis=1).A1  
        m = A.sum() / 2  
        expected = sparse.csr_matrix((np.outer(degrees, degrees) / (2 * m)).astype(A.dtype))  
        B = A - expected  
        return B

class BHC(Simple):
    def make_clustering_matrix(self, sadj):
        r=None
        if r is None:
            r = np.sqrt(sadj.mean() * sadj.shape[0]) 
    
        n = sadj.shape[0]  
        d = sadj.sum(axis=1).A1  
        H = (r**2 - 1) * sparse.eye(n) - r * sadj + sparse.diags(d)  
        return H
    
    
def signed_power_mean_laplacian(A_pos, A_neg, p, epsilon=1e-6):
    pass
    
class SRSC(Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class GMSC(Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class SPMSC(Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class OSC(Overlapping):
    def make_clustering_matrix(self, sadj):
        pass
    
class CSC(Overlapping):
    def make_clustering_matrix(self, sadj):
        pass
    
class DDSC(Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class BSC(Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class RWSC(Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class PMLC(Multiview):
    def make_clustering_matrix(self, sadj):
        pass
    
class SCC(Cocommunity):
    def make_clustering_matrix(self, sadj):
        pass
    
class HSC(Hierarchical):
    def make_clustering_matrix(self, sadj):
        pass

## End subsubclass implementations
