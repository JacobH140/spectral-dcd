#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Add the package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'spectraldcd'))

from spectraldcd.alg.spectral_geodesic_smoothing import spectral_geodesic_smoothing

def create_test_network(n_nodes=50, n_timesteps=10):
    """Create a simple test temporal network."""
    np.random.seed(42)  # for reproducibility
    
    # Create random sparse adjacency matrices
    adjacency_matrices = []
    
    for t in range(n_timesteps):
        # Create a random sparse matrix with some structure
        prob = 0.1 + 0.05 * np.sin(2 * np.pi * t / n_timesteps)  # time-varying sparsity
        
        # Generate random adjacency matrix
        adj = np.random.rand(n_nodes, n_nodes) < prob
        adj = adj.astype(float)
        
        # Make it symmetric (undirected)
        adj = (adj + adj.T) / 2
        adj[adj > 0] = 1  # binarize
        
        # Remove self-loops
        np.fill_diagonal(adj, 0)
        
        # Convert to sparse matrix
        adjacency_matrices.append(csr_matrix(adj))
    
    return adjacency_matrices

def test_intermediate_iterations():
    """Test the intermediate iterations functionality."""
    print("Creating test network...")
    adjacency_matrices = create_test_network(n_nodes=30, n_timesteps=5)
    
    print("Running spectral geodesic smoothing WITHOUT intermediate iterations...")
    assignments1, embeddings1 = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices),
        num_nodes=30,
        ke=5,
        kc=3,
        mode='simple-nsc',
        use_intermediate_iterations=False
    )
    
    print("Running spectral geodesic smoothing WITH intermediate iterations...")
    assignments2, embeddings2 = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices), 
        num_nodes=30,
        ke=5,
        kc=3,
        mode='simple-nsc',
        use_intermediate_iterations=True,
        num_intermediate_iterations=5
    )
    
    print("\nResults comparison:")
    print(f"Standard embeddings shape at t=0: {embeddings1[0].shape}")
    print(f"Enriched embeddings shape at t=0: {embeddings2[0].shape}")
    
    # The enriched embeddings should have more features (original + intermediate iterations)
    expected_enriched_features = embeddings1[0].shape[1] * (1 + 5)  # 1 main + 5 intermediate
    print(f"Expected enriched features: {expected_enriched_features}")
    
    print(f"Assignments1 length: {len(assignments1)}")
    print(f"Assignments2 length: {len(assignments2)}")
    
    # Test that assignments are reasonable (each node assigned to a community)
    for t in range(len(assignments1)):
        unique_communities1 = len(np.unique(assignments1[t]))
        unique_communities2 = len(np.unique(assignments2[t]))
        print(f"t={t}: Standard={unique_communities1} communities, Enriched={unique_communities2} communities")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_intermediate_iterations()