#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Add the package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'spectraldcd'))

from spectraldcd.alg.spectral_geodesic_smoothing import spectral_geodesic_smoothing

def create_community_network(n_nodes=60, n_timesteps=8, n_communities=3):
    """Create a temporal network with evolving community structure."""
    np.random.seed(123)
    
    nodes_per_community = n_nodes // n_communities
    adjacency_matrices = []
    
    for t in range(n_timesteps):
        adj = np.zeros((n_nodes, n_nodes))
        
        # Create within-community edges (stronger)
        for c in range(n_communities):
            start_idx = c * nodes_per_community
            end_idx = (c + 1) * nodes_per_community
            
            # Intra-community probability that varies with time
            intra_prob = 0.3 + 0.2 * np.sin(2 * np.pi * t / n_timesteps + c)
            
            for i in range(start_idx, end_idx):
                for j in range(i+1, end_idx):
                    if np.random.rand() < intra_prob:
                        adj[i, j] = adj[j, i] = 1
        
        # Create between-community edges (weaker)  
        inter_prob = 0.05
        for c1 in range(n_communities):
            for c2 in range(c1+1, n_communities):
                start1, end1 = c1 * nodes_per_community, (c1 + 1) * nodes_per_community
                start2, end2 = c2 * nodes_per_community, (c2 + 1) * nodes_per_community
                
                for i in range(start1, end1):
                    for j in range(start2, end2):
                        if np.random.rand() < inter_prob:
                            adj[i, j] = adj[j, i] = 1
        
        adjacency_matrices.append(csr_matrix(adj))
    
    return adjacency_matrices

def demo_intermediate_iterations():
    """Demonstrate the intermediate iterations functionality."""
    print("=== Spectral Geodesic Smoothing with Intermediate Iterations Demo ===\n")
    
    print("Creating temporal community network...")
    adjacency_matrices = create_community_network(n_nodes=60, n_timesteps=6, n_communities=3)
    
    print(f"Network stats:")
    for t, adj in enumerate(adjacency_matrices):
        n_edges = adj.nnz // 2  # undirected, so divide by 2
        density = n_edges / (60 * 59 / 2)
        print(f"  t={t}: {n_edges} edges, density={density:.3f}")
    
    print("\n1. Standard spectral geodesic smoothing:")
    assignments_standard, embeddings_standard = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices),
        num_nodes=60,
        ke=8,
        kc=3,
        mode='simple-nsc',
        use_intermediate_iterations=False
    )
    
    print("\n2. With intermediate iterations (last 10 iterations):")
    assignments_enriched, embeddings_enriched = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices),
        num_nodes=60,
        ke=8,
        kc=3,
        mode='simple-nsc',
        use_intermediate_iterations=True,
        num_intermediate_iterations=10
    )
    
    print("\n=== Results Comparison ===")
    print(f"Standard embedding dimensions: {embeddings_standard[0].shape[1]} features")
    print(f"Enriched embedding dimensions: {embeddings_enriched[0].shape[1]} features") 
    print(f"Enhancement factor: {embeddings_enriched[0].shape[1] / embeddings_standard[0].shape[1]:.1f}x")
    
    print(f"\nCommunity detection results:")
    for t in range(len(assignments_standard)):
        n_communities_std = len(np.unique(assignments_standard[t]))
        n_communities_enr = len(np.unique(assignments_enriched[t]))
        print(f"  t={t}: Standard={n_communities_std}, Enriched={n_communities_enr} communities")
    
    print("\n=== Only geodesic embeddings (no clustering) ===")
    embeddings_only_standard = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices),
        num_nodes=60,
        ke=8,
        mode='simple-nsc',
        return_geo_embeddings_only=True,
        use_intermediate_iterations=False
    )
    
    embeddings_only_enriched = spectral_geodesic_smoothing(
        sadj_list=adjacency_matrices,
        T=len(adjacency_matrices),
        num_nodes=60,
        ke=8,
        mode='simple-nsc',
        return_geo_embeddings_only=True,
        use_intermediate_iterations=True,
        num_intermediate_iterations=15
    )
    
    print(f"Embeddings-only standard shape: {embeddings_only_standard[0].shape}")
    print(f"Embeddings-only enriched shape: {embeddings_only_enriched[0].shape}")
    
    # Compute some basic statistics
    print(f"\nEmbedding norms comparison (t=0):")
    norm_std = np.linalg.norm(embeddings_standard[0], axis=1).mean()
    norm_enr = np.linalg.norm(embeddings_enriched[0], axis=1).mean()
    print(f"  Standard: {norm_std:.3f}")
    print(f"  Enriched: {norm_enr:.3f}")
    
    print("\n=== Demo completed successfully! ===")
    print("\nKey benefits of intermediate iterations:")
    print("• Richer positional encodings with geodesic iteration history")
    print("• Enhanced feature dimensionality for downstream tasks")
    print("• Better capture of geodesic manifold evolution")
    print("• Improved representational power for temporal networks")

if __name__ == "__main__":
    demo_intermediate_iterations()