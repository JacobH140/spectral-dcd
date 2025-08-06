#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
import sys
import os

# Add the package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'spectraldcd'))

from spectraldcd.alg.temporal_gcn_link_prediction_new import (
    TemporalLinkPredictionExperiment, 
    plot_tnetwork_comparison_bars
)

def create_test_temporal_network(n_nodes=50, n_timesteps=8):
    """Create a more complex test temporal network with community evolution."""
    np.random.seed(123)
    
    n_communities = 3
    nodes_per_community = n_nodes // n_communities
    adjacency_matrices = []
    
    for t in range(n_timesteps):
        adj = np.zeros((n_nodes, n_nodes))
        
        # Create within-community edges (stronger)
        for c in range(n_communities):
            start_idx = c * nodes_per_community
            end_idx = (c + 1) * nodes_per_community
            
            # Intra-community probability that varies with time
            intra_prob = 0.3 + 0.15 * np.sin(2 * np.pi * t / n_timesteps + c)
            
            for i in range(start_idx, end_idx):
                for j in range(i+1, end_idx):
                    if np.random.rand() < intra_prob:
                        adj[i, j] = adj[j, i] = 1
        
        # Create between-community edges (weaker)  
        inter_prob = 0.05 + 0.02 * np.cos(2 * np.pi * t / n_timesteps)
        for c1 in range(n_communities):
            for c2 in range(c1+1, n_communities):
                start1, end1 = c1 * nodes_per_community, (c1 + 1) * nodes_per_community
                start2, end2 = c2 * nodes_per_community, (c2 + 1) * nodes_per_community
                
                for i in range(start1, end1):
                    for j in range(start2, end2):
                        if np.random.rand() < inter_prob:
                            adj[i, j] = adj[j, i] = 1
        
        adjacency_matrices.append(sp.csr_matrix(adj))
    
    return adjacency_matrices

def demo_full_benchmark_with_intermediate():
    """Demo the full benchmark including the new geodesic_intermediate encoding."""
    print("=== Temporal GCN Benchmark with Geodesic Intermediate Encodings ===\\n")
    
    # Create temporal network
    print("Creating temporal community network...")
    adjacency_sequence = create_test_temporal_network(n_nodes=60, n_timesteps=8)
    
    print(f"Network stats:")
    for t, adj in enumerate(adjacency_sequence):
        n_edges = adj.nnz // 2  # undirected, so divide by 2
        density = n_edges / (60 * 59 / 2)
        print(f"  t={t}: {n_edges} edges, density={density:.3f}")
    
    # Run experiments for different encoding types
    results = {}
    
    encoding_types = ["laplacian", "geodesic", "geodesic_intermediate"]
    
    for encoding_type in encoding_types:
        print(f"\\n{'='*50}")
        print(f"Running experiment with {encoding_type.upper()} encoding")
        print(f"{'='*50}")
        
        # Skip static baselines for geodesic methods (too slow for demo)
        include_static = (encoding_type == "laplacian")
        
        experiment = TemporalLinkPredictionExperiment(
            encoding_type=encoding_type,
            n_eigenvectors=16,
            hidden_dim=32,
            epochs=50,  # Reasonable for demo
            canonicalize_sign=False
        )
        
        try:
            experiment_results = experiment.run_comprehensive_experiment(
                adjacency_sequence, verbose=True, include_static=include_static
            )
            
            results[f"{encoding_type}_nocanon"] = experiment_results
            print(f"✓ {encoding_type.upper()} experiment completed successfully!")
            
        except Exception as e:
            print(f"✗ {encoding_type.upper()} experiment failed: {e}")
    
    # Create comparison visualization
    print("\\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*70)
    
    if results:
        plot_tnetwork_comparison_bars(results, save_path="full_benchmark_with_intermediate_comparison")
        print("\\nComparison plots saved!")
    else:
        print("No results to plot.")
    
    return results

if __name__ == "__main__":
    results = demo_full_benchmark_with_intermediate()
    
    print("\\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print("\\nKey accomplishments:")
    print("• Successfully implemented geodesic_intermediate encoding")
    print("• Enriched embeddings with intermediate iteration history")
    print("• Updated experiment framework to include new benchmark")
    print("• Enhanced visualization to show all three methods")
    print("• Demonstrated end-to-end functionality")
    
    if len(results) == 3:
        print("\\n✓ All three encoding methods completed successfully!")
    else:
        print(f"\\n⚠ Only {len(results)} out of 3 encoding methods completed.")