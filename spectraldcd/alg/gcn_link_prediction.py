import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import networkx as nx
from typing import Tuple, Optional, List, Union


class LaplacianEigenvectorEncoder:
    """Encodes node features using Laplacian eigenvectors."""
    
    def __init__(self, n_eigenvectors: int = 64, normalized: bool = True):
        self.n_eigenvectors = n_eigenvectors
        self.normalized = normalized
        
    def fit_transform(self, adjacency_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Compute Laplacian eigenvectors as node features.
        
        Args:
            adjacency_matrix: Sparse adjacency matrix of the graph
            
        Returns:
            Node features as Laplacian eigenvectors
        """
        # Compute degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        
        # Handle isolated nodes
        degrees[degrees == 0] = 1
        
        # Compute Laplacian
        if self.normalized:
            # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            degree_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
            laplacian = sp.eye(adjacency_matrix.shape[0]) - degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
        else:
            # Combinatorial Laplacian: L = D - A
            degree_matrix = sp.diags(degrees)
            laplacian = degree_matrix - adjacency_matrix
        
        # Compute smallest eigenvectors (excluding the constant eigenvector)
        try:
            n_eigs = min(self.n_eigenvectors + 1, adjacency_matrix.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigs, which='SM')
            
            # Remove the smallest eigenvalue (should be close to 0) and its eigenvector
            eigenvectors = eigenvectors[:, 1:]
            
            # If we have fewer eigenvectors than requested, pad with zeros
            if eigenvectors.shape[1] < self.n_eigenvectors:
                padding = np.zeros((eigenvectors.shape[0], self.n_eigenvectors - eigenvectors.shape[1]))
                eigenvectors = np.hstack([eigenvectors, padding])
            
            return eigenvectors[:, :self.n_eigenvectors]
            
        except Exception as e:
            print(f"Warning: Could not compute eigenvectors, using degree-based features. Error: {e}")
            # Fallback to degree-based features
            degree_features = degrees.reshape(-1, 1)
            # Normalize degrees
            degree_features = (degree_features - degree_features.mean()) / (degree_features.std() + 1e-8)
            
            # Pad to required dimension
            if degree_features.shape[1] < self.n_eigenvectors:
                padding = np.zeros((degree_features.shape[0], self.n_eigenvectors - 1))
                degree_features = np.hstack([degree_features, padding])
            
            return degree_features


class GCNLinkPredictor(nn.Module):
    """Graph Convolutional Network for link prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.5):
        super(GCNLinkPredictor, self).__init__()
        
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
        
        # Final embedding dimension
        self.embedding_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get node embeddings.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation/dropout to last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_links(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict link probabilities using dot product of embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices to predict [2, num_edges]
            
        Returns:
            Link probabilities [num_edges]
        """
        row, col = edge_index
        return torch.sigmoid((embeddings[row] * embeddings[col]).sum(dim=-1))


class SBMLinkPrediction:
    """Main class for link prediction on Stochastic Block Model graphs."""
    
    def __init__(self, 
                 encoding_method: str = 'laplacian',
                 n_eigenvectors: int = 64,
                 gcn_hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.5,
                 learning_rate: float = 0.01,
                 epochs: int = 200,
                 device: str = 'auto'):
        """
        Initialize the SBM Link Prediction model.
        
        Args:
            encoding_method: 'laplacian' for Laplacian eigenvectors, 'identity' for identity features
            n_eigenvectors: Number of eigenvectors to use for Laplacian encoding
            gcn_hidden_dims: Hidden dimensions for GCN layers
            dropout: Dropout rate
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.encoding_method = encoding_method
        self.n_eigenvectors = n_eigenvectors
        self.gcn_hidden_dims = gcn_hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.encoder = None
        self.model = None
        self.optimizer = None
        
    def _prepare_features(self, adjacency_matrix: sp.csr_matrix) -> np.ndarray:
        """Prepare node features based on encoding method."""
        if self.encoding_method == 'laplacian':
            if self.encoder is None:
                self.encoder = LaplacianEigenvectorEncoder(self.n_eigenvectors)
            return self.encoder.fit_transform(adjacency_matrix)
        
        elif self.encoding_method == 'identity':
            # Use identity features (one-hot encoding of node IDs)
            n_nodes = adjacency_matrix.shape[0]
            return np.eye(n_nodes)
        
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
    
    def _create_pytorch_data(self, adjacency_matrix: sp.csr_matrix, 
                           test_ratio: float = 0.2) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """Convert scipy sparse matrix to PyTorch Geometric data with train/test split."""
        # Prepare node features
        features = self._prepare_features(adjacency_matrix)
        
        # Convert to edge list
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        
        # Create train/test split for edges
        num_edges = edge_index.shape[1]
        edge_indices = np.arange(num_edges)
        train_edges, test_edges = train_test_split(edge_indices, test_size=test_ratio, random_state=42)
        
        train_edge_index = edge_index[:, train_edges]
        test_edge_index = edge_index[:, test_edges]
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=train_edge_index,
            num_nodes=adjacency_matrix.shape[0]
        )
        
        return data, train_edge_index, test_edge_index
    
    def fit(self, adjacency_matrix: sp.csr_matrix, test_ratio: float = 0.2, verbose: bool = True):
        """
        Train the GCN model for link prediction.
        
        Args:
            adjacency_matrix: Input graph as sparse adjacency matrix
            test_ratio: Ratio of edges to use for testing
            verbose: Whether to print training progress
        """
        # Prepare data
        data, train_edge_index, test_edge_index = self._create_pytorch_data(adjacency_matrix, test_ratio)
        data = data.to(self.device)
        train_edge_index = train_edge_index.to(self.device)
        test_edge_index = test_edge_index.to(self.device)
        
        # Initialize model
        input_dim = data.x.shape[1]
        self.model = GCNLinkPredictor(input_dim, self.gcn_hidden_dims, self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Generate negative edges for training
        train_neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edge_index.shape[1]
        ).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # Get node embeddings
            embeddings = self.model(data.x, train_edge_index)
            
            # Predict positive and negative edges
            pos_pred = self.model.predict_links(embeddings, train_edge_index)
            neg_pred = self.model.predict_links(embeddings, train_neg_edge_index)
            
            # Create labels
            pos_labels = torch.ones(pos_pred.size(0), device=self.device)
            neg_labels = torch.zeros(neg_pred.size(0), device=self.device)
            
            # Compute loss
            loss = F.binary_cross_entropy(
                torch.cat([pos_pred, neg_pred]),
                torch.cat([pos_labels, neg_labels])
            )
            
            loss.backward()
            self.optimizer.step()
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}')
        
        # Store test data for evaluation
        self.test_data = data
        self.test_edge_index = test_edge_index
        self.train_edge_index = train_edge_index
    
    def evaluate(self) -> dict:
        """Evaluate the trained model on test edges."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = self.model(self.test_data.x, self.train_edge_index)
            
            # Generate negative test edges
            test_neg_edge_index = negative_sampling(
                edge_index=self.train_edge_index,
                num_nodes=self.test_data.num_nodes,
                num_neg_samples=self.test_edge_index.shape[1]
            ).to(self.device)
            
            # Predict test edges
            pos_pred = self.model.predict_links(embeddings, self.test_edge_index)
            neg_pred = self.model.predict_links(embeddings, test_neg_edge_index)
            
            # Combine predictions and labels
            y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
            y_true = np.concatenate([
                np.ones(pos_pred.size(0)),
                np.zeros(neg_pred.size(0))
            ])
            
            # Compute metrics
            auc_score = roc_auc_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred > 0.5)
            
            return {
                'auc': auc_score,
                'accuracy': accuracy,
                'n_test_edges': len(y_true) // 2
            }
    
    def predict_new_links(self, adjacency_matrix: sp.csr_matrix, 
                         node_pairs: Optional[List[Tuple[int, int]]] = None,
                         top_k: int = 100) -> List[Tuple[int, int, float]]:
        """
        Predict new links for a given graph.
        
        Args:
            adjacency_matrix: Input graph
            node_pairs: Specific node pairs to evaluate. If None, evaluates all possible pairs.
            top_k: Number of top predictions to return
            
        Returns:
            List of (node_i, node_j, probability) tuples sorted by probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Prepare features
        features = self._prepare_features(adjacency_matrix)
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        
        # Create data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            num_nodes=adjacency_matrix.shape[0]
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
            
            if node_pairs is None:
                # Generate all possible node pairs (excluding existing edges)
                existing_edges = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
                n_nodes = adjacency_matrix.shape[0]
                all_pairs = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)
                           if (i, j) not in existing_edges and (j, i) not in existing_edges]
                node_pairs = all_pairs
            
            # Predict for all pairs
            predictions = []
            for i, j in node_pairs:
                edge_to_predict = torch.tensor([[i], [j]], dtype=torch.long, device=self.device)
                prob = self.model.predict_links(embeddings, edge_to_predict).item()
                predictions.append((i, j, prob))
            
            # Sort by probability and return top_k
            predictions.sort(key=lambda x: x[2], reverse=True)
            return predictions[:top_k]


def demo_sbm_link_prediction():
    """Demonstration of GCN link prediction on SBM data."""
    from ..experiments.dynamic_simplesbm import sbm_dynamic_model_2
    
    print("Generating SBM data...")
    # Generate a simple SBM graph
    adjacency_all, labels_all = sbm_dynamic_model_2(
        N=100, k=3, pin=[0.3, 0.3, 0.3], pout=0.05, 
        p_switch=0.0, T=1, Totalsims=1, base_seed=42
    )
    
    # Use the first (and only) graph
    adj_matrix = sp.csr_matrix(adjacency_all[0, 0])
    
    print(f"Graph has {adj_matrix.shape[0]} nodes and {adj_matrix.nnz // 2} edges")
    
    # Test both encoding methods
    for encoding in ['laplacian', 'identity']:
        print(f"\n--- Testing with {encoding} encoding ---")
        
        # Initialize model
        model = SBMLinkPrediction(
            encoding_method=encoding,
            n_eigenvectors=32,
            gcn_hidden_dims=[64, 32],
            epochs=100,
            learning_rate=0.01
        )
        
        # Train model
        print("Training model...")
        model.fit(adj_matrix, test_ratio=0.2, verbose=False)
        
        # Evaluate
        results = model.evaluate()
        print(f"Results: AUC = {results['auc']:.4f}, Accuracy = {results['accuracy']:.4f}")
        
        # Predict top new links
        print("Top 5 predicted new links:")
        predictions = model.predict_new_links(adj_matrix, top_k=5)
        for i, j, prob in predictions:
            print(f"  Nodes {i}-{j}: {prob:.4f}")


if __name__ == "__main__":
    demo_sbm_link_prediction()