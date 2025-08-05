import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union, Dict
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MAPCanonicalizer:
    """
    Implements the Maximal Axis Projection (MAP) canonicalization from the paper:
    "Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding"
    """
    def __init__(self, c: float = 0.1):
        self.c = c

    def canonicalize_signs(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Applies MAP-sign to each eigenvector to resolve sign ambiguity.

        Args:
            eigenvectors: A numpy array of shape [num_nodes, n_eigenvectors].

        Returns:
            A new numpy array with canonicalized signs.
        """
        canonicalized_eigenvectors = eigenvectors.copy()
        for i in range(eigenvectors.shape[1]):
            u = canonicalized_eigenvectors[:, i]
            canonicalized_eigenvectors[:, i] = self._canonicalize_single_eigenvector(u)
        return canonicalized_eigenvectors

    def _canonicalize_single_eigenvector(self, u: np.ndarray) -> np.ndarray:
        """Applies MAP-sign to a single eigenvector."""
        # Normalize the eigenvector to be safe
        u = u / (np.linalg.norm(u) + 1e-9)
        
        # 1. Axis projection and grouping
        # For a single eigenvector, the projection of e_i is just u_i.
        proj_angles = np.abs(u)
        
        # Group by angle
        unique_angles = np.unique(proj_angles)[::-1] # descending
        
        # 2. Find non-orthogonal axis and canonize
        for angle in unique_angles:
            indices = np.where(proj_angles == angle)[0]
            
            # Create summary vector x_h
            x_h = np.zeros_like(u)
            x_h[indices] = 1
            
            # Add a small constant to break ties, as suggested in the paper
            x_h += self.c * np.ones_like(u)

            # Check for non-orthogonality
            dot_product = u.T @ x_h
            
            if np.abs(dot_product) > 1e-9:
                # Canonize the sign
                return u * np.sign(dot_product)
        
        # If all summary vectors are orthogonal, return original vector
        return u


class LaplacianEigenvectorEncoder:
    """Encodes node features using Laplacian eigenvectors."""
    
    def __init__(self, n_eigenvectors: int = 64, normalized: bool = True, canonicalize_sign: bool = False):
        self.n_eigenvectors = n_eigenvectors
        self.normalized = normalized
        self.canonicalize_sign = canonicalize_sign
        if self.canonicalize_sign:
            self.canonicalizer = MAPCanonicalizer()
        
    def fit_transform(self, adjacency_matrix: sp.csr_matrix) -> np.ndarray:
        """Compute Laplacian eigenvectors as node features."""
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        
        if self.normalized:
            degree_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
            laplacian = sp.eye(adjacency_matrix.shape[0]) - degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
        else:
            degree_matrix = sp.diags(degrees)
            laplacian = degree_matrix - adjacency_matrix
        
        try:
            n_eigs = min(self.n_eigenvectors + 1, adjacency_matrix.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigs, which='SM')
            eigenvectors = eigenvectors[:, 1:]
            
            # Apply sign canonicalization if enabled
            if self.canonicalize_sign:
                eigenvectors = self.canonicalizer.canonicalize_signs(eigenvectors)

            if eigenvectors.shape[1] < self.n_eigenvectors:
                padding = np.zeros((eigenvectors.shape[0], self.n_eigenvectors - eigenvectors.shape[1]))
                eigenvectors = np.hstack([eigenvectors, padding])
            
            return eigenvectors[:, :self.n_eigenvectors]
            
        except Exception as e:
            print(f"Warning: Could not compute eigenvectors, using degree-based features. Error: {e}")
            degree_features = degrees.reshape(-1, 1)
            degree_features = (degree_features - degree_features.mean()) / (degree_features.std() + 1e-8)
            
            if degree_features.shape[1] < self.n_eigenvectors:
                padding = np.zeros((degree_features.shape[0], self.n_eigenvectors - 1))
                degree_features = np.hstack([degree_features, padding])
            
            return degree_features


class GeodesicTemporalEncoder:
    """Encodes node features using geodesic smoothing across time."""
    
    def __init__(self, n_eigenvectors: int = 64, canonicalize_sign: bool = False):
        self.n_eigenvectors = n_eigenvectors
        self.canonicalize_sign = canonicalize_sign
        self.embeddings_sequence = None
        if self.canonicalize_sign:
            self.canonicalizer = MAPCanonicalizer()
        
    def fit_transform_sequence(self, adjacency_sequence: List[sp.csr_matrix]) -> List[np.ndarray]:
        """Compute geodesically smoothed embeddings for temporal sequence."""
        try:
            # Import the spectral geodesic smoothing function
            from .spectral_geodesic_smoothing import spectral_geodesic_smoothing
        except ImportError:
            try:
                from spectral_geodesic_smoothing import spectral_geodesic_smoothing
            except ImportError:
                print("Warning: Could not import spectral_geodesic_smoothing, falling back to individual Laplacian encodings")
                return self._fallback_encoding(adjacency_sequence)
        
        try:
            T = len(adjacency_sequence)
            num_nodes = adjacency_sequence[0].shape[0]
            
            print(f"Debug: Starting geodesic smoothing with T={T}, num_nodes={num_nodes}, ke={self.n_eigenvectors}")
            
            # Run spectral geodesic smoothing to get temporally smoothed embeddings
            embeddings_sequence = spectral_geodesic_smoothing(
                adjacency_sequence, 
                T=T, 
                num_nodes=num_nodes, 
                ke=self.n_eigenvectors,
                stable_communities=False,
                mode='simple-nsc',
                return_geo_embeddings_only=True,
            )
            
            print(f"Debug: Geodesic smoothing succeeded, got {len(embeddings_sequence)} embeddings")
            
            # Process embeddings to ensure consistent dimensionality and apply canonicalization
            processed_embeddings = []
            for t in range(T):
                embedding = embeddings_sequence[t]
                
                # Canonicalize signs if enabled
                if self.canonicalize_sign:
                    embedding = self.canonicalizer.canonicalize_signs(embedding)

                # Ensure we have the right number of dimensions
                if embedding.shape[1] < self.n_eigenvectors:
                    padding = np.zeros((embedding.shape[0], self.n_eigenvectors - embedding.shape[1]))
                    embedding = np.hstack([embedding, padding])
                elif embedding.shape[1] > self.n_eigenvectors:
                    embedding = embedding[:, :self.n_eigenvectors]
                
                processed_embeddings.append(embedding)
            
            self.embeddings_sequence = processed_embeddings
            return processed_embeddings
            
        except Exception as e:
            print(f"Warning: Geodesic smoothing failed ({e}), falling back to individual Laplacian encodings")
            return self._fallback_encoding(adjacency_sequence)
    
    def _fallback_encoding(self, adjacency_sequence: List[sp.csr_matrix]) -> List[np.ndarray]:
        """Fallback to individual Laplacian encodings if geodesic smoothing fails."""
        encoder = LaplacianEigenvectorEncoder(self.n_eigenvectors, canonicalize_sign=self.canonicalize_sign)
        embeddings = []
        for adj_matrix in adjacency_sequence:
            embedding = encoder.fit_transform(adj_matrix)
            embeddings.append(embedding)
        return embeddings
        
    def get_embedding_at_time(self, t: int) -> np.ndarray:
        """Get embedding for specific timestep."""
        if self.embeddings_sequence is None:
            raise ValueError("Must call fit_transform_sequence first")
        return self.embeddings_sequence[t]


class TemporalGCNCell(nn.Module):
    """Single temporal GCN cell with GRU-like update mechanism."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5, input_dropout: float = None):
        super(TemporalGCNCell, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_dropout = input_dropout if input_dropout is not None else dropout
        
        # GCN layers for current timestep
        self.gcn_input = GCNConv(input_dim, hidden_dim)
        self.gcn_hidden = GCNConv(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.layer_norm_hidden = nn.LayerNorm(hidden_dim)
        
        # Temporal update mechanism (GRU-like)
        self.update_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.candidate_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal GCN cell.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            hidden_state: Previous hidden state [num_nodes, hidden_dim]
            
        Returns:
            New hidden state [num_nodes, hidden_dim]
        """
        # Process current graph structure
        current_embedding = self.gcn_input(x, edge_index)
        current_embedding = self.layer_norm_input(current_embedding)
        current_embedding = F.relu(current_embedding)
        current_embedding = F.dropout(current_embedding, p=self.input_dropout, training=self.training)
        
        current_embedding = self.gcn_hidden(current_embedding, edge_index)
        current_embedding = self.layer_norm_hidden(current_embedding)
        
        # Initialize hidden state if None
        if hidden_state is None:
            hidden_state = torch.zeros_like(current_embedding)
        
        # Temporal update mechanism (GRU-like)
        combined = torch.cat([current_embedding, hidden_state], dim=1)
        
        # Update gate: how much to update the hidden state
        update = torch.sigmoid(self.update_gate(combined))
        
        # Reset gate: how much of the previous state to forget
        reset = torch.sigmoid(self.reset_gate(combined))
        
        # Candidate state: new information to potentially add
        reset_hidden = reset * hidden_state
        candidate_input = torch.cat([current_embedding, reset_hidden], dim=1)
        candidate = torch.tanh(self.candidate_gate(candidate_input))
        
        # Final hidden state: interpolation between previous state and candidate
        new_hidden = (1 - update) * hidden_state + update * candidate
        
        return new_hidden


class TemporalGCNLinkPredictor(nn.Module):
    """Temporal Graph Convolutional Network for dynamic link prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5, input_dropout: float = None):
        super(TemporalGCNLinkPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of temporal GCN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else hidden_dim
            # Use input_dropout for first layer, regular dropout for others
            cell_dropout = input_dropout if i == 0 and input_dropout is not None else dropout
            self.cells.append(TemporalGCNCell(cell_input_dim, hidden_dim, dropout, cell_dropout))
        
        # Final projection layer
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_sequence: List[torch.Tensor], edge_index_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through temporal sequence.
        
        Args:
            x_sequence: List of node features for each timestep
            edge_index_sequence: List of edge indices for each timestep
            
        Returns:
            List of node embeddings for each timestep
        """
        batch_size = len(x_sequence)
        
        # Initialize hidden states for each layer
        hidden_states = [None] * self.num_layers
        
        embeddings_sequence = []
        
        for t in range(batch_size):
            x_t = x_sequence[t]
            edge_index_t = edge_index_sequence[t]
            
            # Forward through each temporal layer
            layer_input = x_t
            for layer_idx, cell in enumerate(self.cells):
                hidden_states[layer_idx] = cell(layer_input, edge_index_t, hidden_states[layer_idx])
                layer_input = hidden_states[layer_idx]
            
            # Final embedding for timestep t
            embedding_t = self.output_projection(hidden_states[-1])
            embeddings_sequence.append(embedding_t)
        
        return embeddings_sequence
    
    def predict_links(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link probabilities using dot product of embeddings."""
        row, col = edge_index
        return torch.sigmoid((embeddings[row] * embeddings[col]).sum(dim=-1))


class StaticGCNLinkPredictor(nn.Module):
    """Static GCN for comparison (separate model per timestep)."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.5):
        super(StaticGCNLinkPredictor, self).__init__()
        
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
        
        self.embedding_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass to get node embeddings."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def predict_links(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link probabilities using dot product of embeddings."""
        row, col = edge_index
        return torch.sigmoid((embeddings[row] * embeddings[col]).sum(dim=-1))


class TemporalLinkPredictionExperiment:
    """Main experiment class for comparing temporal vs static approaches."""
    
    def __init__(self, 
                 encoding_type: str = "laplacian",  # "laplacian", "identity", "none", or "geodesic"
                 n_eigenvectors: int = 32,
                 canonicalize_sign: bool = False,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 input_dropout: float = None,
                 learning_rate: float = 0.01,
                 weight_decay: float = 0.0,
                 epochs: int = 200,
                 device: str = 'auto'):
        """
        Initialize experiment.
        
        Args:
            encoding_type: Type of node encoding ("laplacian", "identity", "none", or "geodesic")
            n_eigenvectors: Number of eigenvectors for encoding
            canonicalize_sign: Whether to apply sign canonicalization
            hidden_dim: Hidden dimension for temporal GCN
            num_layers: Number of temporal layers
            dropout: Dropout rate
            input_dropout: Dropout rate for first layer (defaults to dropout if None)
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            epochs: Training epochs
            device: Computing device
        """
        self.encoding_type = encoding_type
        self.n_eigenvectors = n_eigenvectors
        self.canonicalize_sign = canonicalize_sign
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.encoder = None
        self.geodesic_encoder = None
        if encoding_type == "laplacian":
            self.encoder = LaplacianEigenvectorEncoder(n_eigenvectors, canonicalize_sign=canonicalize_sign)
        elif encoding_type == "geodesic":
            self.geodesic_encoder = GeodesicTemporalEncoder(n_eigenvectors, canonicalize_sign=canonicalize_sign)
    
    def _prepare_features(self, adjacency_matrix: sp.csr_matrix) -> np.ndarray:
        """Prepare node features based on encoding method (for single timestep)."""
        if self.encoding_type == "laplacian":
            return self.encoder.fit_transform(adjacency_matrix)
        elif self.encoding_type == "identity":
            # Use identity features (one-hot encoding of node IDs)
            n_nodes = adjacency_matrix.shape[0]
            return np.eye(n_nodes)
        elif self.encoding_type == "geodesic":
            # For geodesic encoding, we need the full sequence - this should not be called directly
            # Instead, use _prepare_geodesic_features_sequence
            raise ValueError("For geodesic encoding, use _prepare_geodesic_features_sequence instead")
        else:  # encoding_type == "none"
            # Use minimal constant features (just a single dimension)
            n_nodes = adjacency_matrix.shape[0]
            return np.ones((n_nodes, 1))
    
    def _prepare_geodesic_features_sequence(self, adjacency_sequence: List[sp.csr_matrix]) -> List[np.ndarray]:
        """Prepare geodesic features for entire temporal sequence."""
        if self.encoding_type != "geodesic":
            raise ValueError("This method should only be called for geodesic encoding")
        
        return self.geodesic_encoder.fit_transform_sequence(adjacency_sequence)
    
    def _split_temporal_snapshots(self, adjacency_sequence: List[sp.csr_matrix]) -> Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[sp.csr_matrix]]:
        """Split temporal snapshots into train/val/test sets (70%/15%/15%)."""
        T = len(adjacency_sequence)
        
        # Calculate split indices
        train_end = int(0.7 * T)
        val_end = int(0.85 * T)
        
        # Ensure we have at least 1 snapshot in each split
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        val_end = min(T - 1, val_end)  # Ensure test has at least 1 snapshot
        
        train_snapshots = adjacency_sequence[:train_end]
        val_snapshots = adjacency_sequence[train_end:val_end]
        test_snapshots = adjacency_sequence[val_end:]
        
        print(f"Temporal split: {len(train_snapshots)} train, {len(val_snapshots)} val, {len(test_snapshots)} test snapshots")
        
        return train_snapshots, val_snapshots, test_snapshots
    
    def _create_temporal_data(self, adjacency_sequence: List[sp.csr_matrix]) -> List[Data]:
        """Convert sequence of adjacency matrices to temporal PyTorch data."""
        data_sequence = []
        
        # Handle geodesic encoding differently since it needs the full sequence
        if self.encoding_type == "geodesic":
            features_sequence = self._prepare_geodesic_features_sequence(adjacency_sequence)
            
            for i, adj_matrix in enumerate(adjacency_sequence):
                features = features_sequence[i]
                edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
                
                data = Data(
                    x=torch.tensor(features, dtype=torch.float),
                    edge_index=edge_index,
                    num_nodes=adj_matrix.shape[0]
                )
                
                data_sequence.append(data)
        else:
            # For other encoding types, process individually
            for adj_matrix in adjacency_sequence:
                features = self._prepare_features(adj_matrix)
                edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
                
                data = Data(
                    x=torch.tensor(features, dtype=torch.float),
                    edge_index=edge_index,
                    num_nodes=adj_matrix.shape[0]
                )
                
                data_sequence.append(data)
        
        return data_sequence
    
    def train_temporal_model(self, adjacency_sequence: List[sp.csr_matrix], 
                             verbose: bool = True) -> Dict:
        """Train temporal GCN model using temporal link forecasting."""
        print("Training Temporal GCN with Link Forecasting...")
        
        # Split snapshots temporally
        train_snapshots, val_snapshots, test_snapshots = self._split_temporal_snapshots(adjacency_sequence)
        
        # Prepare data for all snapshots
        all_data = self._create_temporal_data(adjacency_sequence)
        train_data = all_data[:len(train_snapshots)]
        val_data = all_data[len(train_snapshots):len(train_snapshots)+len(val_snapshots)]
        test_data = all_data[len(train_snapshots)+len(val_snapshots):]
        
        # Move to device
        for i in range(len(all_data)):
            all_data[i] = all_data[i].to(self.device)
        
        # Initialize model
        input_dim = all_data[0].x.shape[1]
        model = TemporalGCNLinkPredictor(input_dim, self.hidden_dim, self.num_layers, self.dropout, self.input_dropout).to(self.device)
        
        if verbose:
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Data device: {all_data[0].x.device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        best_model_state = None
        best_val_auc = 0.0
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass through entire training sequence to maintain temporal state
            x_sequence = [data.x for data in train_data]
            edge_index_sequence = [data.edge_index for data in train_data]
            embeddings_sequence = model(x_sequence, edge_index_sequence)
            
            epoch_loss = 0.0
            
            # Process training snapshots: predict t+1 from embeddings at t
            for t in range(len(train_data) - 1):
                current_embeddings = embeddings_sequence[t]
                next_data = train_data[t + 1]
                
                # Generate positive and negative edges for next snapshot
                pos_edges = next_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=next_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                # Predict links for next snapshot using current embeddings
                pos_pred = model.predict_links(current_embeddings, pos_edges)
                neg_pred = model.predict_links(current_embeddings, neg_edges)
                
                # Labels
                pos_labels = torch.ones(pos_pred.size(0), device=self.device)
                neg_labels = torch.zeros(neg_pred.size(0), device=self.device)
                
                # Loss for this step
                loss_t = F.binary_cross_entropy(
                    torch.cat([pos_pred, neg_pred]),
                    torch.cat([pos_labels, neg_labels])
                )
                epoch_loss += loss_t
            
            if epoch_loss > 0:
                epoch_loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss.item():.4f}')
            
            # Validation after each epoch
            if (epoch + 1) % 10 == 0:  # Validate every 10 epochs
                val_auc = self._validate_temporal_model(model, train_data, val_data)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict().copy()
                    if verbose:
                        print(f'New best validation AUC: {val_auc:.4f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        test_results = self._evaluate_temporal_model(model, train_data + val_data, test_data)
        
        return {
            'model': model,
            'best_val_auc': best_val_auc,
            'results': test_results,
            'train_snapshots': len(train_snapshots),
            'val_snapshots': len(val_snapshots),
            'test_snapshots': len(test_snapshots)
        }
    
    def _validate_temporal_model(self, model, train_data, val_data):
        """Validate temporal model on validation snapshots."""
        model.eval()
        with torch.no_grad():
            val_aucs = []
            
            # Process full sequence up to validation to get proper temporal state
            full_history = train_data + val_data
            x_sequence = [data.x for data in full_history]
            edge_index_sequence = [data.edge_index for data in full_history]
            embeddings_sequence = model(x_sequence, edge_index_sequence)
            
            # Evaluate each validation snapshot
            for t in range(len(val_data)):
                val_idx = len(train_data) + t
                if val_idx == 0:  # Skip if no history
                    continue
                
                # Use embedding from previous timestep to predict current
                prev_embeddings = embeddings_sequence[val_idx - 1]
                current_val_data = val_data[t]
                
                pos_edges = current_val_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=current_val_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                pos_pred = model.predict_links(prev_embeddings, pos_edges)
                neg_pred = model.predict_links(prev_embeddings, neg_edges)
                
                y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
                y_true = np.concatenate([
                    np.ones(pos_pred.size(0)),
                    np.zeros(neg_pred.size(0))
                ])
                
                if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(y_true, y_pred)
                    val_aucs.append(auc)
        
        model.train()
        return np.mean(val_aucs) if val_aucs else 0.5
    
    def _evaluate_temporal_model(self, model, history_data, test_data):
        """Evaluate temporal model on test snapshots."""
        model.eval()
        results = []
        
        with torch.no_grad():
            # Process full sequence to get proper temporal state
            full_sequence = history_data + test_data
            x_sequence = [data.x for data in full_sequence]
            edge_index_sequence = [data.edge_index for data in full_sequence]
            embeddings_sequence = model(x_sequence, edge_index_sequence)
            
            # Evaluate each test snapshot
            for t in range(len(test_data)):
                test_idx = len(history_data) + t
                if test_idx == 0:  # Skip if no history
                    continue
                
                # Use embedding from previous timestep to predict current
                prev_embeddings = embeddings_sequence[test_idx - 1]
                current_test_data = test_data[t]
                
                pos_edges = current_test_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=current_test_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                pos_pred = model.predict_links(prev_embeddings, pos_edges)
                neg_pred = model.predict_links(prev_embeddings, neg_edges)
                
                y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
                y_true = np.concatenate([
                    np.ones(pos_pred.size(0)),
                    np.zeros(neg_pred.size(0))
                ])
                
                if len(np.unique(y_true)) > 1:
                    auc_score = roc_auc_score(y_true, y_pred)
                    ap_score = average_precision_score(y_true, y_pred)
                    accuracy = accuracy_score(y_true, y_pred > 0.5)
                    
                    results.append({
                        'timestep': t,
                        'auc': auc_score,
                        'ap': ap_score,
                        'accuracy': accuracy,
                        'n_test_edges': pos_edges.shape[1]
                    })
        
        return results
    
    def train_static_models(self, adjacency_sequence: List[sp.csr_matrix],
                            verbose: bool = True) -> Dict:
        """Train static GCN models using temporal forecasting (no temporal dynamics)."""
        print("Training Static GCN models with Link Forecasting...")
        
        # Use same temporal split as temporal model
        train_snapshots, val_snapshots, test_snapshots = self._split_temporal_snapshots(adjacency_sequence)
        
        # Prepare data for all snapshots
        all_data = self._create_temporal_data(adjacency_sequence)
        train_data = all_data[:len(train_snapshots)]
        test_data = all_data[len(train_snapshots)+len(val_snapshots):]
        
        # Move to device
        for i in range(len(all_data)):
            all_data[i] = all_data[i].to(self.device)
        
        # Train individual static models for each training snapshot
        models = []
        for t in range(len(train_data)):
            if verbose:
                print(f"Training model for timestep {t}...")
            
            current_data = train_data[t]
            
            # Initialize model
            input_dim = current_data.x.shape[1]
            model = StaticGCNLinkPredictor(input_dim, [self.hidden_dim, self.hidden_dim//2], self.dropout).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            # Training loop - train on current snapshot to predict future snapshots
            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                
                embeddings = model(current_data.x, current_data.edge_index)
                
                # Train to predict multiple future snapshots if available
                epoch_loss = 0.0
                n_targets = 0
                
                # Predict next few snapshots from current embedding
                for future_t in range(t + 1, min(t + 3, len(train_data))):  # Predict up to 2 steps ahead
                    target_data = train_data[future_t]
                    
                    pos_edges = target_data.edge_index
                    neg_edges = negative_sampling(
                        edge_index=pos_edges,
                        num_nodes=target_data.num_nodes,
                        num_neg_samples=pos_edges.shape[1]
                    ).to(self.device)
                    
                    pos_pred = model.predict_links(embeddings, pos_edges)
                    neg_pred = model.predict_links(embeddings, neg_edges)
                    
                    pos_labels = torch.ones(pos_pred.size(0), device=self.device)
                    neg_labels = torch.zeros(neg_pred.size(0), device=self.device)
                    
                    loss = F.binary_cross_entropy(
                        torch.cat([pos_pred, neg_pred]),
                        torch.cat([pos_labels, neg_labels])
                    )
                    epoch_loss += loss
                    n_targets += 1
                
                if n_targets > 0:
                    epoch_loss = epoch_loss / n_targets
                    epoch_loss.backward()
                    optimizer.step()
            
            models.append(model)
        
        # Evaluation on test set - use the last (most recent) trained model for all predictions
        results = []
        if len(models) > 0:
            # Use the model trained on the most recent training snapshot 
            final_model = models[-1]
            final_train_data = train_data[-1]
            
            final_model.eval()
            with torch.no_grad():
                # Get embeddings from the final training snapshot
                base_embeddings = final_model(final_train_data.x, final_train_data.edge_index)
                
                for t in range(len(test_data)):
                    current_test_data = test_data[t]
                    
                    pos_edges = current_test_data.edge_index
                    neg_edges = negative_sampling(
                        edge_index=pos_edges,
                        num_nodes=current_test_data.num_nodes,
                        num_neg_samples=pos_edges.shape[1]
                    ).to(self.device)
                    
                    # Use the final model's embeddings to predict test links
                    pos_pred = final_model.predict_links(base_embeddings, pos_edges)
                    neg_pred = final_model.predict_links(base_embeddings, neg_edges)
                    
                    y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
                    y_true = np.concatenate([
                        np.ones(pos_pred.size(0)),
                        np.zeros(neg_pred.size(0))
                    ])
                    
                    if len(np.unique(y_true)) > 1:
                        auc_score = roc_auc_score(y_true, y_pred)
                        ap_score = average_precision_score(y_true, y_pred)
                        accuracy = accuracy_score(y_true, y_pred > 0.5)
                        
                        results.append({
                            'timestep': t,
                            'auc': auc_score,
                            'ap': ap_score,
                            'accuracy': accuracy,
                            'n_test_edges': pos_edges.shape[1]
                        })
        
        return {
            'models': models,
            'results': results,
            'train_snapshots': len(train_snapshots),
            'val_snapshots': len(val_snapshots),
            'test_snapshots': len(test_snapshots)
        }
    
    def run_comprehensive_experiment(self, adjacency_sequence: List[sp.csr_matrix], 
                                     verbose: bool = True, include_static: bool = True) -> Dict:
        """Run comprehensive comparison experiment."""
        print(f"Running experiment with {self.encoding_type.upper()} encoding...")
        
        # Train temporal model
        temporal_results = self.train_temporal_model(adjacency_sequence, verbose)
        
        # Train static models (optional)
        static_results = None
        if include_static:
            static_results = self.train_static_models(adjacency_sequence, verbose)
        else:
            print("Skipping static baseline training...")
        
        return {
            'temporal': temporal_results,
            'static': static_results,
            'experiment_config': {
                'encoding_type': self.encoding_type,
                'n_eigenvectors': self.n_eigenvectors,
                'canonicalize_sign': self.canonicalize_sign,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'include_static': include_static
            }
        }
    
    def plot_results(self, experiment_results: Dict, save_path: Optional[str] = None):
        """Plot comparison results."""
        temporal_results = experiment_results['temporal']['results']
        static_results = experiment_results['static']
        
        # Get encoding type for labels
        encoding_name = self.encoding_type.title()
        if self.canonicalize_sign:
            encoding_name += " (Canon.)"

        # Create comparison DataFrame
        temporal_df = pd.DataFrame(temporal_results)
        temporal_df['method'] = f'Temporal GCN ({encoding_name})'
        
        if static_results is not None:
            static_df = pd.DataFrame(static_results['results'])
            static_df['method'] = f'Static GCN ({encoding_name})'
            combined_df = pd.concat([temporal_df, static_df], ignore_index=True)
        else:
            combined_df = temporal_df
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC comparison
        sns.lineplot(data=combined_df, x='timestep', y='auc', hue='method', ax=axes[0,0])
        axes[0,0].set_title(f'AUC Score Over Time - {encoding_name} Encoding')
        axes[0,0].set_ylabel('AUC')
        
        # AP comparison
        sns.lineplot(data=combined_df, x='timestep', y='ap', hue='method', ax=axes[0,1])
        axes[0,1].set_title(f'Average Precision Over Time - {encoding_name} Encoding')
        axes[0,1].set_ylabel('Average Precision')
        
        # Accuracy comparison
        sns.lineplot(data=combined_df, x='timestep', y='accuracy', hue='method', ax=axes[1,0])
        axes[1,0].set_title(f'Accuracy Over Time - {encoding_name} Encoding')
        axes[1,0].set_ylabel('Accuracy')
        
        # Box plot comparison
        metrics_melted = combined_df.melt(id_vars=['method', 'timestep'], 
                                          value_vars=['auc', 'ap', 'accuracy'],
                                          var_name='metric', value_name='score')
        sns.boxplot(data=metrics_melted, x='metric', y='score', hue='method', ax=axes[1,1])
        axes[1,1].set_title(f'Overall Performance Distribution - {encoding_name} Encoding')
        axes[1,1].set_ylabel('Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"Encoding: {encoding_name}")
        
        temporal_mean = temporal_df[['auc', 'ap', 'accuracy']].mean()
        
        print(f"\nTemporal GCN Average Performance:")
        print(f"  AUC: {temporal_mean['auc']:.4f}")
        print(f"  AP:  {temporal_mean['ap']:.4f}")
        print(f"  Acc: {temporal_mean['accuracy']:.4f}")
        
        if static_results is not None:
            static_mean = static_df[['auc', 'ap', 'accuracy']].mean()
            
            print(f"\nStatic GCN Average Performance:")
            print(f"  AUC: {static_mean['auc']:.4f}")
            print(f"  AP:  {static_mean['ap']:.4f}")
            print(f"  Acc: {static_mean['accuracy']:.4f}")
            
            print(f"\nImprovement (Temporal - Static):")
            print(f"  AUC: {temporal_mean['auc'] - static_mean['auc']:+.4f}")
            print(f"  AP:  {temporal_mean['ap'] - static_mean['ap']:+.4f}")
            print(f"  Acc: {temporal_mean['accuracy'] - static_mean['accuracy']:+.4f}")
        else:
            print("\nStatic baselines were skipped.")
        
        return combined_df


def run_full_comparison_experiment():
    """Run full comparison experiment with temporal link forecasting."""
    try:
        from ..experiments.dynamic_simplesbm import sbm_dynamic_model_2
    except ImportError:
        # If running as script, use absolute import
        import sys
        import os
        # A simple workaround for the import error if the script is not in a package
        def sbm_dynamic_model_2(N, k, pin, pout, p_switch, Totalsims, T, base_seed, try_sparse):
            # This is a placeholder. In a real scenario, the actual sbm_dynamic_model_2 would be here.
            np.random.seed(base_seed)
            adj_sequence = []
            for _ in range(T):
                adj = sp.random(N, N, density=0.1, format='csr')
                adj_sequence.append(adj)
            return [adj_sequence], None
        #sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        #from spectraldcd.experiments.dynamic_simplesbm import sbm_dynamic_model_2
    
    print("Generating dynamic SBM data...")
    # Generate dynamic SBM data
    
    d = 100
    k = 4
    pin = [0.4]*k
    pout = 0.2
    p_switch = 0.01
    n_sims =1
    base_seed = 4
    T = 50
    
    # Learning hyperparameters
    n_eigenvectors = 8
    hidden_dim = 32
    num_layers = 2
    dropout = 0.4
    input_dropout = 0.1  # Dropout for first layer (set to None to use same as dropout)
    learning_rate = 1e-4
    weight_decay = 1e-5  # L2 regularization strength
    epochs = 500
    include_static = False  # Set to True to include static baselines (takes much longer)
    
    adjacency_all, _ = sbm_dynamic_model_2(
        N=d, k=k, pin=pin, pout=pout, p_switch=p_switch,
        Totalsims=n_sims, T=T, base_seed=base_seed, try_sparse=True,)
    
    print(f"Generated {len(adjacency_all[0])} snapshots with {adjacency_all[0][0].shape[0]} nodes each")
    print(f"SBM parameters: d={d}, k={k}, pin={pin}, pout={pout}, p_switch={p_switch}, n_sims={n_sims}, base_seed={base_seed}")
    print(f"Learning parameters: n_eigenvectors={n_eigenvectors}, hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, input_dropout={input_dropout}, learning_rate={learning_rate}, weight_decay={weight_decay}, epochs={epochs}, include_static={include_static}")
    
    
    
    # Convert to list of sparse matrices
    adjacency_sequence = [sp.csr_matrix(adj) for adj in adjacency_all[0]]
    
    print(f"Generated {len(adjacency_sequence)} timesteps with {adjacency_sequence[0].shape[0]} nodes")
    
    # Run experiments with laplacian and geodesic encoding methods
    results = {}
    
    for encoding_type in ["laplacian", "geodesic"]:
        for canonicalize in [False, True]:
            print(f"\n{'='*50}")
            print(f"Running experiment with {encoding_type.upper()} encoding and canonicalization={canonicalize}")
            print(f"{'='*50}")
            
            experiment = TemporalLinkPredictionExperiment(
                encoding_type=encoding_type,
                n_eigenvectors=n_eigenvectors,
                canonicalize_sign=canonicalize,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                input_dropout=input_dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs
            )
            
            experiment_results = experiment.run_comprehensive_experiment(
                adjacency_sequence, verbose=True, include_static=include_static
            )
            
            results[f"{encoding_type}_{'canon' if canonicalize else 'nocanon'}"] = experiment_results
            
            # Plot results
            experiment.plot_results(experiment_results, 
                                    save_path=f"temporal_gcn_comparison_{encoding_type}_{'canon' if canonicalize else 'nocanon'}.png")

    # Final summary of all experiments
    print(f"\n{'='*70}")
    print("FINAL SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")

    summary_data = []
    for key, res in results.items():
        encoding, canon_str = key.split('_')
        canon = "Canon" if canon_str == "canon" else "No Canon"
        
        temporal_mean = pd.DataFrame(res['temporal']['results'])[['auc', 'ap', 'accuracy']].mean()
        summary_data.append({
            "Encoding": encoding.title(),
            "Canonicalization": canon,
            "AUC": temporal_mean['auc'],
            "AP": temporal_mean['ap'],
            "Accuracy": temporal_mean['accuracy']
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    return results


if __name__ == "__main__":
    # Run the comprehensive experiment
    results = run_full_comparison_experiment()