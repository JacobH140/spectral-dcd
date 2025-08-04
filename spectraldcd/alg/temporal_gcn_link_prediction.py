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


class LaplacianEigenvectorEncoder:
    """Encodes node features using Laplacian eigenvectors."""
    
    def __init__(self, n_eigenvectors: int = 64, normalized: bool = True):
        self.n_eigenvectors = n_eigenvectors
        self.normalized = normalized
        
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


class TemporalGCNCell(nn.Module):
    """Single temporal GCN cell with GRU-like update mechanism."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5):
        super(TemporalGCNCell, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # GCN layers for current timestep
        self.gcn_input = GCNConv(input_dim, hidden_dim)
        self.gcn_hidden = GCNConv(hidden_dim, hidden_dim)
        
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
        current_embedding = F.relu(current_embedding)
        current_embedding = F.dropout(current_embedding, p=self.dropout, training=self.training)
        
        current_embedding = self.gcn_hidden(current_embedding, edge_index)
        
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
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5):
        super(TemporalGCNLinkPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of temporal GCN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(TemporalGCNCell(cell_input_dim, hidden_dim, dropout))
        
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
                 encoding_type: str = "laplacian",  # "laplacian", "identity", or "none"
                 n_eigenvectors: int = 32,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 learning_rate: float = 0.01,
                 epochs: int = 200,
                 device: str = 'auto'):
        """
        Initialize experiment.
        
        Args:
            encoding_type: Type of node encoding ("laplacian", "identity", or "none")
            n_eigenvectors: Number of eigenvectors for encoding
            hidden_dim: Hidden dimension for temporal GCN
            num_layers: Number of temporal layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            device: Computing device
        """
        self.encoding_type = encoding_type
        self.n_eigenvectors = n_eigenvectors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.encoder = None
        if encoding_type == "laplacian":
            self.encoder = LaplacianEigenvectorEncoder(n_eigenvectors)
    
    def _prepare_features(self, adjacency_matrix: sp.csr_matrix) -> np.ndarray:
        """Prepare node features based on encoding method."""
        if self.encoding_type == "laplacian":
            return self.encoder.fit_transform(adjacency_matrix)
        elif self.encoding_type == "identity":
            # Use identity features (one-hot encoding of node IDs)
            n_nodes = adjacency_matrix.shape[0]
            return np.eye(n_nodes)
        else:  # encoding_type == "none"
            # Use minimal constant features (just a single dimension)
            n_nodes = adjacency_matrix.shape[0]
            return np.ones((n_nodes, 1))
    
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
        for data in all_data:
            data = data.to(self.device)
        
        # Initialize model
        input_dim = all_data[0].x.shape[1]
        model = TemporalGCNLinkPredictor(input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        best_model_state = None
        best_val_auc = 0.0
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            epoch_loss = 0.0
            
            # Process training snapshots: predict t+1 from t
            for t in range(len(train_data) - 1):
                current_data = train_data[t]
                next_data = train_data[t + 1]
                
                # Forward pass through current snapshot to get embedding
                embeddings = model([current_data.x], [current_data.edge_index])[0]
                
                # Generate positive and negative edges for next snapshot
                pos_edges = next_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=next_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                # Predict links for next snapshot
                pos_pred = model.predict_links(embeddings, pos_edges)
                neg_pred = model.predict_links(embeddings, neg_edges)
                
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
            # Warm up model with training data
            all_warmup_data = train_data
            
            val_aucs = []
            # Predict each validation snapshot from the state after processing all previous data
            for t in range(len(val_data)):
                # Process all data up to current validation snapshot
                history_data = all_warmup_data + val_data[:t]
                
                if len(history_data) == 0:
                    continue
                    
                # Get embedding from last historical snapshot
                last_data = history_data[-1]
                embeddings = model([last_data.x], [last_data.edge_index])[0]
                
                # Predict current validation snapshot
                current_val_data = val_data[t]
                pos_edges = current_val_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=current_val_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                pos_pred = model.predict_links(embeddings, pos_edges)
                neg_pred = model.predict_links(embeddings, neg_edges)
                
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
            for t in range(len(test_data)):
                # Process all historical data + test data up to t-1
                all_history = history_data + test_data[:t]
                
                if len(all_history) == 0:
                    continue
                
                # Get embedding from last historical snapshot
                last_data = all_history[-1]
                embeddings = model([last_data.x], [last_data.edge_index])[0]
                
                # Predict current test snapshot
                current_test_data = test_data[t]
                pos_edges = current_test_data.edge_index
                neg_edges = negative_sampling(
                    edge_index=pos_edges,
                    num_nodes=current_test_data.num_nodes,
                    num_neg_samples=pos_edges.shape[1]
                ).to(self.device)
                
                pos_pred = model.predict_links(embeddings, pos_edges)
                neg_pred = model.predict_links(embeddings, neg_edges)
                
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
                          test_ratio: float = 0.2, verbose: bool = True) -> Dict:
        """Train separate static GCN models for each timestep."""
        print("Training Static GCN models...")
        
        models = []
        results = []
        all_losses = []
        
        for t, adj_matrix in enumerate(adjacency_sequence):
            if verbose:
                print(f"Training model for timestep {t}...")
            
            # Prepare data for this timestep
            features = self._prepare_features(adj_matrix)
            edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
            
            # Train/test split
            num_edges = edge_index.shape[1]
            edge_indices = np.arange(num_edges)
            train_edges, test_edges = train_test_split(edge_indices, test_size=test_ratio, random_state=42)
            
            train_edge_index = edge_index[:, train_edges].to(self.device)
            test_edge_index = edge_index[:, test_edges].to(self.device)
            
            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=train_edge_index,
                num_nodes=adj_matrix.shape[0]
            ).to(self.device)
            
            # Initialize model
            input_dim = features.shape[1]
            model = StaticGCNLinkPredictor(input_dim, [self.hidden_dim, self.hidden_dim//2], self.dropout).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            model.train()
            losses = []
            
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                
                embeddings = model(data.x, data.edge_index)
                
                # Generate negative edges
                neg_edge_index = negative_sampling(
                    edge_index=train_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=train_edge_index.shape[1]
                ).to(self.device)
                
                # Predictions
                pos_pred = model.predict_links(embeddings, train_edge_index)
                neg_pred = model.predict_links(embeddings, neg_edge_index)
                
                # Labels
                pos_labels = torch.ones(pos_pred.size(0), device=self.device)
                neg_labels = torch.zeros(neg_pred.size(0), device=self.device)
                
                # Loss
                loss = F.binary_cross_entropy(
                    torch.cat([pos_pred, neg_pred]),
                    torch.cat([pos_labels, neg_labels])
                )
                
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            all_losses.append(losses)
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                embeddings = model(data.x, data.edge_index)
                
                test_neg_edge_index = negative_sampling(
                    edge_index=train_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=test_edge_index.shape[1]
                ).to(self.device)
                
                pos_pred = model.predict_links(embeddings, test_edge_index)
                neg_pred = model.predict_links(embeddings, test_neg_edge_index)
                
                y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
                y_true = np.concatenate([
                    np.ones(pos_pred.size(0)),
                    np.zeros(neg_pred.size(0))
                ])
                
                auc_score = roc_auc_score(y_true, y_pred)
                ap_score = average_precision_score(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred > 0.5)
                
                results.append({
                    'timestep': t,
                    'auc': auc_score,
                    'ap': ap_score,
                    'accuracy': accuracy,
                    'n_test_edges': len(y_true) // 2
                })
            
            models.append(model)
        
        return {
            'models': models,
            'results': results,
            'losses': all_losses
        }
    
    def run_comprehensive_experiment(self, adjacency_sequence: List[sp.csr_matrix], 
                                   test_ratio: float = 0.2, verbose: bool = True) -> Dict:
        """Run comprehensive comparison experiment."""
        print(f"Running experiment with {self.encoding_type.upper()} encoding...")
        
        # Train temporal model
        temporal_results = self.train_temporal_model(adjacency_sequence, test_ratio, verbose)
        
        # Train static models
        static_results = self.train_static_models(adjacency_sequence, test_ratio, verbose)
        
        return {
            'temporal': temporal_results,
            'static': static_results,
            'experiment_config': {
                'encoding_type': self.encoding_type,
                'n_eigenvectors': self.n_eigenvectors,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'test_ratio': test_ratio
            }
        }
    
    def plot_results(self, experiment_results: Dict, save_path: Optional[str] = None):
        """Plot comparison results."""
        temporal_results = experiment_results['temporal']['results']
        static_results = experiment_results['static']['results']
        
        # Create comparison DataFrame
        temporal_df = pd.DataFrame(temporal_results)
        temporal_df['method'] = 'Temporal GCN'
        
        static_df = pd.DataFrame(static_results)
        static_df['method'] = 'Static GCN'
        
        combined_df = pd.concat([temporal_df, static_df], ignore_index=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC comparison
        sns.lineplot(data=combined_df, x='timestep', y='auc', hue='method', ax=axes[0,0])
        axes[0,0].set_title('AUC Score Over Time')
        axes[0,0].set_ylabel('AUC')
        
        # AP comparison
        sns.lineplot(data=combined_df, x='timestep', y='ap', hue='method', ax=axes[0,1])
        axes[0,1].set_title('Average Precision Over Time')
        axes[0,1].set_ylabel('Average Precision')
        
        # Accuracy comparison
        sns.lineplot(data=combined_df, x='timestep', y='accuracy', hue='method', ax=axes[1,0])
        axes[1,0].set_title('Accuracy Over Time')
        axes[1,0].set_ylabel('Accuracy')
        
        # Box plot comparison
        metrics_melted = combined_df.melt(id_vars=['method', 'timestep'], 
                                        value_vars=['auc', 'ap', 'accuracy'],
                                        var_name='metric', value_name='score')
        sns.boxplot(data=metrics_melted, x='metric', y='score', hue='method', ax=axes[1,1])
        axes[1,1].set_title('Overall Performance Distribution')
        axes[1,1].set_ylabel('Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print("\n=== EXPERIMENT SUMMARY ===")
        encoding_type = experiment_results['experiment_config']['encoding_type'].title()
        print(f"Encoding: {encoding_type}")
        
        temporal_mean = temporal_df[['auc', 'ap', 'accuracy']].mean()
        static_mean = static_df[['auc', 'ap', 'accuracy']].mean()
        
        print(f"\nTemporal GCN Average Performance:")
        print(f"  AUC: {temporal_mean['auc']:.4f}")
        print(f"  AP:  {temporal_mean['ap']:.4f}")
        print(f"  Acc: {temporal_mean['accuracy']:.4f}")
        
        print(f"\nStatic GCN Average Performance:")
        print(f"  AUC: {static_mean['auc']:.4f}")
        print(f"  AP:  {static_mean['ap']:.4f}")
        print(f"  Acc: {static_mean['accuracy']:.4f}")
        
        print(f"\nImprovement (Temporal - Static):")
        print(f"  AUC: {temporal_mean['auc'] - static_mean['auc']:+.4f}")
        print(f"  AP:  {temporal_mean['ap'] - static_mean['ap']:+.4f}")
        print(f"  Acc: {temporal_mean['accuracy'] - static_mean['accuracy']:+.4f}")
        
        return combined_df


def run_full_comparison_experiment():
    """Run full comparison experiment with both encoding methods."""
    try:
        from ..experiments.dynamic_simplesbm import sbm_dynamic_model_2
    except ImportError:
        # If running as script, use absolute import
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from spectraldcd.experiments.dynamic_simplesbm import sbm_dynamic_model_2
    
    print("Generating dynamic SBM data...")
    # Generate dynamic SBM data
    d = 100
    k = 2
    pin = [0.3] * k
    pout = 0.2
    p_switch = 0.01
    T = 50
    n_sims = 1
    base_seed = 4

    adjacency_all, labels_all = sbm_dynamic_model_2(
        N=d, k=k, pin=pin, pout=pout, 
        p_switch=p_switch, T=T, Totalsims=n_sims, base_seed=base_seed, try_sparse=True
    )
    
    # Convert to list of sparse matrices
    adjacency_sequence = [sp.csr_matrix(adj) for adj in adjacency_all[0]]
    
    print(f"Generated {len(adjacency_sequence)} timesteps with {adjacency_sequence[0].shape[0]} nodes")
    
    # Run experiments with all three encoding methods
    results = {}
    
    for encoding_type in ["laplacian", "identity", "none"]:
        print(f"\n{'='*50}")
        print(f"Running experiment with {encoding_type.upper()} encoding")
        print(f"{'='*50}")
        
        experiment = TemporalLinkPredictionExperiment(
            encoding_type=encoding_type,
            n_eigenvectors=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3,
            learning_rate=0.01,
            epochs=150
        )
        
        experiment_results = experiment.run_comprehensive_experiment(
            adjacency_sequence, test_ratio=0.2, verbose=True
        )
        
        results[encoding_type] = experiment_results
        
        # Plot results
        experiment.plot_results(experiment_results, 
                              save_path=f"temporal_gcn_comparison_{encoding_type}.png")
    
    # Compare encoding methods - organize by encoding type
    print(f"\n{'='*70}")
    print("TEMPORAL vs STATIC COMPARISON FOR EACH ENCODING TYPE")
    print(f"{'='*70}")
    
    # Collect all results
    all_results = {}
    for encoding_type in ["laplacian", "identity", "none"]:
        all_results[encoding_type] = {}
        for method in ['temporal', 'static']:
            df = pd.DataFrame(results[encoding_type][method]['results'])
            all_results[encoding_type][method] = df[['auc', 'ap', 'accuracy']].mean()
    
    # Show comparison for each encoding type
    for encoding_type in ["laplacian", "identity", "none"]:
        print(f"\n{encoding_type.upper()} ENCODING COMPARISON:")
        print(f"{'-'*40}")
        temporal_results = all_results[encoding_type]['temporal']
        static_results = all_results[encoding_type]['static']
        
        print(f"  Temporal GCN: AUC={temporal_results['auc']:.4f}, AP={temporal_results['ap']:.4f}, Acc={temporal_results['accuracy']:.4f}")
        print(f"  Static GCN:   AUC={static_results['auc']:.4f}, AP={static_results['ap']:.4f}, Acc={static_results['accuracy']:.4f}")
        print(f"  Improvement:  AUC={temporal_results['auc']-static_results['auc']:+.4f}, AP={temporal_results['ap']-static_results['ap']:+.4f}, Acc={temporal_results['accuracy']-static_results['accuracy']:+.4f}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE - ALL RESULTS")
    print(f"{'='*70}")
    print(f"{'Encoding':<12} {'Method':<8} {'AUC':<8} {'AP':<8} {'Accuracy':<8}")
    print(f"{'-'*50}")
    for encoding_type in ["laplacian", "identity", "none"]:
        for method in ['temporal', 'static']:
            results_data = all_results[encoding_type][method]
            print(f"{encoding_type.title():<12} {method.title():<8} {results_data['auc']:<8.4f} {results_data['ap']:<8.4f} {results_data['accuracy']:<8.4f}")
        print(f"{'-'*50}")
    
    # Show encoding benefits relative to "none" baseline
    print(f"\nENCODING BENEFITS (vs None baseline):")
    print(f"{'-'*50}")
    none_temporal = all_results["none"]["temporal"]
    none_static = all_results["none"]["static"]
    
    for encoding_type in ["laplacian", "identity"]:
        temporal_benefit = all_results[encoding_type]["temporal"]
        static_benefit = all_results[encoding_type]["static"]
        print(f"{encoding_type.title()} Temporal: AUC={temporal_benefit['auc']-none_temporal['auc']:+.4f}, AP={temporal_benefit['ap']-none_temporal['ap']:+.4f}, Acc={temporal_benefit['accuracy']-none_temporal['accuracy']:+.4f}")
        print(f"{encoding_type.title()} Static:   AUC={static_benefit['auc']-none_static['auc']:+.4f}, AP={static_benefit['ap']-none_static['ap']:+.4f}, Acc={static_benefit['accuracy']-none_static['accuracy']:+.4f}")
        print()
    
    return results


if __name__ == "__main__":
    # Run the comprehensive experiment
    results = run_full_comparison_experiment()