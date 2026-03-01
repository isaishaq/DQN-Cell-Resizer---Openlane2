"""
Graph Neural Network (GNN) based DQN for Cell Sizing
====================================================

Uses GNN to process timing graph structure for better learning.

Why GNN?
--------
1. Timing paths are naturally graphs (cells connected by timing arcs)
2. GNN can capture structural dependencies between cells
3. Handles variable-sized paths elegantly
4. Can learn which cell modifications affect downstream timing
5. Better generalization across different designs

Architecture:
-------------
Input: Timing graph (nodes=cells, edges=timing arcs)
       ↓
   Graph Convolution Layers (message passing)
       ↓
   Global Pooling (aggregate node features)
       ↓
   MLP (map to Q-values)
       ↓
Output: Q(state, action) for each action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch Geometric not installed.")
    print("          Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


# ============================================================================
# GNN-based DQN Network
# ============================================================================

class GraphConvolutionDQN(nn.Module):
    """
    GNN-based Q-network using Graph Convolution.
    
    Architecture:
    1. Multiple GCN layers for message passing
    2. Global pooling to aggregate node information
    3. MLP head for Q-value prediction
    """
    
    def __init__(self,
                 node_feature_dim: int = 9,
                 edge_feature_dim: int = 3,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 num_actions: int = 30,
                 global_feature_dim: int = 3,
                 dropout: float = 0.1):
        """
        Initialize GNN-based DQN.
        
        Args:
            node_feature_dim: Dimension of node (cell) features
            edge_feature_dim: Dimension of edge (timing arc) features
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of graph convolution layers
            num_actions: Number of possible actions
            global_feature_dim: Dimension of global features (WNS, TNS, etc.)
            dropout: Dropout rate
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN")
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Input embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization for each GNN layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Global feature processing
        self.global_fc = nn.Linear(global_feature_dim, hidden_dim)
        
        # MLP head for Q-value prediction
        # Input: pooled graph features + global features
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - global_features: Global design features
                - batch: Batch assignment (for batched graphs)
        
        Returns:
            Q-values: [batch_size, num_actions]
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Graph convolution layers with skip connections
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            x_residual = x
            x = gnn_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Skip connection
            if i > 0:
                x = x + x_residual
        
        # Global pooling: aggregate node features to graph-level
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)    # [batch_size, hidden_dim]
        
        # Process global features
        global_feat = self.global_fc(data.global_features)
        global_feat = F.relu(global_feat)
        
        # Concatenate pooled features and global features
        graph_features = torch.cat([x_mean, x_max, global_feat], dim=1)
        
        # Q-value prediction
        q_values = self.q_network(graph_features)
        
        return q_values


class GraphAttentionDQN(nn.Module):
    """
    GNN-based Q-network using Graph Attention Networks (GAT).
    
    GAT learns attention weights for message passing,
    which can help identify critical cell dependencies.
    """
    
    def __init__(self,
                 node_feature_dim: int = 9,
                 edge_feature_dim: int = 3,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 num_actions: int = 30,
                 global_feature_dim: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize GAT-based DQN.
        
        Args:
            num_heads: Number of attention heads
            Other args same as GraphConvolutionDQN
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN")
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Input embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, 
                           heads=num_heads, dropout=dropout)
                )
            else:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                           heads=num_heads, dropout=dropout)
                )
        
        self.dropout = nn.Dropout(dropout)
        
        # Global feature processing
        self.global_fc = nn.Linear(global_feature_dim, hidden_dim)
        
        # Q-network head
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass (same interface as GraphConvolutionDQN)."""
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Node embedding
        x = self.node_embedding(x)
        x = F.elu(x)
        
        # Graph attention layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Global features
        global_feat = self.global_fc(data.global_features)
        global_feat = F.relu(global_feat)
        
        # Concatenate and predict Q-values
        graph_features = torch.cat([x_mean, x_max, global_feat], dim=1)
        q_values = self.q_network(graph_features)
        
        return q_values


# ============================================================================
# DQN Agent with GNN
# ============================================================================

class GNNDQNAgent:
    """
    DQN Agent using GNN for Q-function approximation.
    """
    
    def __init__(self,
                 node_feature_dim: int = 9,
                 edge_feature_dim: int = 3,
                 num_actions: int = 30,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 use_attention: bool = False,
                 device: str = 'cpu'):
        """
        Initialize GNN-DQN agent.
        
        Args:
            use_attention: If True, use GAT instead of GCN
            device: 'cpu' or 'cuda'
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        # Create Q-networks
        network_class = GraphAttentionDQN if use_attention else GraphConvolutionDQN
        
        self.q_network = network_class(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(device)
        
        # Target network for stable training
        self.target_network = network_class(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        print(f"[INFO] GNN-DQN Agent initialized")
        print(f"       Network type: {'GAT' if use_attention else 'GCN'}")
        print(f"       Device: {device}")
        print(f"       Parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def select_action(self, state: Data, valid_actions_mask: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: PyG Data object representing current state
            valid_actions_mask: Boolean tensor of valid actions
        
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < self.epsilon:
            # Random valid action
            valid_indices = torch.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                return valid_indices[torch.randint(len(valid_indices), (1,))].item()
            return 0
        
        # Greedy action from Q-network
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.q_network(state)
            
            # Mask invalid actions
            q_values = q_values.squeeze()
            q_values[~valid_actions_mask] = -float('inf')
            
            action = q_values.argmax().item()
        
        return action
    
    def train_step(self, batch_data: Dict) -> float:
        """
        Perform one training step.
        
        Args:
            batch_data: Dict with 'states', 'actions', 'rewards', 'next_states', 'dones'
        
        Returns:
            Loss value
        """
        # Unpack batch
        states = batch_data['states']  # List of PyG Data objects
        actions = torch.tensor(batch_data['actions'], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=self.device)
        next_states = batch_data['next_states']
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=self.device)
        
        # Batch graphs
        states_batch = Batch.from_data_list(states).to(self.device)
        next_states_batch = Batch.from_data_list(next_states).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states_batch)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states_batch)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"[INFO] Model saved to: {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[INFO] Model loaded from: {path}")


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Demonstrate GNN-DQN usage."""
    import torch
    from torch_geometric.data import Data
    
    print("\n" + "="*60)
    print("GNN-DQN Example Usage")
    print("="*60)
    
    # Create example graph state
    # Suppose we have a timing path with 5 cells
    num_nodes = 5
    node_features = torch.randn(num_nodes, 9)  # 9 node features
    
    # Edges: 0->1->2->3->4 (sequential path)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=torch.long)
    
    # Global features (WNS, TNS, violations)
    global_features = torch.tensor([-3.5, -50.0, 25.0])
    
    # Create PyG Data object
    state = Data(
        x=node_features,
        edge_index=edge_index,
        global_features=global_features
    )
    
    print(f"\nExample state:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Node features shape: {node_features.shape}")
    
    # Create GNN-DQN agent
    agent = GNNDQNAgent(
        node_feature_dim=9,
        num_actions=30,
        hidden_dim=64,
        use_attention=False
    )
    
    # Forward pass
    with torch.no_grad():
        q_values = agent.q_network(state)
    
    print(f"\nQ-values shape: {q_values.shape}")
    print(f"Q-values (first 5): {q_values[0, :5]}")
    
    # Select action
    valid_mask = torch.ones(30, dtype=torch.bool)
    action = agent.select_action(state, valid_mask)
    print(f"\nSelected action: {action}")
    
    print("\n" + "="*60)
    print("✓ GNN-DQN example complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    if TORCH_GEOMETRIC_AVAILABLE:
        example_usage()
    else:
        print("\n[ERROR] PyTorch Geometric not available.")
        print("Install with:")
        print("  pip install torch-geometric")
        print("  pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html")
