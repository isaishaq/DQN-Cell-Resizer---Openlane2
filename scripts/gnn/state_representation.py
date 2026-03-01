"""
State Representation for DRL-based Cell Sizing
==============================================

Provides multiple state representations:
1. Flat vector (for MLP)
2. Graph representation (for GNN)
3. Hierarchical features
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx


@dataclass
class CellFeatures:
    """Features for a single cell."""
    instance_name: str
    cell_type: str
    base_type: str
    drive_strength: int
    
    # Timing features
    delay: float
    input_slew: float
    output_slew: float
    input_cap: float
    output_cap: float
    
    # Connectivity features
    fanout: int
    fanin: int
    
    # Path features
    slack_contribution: float
    is_on_critical_path: bool
    num_paths_through: int
    
    # Position in path
    path_depth: int  # Distance from startpoint
    stages_to_endpoint: int


@dataclass
class PathFeatures:
    """Features for a timing path."""
    startpoint: str
    endpoint: str
    slack: float
    data_arrival_time: float
    data_required_time: float
    num_stages: int
    is_violated: bool
    cells: List[CellFeatures]


class StateRepresentation:
    """
    Extract state representation from timing data.
    
    Supports multiple representations:
    - Flat vector for MLP
    - Graph for GNN
    - Sequence for RNN
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize state representation.
        
        Args:
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        
        # Normalization constants
        self.norm_constants = {
            'clock_period': 16.0,
            'max_delay': 2.0,
            'max_slew': 2.0,
            'max_cap': 0.5,
            'max_fanout': 20.0,
            'max_drive_strength': 16.0,
            'max_slack': 10.0
        }
    
    def extract_flat_state(self, timing_data: Dict, 
                          actionable_cells: List,
                          top_k_cells: int = 10) -> np.ndarray:
        """
        Extract flat state vector for MLP-based DQN.
        
        State structure:
        - Global metrics: [WNS, TNS, num_violations, num_paths, avg_path_depth]
        - Top-K cell features: [drive_strength, fanout, delay, slew, cap, slack_contrib]
        
        Args:
            timing_data: Parsed timing report
            actionable_cells: List of actionable Cell objects
            top_k_cells: Number of cells to include
            
        Returns:
            State vector of shape (5 + top_k_cells * 6,)
        """
        metrics = timing_data['global_metrics']
        
        # Global features (5)
        global_features = [
            metrics['wns'] / self.norm_constants['clock_period'],
            metrics['tns'] / 100.0,
            metrics['num_violations'] / 100.0,
            metrics['num_paths'] / 1000.0,
            0.0  # Can add average path depth
        ]
        
        # Cell features (top_k_cells * 6)
        cell_features = []
        for i in range(top_k_cells):
            if i < len(actionable_cells):
                cell = actionable_cells[i]
                cell_features.extend([
                    cell.current_drive_strength / self.norm_constants['max_drive_strength'],
                    cell.fanout / self.norm_constants['max_fanout'],
                    cell.delay / self.norm_constants['max_delay'],
                    getattr(cell, 'slew', 0.0) / self.norm_constants['max_slew'],
                    getattr(cell, 'cap', 0.0) / self.norm_constants['max_cap'],
                    cell.slack_contribution / self.norm_constants['max_slack']
                ])
            else:
                # Pad with zeros
                cell_features.extend([0.0] * 6)
        
        state = np.array(global_features + cell_features, dtype=np.float32)
        return state
    
    def extract_graph_state(self, timing_data: Dict,
                           top_n_paths: int = 10) -> Tuple[nx.DiGraph, Dict]:
        """
        Extract graph representation for GNN.
        
        Creates a directed graph where:
        - Nodes = cells + I/O ports
        - Edges = timing arcs (connections between cells)
        - Node features = cell properties
        - Edge features = delay, slew, etc.
        
        Args:
            timing_data: Parsed timing report
            top_n_paths: Number of critical paths to include
            
        Returns:
            graph: NetworkX DiGraph
            node_features: Dict mapping node_id to feature vector
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Process paths to build graph
        paths = timing_data.get('paths', [])
        sorted_paths = sorted(paths, key=lambda p: p.get('slack', 0))
        critical_paths = sorted_paths[:top_n_paths]
        
        node_features = {}
        edge_features = {}
        cell_to_node_id = {}
        node_id_counter = 0
        
        # Add nodes and edges from critical paths
        for path_idx, path in enumerate(critical_paths):
            cells = path.get('cells', [])
            path_slack = path.get('slack', 0.0)
            
            prev_node_id = None
            for cell_idx, cell_data in enumerate(cells):
                instance = cell_data.get('instance_name', '')
                if not instance:
                    continue
                
                # Get or create node ID
                if instance not in cell_to_node_id:
                    node_id = node_id_counter
                    cell_to_node_id[instance] = node_id
                    node_id_counter += 1
                    
                    # Extract node features
                    node_features[node_id] = self._extract_node_features(
                        cell_data, path_slack, is_critical=True
                    )
                    
                    # Add node to graph with attributes
                    G.add_node(node_id,
                              instance=instance,
                              cell_type=cell_data.get('cell_type', ''),
                              drive_strength=cell_data.get('drive_strength', 1))
                else:
                    node_id = cell_to_node_id[instance]
                    # Update features if this path is more critical
                    existing_slack = node_features[node_id][0]
                    if path_slack < existing_slack:
                        node_features[node_id] = self._extract_node_features(
                            cell_data, path_slack, is_critical=True
                        )
                
                # Add edge from previous cell
                if prev_node_id is not None:
                    edge_key = (prev_node_id, node_id)
                    if not G.has_edge(prev_node_id, node_id):
                        # Add edge with features
                        edge_feature = self._extract_edge_features(cell_data)
                        G.add_edge(prev_node_id, node_id)
                        edge_features[edge_key] = edge_feature
                
                prev_node_id = node_id
        
        # Add global graph features as attributes
        G.graph['wns'] = timing_data['global_metrics']['wns']
        G.graph['tns'] = timing_data['global_metrics']['tns']
        G.graph['num_violations'] = timing_data['global_metrics']['num_violations']
        
        return G, node_features, edge_features
    
    def _extract_node_features(self, cell_data: Dict, 
                               path_slack: float,
                               is_critical: bool) -> np.ndarray:
        """
        Extract features for a graph node (cell).
        
        Features:
        - Timing: delay, slew, slack
        - Electrical: cap, fanout
        - Cell properties: drive strength, cell type encoding
        - Path properties: criticality, position
        """
        features = [
            # Timing features (normalized)
            path_slack / self.norm_constants['clock_period'],
            cell_data.get('delay', 0.0) / self.norm_constants['max_delay'],
            cell_data.get('slew', 0.0) / self.norm_constants['max_slew'],
            cell_data.get('time', 0.0) / self.norm_constants['clock_period'],
            
            # Electrical features
            cell_data.get('cap', 0.0) / self.norm_constants['max_cap'],
            cell_data.get('fanout', 0) / self.norm_constants['max_fanout'],
            
            # Cell properties
            cell_data.get('drive_strength', 1) / self.norm_constants['max_drive_strength'],
            
            # Boolean features
            1.0 if is_critical else 0.0,
            1.0 if cell_data.get('transition', '^') == '^' else -1.0,  # Rising/falling
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_edge_features(self, cell_data: Dict) -> np.ndarray:
        """
        Extract features for a graph edge (timing arc).
        
        Features:
        - Arc delay
        - Slew propagation
        - Capacitive load
        """
        features = [
            cell_data.get('delay', 0.0) / self.norm_constants['max_delay'],
            cell_data.get('slew', 0.0) / self.norm_constants['max_slew'],
            cell_data.get('cap', 0.0) / self.norm_constants['max_cap'],
        ]
        
        return np.array(features, dtype=np.float32)
    
    def convert_to_pyg_data(self, graph: nx.DiGraph,
                           node_features: Dict,
                           edge_features: Dict):
        """
        Convert NetworkX graph to PyTorch Geometric Data format.
        
        Args:
            graph: NetworkX graph
            node_features: Node feature dict
            edge_features: Edge feature dict
            
        Returns:
            PyG Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch Geometric not installed. "
                            "Install with: pip install torch-geometric")
        
        # Convert node features to tensor
        num_nodes = graph.number_of_nodes()
        node_feat_dim = len(next(iter(node_features.values())))
        x = torch.zeros(num_nodes, node_feat_dim, dtype=torch.float32)
        
        for node_id, features in node_features.items():
            x[node_id] = torch.tensor(features, dtype=torch.float32)
        
        # Convert edges to COO format
        edge_index = []
        edge_attr = []
        
        for (src, dst) in graph.edges():
            edge_index.append([src, dst])
            if (src, dst) in edge_features:
                edge_attr.append(edge_features[(src, dst)])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else None
        
        # Global features
        global_features = torch.tensor([
            graph.graph['wns'] / self.norm_constants['clock_period'],
            graph.graph['tns'] / 100.0,
            graph.graph['num_violations'] / 100.0
        ], dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            num_nodes=num_nodes
        )
        
        return data
    
    def extract_hierarchical_state(self, timing_data: Dict,
                                   top_n_paths: int = 10) -> Dict:
        """
        Extract hierarchical state representation.
        
        Hierarchy:
        - Design level: global metrics
        - Path level: per-path features
        - Cell level: per-cell features
        
        Useful for hierarchical RL or attention mechanisms.
        """
        design_features = self._extract_design_features(timing_data)
        
        paths = timing_data.get('paths', [])
        sorted_paths = sorted(paths, key=lambda p: p.get('slack', 0))
        critical_paths = sorted_paths[:top_n_paths]
        
        path_features = []
        cell_features_per_path = []
        
        for path in critical_paths:
            # Path-level features
            path_feat = self._extract_path_features(path)
            path_features.append(path_feat)
            
            # Cell-level features for this path
            cells_feat = []
            for cell_data in path.get('cells', []):
                cell_feat = self._extract_node_features(
                    cell_data, 
                    path.get('slack', 0.0),
                    is_critical=True
                )
                cells_feat.append(cell_feat)
            
            cell_features_per_path.append(cells_feat)
        
        return {
            'design': design_features,
            'paths': path_features,
            'cells': cell_features_per_path
        }
    
    def _extract_design_features(self, timing_data: Dict) -> np.ndarray:
        """Extract design-level features."""
        metrics = timing_data['global_metrics']
        return np.array([
            metrics['wns'] / self.norm_constants['clock_period'],
            metrics['tns'] / 100.0,
            metrics['num_violations'] / 100.0,
            metrics['num_paths'] / 1000.0
        ], dtype=np.float32)
    
    def _extract_path_features(self, path: Dict) -> np.ndarray:
        """Extract path-level features."""
        return np.array([
            path.get('slack', 0.0) / self.norm_constants['clock_period'],
            path.get('data_arrival_time', 0.0) / self.norm_constants['clock_period'],
            path.get('data_required_time', 0.0) / self.norm_constants['clock_period'],
            len(path.get('cells', [])) / 100.0,  # Normalized path length
        ], dtype=np.float32)


# ============================================================================
# Helper Functions
# ============================================================================

def visualize_timing_graph(graph: nx.DiGraph, 
                          node_features: Dict,
                          output_file: str = "timing_graph.png"):
    """
    Visualize timing graph for debugging.
    
    Args:
        graph: NetworkX graph
        node_features: Node features dict
        output_file: Output image file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping visualization")
        return
    
    # Color nodes by slack (criticality)
    node_colors = []
    for node in graph.nodes():
        if node in node_features:
            slack = node_features[node][0]  # First feature is slack
            # Red for negative (critical), green for positive
            color = 'red' if slack < 0 else 'green'
        else:
            color = 'gray'
        node_colors.append(color)
    
    # Draw graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=500, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True)
    
    # Add labels (truncate long names)
    labels = {}
    for node in graph.nodes():
        inst_name = graph.nodes[node].get('instance', str(node))
        labels[node] = inst_name[-15:] if len(inst_name) > 15 else inst_name
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    plt.title("Timing Graph (Red=Critical, Green=Non-critical)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[INFO] Graph visualization saved to: {output_file}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    import sys
    from timing_parser import parse_timing_report
    
    if len(sys.argv) < 2:
        print("Usage: python state_representation.py <timing_report.rpt>")
        sys.exit(1)
    
    report_file = sys.argv[1]
    
    print(f"[INFO] Parsing timing report: {report_file}")
    timing_data = parse_timing_report(report_file)
    
    print(f"\n[INFO] Global metrics:")
    print(f"  WNS: {timing_data['global_metrics']['wns']:.3f} ns")
    print(f"  TNS: {timing_data['global_metrics']['tns']:.3f} ns")
    print(f"  Violations: {timing_data['global_metrics']['num_violations']}")
    
    # Test flat state extraction
    print(f"\n[INFO] Testing flat state representation...")
    state_rep = StateRepresentation()
    
    # Mock actionable cells for demonstration
    from discrete_action_space import Cell
    mock_cells = [
        Cell("cell1", "buf_4", 4, 10, 0.5, -2.0),
        Cell("cell2", "nand2_2", 2, 5, 0.3, -1.5),
    ]
    
    flat_state = state_rep.extract_flat_state(timing_data, mock_cells, top_k_cells=10)
    print(f"  Flat state shape: {flat_state.shape}")
    print(f"  Flat state (first 10): {flat_state[:10]}")
    
    # Test graph state extraction
    print(f"\n[INFO] Testing graph state representation...")
    graph, node_features, edge_features = state_rep.extract_graph_state(
        timing_data, top_n_paths=5
    )
    
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Node feature dim: {len(next(iter(node_features.values())))}")
    if edge_features:
        print(f"  Edge feature dim: {len(next(iter(edge_features.values())))}")
    
    # Visualize graph
    print(f"\n[INFO] Visualizing graph...")
    visualize_timing_graph(graph, node_features, "timing_graph.png")
    
    # Test PyG conversion
    print(f"\n[INFO] Testing PyTorch Geometric conversion...")
    try:
        pyg_data = state_rep.convert_to_pyg_data(graph, node_features, edge_features)
        print(f"  PyG Data:")
        print(f"    x.shape: {pyg_data.x.shape}")
        print(f"    edge_index.shape: {pyg_data.edge_index.shape}")
        if pyg_data.edge_attr is not None:
            print(f"    edge_attr.shape: {pyg_data.edge_attr.shape}")
        print(f"    global_features: {pyg_data.global_features}")
        print(f"  ✓ PyG conversion successful!")
    except ImportError as e:
        print(f"  ✗ PyG not available: {e}")
    
    print(f"\n[INFO] State representation test complete!")
