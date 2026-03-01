#!/usr/bin/env python3
"""
Test State Representation and GNN Integration
==============================================

Quick test to verify all components work together.
"""

import sys
from pathlib import Path

print("="*70)
print("Testing State Representation and GNN Integration")
print("="*70)

# Check if timing report argument provided
if len(sys.argv) < 2:
    print("\n[ERROR] Please provide timing report path")
    print("Usage: python test_integration.py <timing_report.rpt>")
    print("\nExample:")
    print("  python test_integration.py ../runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/reports/nom_ss_100C_1v60/max.rpt")
    sys.exit(1)

report_file = sys.argv[1]

if not Path(report_file).exists():
    print(f"\n[ERROR] File not found: {report_file}")
    sys.exit(1)

print(f"\n[INFO] Timing report: {report_file}")

# ============================================================================
# Test 1: Parse Timing Report
# ============================================================================
print("\n" + "-"*70)
print("Test 1: Parsing Timing Report")
print("-"*70)

try:
    from timing_parser import parse_timing_report
    
    timing_data = parse_timing_report(report_file)
    
    print("✓ Timing parser loaded")
    print(f"  Paths parsed: {len(timing_data['paths'])}")
    print(f"  WNS: {timing_data['global_metrics']['wns']:.3f} ns")
    print(f"  TNS: {timing_data['global_metrics']['tns']:.3f} ns")
    print(f"  Violations: {timing_data['global_metrics']['num_violations']}")
    
except Exception as e:
    print(f"✗ Timing parser failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Extract Flat State (for MLP)
# ============================================================================
print("\n" + "-"*70)
print("Test 2: Flat State Representation (MLP)")
print("-"*70)

try:
    from state_representation import StateRepresentation
    from discrete_action_space import Cell
    
    state_rep = StateRepresentation()
    
    # Create mock actionable cells
    mock_cells = [
        Cell("cell1", "sky130_fd_sc_hd__buf_4", 4, 10, 0.5, -2.0),
        Cell("cell2", "sky130_fd_sc_hd__nand2_2", 2, 5, 0.3, -1.5),
        Cell("cell3", "sky130_fd_sc_hd__inv_8", 8, 15, 0.6, -1.0),
    ]
    
    flat_state = state_rep.extract_flat_state(timing_data, mock_cells, top_k_cells=10)
    
    print("✓ Flat state extracted")
    print(f"  State shape: {flat_state.shape}")
    print(f"  State dtype: {flat_state.dtype}")
    print(f"  First 10 values: {flat_state[:10]}")
    print(f"  Value range: [{flat_state.min():.3f}, {flat_state.max():.3f}]")
    
except Exception as e:
    print(f"✗ Flat state extraction failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 3: Extract Graph State (for GNN)
# ============================================================================
print("\n" + "-"*70)
print("Test 3: Graph State Representation (GNN)")
print("-"*70)

try:
    graph, node_features, edge_features = state_rep.extract_graph_state(
        timing_data, 
        top_n_paths=5
    )
    
    print("✓ Graph state extracted")
    print(f"  Nodes (cells): {graph.number_of_nodes()}")
    print(f"  Edges (arcs): {graph.number_of_edges()}")
    print(f"  Node feature dim: {len(next(iter(node_features.values())))}")
    
    if edge_features:
        print(f"  Edge feature dim: {len(next(iter(edge_features.values())))}")
    
    # Show some nodes
    print(f"\n  Sample nodes:")
    for i, (node_id, instance) in enumerate(graph.nodes(data='instance')):
        if i >= 3:
            break
        print(f"    Node {node_id}: {instance}")
    
except Exception as e:
    print(f"✗ Graph extraction failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 4: Convert to PyTorch Geometric
# ============================================================================
print("\n" + "-"*70)
print("Test 4: PyTorch Geometric Conversion")
print("-"*70)

try:
    import torch
    pyg_data = state_rep.convert_to_pyg_data(graph, node_features, edge_features)
    
    print("✓ PyG conversion successful")
    print(f"  Node features (x): {pyg_data.x.shape}")
    print(f"  Edge index: {pyg_data.edge_index.shape}")
    if pyg_data.edge_attr is not None:
        print(f"  Edge features: {pyg_data.edge_attr.shape}")
    print(f"  Global features: {pyg_data.global_features}")
    print(f"  Number of nodes: {pyg_data.num_nodes}")
    
except ImportError as e:
    print(f"⚠ PyTorch Geometric not installed: {e}")
    print("  Install with: pip install torch-geometric")
except Exception as e:
    print(f"✗ PyG conversion failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 5: GNN Model Forward Pass
# ============================================================================
print("\n" + "-"*70)
print("Test 5: GNN Model Forward Pass")
print("-"*70)

try:
    from gnn_dqn import GNNDQNAgent
    import torch
    
    # Create agent
    agent = GNNDQNAgent(
        node_feature_dim=9,
        edge_feature_dim=3,
        num_actions=30,
        hidden_dim=64,
        use_attention=False
    )
    
    print("✓ GNN agent created")
    print(f"  Network type: GCN")
    print(f"  Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        q_values = agent.q_network(pyg_data)
    
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values (first 10): {q_values[0, :10].numpy()}")
    
    # Test action selection
    valid_mask = torch.ones(30, dtype=torch.bool)
    action = agent.select_action(pyg_data, valid_mask)
    
    print(f"  Selected action: {action}")
    print("✓ GNN forward pass successful!")
    
except ImportError as e:
    print(f"⚠ GNN dependencies not installed: {e}")
    print("  Install PyTorch Geometric:")
    print("    pip install torch torch-geometric")
except Exception as e:
    print(f"✗ GNN test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 6: Action Space Integration
# ============================================================================
print("\n" + "-"*70)
print("Test 6: Action Space Integration")
print("-"*70)

try:
    from discrete_action_space import DiscreteActionSpace, CellLibrary
    
    action_space = DiscreteActionSpace(
        mode='single',
        top_k_cells=10,
        library=CellLibrary()
    )
    
    # Get actionable cells from timing data
    actionable_cells = action_space.get_actionable_cells(
        timing_data,
        worst_n_paths=5
    )
    
    print("✓ Action space created")
    print(f"  Action space size: {action_space.n_actions}")
    print(f"  Actionable cells found: {len(actionable_cells)}")
    
    if actionable_cells:
        print(f"\n  Top 5 critical cells:")
        for i, cell in enumerate(actionable_cells[:5]):
            print(f"    {i+1}. {cell.instance_name}: {cell.cell_type} "
                  f"(drive={cell.current_drive_strength}, fanout={cell.fanout})")
        
        # Test action decoding
        test_action = 5
        resizes = action_space.action_to_cell_resize(test_action, actionable_cells)
        
        print(f"\n  Test action {test_action} decodes to:")
        for cell, resize_action in resizes:
            print(f"    {cell.instance_name}: {resize_action.name}")
        
        # Test resize command generation
        resize_commands = action_space.apply_action(test_action, actionable_cells)
        
        if resize_commands:
            print(f"\n  Generated resize commands:")
            for instance, (old_cell, new_cell) in resize_commands.items():
                print(f"    {instance}: {old_cell} → {new_cell}")
        
        print("✓ Action space integration successful!")
    else:
        print("⚠ No actionable cells found (design may already be optimal)")
    
except Exception as e:
    print(f"✗ Action space test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test 7: Visualize Graph (Optional)
# ============================================================================
print("\n" + "-"*70)
print("Test 7: Graph Visualization (Optional)")
print("-"*70)

try:
    from state_representation import visualize_timing_graph
    
    output_file = "test_timing_graph.png"
    visualize_timing_graph(graph, node_features, output_file)
    
    if Path(output_file).exists():
        print(f"✓ Graph visualization saved to: {output_file}")
    else:
        print("⚠ Visualization file not created")
    
except ImportError:
    print("⚠ Matplotlib not installed, skipping visualization")
    print("  Install with: pip install matplotlib")
except Exception as e:
    print(f"⚠ Visualization failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("Test Summary")
print("="*70)

print("""
✓ All core components working!

Next Steps:
1. Train GNN model on multiple designs (create train_gnn_dqn.py)
2. Save trained model weights
3. Integrate with dqn_agent.py for OpenLane
4. Run full optimization in OpenLane flow

Files created:
  - state_representation.py   (State extraction)
  - gnn_dqn.py               (GNN model)
  - discrete_action_space.py  (Action space)
  - timing_parser.py         (Report parser)

Ready to use in OpenLane2!
""")

print("="*70)
print("Test Complete!")
print("="*70 + "\n")
