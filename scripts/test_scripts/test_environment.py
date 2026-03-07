#!/usr/bin/env python3
"""
Test Environment Integration
=============================

Quick test to verify rl_environment.py works with your design.
"""

import sys
import numpy as np
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rl_environment import CellSizingEnv


def test_environment(design_dir: str, config_file: str):
    """Test basic environment functionality."""
    
    print("="*70)
    print("Testing CellSizingEnv Integration")
    print("="*70)
    
    # Initialize environment
    print("\n1. Initializing environment...")
    try:
        env = CellSizingEnv(
            design_dir=design_dir,
            config_file=config_file,
            max_steps=10,
            top_k_cells=10
        )
        print("   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        return False
    
    # Test reset
    print("\n2. Testing reset()...")
    try:
        state = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   State shape: {state.shape}")
        print(f"   Expected: (45,)")
        print(f"   Initial WNS: {env.initial_wns:.3f}")
        print(f"   Actionable cells: {len(env.actionable_cells)}")
    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test step
    print("\n3. Testing step() with random actions...")
    try:
        for step in range(3):
            action = np.random.randint(0, env.action_space.n)
            next_state, reward, done, info = env.step(action)
            
            print(f"\n   Step {step+1}:")
            print(f"     Action: {action}")
            print(f"     Next state shape: {next_state.shape}")
            print(f"     Reward: {reward:.2f}")
            print(f"     Done: {done}")
            print(f"     Info: {info}")
            
            if done:
                print("     Episode finished early (timing met or max steps)")
                break
        
        print("\n   ✓ Step execution successful")
    except Exception as e:
        print(f"\n   ✗ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test close
    print("\n4. Testing close()...")
    try:
        env.close()
        print("   ✓ Environment closed cleanly")
    except Exception as e:
        print(f"   ✗ Close failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nEnvironment is ready for training.")
    print("\nNext steps:")
    print("  1. Review any warnings or errors above")
    print("  2. If using placeholders, implement OpenROAD integration")
    print("  3. Run training: python3 train_dqn.py --designs designs.txt --episodes 10")
    
    return True


def test_with_mock_data():
    """Test with mock data (no OpenROAD required)."""
    
    print("\n" + "="*70)
    print("Mock Data Test (No OpenROAD Required)")
    print("="*70)
    
    # Create temporary mock design directory
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix="test_design_")
    print(f"\nUsing temporary directory: {temp_dir}")
    
    # Create mock config
    import json
    config_path = Path(temp_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "DESIGN_NAME": "test_design",
            "CLOCK_PERIOD": 10.0
        }, f)
    
    print("\nTesting environment with mock data...")
    print("(This will use placeholder functions - should complete without OpenROAD)")
    
    try:
        result = test_environment(temp_dir, str(config_path))
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test CellSizingEnv integration"
    )
    parser.add_argument(
        '--design-dir',
        help='Path to design directory (default: run mock test)'
    )
    parser.add_argument(
        '--config',
        help='Path to config file'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run with mock data only (no OpenROAD)'
    )
    
    args = parser.parse_args()
    
    if args.mock or (args.design_dir is None and args.config is None):
        # Run mock test
        print("Running mock test (no real design required)...")
        success = test_with_mock_data()
    elif args.design_dir and args.config:
        # Run with real design
        print(f"Testing with design: {args.design_dir}")
        success = test_environment(args.design_dir, args.config)
    else:
        print("Error: Must provide both --design-dir and --config, or use --mock")
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)
