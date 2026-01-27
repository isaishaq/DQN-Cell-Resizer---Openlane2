#!/usr/bin/env python3
"""
DQN-only Flow for OpenLane 2

This flow resumes from an existing run's OpenROAD.STAPostPNR state,
runs the DQN resizer, and re-runs STA to measure improvements.

Usage:
    python -m openlane designs/picorv_test/config.json \
        --flow designs/picorv_test/scripts/dqn_flow.py \
        --last-run
"""

from openlane.flows import SequentialFlow
from openlane.steps import OpenROAD
from openlane.config import Variable
from typing import Optional
from decimal import Decimal

# Import your DQN step (create this next)
try:
    from .dqn_resizer_step import DQNResizer
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from dqn_resizer_step import DQNResizer


class DQNFlow(SequentialFlow):
    """
    A minimal flow for running DQN-based cell resizing.
    
    This flow assumes you already have a completed run up to 
    OpenROAD.STAPostPNR and will:
    1. Load the existing state (ODB, SPEF, netlists)
    2. Run DQN-based cell resizer
    3. Re-run STA to measure improvements
    
    To use:
    -------
    # First, complete a baseline run to OpenROAD.STAPostPNR
    python -m openlane designs/picorv_test/config.json --tag baseline
    
    # Then run DQN optimization
    python -m openlane designs/picorv_test/config.json \\
        --flow designs/picorv_test/scripts/dqn_flow.py \\
        --last-run \\
        --tag dqn_experiment \\
        -p DQN_MAX_ITERATIONS=30 \\
        -p DQN_TRAINING_MODE=False
    """
    
    Steps = [
        # Your DQN resizer step
        DQNResizer,
        
        # Re-run extraction and STA to measure improvements
        OpenROAD.RCX,
        OpenROAD.STAPostPNR,
        
        # Optional: Generate reports
        OpenROAD.IRDropReport,
    ]
    
    config_vars = SequentialFlow.config_vars + [
        Variable(
            "DQN_MODEL_PATH",
            Optional[str],
            "Path to pre-trained DQN model. If None, uses random/untrained policy.",
            default=None,
        ),
        Variable(
            "DQN_MAX_ITERATIONS",
            int,
            "Maximum number of resizing iterations for DQN agent.",
            default=50,
        ),
        Variable(
            "DQN_TARGET_SLACK",
            Decimal,
            "Target slack margin in nanoseconds. DQN will try to achieve this.",
            default=0.0,
            units="ns",
        ),
        Variable(
            "DQN_POWER_WEIGHT",
            Decimal,
            "Weight for power optimization (0-1). Higher values prioritize power reduction.",
            default=0.3,
        ),
        Variable(
            "DQN_TRAINING_MODE",
            bool,
            "Enable training mode (exploration + model updates). Slower but adapts to design.",
            default=False,
        ),
        Variable(
            "DQN_EPSILON",
            Decimal,
            "Exploration rate for epsilon-greedy policy (0-1). Only used in training mode.",
            default=0.1,
        ),
        Variable(
            "DQN_LEARNING_RATE",
            Decimal,
            "Learning rate for DQN optimizer. Only used in training mode.",
            default=1e-4,
        ),
        Variable(
            "DQN_BATCH_SIZE",
            int,
            "Batch size for experience replay. Only used in training mode.",
            default=32,
        ),
        Variable(
            "DQN_MEMORY_SIZE",
            int,
            "Size of replay memory buffer.",
            default=10000,
        ),
        Variable(
            "DQN_SAVE_MODEL",
            bool,
            "Save the trained/updated model after completion.",
            default=True,
        ),
        Variable(
            "DQN_SAVE_MODEL_PATH",
            Optional[str],
            "Where to save the model. If None, saves to run directory.",
            default=None,
        ),
    ]
    
    # Gating variables - control which steps run
    gating_config_vars = {
        "OpenROAD.RCX": ["RUN_SPEF_EXTRACTION"],
        "OpenROAD.STAPostPNR": ["RUN_MCSTA"],
        "OpenROAD.IRDropReport": ["RUN_IRDROP_REPORT"],
    }


# Convenience alias
DQNOptimizationFlow = DQNFlow


if __name__ == "__main__":
    # import sys
    # print("This is a flow definition file.")
    # print("To use it, run:")
    # print("  python -m openlane designs/picorv_test/config.json \\")
    # print("      --flow designs/picorv_test/scripts/dqn_flow.py \\")
    # print("      --last-run")
    # sys.exit(0)
    flow = DQNFlow(
        {
            
        },

    )
    flow.start()
