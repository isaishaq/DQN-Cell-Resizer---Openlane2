from openlane.steps import Step, ViewsUpdate, MetricsUpdate
from openlane.steps.openroad import OpenROADStep
from openlane.state import State, DesignFormat
from openlane.config import Variable
from typing import Tuple, Dict, Any
import os
from decimal import Decimal

@Step.factory.register()
class DQNResizer(OpenROADStep):
    """
    Deep Q-Learning based cell resizer for timing optimization.
    
    Uses reinforcement learning to intelligently resize cells
    to optimize timing and power trade-offs.
    """
    
    id = "OpenROAD.DQNResizer"
    name = "DQN-Based Cell Resizer"
    
    inputs = [DesignFormat.ODB, DesignFormat.SPEF]
    outputs = [DesignFormat.ODB, DesignFormat.DEF]
    
    config_vars = OpenROADStep.config_vars + [
        Variable(
            "DQN_MODEL_PATH",
            str,
            "Path to pre-trained DQN model",
            default="models/dqn_resizer.pth",
        ),
        Variable(
            "DQN_MAX_ITERATIONS",
            int,
            "Maximum number of resizing iterations",
            default=50,
        ),
        Variable(
            "DQN_TARGET_SLACK",
            Decimal,
            "Target slack margin in nanoseconds",
            default=0.0,
            units="ns",
        ),
        Variable(
            "DQN_POWER_WEIGHT",
            Decimal,
            "Weight for power optimization (0-1, 1 = prioritize power)",
            default=0.3,
        ),
        Variable(
            "DQN_TRAINING_MODE",
            bool,
            "Enable training mode (exploration + model updates)",
            default=True,
        ),
        Variable(
            "DQN_AGENT_SCRIPT",
            str,
            "Path to the Python DQN agent script",
            default="scripts/dqn_agent.py",
        ),
    ]
    
    def get_script_path(self):
        # Point to your custom TCL script
        return os.path.join(
            os.path.dirname(__file__), 
            "dqn_resizer.tcl"
        )
    
    def run(self, state_in: State, **kwargs) -> Tuple[ViewsUpdate, MetricsUpdate]:
        """
        Main execution: Run DQN-based resizing
        """
        kwargs, env = self.extract_env(kwargs)
        
        # Pass DQN configuration to the environment
        env["DQN_MODEL_PATH"] = self.config["DQN_MODEL_PATH"] or ""
        env["DQN_MAX_ITERATIONS"] = str(self.config["DQN_MAX_ITERATIONS"])
        env["DQN_TARGET_SLACK"] = str(self.config["DQN_TARGET_SLACK"])
        env["DQN_POWER_WEIGHT"] = str(self.config["DQN_POWER_WEIGHT"])
        env["DQN_TRAINING_MODE"] = "1" if self.config["DQN_TRAINING_MODE"] else "0"
        
        # Point to Python DQN agent script
        env["DQN_AGENT_SCRIPT"] = str(self.config["DQN_AGENT_SCRIPT"]) or ""
        # env["DQN_AGENT_SCRIPT"] = os.path.join(
        #     os.path.dirname(__file__),
        #     "dqn_agent.py"
        # )
        
        return super().run(state_in, env=env, **kwargs)