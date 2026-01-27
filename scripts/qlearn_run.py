from openlane.steps import Step, State
from openlane.common import Config

# Path to the state_out.json from the CTS step
previous_state_path = "./runs/RUN_TIMESTAMP/steps/OpenROAD.CTS/state_out.json"
design_config_path = "./runs/RUN_TIMESTAMP/config.json"

# Load the existing state
input_state = State.load(previous_state_path)
config = Config.load(design_config_path)

# Initialize your custom Q-learning step
my_q_step = QLearningGateSizing(config=config)

# Run only this step from the saved post-CTS point
final_state = my_q_step.start(state_in=input_state)
