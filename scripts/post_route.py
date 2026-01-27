from openlane.steps import Step
from openlane.state import State
import pathlib

# Load the state from a specific JSON file
input_state_path = pathlib.Path("/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2025-11-23_17-01-02/26-odb-applydeftemplate/state_out.json")
# The load method handles deserialization
my_step = Step.load(config="/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2025-11-23_17-01-02/resolved.json", state_in=input_state_path)

# You can then run the step in your Python script
# my_step.start()