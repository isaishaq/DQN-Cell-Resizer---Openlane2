import subprocess
import os 
script_path = "/home/isaishaq/openlane2/designs/picorv_test/scripts/tcl/get_timing_metrics.tcl"
command = ['openroad', '-no_splash', '-exit', script_path]
env = os.environ.copy()
#command = script_path
result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

print(result.stdout)