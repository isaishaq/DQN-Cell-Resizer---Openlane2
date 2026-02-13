# Get timing metrics from OpenROAD
# Environment variables expected:
#   METRICS_FILE - Output JSON file path

source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# Load design with parasitics
puts "\[INFO\] Loading corners"
load_rsz_corners
puts "\[INFO\] Loading ODB"
read_current_odb

# Get timing metrics
set wns [sta::worst_slack]
set tns [sta::total_negative_slack]

# Output as JSON to file specified by env var
set metrics_file $::env(METRICS_FILE)
set fp [open $metrics_file w]
puts $fp "{\n  \"wns\": $wns,\n  \"tns\": $tns,\n  \"power\": 0.0\n}"
close $fp

puts "\[INFO\] Timing metrics saved to $metrics_file"
