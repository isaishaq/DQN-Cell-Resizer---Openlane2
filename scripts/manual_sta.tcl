# Read the ODB database
read_db /home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44/final/odb/picorv32a.odb

# Read Liberty timing files
read_liberty /home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44/final/lib/max_ss_100C_1v60/picorv32a__max_ss_100C_1v60.lib

# Read LEF files  
read_lef /path/to/tech.lef
read_lef /path/to/cells.lef

# Read SDC constraints
read_sdc /home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44/final/sdc/picorv32a.sdc

# Set propagated clock mode
set_propagated_clock [all_clocks]

# Estimate parasitics (choose one based on your stage):
estimate_parasitics -placement       # After placement
# OR
estimate_parasitics -global_routing  # After global routing

# Get timing reports
report_checks -path_delay max -format full_clock_expanded  # Setup
report_checks -path_delay min -format full_clock_expanded  # Hold

# Get slack values
puts "Worst Setup Slack: [worst_slack -max]"
puts "Worst Hold Slack: [worst_slack -min]"
puts "Total Negative Slack (Setup): [total_negative_slack -max]"
puts "Total Negative Slack (Hold): [total_negative_slack -min]"

# Report violations
report_check_types -max_slew -max_capacitance -max_fanout -violators

# Get detailed path info
report_checks -path_delay max -group_count 10 -format full_clock_expanded

# Open GUI to visualize (optional)
# gui::show