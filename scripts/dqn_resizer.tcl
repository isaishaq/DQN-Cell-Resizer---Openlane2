# Load OpenROAD utilities
set ::env(RSZ_DONT_TOUCH_RX) False
set ::env(DPL_CELL_PADDING) 2
set ::env(PL_MAX_DISPLACEMENT_X) 20
set ::env(PL_MAX_DISPLACEMENT_Y) 10
set ::env(DPL_CELL_PADDING) 0
set ::env(RSZ_CORNER_0) "nom_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set ::env(RSZ_CORNER_1) "nom_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
set ::env(RSZ_CORNER_2) "nom_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"

source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# Load design with parasitics
puts "\[INFO\] Loading corners"
load_rsz_corners
puts "\[INFO\] Loading ODB"
read_current_odb

set_propagated_clock [all_clocks]
set_dont_touch_objects

# Load SPEF parasitics for accurate timing
if { [info exists ::env(CURRENT_SPEF)] } {
    foreach corner $::env(STA_CORNERS) {
        if { [info exists ::env(CURRENT_SPEF_$corner)] } {
            read_spef -corner $corner $::env(CURRENT_SPEF_$corner)
        }
    }
}

# Set RC values
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl

puts "\[INFO\] ============================================================"
puts "\[INFO\] Multi-Corner STA Report (Before DQN Training)"
puts "\[INFO\] ============================================================"

# Get all defined corners
set all_corners [sta::corners]
puts "\[INFO\] Found [llength $all_corners] corner(s) defined"

# Report timing for each corner
foreach corner $all_corners {
    set corner_name [$corner name]
    puts "\n"
    puts "\[INFO\] ============================================================"
    puts "\[INFO\] Corner: $corner_name"
    puts "\[INFO\] ============================================================"
    
    # Report setup (max delay) worst slack
    set ws_max [worst_slack -corner $corner_name -max]
    puts "\[INFO\] Setup Worst Slack: $ws_max ns"
    
    # Report hold (min delay) worst slack
    set ws_min [worst_slack -corner $corner_name -min]
    puts "\[INFO\] Hold Worst Slack: $ws_min ns"
    
    # Report Total Negative Slack (TNS) for setup
    set tns_max [total_negative_slack -corner $corner_name -max]
    puts "\[INFO\] Setup TNS: $tns_max ns"
    
    # Report Total Negative Slack (TNS) for hold
    set tns_min [total_negative_slack -corner $corner_name -min]
    puts "\[INFO\] Hold TNS: $tns_min ns"
    
    # Compute WNS from WS
    set wns_max 0.0
    if { $ws_max < 0 } {
        set wns_max $ws_max
    }
    set wns_min 0.0
    if { $ws_min < 0 } {
        set wns_min $ws_min
    }
    puts "\[INFO\] Setup WNS: $wns_max ns"
    puts "\[INFO\] Hold WNS: $wns_min ns"
    
    # Report critical path for setup
    puts "\n\[INFO\] Critical Path (Setup - Max Delay):"
    puts "------------------------------------------------------------"
    report_checks -path_delay max -fields {slew cap input nets fanout} \
                  -format full_clock_expanded -corner $corner_name -digits 4
    
    # Report critical path for hold
    puts "\n\[INFO\] Critical Path (Hold - Min Delay):"
    puts "------------------------------------------------------------"
    report_checks -path_delay min -fields {slew cap input nets fanout} \
                  -format full_clock_expanded -corner $corner_name -digits 4
    
    # Report design checks
    puts "\n\[INFO\] Design Rule Violations:"
    puts "------------------------------------------------------------"
    report_check_types -max_slew -max_capacitance -max_fanout -violators -corner $corner_name
    
    # Report power
    puts "\n\[INFO\] Power Analysis:"
    puts "------------------------------------------------------------"
    report_power -corner $corner_name
    
    puts "\[INFO\] ============================================================\n"
}

# Summary across all corners
puts "\[INFO\] ============================================================"
puts "\[INFO\] Multi-Corner Summary"
puts "\[INFO\] ============================================================"
set worst_setup_slack 1e30
set worst_hold_slack 1e30
set worst_setup_corner "N/A"
set worst_hold_corner "N/A"

foreach corner $all_corners {
    set corner_name [$corner name]
    set ws_max [worst_slack -corner $corner_name -max]
    set ws_min [worst_slack -corner $corner_name -min]
    
    if { $ws_max < $worst_setup_slack } {
        set worst_setup_slack $ws_max
        set worst_setup_corner $corner_name
    }
    
    if { $ws_min < $worst_hold_slack } {
        set worst_hold_slack $ws_min
        set worst_hold_corner $corner_name
    }
}

puts "\[INFO\] Worst Setup Slack: $worst_setup_slack ns (Corner: $worst_setup_corner)"
puts "\[INFO\] Worst Hold Slack: $worst_hold_slack ns (Corner: $worst_hold_corner)"
puts "\[INFO\] ============================================================\n"

puts "\[INFO\] Starting DQN-based resizing..."

# Call Python DQN agent via OpenROAD's Python interface
# This is where the magic happens! -->
# set python_script $::env(DQN_AGENT_SCRIPT)

# Export current database for Python to access
set temp_odb "/tmp/dqn_temp_[pid].odb"
write_db $temp_odb

# Run DQN agent
# set dqn_result [exec python3 $python_script \
#     --odb $temp_odb \
#     --model $::env(DQN_MODEL_PATH) \
#     --max-iterations $::env(DQN_MAX_ITERATIONS) \
#     --target-slack $::env(DQN_TARGET_SLACK) \
#     --power-weight $::env(DQN_POWER_WEIGHT) \
#     --training $::env(DQN_TRAINING_MODE)]

# puts "\[INFO\] DQN Result: $dqn_result"

# Reload modified database
# read_db $temp_odb
# file delete $temp_odb

# Re-estimate parasitics after resizing
estimate_parasitics -placement

# Legalize placement if needed
source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl

unset_dont_touch_objects

# Final STA
report_checks -path_delay min_max -format full_clock_expanded

write_views