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

    # User testing hook
    # puts "TCL_ENV_IN -> \n$::env(_TCL_ENV_IN)"
    # puts "PNR_EXCLUDED_CELLS -> \n$::env(_PNR_EXCLUDED_CELLS)"
    # puts "SDC_IN -> \n$::env(_SDC_IN)"
    #gui::show

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

# Call Python DQN agent - pass data via environment variables
# This follows the OpenLane 2 pattern of env-based communication

# Set environment variables for Python script
set ::env(DQN_ODB_PATH) "$::env(CURRENT_ODB)"
set ::env(DQN_WORK_DIR) "$::env(STEP_DIR)/dqn_work"
set ::env(DQN_SCRIPTS_DIR) "[file dirname $::env(DQN_AGENT_SCRIPT)]/tcl"

# Ensure work directory exists
file mkdir $::env(DQN_WORK_DIR)

# Write current database for Python to access
puts "\[INFO\] Exporting current database..."
write_db $::env(DQN_ODB_PATH)

# Run DQN agent with work directory
puts "\[INFO\] Launching DQN agent..."
puts "\[INFO\] ODB Path: $::env(DQN_ODB_PATH)"
puts "\[INFO\] Work Dir: $::env(DQN_WORK_DIR)"
puts "\[INFO\] Scripts Dir: $::env(DQN_SCRIPTS_DIR)"

set dqn_cmd [list python3 $::env(DQN_AGENT_SCRIPT) \
    --odb $::env(DQN_ODB_PATH) \
    --work-dir $::env(DQN_WORK_DIR) \
    --model $::env(DQN_MODEL_PATH) \
    --max-iterations $::env(DQN_MAX_ITERATIONS) \
    --target-slack $::env(DQN_TARGET_SLACK) \
    --power-weight $::env(DQN_POWER_WEIGHT) \
    --training $::env(DQN_TRAINING_MODE)]

if { [catch {exec {*}$dqn_cmd >@ stdout 2>@ stderr} dqn_result] } {
    puts "\[ERROR\] DQN agent failed: $dqn_result"
    puts "\[ERROR\] Continuing with original database..."
} else {
    puts "\[INFO\] DQN agent completed successfully"
    
    # Reload modified database (agent saves back to the same path)
    if { [file exists $::env(DQN_ODB_PATH)] } {
        puts "\[INFO\] Reloading modified database..."
        read_db $::env(DQN_ODB_PATH)
        
        # Invalidate timing cache - force recalculation
        puts "\[INFO\] Updating timing after DQN resizing..."
    }
}

# Re-estimate parasitics after resizing
estimate_parasitics -placement

# Legalize placement if needed
source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl

unset_dont_touch_objects

# Final STA
report_checks -path_delay min_max -format full_clock_expanded

write_views