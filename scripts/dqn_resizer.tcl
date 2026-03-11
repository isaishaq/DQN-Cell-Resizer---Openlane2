# RUN DIR
set ::env(STEP_DIR) "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/"

# Load OpenROAD utilities
set ::env(RSZ_DONT_TOUCH_RX) False
set ::env(DPL_CELL_PADDING) 2
set ::env(PL_MAX_DISPLACEMENT_X) 20
set ::env(PL_MAX_DISPLACEMENT_Y) 10
set ::env(RSZ_CORNER_0) "nom_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set ::env(RSZ_CORNER_1) "nom_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
set ::env(RSZ_CORNER_2) "nom_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"


# Utils
set ::env(TCL_UTILS_DIR) /home/isaishaq/openlane2/designs/picorv_test/scripts/tcl

source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# SDC overrides
# set ::env(CLOCK_PERIOD) 16
# set ::env(CLOCK_UNCERTAINTY) 4.0

# DQN Paths
# set ::env(DQN_MODEL_PATH) "$::env(STEP_DIR)/model/dqn_model.pth"
# if {![info exists ::env(DQN_AGENT_SCRIPT)]} {
#     set ::env(DQN_AGENT_SCRIPT) "/home/isaishaq/openlane2/designs/picorv_test/scripts/dqn_agent.py"
# } elseif {![info exists ::env(HEURISTIC_AGENT_SCRIPT)]} {
#     set ::env(HEURISTIC_AGENT_SCRIPT) "/home/isaishaq/openlane2/designs/picorv_test/scripts/heuristic_agent.py"
# }

set ::env(HEURISTIC_AGENT_SCRIPT) "/home/isaishaq/openlane2/designs/picorv_test/scripts/heuristic_agent.py"

# Agent selection: DQN if model exists, else heuristic
if {[file exists $::env(DQN_AGENT_SCRIPT)]} {
    set ::env(ACTIVE_AGENT) "dqn"
    set ::env(AGENT_SCRIPT) $::env(DQN_AGENT_SCRIPT)
    puts "\[INFO\] Using DQN agent with trained model: $::env(DQN_MODEL_PATH)"
} else {
    set ::env(ACTIVE_AGENT) "heuristic"
    set ::env(AGENT_SCRIPT) $::env(HEURISTIC_AGENT_SCRIPT)
    puts "\[INFO\] No trained model found - using heuristic agent"
}
# Create output directories
file mkdir "$::env(STEP_DIR)/reports"
file mkdir "$::env(STEP_DIR)/actions"
file mkdir "$::env(STEP_DIR)/model"

# Load design with parasitics
puts "\[INFO\] Loading corners"
load_rsz_corners
puts "\[INFO\] Loading ODB"
read_current_odb


set_propagated_clock [all_clocks]
set_dont_touch_objects

estimate_parasitics -global_routing

# Set RC values
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl


puts "\[INFO\] Starting cell resizing optimization..."
puts "  Agent: $::env(ACTIVE_AGENT)"
puts "  Max iterations: [expr {[info exists ::env(DQN_MAX_ITERATIONS)] ? $::env(DQN_MAX_ITERATIONS) : 50}]"
puts "  Target WNS: 0.0 ns"
if {$::env(ACTIVE_AGENT) eq "dqn"} {
    puts "  Model: $::env(DQN_MODEL_PATH)"
} else {
    puts "  Strategy: balanced (heuristic)"
}

set dqn_max_iters [expr {[info exists ::env(DQN_MAX_ITERATIONS)] ? $::env(DQN_MAX_ITERATIONS) : 50}]

for {set iter 1} {$iter <= $dqn_max_iters} {incr iter} {
    set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um
    set ::env(CURRENT_ITERATION) $iter

    set corner_name "nom_ss_100C_1v60"; # hardcoded for now, can be looped over corners if needed
    set ::env(REPORT_FOLDER) "$::env(STEP_DIR)/reports/iter${::env(CURRENT_ITERATION)}_$corner_name"
    source $::env(TCL_UTILS_DIR)/get_timing_metrics.tcl

    # Call appropriate agent (DQN or Heuristic)
    set actions_file "$::env(STEP_DIR)/actions/actions_iter${iter}.txt"
    set model_file "$::env(DQN_MODEL_PATH)"
    set report_file "$::env(REPORT_FOLDER)/max.rpt"
    
    if {$::env(ACTIVE_AGENT) eq "dqn"} {
        puts "\[INFO\] Running DQN agent: $::env(DQN_AGENT_SCRIPT)"
        exec python3 $::env(DQN_AGENT_SCRIPT) \
            --timing-report $report_file \
            --output-actions $actions_file \
            --model $model_file \
            --iteration $iter \
            --verbose \
            --state-log "$::env(STEP_DIR)/logs/latest_run.log" \
            --epsilon 0.1 \
            >@ stdout 2>@ stderr
    } else {
        puts "\[INFO\] Running heuristic agent: $::env(HEURISTIC_AGENT_SCRIPT)"
        exec python3 $::env(HEURISTIC_AGENT_SCRIPT) \
            --timing-report $report_file \
            --output-actions $actions_file \
            --strategy balanced \
            --iteration $iter \
            --verbose \
            >@ stdout 2>@ stderr
    }
    

    # Read and apply actions
    puts "\[INFO\] Applying DQN actions from $actions_file"
    set num_resizes 0
    set num_skipped 0
    
    if {[file exists $actions_file]} {
        set fp [open $actions_file r]
        while {[gets $fp line] >= 0} {
            # Skip comments and empty lines
            if {[string match "#*" $line]} { continue }
            if {$line eq ""} { continue }
            
            # Parse: instance_name new_cell_type
            set parts [split $line]
            if {[llength $parts] != 2} { continue }
            
            set instance [lindex $parts 0]
            set new_cell [lindex $parts 1]
            
            # Apply resize
            set inst [get_cells -quiet $instance]
            if {$inst != ""} {
                set ref_lib [get_lib_cells -quiet $new_cell]
                if {$ref_lib != ""} {
                    replace_cell $inst $new_cell
                    puts "  ✓ Resized $instance -> $new_cell"
                    incr num_resizes
                } else {
                    puts "  ✗ Cell type $new_cell not found in library"
                    incr num_skipped
                }
            } else {
                puts "  ✗ Instance $instance not found"
                incr num_skipped
            }
        }
        close $fp
    } else {
        puts "\[WARNING\] Actions file not found: $actions_file"
    }
    
    puts "\[INFO\] Applied $num_resizes resizes, skipped $num_skipped"
    
    # Update timing after resizing
    estimate_parasitics -global_routing

    set tns [total_negative_slack -corner $corner_name -max]
    set wns [worst_slack -corner $corner_name -max]
    
    puts "\[INFO\] Iteration $iter complete: WNS=$wns, TNS=$tns"
    
    # Break if timing is met
    if {$wns >= 0.0} {
        puts "\[SUCCESS\] ✓ Timing constraints met! WNS = $wns"
        break
    }
    
    # Break if no actions taken
    # if {$num_resizes == 0 && $::env(ACTIVE_AGENT) eq "heuristic"} {
    #     puts "\[INFO\] No more actions available - converged"
    #     break
    # }

}

# ============================================================================
# Post-processing
# ============================================================================

puts "\n========================================="
puts "Cell Resizing Complete ($::env(ACTIVE_AGENT))"
puts "========================================="

# Re-estimate parasitics one final time
estimate_parasitics -global_routing

# Legalize placement if needed (cells may have changed size)
puts "\[INFO\] Legalizing placement..."
source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl

# Unset dont_touch
unset_dont_touch_objects

# Final timing report
puts "\[INFO\] Generating final timing report..."
set final_report_dir "$::env(STEP_DIR)/reports/final"
file mkdir $final_report_dir

report_checks -path_delay max -format full_clock_expanded \
    > $final_report_dir/timing_final_max.rpt
report_checks -path_delay min -format full_clock_expanded \
    > $final_report_dir/timing_final_min.rpt

set final_wns [sta::worst_slack -max]
set final_tns [sta::total_negative_slack -max]
# set final_area [sta::design_area]

puts "\n========================================="
puts "Final Results:"
puts "  WNS: $final_wns ns"
puts "  TNS: $final_tns ns"
# puts "  Area: $final_area um²"
puts "========================================="

# Write output views
puts "\[INFO\] Writing output database..."
write_views

puts "\[SUCCESS\] Cell resizing flow complete!"
puts "Agent used: $::env(ACTIVE_AGENT)"