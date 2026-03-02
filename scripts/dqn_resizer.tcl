# RUN DIR
set ::env(STEP_DIR) "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-03-01_15-10-18/74-dqn-resizer-test/"

# Load OpenROAD utilities
set ::env(RSZ_DONT_TOUCH_RX) False
set ::env(DPL_CELL_PADDING) 2
set ::env(PL_MAX_DISPLACEMENT_X) 20
set ::env(PL_MAX_DISPLACEMENT_Y) 10
# set ::env(RSZ_CORNER_0) "nom_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
# set ::env(RSZ_CORNER_1) "nom_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
# set ::env(RSZ_CORNER_2) "nom_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"


# Utils
set ::env(TCL_UTILS_DIR) /home/isaishaq/openlane2/designs/picorv_test/scripts/tcl

source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# SDC overrides
set ::env(CLOCK_PERIOD) 16

# Path
set ::env(DQN_MODEL_PATH) "$::env(STEP_DIR)/model/dqn_model.pth"

# Load design with parasitics
puts "\[INFO\] Loading corners"
load_rsz_corners
puts "\[INFO\] Loading ODB"
read_current_odb

# puts "SDC file -> $::env(_TCL_ENV_IN)"

set_propagated_clock [all_clocks]
set_dont_touch_objects

estimate_parasitics -global_routing

# Set RC values
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl


puts "\[INFO\] Training DQN agent for up to 50 iterations..."
for {set iter 1} {$iter <= 5} {incr iter} {
    set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um
    set ::env(CURRENT_ITERATION) $iter

    set corner_name "nom_ss_100C_1v60"; # hardcoded for now, can be looped over corners if needed
    set ::env(REPORT_FOLDER) "$::env(STEP_DIR)/reports/iter${::env(CURRENT_ITERATION)}_$corner_name"
    source $::env(TCL_UTILS_DIR)/get_timing_metrics.tcl

    # Call Python DQN agent - pass data via environment variables
    set actions_file "$::env(STEP_DIR)/actions/actions_iter${iter}.txt"
    set model_file "$::env(DQN_MODEL_PATH)"
    set report_file "$::env(REPORT_FOLDER)/max.rpt"
    
    exec python3 $::env(DQN_AGENT_SCRIPT) \
        --timing-report $report_file \
        --output-actions $actions_file \
        --model $model_file \
        --iteration $iter \
        --verbose \
        --epsilon 0.8
    
    # # Read and apply actions
    # set fp [open $actions_file r]
    # while {[gets $fp line] >= 0} {
    #     # Skip comments
    #     if {[string match "#*" $line]} { continue }
    #     if {$line eq ""} { continue }
        
    #     # Parse: instance_name new_cell_type
    #     set parts [split $line]
    #     set instance [lindex $parts 0]
    #     set new_cell [lindex $parts 1]
        
    #     # Apply resize
    #     # set inst [odb::dbInst_find $block $instance]
    #     set inst [get_cells $instance]
    #     # Try if not WARNING STA-0349: Cannot find instance $instance in design
    #     if {$inst != ""} {
    #         set ref_lib [get_lib_cells $new_cell]
    #         if { [llength $ref_lib] != 0 } {
    #             replace_cell $inst $new_cell
    #             puts "Resized $instance to $new_cell"
    #         }
    #     } else {
    #         puts "WARNING: Instance $instance not found in design, skipping resize"
    #     }
    # }
    # close $fp
    
    # # Update timing
    # # sta::update_timing
    
    # # Check convergence
    # set wns [sta::worst_slack -max]
    # if {$wns >= 0} {
    #     puts "Converged! WNS = $wns"
    #     break
    # }

}




# Re-estimate parasitics after resizing
# estimate_parasitics -global_routing

# Legalize placement if needed
# source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl
# unset_dont_touch_objects

# Final STA
# report_checks -path_delay min_max -format full_clock_expanded

# write_views