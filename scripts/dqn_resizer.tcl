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

# SDC overrides
set ::env(CLOCK_PERIOD) 16

# Utils
set ::env(TCL_UTILS_DIR) /home/isaishaq/openlane2/designs/picorv_test/scripts/tcl

source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

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


puts "\[INFO\] Training DQN agent for up to 50 iterations..."
for {set iter 1} {$iter <= 5} {incr iter} {
    set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um
    set ::env(CURRENT_ITERATION) $iter
    source $::env(TCL_UTILS_DIR)/get_timing_metrics.tcl

    # Call Python DQN agent - pass data via environment variables

}




# Re-estimate parasitics after resizing
estimate_parasitics -global_routing

# Legalize placement if needed
# source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl
# unset_dont_touch_objects

# Final STA
# report_checks -path_delay min_max -format full_clock_expanded

# write_views