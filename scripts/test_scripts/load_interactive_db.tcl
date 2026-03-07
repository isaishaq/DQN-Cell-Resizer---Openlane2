# Change this depends on the step you want to load
set step_dir "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44/74-dqn-resizer-test/"
set ::env(CURRENT_SPEF) "/home/isaishaq/openlane2/designs/picorv_test/runs/RUN_2026-01-17_18-33-44/52-openroad-rcx/"
set ::env(CURRENT_SPEF_nom_tt_025C_1v80) "$::env(CURRENT_SPEF)/nom/picorv32a.nom.spef"
set ::env(CURRENT_SPEF_nom_ss_100C_1v60) "$::env(CURRENT_SPEF)/nom/picorv32a.nom.spef"



# Load OpenROAD utilities
set ::env(RSZ_DONT_TOUCH_RX) False
set ::env(DPL_CELL_PADDING) 2
set ::env(PL_MAX_DISPLACEMENT_X) 20
set ::env(PL_MAX_DISPLACEMENT_Y) 10
set ::env(DPL_CELL_PADDING) 0
set ::env(RSZ_CORNER_0) "nom_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set ::env(RSZ_CORNER_1) "nom_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
set ::env(RSZ_CORNER_2) "nom_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"
# set ::env(RSZ_CORNER_3) "min_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
# set ::env(RSZ_CORNER_4) "min_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
# set ::env(RSZ_CORNER_5) "min_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"
# set ::env(RSZ_CORNER_6) "max_tt_025C_1v80 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
# set ::env(RSZ_CORNER_7) "max_ss_100C_1v60 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib"
# set ::env(RSZ_CORNER_8) "max_ff_n40C_1v95 /home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib"


set ::env(SCRIPTS_DIR) /home/isaishaq/openlane2/openlane/scripts/
set ::env(_TCL_ENV_IN) ${step_dir}/_env.tcl
source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# SDC overrides
set ::env(CLOCK_PERIOD) 19

set ::env(_PNR_EXCLUDED_CELLS) {
sky130_fd_sc_hd__lpflow_decapkapwr_12 sky130_fd_sc_hd__lpflow_decapkapwr_6 sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_4
sky130_fd_sc_hd__lpflow_isobufsrc_2 sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_2 sky130_fd_sc_hd__clkdlybuf4s15_1
sky130_fd_sc_hd__lpflow_inputiso1n_1 sky130_fd_sc_hd__lpflow_clkbufkapwr_2 sky130_fd_sc_hd__probe_p_8
sky130_fd_sc_hd__lpflow_clkbufkapwr_4 sky130_fd_sc_hd__lpflow_decapkapwr_8 sky130_fd_sc_hd__lpflow_clkinvkapwr_1
sky130_fd_sc_hd__xnor3_2 sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_4 sky130_fd_sc_hd__o311ai_0
sky130_fd_sc_hd__lpflow_clkbufkapwr_16 sky130_fd_sc_hd__buf_16 sky130_fd_sc_hd__lpflow_inputisolatch_1
sky130_fd_sc_hd__lpflow_isobufsrc_4 sky130_fd_sc_hd__lpflow_clkinvkapwr_4 sky130_fd_sc_hd__lpflow_clkinvkapwr_8
sky130_fd_sc_hd__xnor3_1 sky130_fd_sc_hd__or2_0 sky130_fd_sc_hd__lpflow_clkinvkapwr_2 sky130_fd_sc_hd__mux4_4
sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_4 sky130_fd_sc_hd__lpflow_decapkapwr_3 sky130_fd_sc_hd__lpflow_inputiso0p_1
sky130_fd_sc_hd__lpflow_bleeder_1 sky130_fd_sc_hd__and2_0 sky130_fd_sc_hd__lpflow_clkbufkapwr_1 sky130_fd_sc_hd__xnor3_4
sky130_fd_sc_hd__lpflow_isobufsrc_16 sky130_fd_sc_hd__lpflow_clkinvkapwr_16 sky130_fd_sc_hd__o21ai_0
sky130_fd_sc_hd__xor3_1 sky130_fd_sc_hd__xor3_4 sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_1 sky130_fd_sc_hd__fa_4
sky130_fd_sc_hd__xor3_2 sky130_fd_sc_hd__lpflow_inputiso1p_1 sky130_fd_sc_hd__clkdlybuf4s18_1
sky130_fd_sc_hd__lpflow_decapkapwr_4 sky130_fd_sc_hd__lpflow_isobufsrc_1 sky130_fd_sc_hd__lpflow_inputiso0n_1
sky130_fd_sc_hd__lpflow_isobufsrc_8 sky130_fd_sc_hd__probec_p_8 sky130_fd_sc_hd__lpflow_lsbuf_lh_isowell_tap_1
sky130_fd_sc_hd__a21boi_0 sky130_fd_sc_hd__lpflow_lsbuf_lh_hl_isowell_tap_2 sky130_fd_sc_hd__a2111oi_0
sky130_fd_sc_hd__lpflow_isobufsrckapwr_16 sky130_fd_sc_hd__lpflow_clkbufkapwr_8
}

set ::env(_SDC_IN) /nix/store/ss2cw3sxbrwwx9jl0rrppbw4kgcmgi2n-python3-3.11.9-env/lib/python3.11/site-packages/openlane/scripts/base.sdc

# Load design with parasitics
puts "\[INFO\] Loading corners"
load_rsz_corners
puts "\[INFO\] Loading ODB"
read_current_odb

set_propagated_clock [all_clocks]
set_dont_touch_objects

estimate_parasitics -global_routing
read_spef -corner nom_ss_100C_1v60 $::env(CURRENT_SPEF_nom_ss_100C_1v60)

# Load SPEF parasitics for accurate timing
# if { [info exists ::env(CURRENT_SPEF)] } {
#     foreach corner $::env(STA_CORNERS) {
#         if { [info exists ::env(CURRENT_SPEF_$corner)] } {
#             puts "\[INFO\] Reading SPEF for corner $corner from $::env(CURRENT_SPEF_$corner)…"
#             read_spef -corner $corner $::env(CURRENT_SPEF_$corner)
#         }
#     }
# }

# Set RC values
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl
