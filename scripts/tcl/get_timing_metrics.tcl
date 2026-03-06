# Get timing metrics from OpenROAD
# Environment variables expected:
#   METRICS_FILE - Output JSON file path

# source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
# source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl

# # Load design with parasitics
# puts "\[INFO\] Loading corners"
# load_rsz_corners
# puts "\[INFO\] Loading ODB"
# read_current_odb

set sta_report_default_digits 6
if {$env(STEP_DIR) eq ""} {
    puts "\[ERROR\] STEP_DIR environment variable is not set. Please set it to the step directory containing the ODB and SPEF files."
    exit 1
}

# Print available corners
foreach corner [sta::corners] {
    puts "\[INFO\] Available corner: [[lindex $corner 0] name]"
}

# Enable only for 1 corner -> nom_ss_100C_1v60

foreach corner [sta::corners] {
    # Enable only for 1 corner -> nom_ss_100C_1v60
    if { [$corner name] ne "nom_ss_100C_1v60" } {
        puts "\[INFO\] Skipping corner [[lindex $corner 0] name] (only nom_ss_100C_1v60 is enabled)"
        continue
    }

    set corner_name [[lindex $corner 0] name]
    puts "Printing metrics for corner: $corner_name"
    sta::set_cmd_corner $corner

    set folder_path $env(STEP_DIR)/reports/iter${env(CURRENT_ITERATION)}_$corner_name
    file mkdir $folder_path
    
    # If file exists, remove all files inside this folder to avoid appending to old data
    if {[file exists $folder_path]} {
        file delete -force [glob -nocomplain $folder_path/*]
    }

    set clocks [sta::sort_by_name [sta::all_clocks]]

    puts "Creating min.rpt"
    report_checks -sort_by_slack -path_delay min -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 100 -corner [$corner name] -digits 3 > $folder_path/min.rpt
    set tns [total_negative_slack -corner [$corner name] -min]
    set file [open $folder_path/tns_min.rpt w]
    puts $file "TNS: $tns" 
    close $file

    puts "Creating max.rpt"
    report_checks -sort_by_slack -path_delay max -fields {slew cap input nets fanout} -format full_clock_expanded -group_count 100 -corner [$corner name] -digits 3 > $folder_path/max.rpt
    set tns [total_negative_slack -corner [$corner name] -max]
    set file [open $folder_path/tns_max.rpt w]
    puts $file "TNS: $tns"
    close $file



    puts "Creating checks.rpt"
    report_checks -unconstrained -fields {slew cap input nets fanout} -format full_clock_expanded -corner [$corner name] > $folder_path/checks.rpt


    puts "Creating slack_max.rpt"
    report_checks -slack_max -0.01 -fields {slew cap input nets fanout} -format full_clock_expanded -corner [$corner name] > $folder_path/slack_max.rpt

    puts "Creating violators.rpt"
    report_check_types -max_slew -max_capacitance -max_fanout -violators -corner [$corner name] > $folder_path/violators.rpt

    puts "Creating parasitic_annotation.rpt"
    report_parasitic_annotation -report_unannotated > $folder_path/parasitic_annotation.rpt

}

# # Get timing metrics
# set wns [sta::worst_slack]
# set tns [sta::total_negative_slack]

# # Output as JSON to file specified by env var
# set metrics_file $::env(METRICS_FILE)
# set fp [open $metrics_file w]
# puts $fp "{\n  \"wns\": $wns,\n  \"tns\": $tns,\n  \"power\": 0.0\n}"
# close $fp

# puts "\[INFO\] Timing metrics saved to $metrics_file"
