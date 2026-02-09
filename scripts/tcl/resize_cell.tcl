# Resize a cell to a different drive strength
# Environment variables expected:
#   CELL_INSTANCE_NAME - Name of the cell instance to resize
#   NEW_MASTER_NAME - Name of the new master cell
#   OUTPUT_FILE - Output file path (success/failure)

set inst_name $::env(CELL_INSTANCE_NAME)
set new_master_name $::env(NEW_MASTER_NAME)

# Get the instance
set inst [[[ord::get_db] getChip] getBlock]::findInst $inst_name]

if { $inst == "NULL" } {
    set fp [open $::env(OUTPUT_FILE) w]
    puts $fp "ERROR: Instance not found"
    close $fp
    exit 1
}

# Get the new master
set lib [lindex [[[ord::get_db] getChip] getBlock]::getLibs] 0]
set new_master [$lib findMaster $new_master_name]

if { $new_master == "NULL" } {
    set fp [open $::env(OUTPUT_FILE) w]
    puts $fp "ERROR: Master not found"
    close $fp
    exit 1
}

# Swap the master
$inst swapMaster $new_master

set fp [open $::env(OUTPUT_FILE) w]
puts $fp "SUCCESS"
close $fp

puts "\[INFO\] Resized $inst_name to $new_master_name"
