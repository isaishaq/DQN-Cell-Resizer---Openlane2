# Get Power metrics
# Environment variables expected:
#   OUTPUT_FILE - Output file path

# For now, return 0.0 as placeholder
# In real implementation, you would use report_power
set power 0.0

set fp [open $::env(OUTPUT_FILE) w]
puts $fp $power
close $fp

puts "\[INFO\] Power: $power W"
