# Get Total Negative Slack
# Environment variables expected:
#   OUTPUT_FILE - Output file path

set tns [sta::total_negative_slack]

set fp [open $::env(OUTPUT_FILE) w]
puts $fp $tns
close $fp

puts "\[INFO\] TNS: $tns ns"
