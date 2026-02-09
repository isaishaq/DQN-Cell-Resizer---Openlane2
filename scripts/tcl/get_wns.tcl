# Get Worst Negative Slack
# Environment variables expected:
#   OUTPUT_FILE - Output file path

set wns [sta::worst_slack]

set fp [open $::env(OUTPUT_FILE) w]
puts $fp $wns
close $fp

puts "\[INFO\] WNS: $wns ns"
