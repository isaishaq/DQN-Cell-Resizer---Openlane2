# Save the database
# Environment variables expected:
#   OUTPUT_PATH - Path to save the database

write_db $::env(OUTPUT_PATH)

puts "\[INFO\] Database saved to $::env(OUTPUT_PATH)"
