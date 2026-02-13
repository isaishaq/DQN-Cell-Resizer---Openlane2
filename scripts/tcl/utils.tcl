proc print_name {objs} {
    foreach obj $objs {
        puts "\[INFO\] Object name: [get_full_name [lindex $obj 0]]"
    }
}