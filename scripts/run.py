from custom_flow import CustomFlow

flow = CustomFlow(
    {
        # Required
        "DESIGN_NAME": "picorv32a",
        "VERILOG_FILES": ["/home/isaishaq/openlane2/designs/picorv_test/src/picorv32a.v"],
        
        # General
        "PDK": "sky130A",
        "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
        "STD_CELL_LIBRARY_OPT": "sky130_fd_sc_hvl", 

        # STA
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": 20,
        "CLOCK_NET": "clk",

        # Floorplanning
        "DESIGN_REPAIR_REMOVE_BUFFERS": True,
        "GPL_CELL_PADDING": 2,
        "DPL_CELL_PADDING": 2,
        "FP_CORE_UTIL": 40,


        "SYNTH_AUTONAME": True,
        #"RUN_POST_GRT_RESIZER_TIMING": True,
        #"GLB_RESIZER_TIMING_OPTIMIZATIONS": True,

        "DQN_TRAINING_MODE": True,
        "DQN_MODEL_PATH": "/home/isaishaq/openlane2/designs/picorv_test/models/dqn_model.pth",
        "DQN_MAX_ITERATIONS": 100,
    },
        design_dir="/home/isaishaq/openlane2/designs/picorv_test",
        #pdk="sky130A",
        #scl="sky130_fd_sc_hd",
        pdk_root="/home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af",
)

flow.start()


