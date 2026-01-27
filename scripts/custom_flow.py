from openlane.flows import SequentialFlow
from openlane.steps import Yosys, Misc, OpenROAD, Magic, Netgen

class MyFlow(SequentialFlow):
    Steps = [
        Yosys.Synthesis,
        OpenROAD.CheckSDCFiles,
        OpenROAD.Floorplan,
        OpenROAD.TapEndcapInsertion,
        OpenROAD.GeneratePDN,
        OpenROAD.IOPlacement,
        OpenROAD.GlobalPlacement,
        OpenROAD.DetailedPlacement,
        OpenROAD.GlobalRouting,
        OpenROAD.DetailedRouting,
        OpenROAD.FillInsertion,
        Magic.StreamOut,
        Magic.DRC,
        Magic.SpiceExtraction,
        Netgen.LVS
    ]


flow = MyFlow(

    {
        "DESIGN_NAME": "picorv32a",
        "VERILOG_FILES": ["/home/isaishaq/openlane2/designs/picorv_test/src/picorv32a.v"],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": 10,
    },
        design_dir="/home/isaishaq/openlane2/designs/picorv_test",
        pdk="sky130A",
        scl="sky130_fd_sc_hd",
        pdk_root="/home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af",
)
flow.start()

