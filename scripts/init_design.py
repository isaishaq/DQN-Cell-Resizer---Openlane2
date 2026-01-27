from openlane.flows import SequentialFlow
from openlane.steps import Yosys, OpenROAD, Odb, Checker, Verilator

class Init_Design(SequentialFlow):
    Steps = [
        Verilator.Lint,
        Checker.LintTimingConstructs,
        Checker.LintErrors,
        Checker.LintWarnings,
        Yosys.JsonHeader,
        Yosys.Synthesis,
        Checker.YosysUnmappedCells,
        Checker.YosysSynthChecks,
        Checker.NetlistAssignStatements,
        OpenROAD.CheckSDCFiles,
        OpenROAD.CheckMacroInstances,
        OpenROAD.STAPrePNR,
        OpenROAD.Floorplan,
        Odb.CheckMacroAntennaProperties,
        Odb.SetPowerConnections,
        Odb.ManualMacroPlacement,
        OpenROAD.CutRows,
        OpenROAD.TapEndcapInsertion,
        Odb.AddPDNObstructions,
        OpenROAD.GeneratePDN,
        Odb.RemovePDNObstructions,
        Odb.AddRoutingObstructions,
        OpenROAD.GlobalPlacementSkipIO,
        OpenROAD.IOPlacement,
        Odb.CustomIOPlacement,
        Odb.ApplyDEFTemplate,
    ]

flow = Init_Design(

    {
        "DESIGN_NAME": "picorv32a",
        "VERILOG_FILES": ["/home/isaishaq/openlane2/designs/picorv_test/src/picorv32a.v"],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": 10,
        "CLOCK_NET": "clk",
        "GLB_RESIZER_TIMING_OPTIMIZATIONS": True,
        "DESIGN_REPAIR_REMOVE_BUFFERS": True,
        "GPL_CELL_PADDING": 2,
        "DPL_CELL_PADDING": 2,
        "FP_CORE_UTIL": 35,
        "SYNTH_AUTONAME": True
    },
        design_dir="/home/isaishaq/openlane2/designs/picorv_test",
        pdk="sky130A",
        scl="sky130_fd_sc_hd",
        pdk_root="/home/isaishaq/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af",
)
flow.start()