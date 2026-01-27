from openlane.flows import SequentialFlow
from openlane.steps import OpenROAD, Odb, Checker

class Compile(SequentialFlow):
    Steps = [
        OpenROAD.GlobalPlacement,
        Odb.WriteVerilogHeader,
        Checker.PowerGridViolations,
        OpenROAD.STAMidPNR,
        OpenROAD.RepairDesignPostGPL,
        Odb.ManualGlobalPlacement,
        OpenROAD.DetailedPlacement,
    ]

flow = Compile()


