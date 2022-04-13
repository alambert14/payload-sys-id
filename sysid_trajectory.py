from pydrake.all import (
    MathematicalProgram,
    MathematicalProgramResult,
)

class SysIDTrajectory:

    def __init__(self):

        self.prog = MathematicalProgram()
