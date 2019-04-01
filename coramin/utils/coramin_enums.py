from enum import IntEnum

class RelaxationSide(IntEnum):
    UNDER = 1
    OVER = 2
    BOTH = 3

class FunctionShape(IntEnum):
    LINEAR = 1
    CONVEX = 2
    CONCAVE = 3
    UNKNOWN = 4
