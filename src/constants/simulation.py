from dataclasses import dataclass


@dataclass
class Classification:
    Empty = 22
    Colliding = 33
    Interior = 44
    Insulated = 55
    Unknown = 66


@dataclass
class Simulation:
    """Defines parameters for the simulation."""

    MinTemperature = -273.15
    MaxTemperature = 100.0
