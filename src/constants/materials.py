from src.constants.colors import ColorRGB
from dataclasses import dataclass


@dataclass
class Material:
    """Defines parameters that represent a material."""

    Conductivity: float = 0.0
    LatentHeat: float = 0.0
    Capacity: float = 0.0
    Density: float = 0.0
    Lambda: float = 0.0
    Color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    Phase: int = 0
    Mu: float = 0.0

    # TODO: proper default variables?!
    E = 1.0
    nu = 1.0
    Zeta: int = 10
    Theta_s: float = 0.0
    Theta_c: float = 0.0


@dataclass
class Water(Material):
    Conductivity = 0.55
    # LatentHeat = 334.4
    LatentHeat = 0.3344
    Capacity = 4.186  # j/dC
    Density = 997.0
    Lambda = 1e32
    Color = ColorRGB.Water
    Phase = 0
    Mu = 0


@dataclass
class Ice(Material):
    # E = 8.5e5
    # nu = 0.1
    E = 5e3
    nu = 0.3
    Conductivity = 2.33
    LatentHeat = 0.0
    Capacity = 2.093  # j/dC
    Density = 917.0
    Color = ColorRGB.Ice
    Phase = 1
    Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    Mu = E / (2 * (1 + nu))
