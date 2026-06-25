from src.configurations.geometries import Geometry

import taichi as ti


@ti.data_oriented
class Configuration:
    """This class represents a starting configuration for the MLS-MPM algorithm."""

    def __init__(
        self,
        name: str,
        geometries: list[Geometry],
        dt=1e-4,
        gravity=[0.0, -9.81, 0.0],  # Gravity
        ambient_temperature=0.0,  # temperature of empty cells
        boundary_temperature=0.0,  # temperature of boundary cells
        information: str = "",
    ):
        self.dt = dt
        self.name = name
        self.gravity = gravity
        self.information = information
        self.ambient_temperature = ambient_temperature
        self.boundary_temperature = boundary_temperature

        self.geometries = geometries
        self.initial_geometries = []
        self.discrete_geometries = []
        self.continuous_geometries = []
        for geometry in geometries:
            if geometry.frame_threshold == 0:
                self.initial_geometries.append(geometry)
            elif geometry.is_continuous:
                self.continuous_geometries.append(geometry)
            else:
                self.discrete_geometries.append(geometry)

        # Sort this by frame_threshold, so only the first element has to be checked against.
        self.continuous_geometries.sort(key=(lambda g: g.frame_threshold))
        self.discrete_geometries.sort(key=(lambda g: g.frame_threshold))

    def volume(self) -> float:
        _volume = 0.0
        for geometry in self.geometries:
            _volume += geometry.volume()
        return _volume
