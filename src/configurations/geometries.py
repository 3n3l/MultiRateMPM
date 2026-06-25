from src.constants import Material

from abc import ABC, abstractmethod
from typing import Tuple

import taichi as ti


class Geometry(ABC):
    def __init__(
        self,
        velocity: Tuple[float, float, float],
        frame_threshold: int,
        temperature: float,
        material: Material,
        is_continuous: bool = False,
    ) -> None:
        self.conductivity = material.Conductivity
        self.latent_heat = material.LatentHeat
        self.capacity = material.Capacity
        self.lambda_0 = material.Lambda
        self.density = material.Density
        self.color = material.Color
        self.phase = material.Phase
        self.mu_0 = material.Mu
        self.frame_threshold = frame_threshold
        self.is_continuous = is_continuous
        self.temperature = temperature
        self.velocity = list(velocity)
        self.material = material

    @abstractmethod
    def in_bounds(self, x: float, y: float, z: float) -> bool:
        pass

    @abstractmethod
    def random_seed(self) -> ti.Vector:
        pass

    @abstractmethod
    def volume(self) -> float:
        pass


class Circle(Geometry):
    def __init__(
        self,
        velocity: Tuple[float, float, float],
        center: Tuple[float, float, float],
        material: Material,
        radius: float,
        temperature: float = 0.0,
        frame_threshold: int = 0,
        is_continuous: bool = False,
    ) -> None:
        super().__init__(velocity, frame_threshold, temperature, material, is_continuous)
        self.x, self.y, self.z = list(center)
        self.squared_radius = radius * radius
        self.radius = radius

    @ti.func
    def in_bounds(self, x: float, y: float, z: float) -> bool:
        return ((self.x - x) ** 2) + ((self.y - y) ** 2) + ((self.z - z) ** 2) <= self.squared_radius

    @ti.func
    def random_seed(self) -> ti.Vector:
        r = self.radius * ti.math.sqrt(ti.random())
        t = 2 * ti.math.pi * ti.random() # theta
        p = 2 * ti.math.pi * ti.random() # phi
        x = (r * ti.sin(t) * ti.cos(p)) + self.x
        y = (r * ti.sin(t) * ti.sin(p)) + self.y
        z = (r * ti.cos(t)) + self.z
        return ti.Vector([x, y, z])

    def volume(self) -> float:
        return (4 / 3) * ti.math.pi * (self.radius ** 3)


class Rectangle(Geometry):
    def __init__(
        self,
        lower_left: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        size: Tuple[float, float, float],
        material: Material,
        temperature: float = 0.0,
        frame_threshold: int = 0,
        is_continuous: bool = False,
    ) -> None:
        super().__init__(velocity, frame_threshold, temperature, material, is_continuous)
        self.width, self.height, self.depth = size
        self.x, self.y, self.z = lower_left
        self.r_bound = self.x + self.width  # right boundary
        self.t_bound = self.y + self.height  # top boundary
        self.b_bound = self.z + self.depth  # back boundary

    @ti.func
    def in_bounds(self, x: float, y: float, z: float) -> bool:
        _in_bounds = self.x <= x <= self.r_bound
        _in_bounds &= self.y <= y <= self.t_bound
        _in_bounds &= self.z <= z <= self.b_bound
        return _in_bounds

    @ti.func
    def random_seed(self) -> ti.Vector:
        x = self.x + ti.random() * self.width
        y = self.y + ti.random() * self.height
        z = self.z + ti.random() * self.depth
        return ti.Vector([x, y, z])

    def volume(self) -> float:
        return self.width * self.height * self.depth
