from src.configurations import Configuration
from abc import ABC

import taichi as ti


@ti.data_oriented
class CollocatedSolver(ABC):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        self.max_particles = max_particles
        self.inv_dx = float(n_grid)
        self.n_grid = n_grid
        self.dx = 1 / n_grid
        self.vol_0_p = vol_0
        self.d = 3  # 3D

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.wx = self.n_grid + self.boundary_width + self.boundary_width
        self.wy = self.wx
        self.wz = self.wx
        self.w_offset = (-self.boundary_width, -self.boundary_width, -self.boundary_width)
        self.negative_boundary = -self.boundary_width
        self.positive_boundary = self.n_grid + self.boundary_width

        # Variables accessed by kernels must be stored in fields:
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.gravity = ti.Vector.field(self.d, dtype=ti.f64, shape=())
        self.dt = ti.field(dtype=ti.f64, shape=())

        # # Properties on cell centers:
        # # self.classification_c = ti.field(dtype=ti.i32, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        # # self.temperature_c = ti.field(dtype=ti.f64, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        # self.velocity_c = ti.Vector.field(self.d, dtype=ti.f64, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        # self.mass_c = ti.field(dtype=ti.f64, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)

        # Properties on particles:
        self.velocity_p = ti.Vector.field(self.d, dtype=ti.f64, shape=max_particles)
        self.position_p = ti.Vector.field(self.d, dtype=ti.f64, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=ti.f64, shape=max_particles)
        self.phase_p = ti.field(dtype=ti.f64, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.f64, shape=max_particles)

    @ti.func
    def compute_quadratic_kernel(self, distance: ti.template()) -> ti.template():  # pyright: ignore
        """
        Quadratic kernels [JST16 eq. 123], with x=fx, fx-1, fx-2).
        Based on https://www.bilibili.com/opus/662560355423092789

        ---
        Arguments:
            - distance: vector, distance between base cell and particle position
        """
        return [0.5 * (1.5 - distance) ** 2, 0.75 - (distance - 1.0) ** 2, 0.5 * (distance - 0.5) ** 2]

    @ti.func
    def compute_quadratic_gradient(self, distance: ti.template()) -> ti.template():  # pyright: ignore
        """
        Quadratic gradients, with x=fx, fx-1, fx-2).
        Based on https://www.bilibili.com/opus/662560355423092789

        ---
        Arguments:
            - distance: vector, distance between base cell and particle position
        """
        return [distance - 1.5, -2.0 * (distance - 1), distance - 0.5]

    def reset(self, configuration: Configuration, quality: float):
        self.gravity[None] = configuration.gravity
        # self.dt[None] = configuration.dt / quality
        self.position_p.fill([42, 42] if self.d == 2 else [42, 42, 42])
        self.n_particles[None] = 0

    def substep(self):
        pass

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        pass
