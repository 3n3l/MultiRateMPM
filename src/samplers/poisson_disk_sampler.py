from src.solvers import CollocatedSolver
from abc import ABC

import taichi as ti


@ti.data_oriented
class PoissonDiskSampler(ABC):
    def __init__(
        self,
        solver: CollocatedSolver,
        r: float = 0.002,
        k: int = 30,
    ) -> None:
        # Some of the solver's constants wills be used:
        self.solver = solver

        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(3)  # Cell size is bounded by this (3D)
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # Initialize an n-dimension background grid to store samples:
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid, self.n_grid))

        # We can't use a resizable list, so we point to the head and tail:
        self._head = ti.field(ti.i32, shape=())
        self._tail = ti.field(ti.i32, shape=())

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y, z = self._point_to_index(base_point)
        xs = (ti.max(0, x - 2), ti.min(self.n_grid, x + 3))  # pyright: ignore
        ys = (ti.max(0, y - 2), ti.min(self.n_grid, y + 3))  # pyright: ignore
        zs = (ti.max(0, z - 2), ti.min(self.n_grid, z + 3))  # pyright: ignore
        distance_min = ti.math.inf  # initialize as maximum possible distance
        for i, j, k in ti.ndrange(xs, ys, zs):
            if (index := self.background_grid[i, j, k]) != -1:
                # We found a point and can compute the distance:
                found_point = self.solver.position_p[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance

        return distance_min < self.r

    @ti.func
    def _in_bounds(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        in_bounds = geometry.in_bounds(point[0], point[1], point[2])  # in geometry bounds
        in_bounds &= 0.0 < point[0] < 1.0  # in simulation bounds
        in_bounds &= 0.0 < point[1] < 1.0  # in simulation bounds
        in_bounds &= 0.0 < point[2] < 1.0  # in simulation bounds
        return in_bounds

    @ti.func
    def _point_to_index(self, point: ti.template()) -> ti.Vector:  # pyright: ignore
        _index = ti.cast((point * self.n_grid), dtype=ti.i32)  # pyright: ignore
        return ti.math.clamp(_index, 0, self.n_grid)

    @ti.func
    def _point_fits(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        point_has_been_found = not self._has_collision(point)  # no collision
        point_has_been_found &= self._in_bounds(point, geometry)  # in bounds
        return point_has_been_found

    @ti.func
    def _can_sample_more_points(self) -> bool:
        return (self._head[None] < self._tail[None]) and (self._head[None] < self.solver.max_particles - 1)

    @ti.func
    def _initialize_grid(self):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            self.background_grid[i, j, k] = -1
        for p in ti.ndrange(self.solver.n_particles[None]):
            index = self._point_to_index(self.solver.position_p[p])
            self.background_grid[index] = p

    @ti.func
    def _generate_point_around(self, prev_position: ti.template()) -> ti.Vector:  # pyright: ignore
        t = ti.random() * 2 * ti.math.pi  # theta
        p = ti.random() * 2 * ti.math.pi  # phi
        x = ti.sin(t) * ti.cos(p)
        y = ti.sin(t) * ti.sin(p)
        z = ti.cos(t)
        offset = ti.Vector([x, y, z])
        offset *= (1 + ti.random()) * self.r
        return prev_position + offset

    @ti.func
    def _generate_initial_point(self, geometry: ti.template()) -> ti.Vector:  # pyright: ignore
        initial_point = geometry.random_seed()

        n_samples = 0  # otherwise this might not halt
        while not self._point_fits(initial_point, geometry) and n_samples < self.k:
            initial_point = geometry.random_seed()
            n_samples += 1

        index = self._point_to_index(initial_point)
        self.background_grid[index] = self._head[None]

        return initial_point

    @ti.kernel
    def add_geometry(self, geometry: ti.template()):  # pyright: ignore
        # Initialize background grid to the current positions:
        self._initialize_grid()

        # Find a good initial point for this sample run:
        initial_point = self._generate_initial_point(geometry)
        self.solver.add_particle(self.solver.n_particles[None], initial_point, geometry)
        self._tail[None] = self.solver.n_particles[None] + 1
        self._head[None] = self.solver.n_particles[None]
        self.solver.n_particles[None] += 1

        while self._can_sample_more_points():
            prev_position = self.solver.position_p[self._head[None]]
            self._head[None] += 1  # Increment on each iteration
            for _ in range(self.k):
                next_position = self._generate_point_around(prev_position)
                next_index = self._point_to_index(next_position)
                if self._point_fits(next_position, geometry):
                    self.background_grid[next_index] = self._tail[None]
                    self.solver.add_particle(self._tail[None], next_position, geometry)
                    self.solver.n_particles[None] += 1  # Increment when point is found
                    self._tail[None] += 1  # Increment when point is found
