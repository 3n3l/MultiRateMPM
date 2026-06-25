from src.configurations import Configuration
from src.constants import Ice, Water
from src.solvers import CollocatedSolver

from typing import override

import taichi.math as tm
import taichi as ti


@ti.data_oriented
class MultiRateMPM(CollocatedSolver):
    def __init__(self, max_particles: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_grid, vol_0)

        # Particle properties:
        # self.theta_c_p = ti.field(dtype=ti.f64, shape=max_particles)
        # self.theta_s_p = ti.field(dtype=ti.f64, shape=max_particles)
        # self.lambda_p = ti.field(dtype=ti.f64, shape=max_particles)
        # self.mu_0_p = ti.field(dtype=ti.f64, shape=max_particles)
        # self.zeta_p = ti.field(dtype=ti.i32, shape=max_particles)
        self.FE_p = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f64, shape=max_particles)
        self.C_p = ti.Matrix.field(3, 3, dtype=ti.f64, shape=max_particles)
        self.damage_p = ti.field(dtype=ti.f64, shape=max_particles)  # phase field for damage: 0 - damaged, 1 - intact

        # cfl = 6.0
        cfl = 0.6

        # dt_water = cfl * self.dx / tm.sqrt(Water.Lambda)
        dt_water = cfl * self.dx / tm.sqrt(Ice.Lambda)
        dt_solid = cfl * self.dx / tm.sqrt(2 * Ice.Mu + Ice.Lambda)
        # dt_water = 1e-3
        # dt_solid = 1e-4
        print(f"dt_water: {dt_water:<.2E}, dt_solid: {dt_solid:<.2E}")
        self.dt[None] = dt_water

        # # Async MPM extension
        # self.T = ti.field(dtype=ti.i32, shape=())  # global accumulated timestep multiplier
        # self.M_prime, self.M = ti.field(dtype=ti.i32, shape=()), ti.field(dtype=ti.i32, shape=())
        # block_size = 8
        # n_block = n_grid // block_size
        # self.Mblock = ti.field(dtype=ti.i32, shape=(n_block, n_block))
        self.sigma_max = ti.field(dtype=ti.f64, shape=())
        self.sig_y_0 = 0.005

        # self.temperature_c = ti.field(dtype=ti.f64, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        w_shape1 = (self.wx, self.wy, self.wz)
        w_offset1 = (-self.boundary_width, -self.boundary_width, -self.boundary_width)
        w_shape2 = (2, self.wx, self.wy, self.wz)
        w_offset2 = (0, -self.boundary_width, -self.boundary_width, -self.boundary_width)
        self.grids_v = ti.Vector.field(self.d, dtype=ti.f64, shape=w_shape2, offset=w_offset2)
        self.grids_m = ti.field(dtype=ti.f64, shape=w_shape2, offset=w_offset2)
        self.grids_n = ti.Vector.field(self.d, dtype=ti.f64, shape=w_shape2, offset=w_offset2)
        self.grid_n = ti.Vector.field(self.d, dtype=ti.f64, shape=w_shape1, offset=w_offset1)
        self.grid_d = ti.field(dtype=ti.f64, shape=w_shape1, offset=w_offset1)  # damage phase field for solid

    @ti.func
    @override
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position
        # self.theta_c_p[index] = geometry.material.Theta_c
        # self.theta_s_p[index] = geometry.material.Theta_s
        # self.lambda_p[index] = geometry.material.Lambda
        self.color_p[index] = geometry.material.Color
        self.phase_p[index] = geometry.phase
        # self.zeta_p[index] = geometry.material.Zeta
        # self.mu_0_p[index] = geometry.material.Mu

        # Set properties to default values:
        self.mass_p[index] = self.vol_0_p # TODO: this is using density = 1.0 right now
        # self.mass_p[index] = self.vol_0_p * geometry.density
        self.FE_p[index] = ti.Matrix.identity(ti.f64, 3)
        self.JP_p[index] = 1.0
        self.C_p[index] = ti.Matrix.zero(ti.f64, 3, 3)

    def reset(self, configuration: Configuration, quality: float):
        # self.boundary_temperature[None] = configuration.boundary_temperature
        # self.ambient_temperature[None] = configuration.ambient_temperature
        self.gravity[None] = configuration.gravity
        # self.dt[None] = configuration.dt / quality
        self.damage_p.fill(1)
        self.position_p.fill([42, 42] if self.d == 2 else [42, 42, 42])
        self.n_particles[None] = 0

    @ti.kernel
    def reset_block_mark(self):
        self.Mblock.fill(0)

    # @ti.kernel
    # def reset_grids(self):
    #     for i, j, k in self.mass_c:
    #         self.velocity_c[i, j, k] = 0
    #         self.mass_c[i, j, k] = 0
    #
    #     for p in ti.ndrange(self.n_particles[None]):
    #         self.damage_p[p] = 1
    #
    #     self.sigma_max[None] = 0

    @ti.kernel
    def reset_grids(self):
        self.grids_v.fill(0)
        self.grids_m.fill(0)
        self.grids_n.fill(0)
        self.grid_n.fill(0)
        self.grid_d.fill(1)  # Initialize damage phase field for solid
        self.sigma_max[None] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # Evolve deformation gradient:
            self.FE_p[p] += (self.dt[None] * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # self.FE_p[p] = (ti.Matrix.identity(ti.f64, self.d) + self.dt[None] * self.C_p[p]) @ self.FE_p[p]

            U, sigma, V = ti.svd(self.FE_p[p])
            JE, new_sig = 1.0, sigma
            sig_diag = ti.Vector([sigma[d, d] for d in ti.static(range(self.d))])

            if self.phase_p[p] == Ice.Phase:
                eps = tm.log(sig_diag)
                eps_mean = eps.sum() / self.d  # pyright: ignore
                eps_dev = eps - eps_mean
                eps_dev_norm = eps_dev.norm()
                sig_y = self.sig_y_0 + 0.005 * (1.0 - self.JP_p[p]) ** 0.2  # simple hardening model
                if eps_dev_norm > sig_y:
                    eps_dev *= sig_y / eps_dev_norm
                for d in ti.static(ti.ndrange(self.d)):
                    new_sig[d, d] = tm.exp(eps_mean + eps_dev[d])

            # Evolve local volume change
            for d in ti.static(ti.ndrange(self.d)):
                self.JP_p[p] *= sigma[d, d] / new_sig[d, d]
                sigma[d, d] = new_sig[d, d]
                JE *= new_sig[d, d]

            stress = ti.Matrix.zero(ti.f64, self.d, self.d)
            if self.phase_p[p] == Water.Phase:
                # self.FE_p[p] = ti.Matrix.identity(ti.f64, self.d) (JE ** (1 / self.d))
                self.FE_p[p] = (JE ** (1 / self.d)) * ti.Matrix.identity(ti.f64, self.d)
                stress = 2 * Water.Mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
                stress += ti.Matrix.identity(ti.f64, self.d) * Ice.Lambda * JE * (JE - 1)
            else:
                self.FE_p[p] = U @ sigma @ V.transpose()
                mu, la = Ice.Mu, Ice.Lambda
                I = ti.Matrix.identity(ti.f64, self.d)
                if JE >= 1:
                    # stress = (1 - 0.001) * self.damage_p[p] ** 2 + 0.001
                    # stress *= 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
                    # stress += I * la * JE * (JE - 1)

                    stress = ((1 - 0.001) * self.damage_p[p] ** 2 + 0.001) * (
                        2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose() + I * la * JE * (JE - 1)
                    )
                else:
                    # stress = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
                    # stress += I * la * JE * (JE - 1)

                    stress = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose() + I * la * JE * (JE - 1)

            stress *= -self.dt[None] * self.vol_0_p * 4 * self.inv_dx * self.inv_dx
            if self.phase_p[p] == Ice.Phase:
                eig_vals, _ = ti.sym_eig(stress)
                sig_max = tm.max(eig_vals[0], eig_vals[1], eig_vals[2])
                ti.atomic_max(self.sigma_max[None], sig_max)
                if sig_max > 0.02:
                    self.damage_p[p] = tm.max(0, tm.min(self.damage_p[p], 1.0 + 200.0 * (1.0 - sig_max / 0.02)))  # simple damage model

            # # Clamp singular values to apply plasticity:
            # U, sigma, V = ti.svd(self.FE_p[p])
            # JE = 1.0
            # for d in ti.static(range(3)):
            #     singular_value = float(sigma[d, d])
            #     singular_value = max(singular_value, 1 - self.theta_c_p[p])
            #     singular_value = min(singular_value, 1 + self.theta_s_p[p])
            #     self.JP_p[p] *= sigma[d, d] / singular_value
            #     sigma[d, d] = singular_value
            #     JE *= singular_value

            # # Reconstruct elastic deformation gradient after plasticity
            # self.FE_p[p] = U @ sigma @ V.transpose()

            # # Apply snow stran hardening by adjusting Lame parameters
            # h = ti.max(0.1, ti.min(20, ti.exp(self.zeta_p[p] * (1.0 - self.JP_p[p]))))
            # mu, la = self.mu_0_p[p] * h, self.lambda_p[p] * h

            # # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            # piola_kirchhoff = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore
            # piola_kirchhoff += ti.Matrix.identity(float, 3) * la * JE * (JE - 1)

            # # Cauchy stress times dt and D_inv
            # cauchy_stress = -self.dt[None] * self.vol_0_p * 4 * self.inv_dx * self.inv_dx * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = stress + self.mass_p[p] * self.C_p[p]

            # Lower left corner of the interpolation grid:
            # Based on https://www.bilibili.com/opus/662560355423092789
            base = ti.floor((self.position_p[p] * self.inv_dx - 0.5), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist = self.position_p[p] * self.inv_dx - ti.cast(base, ti.f64)

            # Quadratic kernels:
            w = self.compute_quadratic_kernel(dist)
            wd = self.compute_quadratic_gradient(dist)

            # Rasterize mass and velocity
            # mat_id = ti.cast(ti.i32, self.phase_p[p])
            # mat_id = int(self.phase_p[p])
            mat_id = 0 if self.phase_p[p] == Water.Phase else 1
            mass_p = self.mass_p[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j, k])
                dpos = ti.cast(offset - dist, ti.f64) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                momentum = self.mass_p[p] * self.velocity_p[p] + affine @ dpos  # pyright: ignore
                # self.mass_c[base + offset] += weight * self.mass_p[p]
                # self.velocity_c[base + offset] += weight * v

                self.grids_v[mat_id, base + offset] += weight * momentum
                self.grids_m[mat_id, base + offset] += weight * mass_p
                self.grids_n[mat_id, base + offset] += weight * mass_p * ti.Vector([wd[i][0], wd[j][1], wd[k][2]]).normalized()

                # self.grids_v[mat_id, base[0] + i, base[1] + j, base[2] + k] += weight * momentum
                # self.grids_m[mat_id, base[0] + i, base[1] + j, base[2] + k] += weight * mass_p
                # self.grids_n[mat_id, base[0] + i, base[1] + j, base[2] + k] += weight * mass_p * ti.Vector([wd[i][0], wd[j][1], wd[k][2]]).normalized()

    # @ti.kernel
    # def momentum_to_velocity(self):
    #     for i, j, k in self.mass_c:
    #         # Normalize velocity, add gravity:
    #         if self.mass_c[i, j, k] > 0:
    #             self.velocity_c[i, j, k] /= self.mass_c[i, j, k]
    #             self.velocity_c[i, j, k][1] += self.dt[None] * self.gravity[None]
    #
    #         # Free-slip simulation boundary:
    #         if i < 0 or i > self.n_grid:
    #             self.velocity_c[i, j, k][0] = 0
    #         if j < 0 or j > self.n_grid:
    #             self.velocity_c[i, j, k][1] = 0
    #         if k < 0 or k > self.n_grid:
    #             self.velocity_c[i, j, k][2] = 0

    @ti.kernel
    def grid_op(self):
        for i, j, k in ti.ndrange(self.wx, self.wy, self.wz):
            if self.grids_m[0, i, j, k] > 0 and self.grids_m[1, i, j, k] > 0:  # No need for epsilon here
                # print(self.grids_m[0, i, j, k], self.grids_m[1, i, j, k])
                m_reduced = self.grids_m[0, i, j, k] * self.grids_m[1, i, j, k] / (self.grids_m[0, i, j, k] + self.grids_m[1, i, j, k])
                v_rel = self.grids_v[0, i, j, k] / self.grids_m[0, i, j, k] - self.grids_v[1, i, j, k] / self.grids_m[1, i, j, k]
                n_est = -(self.grids_n[0, i, j, k].normalized() - self.grids_n[1, i, j, k].normalized()).normalized()
                # if self.grids_m[0, i, j, k] < 1e-4: n_est = self.grids_n[1, i, j, k].normalized()
                self.grid_n[i, j, k] = n_est
                # vn = v_rel.dot(n_est)  # pyright: ignore
                vn = v_rel @ n_est
                if vn < 0:
                    f_nor = m_reduced * vn * n_est
                    vt = v_rel - vn * n_est
                    vt_prime = vt * tm.max(0, 1 - 0.3 * abs(vn) / tm.max(abs(vt), 1e-5))  # simple Coulomb friction model
                    self.grids_v[0, i, j, k] -= f_nor + vt_prime * m_reduced  # mv_rel.dot(n_est) * n_est # v_rel * m_reduced # f_nor
                    self.grids_v[1, i, j, k] += f_nor + vt_prime * m_reduced  # mv_rel.dot(n_est) * n_est # v_rel * m_reduced # f_nor

            for n in ti.ndrange(2):
                if self.grids_m[n, i, j, k] > 0:  # No need for epsilon here
                    # Momentum to velocity
                    self.grids_v[n, i, j, k] = self.grids_v[n, i, j, k] / self.grids_m[n, i, j, k]
                    self.grids_v[n, i, j, k] += self.dt[None] * self.gravity[None]  # gravity

                    # if i < 0 and self.grids_v[n, i, j, k][0] < 0:
                    #     self.grids_v[n, i, j, k][0] = 0  # Boundary conditions
                    # if i > self.n_grid and self.grids_v[n, i, j, k][0] > 0:
                    #     self.grids_v[n, i, j, k][0] = 0
                    # if j < 0 and self.grids_v[n, i, j, k][1] < 0:
                    #     self.grids_v[n, i, j, k][1] = 0
                    # if j > self.n_grid and self.grids_v[n, i, j, k][1] > 0:
                    #     self.grids_v[n, i, j, k][1] = 0
                    # if k < 0 and self.grids_v[n, i, j, k][2] < 0:
                    #     self.grids_v[n, i, j, k][2] = 0
                    # if k > self.n_grid and self.grids_v[n, i, j, k][2] > 0:
                    #     self.grids_v[n, i, j, k][2] = 0

                    if i < 0 or i >= self.n_grid:
                        self.grids_v[n, i, j, k][0] = 0
                    if j < 0 or j >= self.n_grid:
                        self.grids_v[n, i, j, k][1] = 0
                    if k < 0 or k >= self.n_grid:
                        self.grids_v[n, i, j, k][2] = 0

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # Lower left corner of the interpolation grid:
            # Based on https://www.bilibili.com/opus/662560355423092789
            base = ti.floor((self.position_p[p] * self.inv_dx - 0.5), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist = self.position_p[p] * self.inv_dx - ti.cast(base, ti.f64)

            # Quadratic kernels:
            w = self.compute_quadratic_kernel(dist)

            C = ti.Matrix.zero(ti.f64, 3, 3)
            v = ti.Vector.zero(ti.f64, 3)
            # mat_id = ti.cast(ti.i32, self.phase_p[p])
            mat_id = 0 if self.phase_p[p] == Water.Phase else 1
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j, k]).cast(ti.f64) - dist
                offset = ti.Vector([i, j, k])
                g_v = self.grids_v[mat_id, base + offset]
                # g_v = self.grids_v[mat_id, base[0] + i, base[1] + j, base[2] + k]
                weight = w[i][0] * w[j][1] * w[k][2]
                C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                v += weight * g_v

            # print(C)
            self.velocity_p[p], self.C_p[p] = v, C
            self.position_p[p] += self.dt[None] * v
            self.position_p[p] = tm.clamp(self.position_p[p], 0, 1)

    @override
    def substep(self):
        self.reset_grids()
        self.particle_to_grid()
        self.grid_op()
        self.grid_to_particle()
