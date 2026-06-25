from src.configurations import Configuration
from src.simulation import BaseSimulation
from src.constants import ColorRGB
from src.samplers import PoissonDiskSampler
from src.solvers import CollocatedSolver

import taichi as ti
import numpy as np


@ti.data_oriented
class GGUI_Simulation(BaseSimulation):
    def __init__(
        self,
        configurations: list[Configuration],
        sampler: PoissonDiskSampler,
        res: tuple[int, int],
        solver: CollocatedSolver,
        quality: float,
        prefix: str,
        name: str,
        initial_configuration: int = 0,
        radius=0.0015,
    ) -> None:
        super().__init__(
            initial_configuration=initial_configuration,
            configurations=configurations,
            radius=radius,
            prefix=prefix,
            quality=quality,
            sampler=sampler,
            solver=solver,
            name=name,
        )

        # GGUI.
        self.window = ti.ui.Window(name, res, fps_limit=self.fps)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color(ColorRGB.Background)
        self.gui = self.window.get_gui()

        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 0.5, 1.95)
        self.camera.lookat(0.5, 0.4, 0.5)
        self.camera.fov(65)
        self.scene.set_camera(self.camera)

    def show_configurations(self) -> None:
        """
        Show all possible configurations inside own subwindow, choosing one will
        load that configuration and reset the solver.
        """
        prev_configuration_id = self.configuration_id
        with self.gui.sub_window("Configurations", 0.01, 0.01, 0.65, 0.64) as subwindow:
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if subwindow.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                _id = self.configuration_id
                configuration = self.configurations[_id]
                self.load_configuration(configuration)
                self.is_paused = True

    def show_parameters(self) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        """
        with self.gui.sub_window("Parameters", 0.01, 0.66, 0.98, 0.33) as subwindow:
            self.solver.ambient_temperature[None] = subwindow.slider_int(
                text="Ambient Temperature",
                old_value=int(self.solver.ambient_temperature[None]),  # pyright: ignore
                minimum=-273,
                maximum=273,
            )
            self.solver.boundary_temperature[None] = subwindow.slider_int(
                text="Boundary Temperature",
                old_value=int(self.solver.boundary_temperature[None]),  # pyright: ignore
                minimum=-273,
                maximum=273,
            )
            # self.solver.gravity[None] = subwindow.slider_float(
            #     text="Gravity",
            #     old_value=float(self.solver.gravity[None]),  # pyright: ignore
            #     minimum=0,
            #     maximum=-9.81,
            # )
            # # NOTE: dt needs to be scaled, otherwise the precision of slider_float is not enough
            # self.solver.dt[None] = (
            #     subwindow.slider_float(
            #         text="10 * dt",
            #         old_value=self.solver.dt[None] * 10,  # pyright: ignore
            #         minimum=1e-3,
            #         maximum=3e-2,
            #     )
            #     / 10
            # )

    def show_buttons(self) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        """
        with self.gui.sub_window("Settings", 0.67, 0.01, 0.32, 0.64) as subwindow:
            if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
                # This button toggles between saving frames and not saving frames.
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.setup_video_manager()
                else:
                    self.create_video()
            if subwindow.button(" Reset Particles "):
                self.reset()
            if subwindow.button(" Start Simulation"):
                self.is_paused = False

            self.should_create_video = subwindow.checkbox("Create Video", self.should_create_video)
            self.should_create_gif = subwindow.checkbox("Create GIF", self.should_create_gif)

            _should_write_particle = self.should_write_particles
            self.should_write_particles = subwindow.checkbox("Export Particles", self.should_write_particles)
            if not _should_write_particle and self.should_write_particles:
                self.setup_ply_writer()

    def show_settings(self) -> None:
        """
        Show settings in a GGUI subwindow, this should be called once per generated frames
        and will only show these settings if the simulation is paused at the moment.
        """
        if not self.is_paused or not self.should_show_settings:
            self.is_showing_settings = False
            return  # don't bother

        self.is_showing_settings = True
        self.show_configurations()
        # self.show_parameters()
        self.show_buttons()

    def handle_events(self) -> None:
        """
        Handle key presses arising from window events.
        """
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in ["h"]:
                self.should_show_settings = not self.should_show_settings
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.setup_video_manager()
                else:
                    self.create_video()
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def render(self) -> None:
        """Render the simulation."""
        self.scene.particles(
            per_vertex_color=self.solver.color_p,
            centers=self.solver.position_p,
            radius=self.radius,
        )

        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        point_color = (0.85, 0.85, 0.85)
        self.scene.point_light(pos=(-1.0, 1.5, -1.0), color=point_color)
        self.scene.point_light(pos=(-1.0, 1.5, 2.0), color=point_color)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.canvas.scene(self.scene)

        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())

        if self.should_write_particles and not self.is_paused and not self.is_showing_settings:
            np_position = np.reshape(self.solver.position_p.to_numpy(), (self.solver.max_particles, 3))
            # np_colors = np.reshape(self.solver.color_p.to_numpy(), (self.solver.max_particles, 3))
            self.writer.add_vertex_pos(np_position[:, 0], np_position[:, 1], np_position[:, 2])
            # self.writer.add_vertex_rgba(np_colors[:, 0], np_colors[:, 1], np_colors[:, 2], 1.0)
            self.writer.export_frame_ascii(self.particle_frame_count, self.particle_output_path)

        self.window.show()

    def run(self) -> None:
        """Run the simulation."""
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
            self.render()
