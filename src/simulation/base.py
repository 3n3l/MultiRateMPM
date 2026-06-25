from src.configurations import Configuration, Geometry
from src.samplers import PoissonDiskSampler
from src.solvers import CollocatedSolver

from abc import abstractmethod
from datetime import datetime

import taichi as ti
import math, os


@ti.data_oriented
class BaseSimulation:
    def __init__(
        self,
        configurations: list[Configuration],
        sampler: PoissonDiskSampler,
        solver: CollocatedSolver,
        quality: float,
        radius: float,
        prefix: str,
        name: str,
        initial_configuration: int = 0,
        fps: int = 60,
    ) -> None:
        """Constructs a Renderer object, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        # State.
        self.quality = quality
        self.is_paused = True
        self.should_write_to_disk = False
        self.should_write_particles = False
        self.should_create_video = True
        self.should_create_gif = False
        self.is_showing_settings = not self.is_paused  # wether the settings are showing
        self.should_show_settings = True  # wether the settings should be shown
        self.radius = radius
        self.fps = fps

        # Name and video/gif prefix
        self.video_prefix = prefix
        self.name = name

        # Solvers and samplers
        self.sampler = sampler
        self.solver = solver

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created frames, videos and GIFs.
        self.frame_parent_dir = ".output"
        if not os.path.exists(self.frame_parent_dir):
            os.makedirs(self.frame_parent_dir)

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created .ply files
        self.particle_frame_count = 0  # separate counter for .ply files
        self.particle_parent_dir = ".particles"
        if not os.path.exists(self.particle_parent_dir):
            os.makedirs(self.particle_parent_dir)
        self.writer = ti.tools.PLYWriter(num_vertices=self.solver.max_particles)

        # Store configurations, sort them and add information:
        self.current_frame = 0
        self.configurations = configurations
        self.configurations.sort(key=lambda c: str.lower(c.name), reverse=False)
        max_length = len(max(self.configurations, key=lambda c: len(c.name)).name)
        for i, c in enumerate(self.configurations):
            name = self.configurations[i].name
            information = self.configurations[i].information
            c.name = f"{f' ({i})':5s} {name:{max_length}s} [{information}]"

        # Load the initial configuration and reset the solver to this configuration.
        self.configuration_id = initial_configuration
        self.load_configuration(configurations[self.configuration_id])

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def has_loadable_geometry(self, geometries: list[Geometry]):
        if len(geometries) <= 0:
            return False

        return self.current_frame == geometries[0].frame_threshold

    def substep(self) -> None:
        self.particle_frame_count += 1
        self.current_frame += 1

        # Discrete geometries will be added once per iteration:
        while self.has_loadable_geometry(self.discrete_geometries):
            self.sampler.add_geometry(self.discrete_geometries.pop(0))

        # Continuous geometries will be added once per substep:
        loadable_geometries = []
        while self.has_loadable_geometry(self.continuous_geometries):
            loadable_geometries.append(self.continuous_geometries.pop(0))

        # for _ in range(math.ceil((0.5 / (self.fps * self.solver.dt[None])))):
        for _ in range(int(2e-3 // self.solver.dt[None])):
            for geometry in loadable_geometries:
                self.sampler.add_geometry(geometry)
            self.solver.substep()

    def setup_video_manager(self) -> None:
        """
        Creates an output directory, a VideoManager in this directory and then dumps frames to this directory.
        """
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        title = f"{self.video_prefix}_{date}"
        frame_output_dir = f"{self.frame_parent_dir}/{title}"
        os.makedirs(frame_output_dir)
        self.video_manager = ti.tools.VideoManager(
            output_dir=frame_output_dir,
            video_filename=title,
            automatic_build=False,
            framerate=self.fps,
        )

    def create_video(self) -> None:
        """
        Converts stored frames in the before created output directory to a video.
        """
        self.video_manager.make_video(gif=self.should_create_gif, mp4=self.should_create_video)

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.configuration = configuration
        self.reset()

    def setup_ply_writer(self) -> None:
        """
        Creates a directory for the .ply files, then exports particle positions into this directory.
        """
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        title = f"{self.video_prefix}_{date}"
        self.particle_output_path = f"{self.particle_parent_dir}/{title}/"
        self.particle_frame_count = 0
        os.makedirs(self.particle_output_path)

    def reset(self) -> None:
        """
        Reset the simulation.
        """
        self.solver.reset(self.configuration, self.quality)
        self.current_frame = 0

        # We copy this, so we can pop from this list and check the length:
        self.discrete_geometries = self.configuration.discrete_geometries.copy()
        self.continuous_geometries = self.configuration.continuous_geometries.copy()

        # Load all the initial geometries into the solver:
        for geometry in self.configuration.initial_geometries:
            self.sampler.add_geometry(geometry)
