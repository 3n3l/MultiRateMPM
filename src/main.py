import sys, os, math

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from src.presets import ice_presets, water_presets, mixed_presets
from src.parsers.parsing import parser, add_configuration
from src.simulation import GGUI_Simulation
from src.samplers import PoissonDiskSampler

from multi_rate_mpm import MultiRateMPM

import taichi as ti


def main():
    configurations = ice_presets + water_presets + mixed_presets
    add_configuration(configurations)
    arguments = parser.parse_args()
    print(parser.epilog)

    # Initialize Taichi on the chosen architecture:
    chosen_arch = ti.cpu
    if arguments.arch.lower() == "cpu":
        chosen_arch = ti.cpu
    elif arguments.arch.lower() == "gpu":
        chosen_arch = ti.gpu
    else:
        chosen_arch = ti.cuda
    ti.init(arch=chosen_arch, default_fp=ti.f64, debug=arguments.debug, verbose=arguments.verbose, unrolling_limit=0)

    initial_configuration = arguments.configuration % len(configurations)
    name = "Two-Way Simulation of Water & Ice"
    prefix = "TWS_MLSMPM"

    d = 3  # arguments.dimension
    q = 2**arguments.quality

    n_grid = math.ceil(128 * q)
    dx = 1 / n_grid
    n_particles_cell = 3
    radius = dx / (2 * (n_particles_cell ** (1 / 3)))
    vol_0 = (0.5 * dx) ** d

    # Make a rough guess of maximum possible amount of particles from volumes:
    max_volume = 0.0
    for configuration in configurations:
        if (volume := configuration.volume()) > max_volume:
            max_volume = volume
    max_particles = math.ceil(max_volume / (vol_0))

    solver = MultiRateMPM(max_particles=max_particles, n_grid=n_grid, vol_0=vol_0)
    poisson_disk_sampler = PoissonDiskSampler(solver=solver, r=radius * 1.5, k=30)

    simulation = GGUI_Simulation(
        initial_configuration=initial_configuration,
        configurations=configurations,
        sampler=poisson_disk_sampler,
        res=(720, 720),
        prefix=prefix,
        solver=solver,
        radius=radius,
        name=name,
        quality=q,
    )
    simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
