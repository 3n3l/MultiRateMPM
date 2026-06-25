from src.configurations import Configuration

from argparse import ArgumentParser, RawTextHelpFormatter

epilog = "\n\033[91m>>> Press R to [R]eset, [S]PACE to pause/unpause the [S]imulation!\033[0m\n"
parser = ArgumentParser(prog="main.py", epilog=epilog, formatter_class=RawTextHelpFormatter)


def add_configuration(configurations: list[Configuration]):
    help = f"Available Configurations:\n{'\n'.join([f'[{i}] -> {c.name}' for i, c in enumerate(configurations)])}"
    parser.add_argument(
        "-c",
        "--configuration",
        default=0,
        nargs="?",
        help=help,
        type=int,
    )


quality_help = "Choose a quality multiplicator for the simulation (higher is better)."
parser.add_argument(
    "-q",
    "--quality",
    default=0,
    choices={-3, -2, -1, 0, 1, 2, 3},
    nargs="?",
    help=quality_help,
    type=int,
)

dimension_help = "Choose the dimension for the simulation."
parser.add_argument(
    "-d",
    "--dimension",
    default=3,
    choices={2, 3},
    nargs="?",
    help=dimension_help,
    type=int,
)

solver_type_help = "Choose the Taichi architecture to run on."
parser.add_argument(
    "-a",
    "--arch",
    default="CPU",
    nargs="?",
    choices=["CPU", "CUDA", "GPU"],
    help=solver_type_help,
)

solver_type_help = "Turn on debugging."
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help=solver_type_help,
)

solver_type_help = "Turn on verbose logging."
parser.add_argument(
    "--verbose",
    default=False,
    action="store_true",
    help=solver_type_help,
)
