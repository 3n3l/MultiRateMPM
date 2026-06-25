from src.configurations import Circle, Rectangle, Configuration
from src.constants import Water

water_presets = [
    Configuration(
        name="Waterjet",
        information="Water",
        dt=5e-3,
        geometries=[
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    center=(0.5, 0.98, 0.5),
                    velocity=(0, -3, 0),
                    radius=0.04,
                )
                for i in range(1, 201)
            ],
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    center=(0.5, 0.975, 0.5),
                    velocity=(0, -3, 0),
                    radius=0.04,
                )
                for i in range(2, 201)
            ],
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    center=(0.5, 0.97, 0.5),
                    velocity=(0, -3, 0),
                    radius=0.04,
                )
                for i in range(3, 201)
            ],
        ],
    ),

    # Configuration(
    #     name="Waterjet & Pool",
    #     information="Water",
    #     dt=1e-3,
    #     geometries=[
    #         Rectangle(
    #             material=Water,  # pyright: ignore
    #             lower_left=(0.0, 0.0),
    #             size=(1.0, 0.1),
    #             velocity=(0, 0),
    #         ),
    #         *[
    #             Rectangle(
    #                 material=Water,  # pyright: ignore
    #                 is_continuous=True,
    #                 frame_threshold=i,
    #                 lower_left=(0.47, 0.94),
    #                 velocity=(0, -2),
    #                 size=(0.06, 0.06),
    #             )
    #             for i in range(3, 203)
    #         ],
    #         *[
    #             Circle(
    #                 material=Water,  # pyright: ignore
    #                 is_continuous=True,
    #                 frame_threshold=i,
    #                 center=(0.5, 0.94),
    #                 velocity=(0, -2),
    #                 radius=0.03,
    #             )
    #             for i in range(3, 203)
    #         ],
    #     ],
    # ),
    Configuration(
        name="Dam Break",
        information="Water",
        dt=5e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0, 0.0),
                size=(0.3, 0.3, 1.0),
                velocity=(0, 0, 0),
            ),
        ],
    ),
    Configuration(
        name="Dam Break, Centered",
        information="Water",
        dt=5e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.35, 0.0, 0.0),
                size=(0.3, 0.3, 1.0),
                velocity=(0, 0, 0),
            ),
        ],
    ),
    Configuration(
        name="Spherefall, Water",
        information="Water",
        dt=5e-3,
        geometries=[
            Circle(
                material=Water,  # pyright: ignore
                center=(0.5, 0.4, 0.5),
                velocity=(0, -3, 0),
                radius=0.1,
            ),
        ],
    ),
    # Configuration(
    #     name="Pool",
    #     information="Water",
    #     dt=1e-3,
    #     geometries=[
    #         Rectangle(
    #             material=Water,  # pyright: ignore
    #             lower_left=(0.0, 0.0),
    #             size=(1.0, 0.25),
    #             velocity=(0, 0),
    #         ),
    #     ],
    # ),
]
