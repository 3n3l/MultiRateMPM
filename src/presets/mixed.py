from src.configurations import Circle, Rectangle, Configuration
from src.constants import Ice, Water

mixed_presets = [
    Configuration(
        dt=3e-4,
        name="Pool & Ice Cubes",
        information="Water, Ice -> Water",
        ambient_temperature=1.0,
        boundary_temperature=1000.0,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                size=(1.0, 0.05, 1.0),
                velocity=(0, 0, 0),
                lower_left=(0, 0, 0),
                frame_threshold=0,
                temperature=100.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08, 0.08),
                velocity=(0, -4, 0),
                lower_left=(0.25, 0.55, 0.25),
                frame_threshold=1,
                temperature=-1.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08, 0.08),
                velocity=(0, -4, 0),
                lower_left=(0.45, 0.45, 0.5),
                frame_threshold=21,
                temperature=-1.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08, 0.08),
                velocity=(0, -4, 0),
                lower_left=(0.65, 0.35, 0.75),
                frame_threshold=41,
                temperature=-1.0,
            ),
        ],
    ),
    # Configuration(
    #     dt=1e-3,
    #     name="Freezing Pool",
    #     information="Water -> Ice",
    #     ambient_temperature=-500.0,
    #     geometries=[
    #         Rectangle(
    #             lower_left=(0.0, 0.0),
    #             material=Water,  # pyright: ignore
    #             temperature=20.0,
    #             size=(1.0, 0.15),
    #             velocity=(0, 0),
    #         ),
    #     ],
    # ),
    # Configuration(
    #     dt=3e-4,
    #     name="Waterjet & Ice Cubes",
    #     information="Water, Ice -> Water",
    #     ambient_temperature=1.0,
    #     geometries=[
    #         *[
    #             Circle(
    #                 is_continuous=True,
    #                 material=Water,  # pyright: ignore
    #                 center=(0.04, 0.96),
    #                 temperature=50.0,
    #                 velocity=(2, -1),
    #                 radius=0.025,
    #                 frame_threshold=i,
    #             )
    #             for i in range(1, 150)
    #         ],
    #         Rectangle(  # BL
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.495, 0.0),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #         Rectangle(  # BM
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.585, 0.0),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #         Rectangle(  # BR
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.675, 0.0),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #         Rectangle(  # ML
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.5445, 0.09),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #         Rectangle(  # MR
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.6345, 0.09),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #         Rectangle(  # TM
    #             material=Ice,  # pyright: ignore
    #             velocity=(0, 0),
    #             lower_left=(0.585, 0.18),
    #             size=(0.08, 0.08),
    #             temperature=-50.0,
    #         ),
    #     ],
    # ),
    # Configuration(
    #     name="Waterjet & Smash",
    #     information="Water, Ice -> Water",
    #     dt=3e-4,
    #     gravity=-9.81,
    #     ambient_temperature=1.0,
    #     boundary_temperature=100.0,
    #     geometries=[
    #         *[
    #             Rectangle(
    #                 material=Water,  # pyright: ignore
    #                 is_continuous=True,
    #                 frame_threshold=i,
    #                 lower_left=(0.47, 0.94),
    #                 temperature=500.0,
    #                 velocity=(0, -3),
    #                 size=(0.06, 0.06),
    #             )
    #             for i in range(9, 209)
    #         ],
    #         *[
    #             Circle(
    #                 material=Water,  # pyright: ignore
    #                 is_continuous=True,
    #                 frame_threshold=i,
    #                 center=(0.5, 0.94),
    #                 temperature=500.0,
    #                 velocity=(0, -4),
    #                 radius=0.03,
    #             )
    #             for i in range(9, 209)
    #         ],
    #         Circle(
    #             material=Ice,  # pyright: ignore
    #             center=(0.2, 0.5),
    #             velocity=(4, 0),
    #             frame_threshold=1,
    #             temperature=-1.0,
    #             radius=0.06,
    #         ),
    #         Circle(
    #             material=Ice,  # pyright: ignore
    #             center=(0.2, 0.6),
    #             velocity=(4, 0),
    #             frame_threshold=11,
    #             temperature=-1.0,
    #             radius=0.06,
    #         ),
    #         Circle(
    #             material=Ice,  # pyright: ignore
    #             center=(0.2, 0.7),
    #             velocity=(4, 0),
    #             frame_threshold=21,
    #             temperature=-1.0,
    #             radius=0.06,
    #         ),
    #     ],
    # ),
    Configuration(
        dt=3e-4,
        name="Spherefall, Water vs. Ice",
        information="Water, Ice",
        ambient_temperature=0.0,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.25, 0.5, 0.5),
                velocity=(0, -3, 0),
                temperature=-100.0,
                radius=0.1,
            ),
            Circle(
                material=Water,  # pyright: ignore
                center=(0.75, 0.5, 0.5),
                velocity=(0, -3, 0),
                temperature=-100.0,
                radius=0.1,
            ),
        ],
    ),
]
