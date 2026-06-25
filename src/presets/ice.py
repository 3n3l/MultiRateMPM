from src.configurations import Circle, Configuration
from src.constants import Ice

ice_presets = [
    Configuration(
        name="Spherefall, Ice",
        information="Ice",
        gravity=-9.81,
        dt=1e-4,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.5, 0.4, 0.5),
                velocity=(0, -3, 0),
                temperature=-100.0,
                radius=0.1,
            ),
        ],
    ),
    Configuration(
        name="Snowball Hits Snowball",
        dt=1e-4,
        gravity=-9.81,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.1, 0.5, 0.5),
                velocity=(3, 0, 0),
                radius=0.08,
            ),
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.9, 0.56, 0.5),
                velocity=(-6, 0, 0),
                radius=0.08,
            ),
        ],
    ),
]
