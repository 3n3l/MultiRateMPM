from dataclasses import dataclass
from taichi import hex_to_rgb


@dataclass
class ColorHEX:
    HeatMap = [0x323296, 0x5050AB, 0x7575BF, 0xDB7F85, 0xD64F58, 0xC73C45]
    Background = 0x007D79  # teal 60
    Magenta = 0xFF7EB6  # magenta 40
    Purple = 0xBE95FF  # purple 40
    Water = 0x78A9FF  # blue 40
    Ice = 0xD0E2FF  # blue 20


@dataclass
class ColorRGB:
    HeatMap = [hex_to_rgb(color) for color in ColorHEX.HeatMap]
    Background = hex_to_rgb(ColorHEX.Background)
    Magenta = hex_to_rgb(ColorHEX.Magenta)
    Purple = hex_to_rgb(ColorHEX.Purple)
    Water = hex_to_rgb(ColorHEX.Water)
    Ice = hex_to_rgb(ColorHEX.Ice)
