import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from backend.services.density import DensityService

density = DensityService()

density.initialize_heatmap(
    (360, 640, 3)
)

print(type(density.heatmap))

print(density.heatmap.shape)

print(type(density.track_history))

print(type(density.last_positions))

print("Density Service Created Successfully")
