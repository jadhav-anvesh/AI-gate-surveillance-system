import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.services.density import DensityService

density = DensityService()

density.initialize_heatmap((360, 640, 3))

print(density.heatmap.shape)

print("Density Service Ready")
