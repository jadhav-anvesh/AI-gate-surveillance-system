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

from backend.core.pipeline_manager import (
    PipelineManager
)

manager = PipelineManager()

print(
    manager.status()
)

manager.start_processing()

print(
    manager.status()
)

manager.stop_processing()

print(
    manager.status()
)

manager.update_statistics(
    {
        "vehicles": 12,
        "speed": 35
    }
)

print(
    manager.get_statistics()
)
manager.start_processing()

print(
    manager.status()
)

manager.stop_processing()

print(
    manager.status()
)

print(
    "PipelineManager Created Successfully"
)
