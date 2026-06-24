import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.services.speed import SpeedService

speed_service = SpeedService(
    frame_rate=10
)

for y in [100, 110, 120, 130, 140]:
    speed_service.update_coordinates(
        1,
        y
    )

speed = speed_service.calculate_speed(
    1
)

print(speed)

speed_service.update_class_speed(
    "CAR",
    speed
)

print(
    speed_service.vehicle_speed_class_map
)

print(
    "Speed Service Created Successfully"
)
