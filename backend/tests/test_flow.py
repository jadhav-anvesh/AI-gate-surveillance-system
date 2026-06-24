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

from backend.services.flow import FlowService

flow = FlowService()

flow.initialize_lanes(4)

flow.update_vehicle_count(
    "CAR",
    0
)

flow.update_vehicle_count(
    "CAR",
    0
)

flow.update_vehicle_count(
    "TRUCK",
    1
)
flow.update_vehicle_scene_count(
    current_second=1,
    vehicle_count=12
)

flow.update_vehicle_scene_count(
    current_second=2,
    vehicle_count=18
)

flow.update_lane_scene_count(
    lane_index=0,
    current_second=1,
    current_total_count=10
)

flow.update_lane_scene_count(
    lane_index=0,
    current_second=2,
    current_total_count=15
)
flow.process_trigger(
    (
        [True, False, True],
        [False, False, False]
    ),
    [0, 1, 0],
    {
        0: "car",
        1: "truck"
    },
    0
)

print(flow.vehicle_distribution_map)
print(flow.vehicle_distribution_map)

print(flow.lane_wise_vehicle_distribution_maps)

print(flow.vehicle_in_scene_map)
print(
    flow.lane_wise_vehicle_in_scene_maps
)
print("Flow Service Created Successfully")
