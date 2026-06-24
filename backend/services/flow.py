class FlowService:

    def __init__(self):

        # Total vehicle counts
        self.vehicle_distribution_map = {}

        # Vehicles seen over time
        self.vehicle_in_scene_map = {}

        # Per-lane counts
        self.lane_wise_vehicle_distribution_maps = []

        # Per-lane time series
        self.lane_wise_vehicle_in_scene_maps = []

    def initialize_lanes(self, num_lanes):

        self.lane_wise_vehicle_distribution_maps = [{}for _ in range(num_lanes)]
        self.lane_wise_vehicle_in_scene_maps = [{}for _ in range(num_lanes)]

    def update_vehicle_count(self, vehicle_class, lane_index):

        # Total vehicle count
        self.vehicle_distribution_map[ vehicle_class ] = self.vehicle_distribution_map.get( vehicle_class, 0 ) + 1

        # Lane-wise count
        self.lane_wise_vehicle_distribution_maps[lane_index][vehicle_class]=(self.lane_wise_vehicle_distribution_maps[lane_index].get(vehicle_class,0)+1)

    def update_vehicle_scene_count(self, current_second, vehicle_count):

        self.vehicle_in_scene_map[current_second] = vehicle_count
    
    def update_lane_scene_count( self, lane_index, current_second, current_total_count):

        lane_map = self.lane_wise_vehicle_in_scene_maps[ lane_index]
        previous_second = current_second - 1
        if previous_second in lane_map:
            lane_map[current_second] = (current_total_count - lane_map[previous_second])
        else:

            lane_map[current_second] = ( current_total_count)

    def process_trigger(
        self,
        trigger,
        class_ids,
        id_cls_map,
        lane_index
    ):

        for isIn, isOut, cls_id in zip(
            trigger[0],
            trigger[1],
            class_ids
        ):

            if isIn or isOut:

                vehicle_class = id_cls_map[
                    cls_id
                ].upper()

                self.update_vehicle_count(
                    vehicle_class,
                    lane_index
                )
    def get_vehicle_distribution(self):

        return self.vehicle_distribution_map

    def get_vehicle_scene_data(self):

        return self.vehicle_in_scene_map

    def get_lane_distribution(self):

        return self.lane_wise_vehicle_distribution_maps

    def get_lane_scene_data(self):

        return self.lane_wise_vehicle_in_scene_maps
