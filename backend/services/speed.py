from collections import defaultdict, deque


class SpeedService:

    def __init__(self, frame_rate):

        self.frame_rate = frame_rate

        self.coordinates_for_speed = defaultdict(
            lambda: deque(maxlen=frame_rate)
        )

        self.vehicle_speed_class_map = {}

        self.vehicle_speed_time_map = {}
        self.current_tracker_speed = {}

    def update_coordinates(
        self,
        tracker_id,
        y_coordinate
    ):

        self.coordinates_for_speed[
            tracker_id
        ].append(
            y_coordinate
        )
    def calculate_speed(
        self,
        tracker_id
    ):
        if len(
            self.coordinates_for_speed[
                tracker_id
            ]
        ) < self.frame_rate / 2:

            return None

        coordinate_start = self.coordinates_for_speed[
            tracker_id
        ][-1]

        coordinate_end = self.coordinates_for_speed[
            tracker_id
        ][0]

        distance = abs(
            coordinate_start
            - coordinate_end
        )

        t_time = (
            len(
                self.coordinates_for_speed[
                    tracker_id
                ]
            )
            / self.frame_rate
        )

        speed = distance / t_time * 3.6
        self.current_tracker_speed[tracker_id] = speed

        return speed
    def update_class_speed(
        self,
        class_name,
        speed
    ):
        if class_name not in self.vehicle_speed_class_map:

            self.vehicle_speed_class_map[
                class_name
            ] = [speed, 1]

        else:

            self.vehicle_speed_class_map[
                class_name
            ][0] += speed

            self.vehicle_speed_class_map[
                class_name
            ][1] += 1
    def update_time_speed(
        self,
        current_second,
        total_speed,
        total_count
    ):
        if total_count > 0:

            self.vehicle_speed_time_map[
                current_second
            ] = (
                total_speed
                / total_count
            )

        else:

            self.vehicle_speed_time_map[
                current_second
            ] = 0

    def get_class_speed_data(self):

        return self.vehicle_speed_class_map

    def get_time_speed_data(self):

        return self.vehicle_speed_time_map

    def get_tracker_speed(self, tracker_id):

        return self.current_tracker_speed.get(tracker_id)
