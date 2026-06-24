import cv2
import numpy as np
from collections import defaultdict


class DensityService:

    def __init__(self):

        self.track_history = defaultdict(lambda: [])
        self.last_positions = {}
        self.heatmap = None

    def initialize_heatmap(self, frame_shape):

        self.heatmap = np.zeros(frame_shape, dtype=np.float32)

    def get_heatmap(self):

        return self.heatmap

    def get_track_history(self):

        return self.track_history

    def get_last_positions(self):

        return self.last_positions

    def calculate_distance(self, point1, point2):

        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def update_heatmap(self, results, track_ids, frame):

        if hasattr(results, "xyxy"):

            xyxy = results.xyxy

            boxes = []

            for box in xyxy:

                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1

                x_center = x1 + width / 2
                y_center = y1 + height / 2

                boxes.append((x_center, y_center, width, height))

        else:

            boxes = results[0].boxes.xywh.cpu()

        for box, track_id in zip(boxes, track_ids):

            x_center, y_center, width, height = box

            current_position = (
                float(x_center),
                float(y_center)
            )

            top_left_x = max(
                0,
                int(x_center - width / 2)
            )

            top_left_y = max(
                0,
                int(y_center - height / 2)
            )

            bottom_right_x = min(
                self.heatmap.shape[1],
                int(x_center + width / 2)
            )

            bottom_right_y = min(
                self.heatmap.shape[0],
                int(y_center + height / 2)
            )

            track = self.track_history[track_id]
            track.append(current_position)

            if len(track) > 1200:
                track.pop(0)

            last_position = self.last_positions.get(track_id)

            if ( last_position and self.calculate_distance( last_position, current_position) > 5):
                self.heatmap[
                    top_left_y:bottom_right_y,
                    top_left_x:bottom_right_x
                ] += 1

            self.last_positions[track_id] = current_position
    
    def generate_overlay(self, frame):

        heatmap_blurred = cv2.GaussianBlur(self.heatmap, (15, 15), 0)

        heatmap_norm = cv2.normalize(
            heatmap_blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_norm,
            cv2.COLORMAP_JET
        )

        if frame.shape != heatmap_color.shape:
            heatmap_color = cv2.resize(
                heatmap_color,
                (frame.shape[1], frame.shape[0])
            )

        overlay = cv2.addWeighted(
            frame,
            0.3,
            heatmap_color,
            0.7,
            0
        )

        return overlay
