import cv2
from queue import Queue
from threading import Thread, Event

from utils import resize_frame_to_360p


class VideoSource:

    def __init__(
        self,
        source,
        target_fps=10,
        queue_size=10
    ):

        self.source = source

        self.target_fps = target_fps

        self.frame_queue = Queue(
            maxsize=queue_size
        )

        self.stop_event = Event()

        self.thread = None

    def frame_grabber(self):

        cap = cv2.VideoCapture(
            self.source
        )

        skip_factor = max(
            1,
            30 // self.target_fps
        )

        frame_count = 0

        while not self.stop_event.is_set():

            ret, frame = cap.read()
            original_frame = frame.copy()
            if not ret:

                cap.release()

                cap = cv2.VideoCapture(
                    self.source
                )

                continue

            frame_count += 1

            if frame_count % skip_factor != 0:
                continue

            frame = resize_frame_to_360p(
                frame
            )

            if not self.frame_queue.full():

                self.frame_queue.put(
                    {
                        "det_frame": frame,
                        "original_frame": original_frame
                    }
                )
        cap.release()

    def start(self):

        self.thread = Thread(
            target=self.frame_grabber,
            daemon=True
        )

        self.thread.start()

    def get_frame(
        self,
        timeout=2.0
    ):

        try:

            return self.frame_queue.get(
                timeout=timeout
            )

        except Exception:

            return None

    def stop(self):

        self.stop_event.set()

        if self.thread:

            self.thread.join()
