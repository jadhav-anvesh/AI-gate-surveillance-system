"""
Video source service.

Reads frames from a video file or camera stream on a background
thread, downsampling to a target FPS and frame size, and exposes
them through a bounded queue for the processing pipeline to consume.
"""

from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional, Union

import cv2

from utils import resize_frame_to_360p

# ----------------------------------------------------------------------
# Tuning constants
# ----------------------------------------------------------------------
ASSUMED_SOURCE_FPS = 30  # baseline FPS assumed when computing the frame skip factor


class VideoSource:
    """Pulls frames from a video source on a background thread."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self, source: Union[str, int], target_fps: int = 10, queue_size: int = 10
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self.frame_queue: Queue = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

    # ------------------------------------------------------------------
    # Frame Capture Loop
    # ------------------------------------------------------------------

    def frame_grabber(self) -> None:
        """
        Background-thread loop: reads frames, throttles to target_fps,
        and pushes {det_frame, original_frame} pairs onto the queue.

        If the source runs out of frames (e.g. end of a video file),
        it's reopened so playback loops rather than the thread exiting.
        """
        cap = cv2.VideoCapture(self.source)
        skip_factor = max(1, ASSUMED_SOURCE_FPS // self.target_fps)
        frame_count = 0

        while not self.stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue

            original_frame = frame.copy()
            frame_count += 1

            if frame_count % skip_factor != 0:
                continue

            frame = resize_frame_to_360p(frame)

            if not self.frame_queue.full():
                self.frame_queue.put(
                    {
                        "det_frame": frame,
                        "original_frame": original_frame,
                    }
                )

        cap.release()

    # ------------------------------------------------------------------
    # Public Controls
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background frame-grabbing thread."""
        self.thread = Thread(target=self.frame_grabber, daemon=True)
        self.thread.start()

    def get_frame(self, timeout: float = 2.0) -> Optional[dict]:
        """Return the next available frame pair, or None if none arrive in time."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
