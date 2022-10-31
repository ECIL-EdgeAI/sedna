import cv2 as cv
import goto
import os
import queue
import subprocess as sp
import time
import threading
from dominate.tags import label
from goto import with_goto
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXTRACT_FREQUENCY = 200
pusher_dict = {}


class StreamPusher(FileSystemEventHandler):
    def __init__(self, watch_path=None, rtmp_url=None, fps=20, width=640, height=480):
        self.rtmp_url = rtmp_url
        self.watch_path = watch_path
        self.raw_frame_q = queue.Queue()

        # Set video params
        self.fps = fps
        self.width = width
        self.height = height
        self.curr_frame = None

        # Set ffmpeg command text
        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        self.rtmp_url]

    def image_watch(self):
        observer = Observer()
        observer.schedule(self, self.watch_path, False)
        observer.start()  # Start observer
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    # Read frame from image
    def on_moved(self, event):
        print(event)
        frame = cv.imread(event.dest_path)
        if frame is None:
            return

        if not self.raw_frame_q.full():
            self.raw_frame_q.put_nowait(frame)

    @with_goto
    def push_frame(self):
        label .begin

        # Configure the pipe to pass commands to the os
        p = sp.Popen(self.command, stdin=sp.PIPE)

        try:
            while True:
                if not self.raw_frame_q.empty():
                    # Take the frame from the queue
                    frame = self.raw_frame_q.get_nowait()
                    self.curr_frame = frame

                if self.curr_frame is None:
                    time.sleep(0.1)
                    continue

                p.stdin.write(self.curr_frame)
        except KeyboardInterrupt:
            p.kill()
        except Exception as e:
            print(e)
            goto .begin

    def extract_video_frames(self, video_path):
        cap = cv.VideoCapture(video_path)
        frame_count = 0
        index = 0

        try:
            while cap.isOpened():
                _, raw_frame = cap.read()
                if raw_frame is None:
                    continue

                if frame_count % EXTRACT_FREQUENCY == 0:
                    save_path = '{}/{}.png'.format(self.watch_path, index)
                    cv.imwrite(save_path, raw_frame)
                    # Make sure the image is completely written to disk
                    new_path = '{}/{}.live.png'.format(self.watch_path, index)
                    os.rename(save_path, new_path)
                    index += 1
                frame_count += 1
        except KeyboardInterrupt:
            cap.release()

    # Run watch and push
    def run(self):
        threads = [
            threading.Thread(target=self.push_frame, daemon=True),
            threading.Thread(target=self.image_watch, daemon=True)
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]


def singleton(cls):
    cls._lock = threading.Lock()
    cls._instance = None

    def __new__(*args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = object.__new__(cls)
        return cls._instance

    cls.__new__ = __new__
    return cls


@singleton
class StreamFactory(object):
    def __init__(self):
        self._lock = threading.Lock()

    def get(self, watch_path, rtmp_url, fps, width, height):
        global pusher_dict
        stream_name = '{}#{}#{}#{}'.format(rtmp_url, fps, width, height)
        with self._lock:
            if stream_name not in pusher_dict:
                pusher_dict[stream_name] = StreamPusher(
                    watch_path=watch_path,
                    rtmp_url=rtmp_url,
                    fps=fps,
                    width=width,
                    height=height
                )
            return pusher_dict[stream_name]
