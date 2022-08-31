import sys
import threading
import multiprocessing
import time
from datetime import datetime
# from math import sqrt
# import keyboard
#
import Detection
import cv2
import numpy as np
from djitellopy import tello
import Tracker
from constants import *
import copy

now = datetime.now()


class Drone(tello.Tello):
    """
    this is a drone class we inherit from the tello.Tello class and added data and functionality to our needs
    """

    def __init__(self, detction_module=None, deep_sort=None):
        """
        calls the super constructor and runs the code needed for use of the object
        """

        # connecting to Tello
        super().__init__(retry_count=6)
        self.connect()
        print(self.get_battery())
        self.streamoff()
        self.streamon()

        # synchronization
        self.buffer_mutex = threading.RLock()  # only one thread can access the buffer
        self.fillCount = threading.Semaphore(0)  # counts how many frames are in the buffer
        self.emptyCount = threading.Semaphore(BUFFER_SIZE)  # prevents threads from accessing the buffer when

        # initializing different classes
        self.frames_receiver = self.FramesReceiver(self)
        self.video_saver = self.VideoSaver(self)
        self.detector = Detection.ObjectsDetector("yolov4-tiny.cfg", "yolov4-tiny.weights", detction_module)

        # initializing Trackers
        ratiooo = 300 / 480
        # ratiooo = 1
        self.tracker = Tracker.ObjectTracking(pid_values_x=(0.27 * ratiooo, 1e-5, 0, FRAME_SHAPE[0] // 2),
                                              pid_values_y=(0.5 * ratiooo, 0, 0.1 * ratiooo, FRAME_SHAPE[1] // 1.9),
                                              pid_values_z=(700 * ratiooo, 0, 0, OPTIMAL_Z_RATIO * 1.2))

        self.circular_tracker = Tracker.ObjectTracking(
            pid_values_x=(0.25 * ratiooo, 1.1e-3 * ratiooo, 0.15 * ratiooo, FRAME_SHAPE[0] // 2),
            pid_values_y=(0.5 * ratiooo, 0, 0.1 * ratiooo, FRAME_SHAPE[1] // 1.9),
            pid_values_z=(100 * ratiooo, 0, 0, OPTIMAL_Z_RATIO))
        self.current_tracker = self.tracker

        # initializing threads
        self.saver_thread = threading.Thread(target=self.video_saver.create_video_loop, name='saver_thread')
        self.saver_thread.start()
        self.receiver_thread = threading.Thread(target=self.frames_receiver.readframes, name='receiver_thread')
        self.receiver_thread.start()

        # initializing for test TODO
        self.frame_counter = 0
        self.bbox_counter = 0
        self.yes_bbox_counter = 0
        self.array_bbox = []
        self.start_time = time.time()

        self.frame_counter_start = False

    class FramesReceiver:
        """
        this class receives frames from the drone and stored them in a array
        """

        def __init__(self, drone):
            """
            gets the drone and inits all the object we need for the task
            :param drone: the drone
            """
            self.drone = drone
            self.image_buffer = []
            self.VideoCap = self.drone.get_video_capture()

        def readframes(self):
            """
            runs in a loop and reads new frames from the drone. this function should be called in a new thread
            :return: None
            """
            while True:
                ret, img = self.VideoCap.read()
                if ret:
                    img = cv2.resize(img, FRAME_SHAPE)
                    self.drone.emptyCount.acquire()
                    self.drone.buffer_mutex.acquire()

                    self.image_buffer.append(img)
                    if self.drone.frame_counter_start:  # TODO frame_counter for test
                        self.drone.frame_counter += 1  # TODO frame_counter for test
                    self.drone.buffer_mutex.release()
                    self.drone.fillCount.release()

        def getFrame(self, consume=True):
            """
            allow reading from the array buffer safely if consume is true we would pop the beginning of the buffer
            else we would just read the last frame
            :param consume:
            :return:
            """

            if consume:
                self.drone.fillCount.acquire()
                self.drone.buffer_mutex.acquire()
                img = self.image_buffer.pop(0)
                self.drone.buffer_mutex.release()
                self.drone.emptyCount.release()
                return img
            else:
                self.drone.fillCount.acquire()
                self.drone.buffer_mutex.acquire()
                img = self.image_buffer[-1]
                self.drone.buffer_mutex.release()
                self.drone.fillCount.release()
                return img.copy()

    class VideoSaver:
        """
        this a local class that deals with saving videos from  the frames_receiver of the drone and storing them
        """

        def __init__(self, drone):
            """
            gets the drone object for the class and creates all the objects necessary for saving a video
            :param drone: the drone
            """
            self.drone = drone
            self.video_writer = cv2.VideoWriter(f'{now.strftime("%H:%M:%S")}.avi',
                                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FRAME_RATE, FRAME_SHAPE)
            # self.video_writer = cv2.VideoWriter(f'{now.strftime("%H:%M:%S")}.mp4',
            #                                     cv2.VideoWriter_fourcc(*'MP4V'), FRAME_RATE, FRAME_SHAPE)

        def create_video_loop(self):
            """
            reads from the array in a loop and saves the video when q is pressed video would be saved. this function is called with a new thread
            :return:
            """
            while True:
                img = self.drone.frames_receiver.getFrame()
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.drone.send_read_command(0, 0, 0, 0)
                    self.drone.land()
                    self.video_writer.release()
                    cv2.destroyAllWindows()
                    break
                # cv2.imshow("Image", img)  # TODO - remove
                self.video_writer.write(img)

    def getFrame(self):
        """
        gets the latest image received from the drone
        """
        return self.frames_receiver.getFrame(consume=False)

    # def emergency_landing_check(self):
    #     if keyboard.is_pressed('q'):
    #         cv2.destroyAllWindows()
    #         self.send_rc_control(0, 0, 0, 0)
    #         self.land()
    #         exit(1)
    #
    def track(self):
        """

        """
        # while True:
        last_detected_person = None
        while (time.time() - self.start_time) < 400:  # TODO - remove leave while True
            bbox = self.detector.find_center(self.getFrame())
            if bbox:
                z_ratio = bbox[6]
                # print(f"z_ratio is {z_ratio}")
                # self.switch_tracker(z_ratio) # TODO - uncomment
                last_detected_person = bbox
            # elif last_detected_person:
            #     print(f"last_detected_person")
            #     self.current_tracker.reset()
            #     self.send_rc_control(0, 0, 0, - SEEK_YAW if last_detected_person[4] > FRAME_SHAPE[0] else SEEK_YAW)
            #     continue
            xResponse, yResponse, zResponse = self.current_tracker.get_rc_commend(bbox)
            # yResponse = 0  # TODO - remove for test
            # zResponse = 0  # TODO - remove for test

            # yResponse,zResponse = 0,0#todo
            self.send_rc_control(0 if self.current_tracker is self.tracker or not bbox else SIDE_MOTION, -zResponse,
                                 -yResponse, xResponse)
            # self.emergency_landing_check()

        self.land()  # TODO - remove
        sys.exit()

    def switch_tracker(self, z_ratio):
        #
        # print(f"{(0.1 > z_ratio or 0.25 < z_ratio) and (self.current_tracker is not self.tracker)=}")
        # print(f"{self.current_tracker is not self.tracker=}")
        # print(f"{self.current_tracker is not self.circular_tracker=}")
        if ((OPTIMAL_Z_RATIO - DELTA_BOUND_CIRCULAR) <= z_ratio <= (OPTIMAL_Z_RATIO + DELTA_BOUND_CIRCULAR)) and (
                self.current_tracker is not self.circular_tracker):
            temp_data_x, temp_data_y, temp_data_z = self.current_tracker.get_temporal_data()
            self.current_tracker = self.circular_tracker
            self.current_tracker.set_temporal_data(temp_data_x, temp_data_y, temp_data_z)
            print(f"switch was made 0.circular")
        elif ((OPTIMAL_Z_RATIO - DELTA_BOUND_TRACKER) > z_ratio or (
                OPTIMAL_Z_RATIO + DELTA_BOUND_TRACKER) < z_ratio) and (
                self.current_tracker is not self.tracker):
            temp_data_x, temp_data_y, temp_data_z = self.current_tracker.get_temporal_data()
            self.current_tracker = self.tracker
            self.current_tracker.set_temporal_data(temp_data_x, temp_data_y, temp_data_z)
            print(f"switch was made 1.tracker")

    def track_test(self):  # TODO for test
        erorr = []
        targetarr = np.array(TARGET)
        last_detected_person = None
        while (time.time() - self.start_time) < 60:  # TODO - remove leave while True
            self.frame_counter_start = True
            self.bbox_counter += 1
            bbox = self.detector.find_center(self.getFrame())
            if bbox:
                cur_pos = np.array(bbox[4:7])
                erorr.append(np.linalg.norm(cur_pos - targetarr))
                self.yes_bbox_counter += 1
                self.array_bbox.append(bbox)
                z_ratio = bbox[6]
                # print(f"z_ratio is {z_ratio}")
                self.switch_tracker(z_ratio)
                last_detected_person = bbox
            elif last_detected_person:
                print(f"last_detected_person")
                self.current_tracker.reset()
                self.send_rc_control(0, 0, 0, - SEEK_YAW if last_detected_person[4] > FRAME_SHAPE[0] else SEEK_YAW)
                continue
            xResponse, yResponse, zResponse = self.current_tracker.get_rc_commend(bbox)
            # yResponse,zResponse = 0,0#todo
            self.send_rc_control(0 if self.current_tracker is self.tracker or not bbox else SIDE_MOTION, -zResponse,
                                 -yResponse, xResponse)
            # self.emergency_landing_check()
        frame_count = copy.deepcopy(self.frame_counter)
        np.save("bb_error", erorr)
        self.land()  # TODO - remove
        return self.bbox_counter, self.yes_bbox_counter, frame_count, self.array_bbox, np.array(erorr)

        # erorr = []
        # targetarr = np.array(TARGET)
        # while (time.time() - self.start_time) < 90:
        #     self.frame_counter_start = True
        #     bbox = self.detector.find_center(self.getFrame())
        #     cur_pos = np.array(bbox[4:7])
        #     erorr.append(np.linalg.norm(cur_pos - targetarr))
        #
        #     self.bbox_counter += 1
        #     if bbox:
        #         self.yes_bbox_counter += 1
        #         self.array_bbox.append(bbox)
        #     xVal, yVal, zVal = self.current_tracker.get_rc_commend(bbox)
        #     self.send_rc_control(0, -zVal, -yVal, xVal)
        #     # self.emergency_landing_check()
        # frame_count = copy.deepcopy(self.frame_counter)
        # np.save("bb_error", erorr)
        # return self.bbox_counter, self.yes_bbox_counter, frame_count, self.array_bbox,np.array(erorr)
