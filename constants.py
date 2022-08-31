import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import urllib
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
FRAME_RATE = 30
BUFFER_SIZE = 10000
FRAME_SHAPE = (640, 480)  # (x, y)
FRAME_SIZE = (FRAME_SHAPE[0] * FRAME_SHAPE[1])

DELTA_BOUND_CIRCULAR = 0.03  # z_ratio in [OPTIMAL_Z_RATIO-delta_bound_circular, OPTIMAL_Z_RATIO+delta_bound_circular] circular_tracker
DELTA_BOUND_TRACKER = 0.09  # z_ratio not in [OPTIMAL_Z_RATIO-delta_bound_tracker, OPTIMAL_Z_RATIO+delta_bound_tracker] tracker

OPTIMAL_Z_RATIO = 0.12  # 0.13 before

SIDE_MOTION = 40
SEEK_YAW = 10  # the speed in which the drone will turn when a tracked person left the frame
"""the point on the image we want the traked person to be on"""
TARGET = FRAME_SHAPE[0] // 2, FRAME_SHAPE[1] // 1.9, OPTIMAL_Z_RATIO




