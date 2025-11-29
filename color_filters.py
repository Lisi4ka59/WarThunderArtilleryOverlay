import cv2
import numpy as np


def prefilter_colors_red(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array(config["red1"])
    upper_red1 = np.array(config["red2"])

    lower_red2 = np.array(config["red3"])
    upper_red2 = np.array(config["red4"])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)

    return cv2.bitwise_and(image, image, mask=mask_red)


def prefilter_colors_yellow(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(config["yellow1"])
    upper = np.array(config["yellow2"])

    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def prefilter_colors_warm_yellow(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(config["warm_yellow1"])
    upper = np.array(config["warm_yellow2"])

    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)


def prefilter_colors_blue(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array(config["blue1"])
    upper = np.array(config["blue2"])

    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)