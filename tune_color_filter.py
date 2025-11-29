import cv2
import json
import numpy as np

import find_enemy_respawn
import find_marker
import find_user_marker
from main import ENEMY_SPAWN_TEMPLATE_PATH, MARKER_TEMPLATE_PATH, draw_boxes, USER_TEMPLATE_PATH

CONFIG_FILE = "color_config.json"


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    print("Saved:", config)


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "red1": [0, 120, 70], "red2": [10, 255, 255],
            "red3": [170, 120, 70], "red4": [180, 255, 255],
            "yellow1": [9, 143, 107], "yellow2": [40, 255, 255],
            "warm_yellow1": [15, 20, 180], "warm_yellow2": [35, 255, 255],
            "blue1": [85, 67, 36], "blue2": [128, 230, 212],
            "enemy_spawn_threshold": 67,
            "marker_yellow_threshold": 35,
            "marker_blue_threshold": 27,
            "marker_user_threshold": 22,
        }


def create_trackbar_window(name):
    cv2.namedWindow(name)

    # Lower
    cv2.createTrackbar("H_low", name, 0, 180, lambda x: None)
    cv2.createTrackbar("S_low", name, 0, 255, lambda x: None)
    cv2.createTrackbar("V_low", name, 0, 255, lambda x: None)

    # Upper
    cv2.createTrackbar("H_high", name, 0, 180, lambda x: None)
    cv2.createTrackbar("S_high", name, 0, 255, lambda x: None)
    cv2.createTrackbar("V_high", name, 0, 255, lambda x: None)

    # Threshold
    cv2.createTrackbar("Threshold", name, 0, 100, lambda x: None)

    return name


def set_trackbar_values(window, low, high, threshold):
    cv2.setTrackbarPos("H_low", window, low[0])
    cv2.setTrackbarPos("S_low", window, low[1])
    cv2.setTrackbarPos("V_low", window, low[2])

    cv2.setTrackbarPos("H_high", window, high[0])
    cv2.setTrackbarPos("S_high", window, high[1])
    cv2.setTrackbarPos("V_high", window, high[2])

    cv2.setTrackbarPos("Threshold", window, threshold)


def get_trackbar_values(window):
    h1 = cv2.getTrackbarPos("H_low", window)
    s1 = cv2.getTrackbarPos("S_low", window)
    v1 = cv2.getTrackbarPos("V_low", window)

    h2 = cv2.getTrackbarPos("H_high", window)
    s2 = cv2.getTrackbarPos("S_high", window)
    v2 = cv2.getTrackbarPos("V_high", window)

    tr = cv2.getTrackbarPos("Threshold", window)

    return np.array([h1, s1, v1]), np.array([h2, s2, v2]), tr


def get_additional_values(window):
    h3 = cv2.getTrackbarPos("H_low_2", window)
    s3 = cv2.getTrackbarPos("S_low_2", window)
    v3 = cv2.getTrackbarPos("V_low_2", window)

    h4 = cv2.getTrackbarPos("H_high_2", window)
    s4 = cv2.getTrackbarPos("S_high_2", window)
    v4 = cv2.getTrackbarPos("V_high_2", window)
    return np.array([h3, s3, v3]), np.array([h4, s4, v4])


def tune_color(image_path, mode):
    """
    mode: 'red', 'yellow', 'warm-yellow', 'blue'
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Не удалось загрузить изображение")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    config = load_config()

    if mode == "red":
        window_name = "Tune Red (enemy spawns markers)"
        low_1 = config["red1"]
        high_1 = config["red2"]
        low_2 = config["red3"]
        high_2 = config["red4"]
        threshold = config["enemy_spawn_threshold"]

        cv2.namedWindow(window_name)

        # Lower
        cv2.createTrackbar("H_low", window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("S_low", window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("V_low", window_name, 0, 255, lambda x: None)

        # Upper
        cv2.createTrackbar("H_high", window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("S_high", window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("V_high", window_name, 0, 255, lambda x: None)

        # Lower
        cv2.createTrackbar("H_low_2", window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("S_low_2", window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("V_low_2", window_name, 0, 255, lambda x: None)

        # Upper
        cv2.createTrackbar("H_high_2", window_name, 0, 180, lambda x: None)
        cv2.createTrackbar("S_high_2", window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("V_high_2", window_name, 0, 255, lambda x: None)

        # Threshold
        cv2.createTrackbar("Threshold", window_name, 0, 100, lambda x: None)

        cv2.setTrackbarPos("H_low", window_name, low_1[0])
        cv2.setTrackbarPos("S_low", window_name, low_1[1])
        cv2.setTrackbarPos("V_low", window_name, low_1[2])

        cv2.setTrackbarPos("H_high", window_name, high_1[0])
        cv2.setTrackbarPos("S_high", window_name, high_1[1])
        cv2.setTrackbarPos("V_high", window_name, high_1[2])

        cv2.setTrackbarPos("H_low_2", window_name, low_2[0])
        cv2.setTrackbarPos("S_low_2", window_name, low_2[1])
        cv2.setTrackbarPos("V_low_2", window_name, low_2[2])

        cv2.setTrackbarPos("H_high_2", window_name, high_2[0])
        cv2.setTrackbarPos("S_high_2", window_name, high_2[1])
        cv2.setTrackbarPos("V_high_2", window_name, high_2[2])

        cv2.setTrackbarPos("Threshold", window_name, threshold)

    elif mode == "yellow":
        window_name = "Tune Yellow (yellow markers)"
        low = config["yellow1"]
        high = config["yellow2"]
        threshold = config["marker_yellow_threshold"]
        create_trackbar_window(window_name)
        set_trackbar_values(window_name, low, high, threshold)
    elif mode == "warm-yellow":
        window_name = "Tune Warm-yellow (user marker)"
        low = config["warm_yellow1"]
        high = config["warm_yellow2"]
        threshold = config["marker_user_threshold"]
        create_trackbar_window(window_name)
        set_trackbar_values(window_name, low, high, threshold)
    elif mode == "blue":
        window_name = "Tune Blue (blue markers)"
        low = config["blue1"]
        high = config["blue2"]
        threshold = config["marker_blue_threshold"]
        create_trackbar_window(window_name)
        set_trackbar_values(window_name, low, high, threshold)
    else:
        raise ValueError("mode must be 'red', 'yellow' or 'blue'")

    print("Нажмите 's' чтобы сохранить, ESC — выйти")

    while True:
        low_val, high_val, threshold = get_trackbar_values(window_name)
        mask = cv2.inRange(hsv, low_val, high_val)
        filtered = cv2.bitwise_and(image, image, mask=mask)

        if mode == "red":
            low_val2, high_val2 = get_additional_values(window_name)
            config["red1"] = low_val.tolist()
            config["red2"] = high_val.tolist()
            config["red3"] = low_val2.tolist()
            config["red4"] = high_val2.tolist()
            config["enemy_spawn_threshold"] = threshold
            mask = cv2.inRange(hsv, low_val, high_val) | cv2.inRange(hsv, low_val2, high_val2)
            filtered = cv2.bitwise_and(image, image, mask=mask)

            enemy_spawns_centres = find_enemy_respawn.find_enemy_respawn_centres(
                image,
                ENEMY_SPAWN_TEMPLATE_PATH,
                config)
            draw_boxes(filtered, enemy_spawns_centres, [0, 255, 0])
            cv2.putText(filtered, f"Max_threshold: {find_enemy_respawn.get_max_threshold(image, ENEMY_SPAWN_TEMPLATE_PATH, config)}%",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        elif mode == "yellow":
            config["yellow1"] = low_val.tolist()
            config["yellow2"] = high_val.tolist()
            config["marker_yellow_threshold"] = threshold
            markers_centres_yellow = find_marker.find_marker_yellow(image, MARKER_TEMPLATE_PATH, config)
            draw_boxes(filtered, markers_centres_yellow, [255, 0, 0])
            cv2.putText(filtered, f"Max_threshold: {find_marker.get_max_val_yellow(image, MARKER_TEMPLATE_PATH, config)}%",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        elif mode == "warm-yellow":
            config["warm_yellow1"] = low_val.tolist()
            config["warm_yellow2"] = high_val.tolist()
            config["marker_user_threshold"] = threshold
            markers_centres_warm_yellow = find_user_marker.find_user_marker_centre(image, USER_TEMPLATE_PATH, config)
            draw_boxes(filtered, markers_centres_warm_yellow, [255, 0, 255])
            cv2.putText(filtered, f"Max_threshold: {find_user_marker.get_max_threshold(image, USER_TEMPLATE_PATH, config)}%",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        elif mode == "blue":
            config["blue1"] = low_val.tolist()
            config["blue2"] = high_val.tolist()
            config["marker_blue_threshold"] = threshold
            markers_centres_blue = find_marker.find_marker_blue(image, MARKER_TEMPLATE_PATH, config)
            draw_boxes(filtered, markers_centres_blue, [0, 0, 255])
            cv2.putText(filtered, f"Max_threshold: {find_marker.get_max_val_blue(image, MARKER_TEMPLATE_PATH, config)}%",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        cv2.imshow("Original", image)
        cv2.imshow("Filtered", filtered)

        key = cv2.waitKey(30)

        # Save config
        if key == ord('s'):
            save_config(config)
            print("Настройки сохранены!")

        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()


# ------------------- Демонстрация ---------------------
# tune_color("minimap.png", "red")
# tune_color("minimap.png", "yellow")
tune_color("minimap.png", "warm-yellow")
# tune_color("blue_marker_color_tune_tester.png", "blue")
