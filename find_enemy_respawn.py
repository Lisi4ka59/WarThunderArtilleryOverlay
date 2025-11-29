import cv2
import numpy as np

from color_filters import prefilter_colors_red


def find_enemy_respawn_centres(image: np.ndarray,
                               template_path: str,
                               config,
                               min_distance: int = 40,
                               score_drop_threshold: float = 0.02
                               ):
    """
    Ищет одно или два совпадения шаблона на изображении.

    :param config: конфиг для цветовой фильтрации
    :param image: изображение (BGR numpy array).
    :param template_path: путь к шаблону.
    :param min_distance: минимальное расстояние между найденными центрами.
    :param score_drop_threshold: порог, при котором второе совпадение отбрасывается.
    :return: список центров [(x1, y1), (x2, y2)].
    """
    filtered_image = prefilter_colors_red(image, config)

    template = cv2.imread(template_path)
    if template is None:
        raise RuntimeError(f"Не удалось загрузить шаблон: {template_path}")

    result = cv2.matchTemplate(filtered_image, template, cv2.TM_CCOEFF_NORMED)

    h, w = template.shape[:2]

    # --- 1. Находим первое совпадение ---
    min_val, max_val1, min_loc, max_loc1 = cv2.minMaxLoc(result)
    if max_val1 < config["enemy_spawn_threshold"] / 100:
        return []
    top_left1 = max_loc1
    center1 = (top_left1[0] + w // 2, top_left1[1] + h // 2)

    # Зануляем область вокруг первого совпадения, чтобы найти второе
    x1, y1 = top_left1
    result[max(0, y1 - h):min(result.shape[0], y1 + h),
    max(0, x1 - w):min(result.shape[1], x1 + w)] = -1.0

    # --- 2. Находим второе совпадение ---
    min_val, max_val2, min_loc, max_loc2 = cv2.minMaxLoc(result)
    top_left2 = max_loc2
    center2 = (top_left2[0] + w // 2, top_left2[1] + h // 2)

    centers = [center1]

    # --- 3. Фильтр по расстоянию ---
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = np.hypot(dx, dy)

    # --- 4. Фильтр по качеству совпадения ---
    if distance >= min_distance and abs(max_val2 - max_val1) <= score_drop_threshold:
        centers.append(center2)

    return centers


def get_max_threshold(image, template_path, config):
    filtered_image = prefilter_colors_red(image, config)
    template = cv2.imread(template_path)
    result = cv2.matchTemplate(filtered_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val1, min_loc, max_loc1 = cv2.minMaxLoc(result)
    return round(max_val1 * 100, 1)
