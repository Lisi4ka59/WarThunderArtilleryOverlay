import cv2
import numpy as np

from color_filters import prefilter_colors_warm_yellow


def find_user_marker_centre(image: np.ndarray,
                              template_path: str,
                              config):
    """
    Ищет ОДНО лучшее совпадение шаблона на изображении.
    Возвращает центр найденного шаблона или пустой список, если ничего не найдено.
    """

    # --- 1. Цветовая фильтрация ---
    filtered_image = prefilter_colors_warm_yellow(image, config)

    # --- 2. Загружаем шаблон ---
    template = cv2.imread(template_path)
    if template is None:
        raise RuntimeError(f"Не удалось загрузить шаблон: {template_path}")

    h, w = template.shape[:2]

    # --- 3. Шаблонное сопоставление ---
    result = cv2.matchTemplate(filtered_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # --- 4. Проверка порога ---
    if max_val < config["marker_user_threshold"] / 100:
        return []

    # --- 5. Координаты центра ---
    top_left = max_loc
    center = (top_left[0] + w // 2, top_left[1] + h // 2)

    return [center]


def get_max_threshold(image, template_path, config):
    filtered_image = prefilter_colors_warm_yellow(image, config)
    template = cv2.imread(template_path)
    result = cv2.matchTemplate(filtered_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return round(max_val * 100, 1)