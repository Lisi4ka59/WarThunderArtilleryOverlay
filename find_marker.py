import cv2

from color_filters import prefilter_colors_yellow, prefilter_colors_blue


def find_marker_centres(image, template_path, max_matches=5, min_distance=5, threshold=90):
    """
    Ищет до max_matches совпадений шаблона по форме (цвет игнорируется).
    Возвращает список центров найденных совпадений.
    """

    # Загружаем template и изображение
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Не удалось загрузить template: {template_path}")

    # MatchTemplate по границам
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    matches = []
    used_points = []

    while len(matches) < max_matches:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold / 100:
            break  # больше нет хороших совпадений

        x, y = max_loc
        h, w = template.shape[:2]

        center = (x + w // 2, y + h // 2)

        # Проверяем расстояние от уже найденных
        too_close = False
        for (cx, cy) in used_points:
            if (center[0] - cx) ** 2 + (center[1] - cy) ** 2 < min_distance ** 2:
                too_close = True
                break

        if not too_close:
            matches.append(center)
            used_points.append(center)

        # Чтобы мешающий пиксель не участвовал дальше — затираем его область
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            -1,
            thickness=-1
        )

    return matches


def find_marker_yellow(image, template_path, config, max_matches=5, min_distance=5):
    return find_marker_centres(
        prefilter_colors_yellow(image, config),
        template_path,
        max_matches,
        min_distance,
        config["marker_yellow_threshold"],
    )


def find_marker_blue(image, template_path, config, max_matches=5, min_distance=5):
    return find_marker_centres(
        prefilter_colors_blue(image, config),
        template_path,
        max_matches,
        min_distance,
        config["marker_blue_threshold"],
    )


def get_max_val_yellow(image, template_path, config):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    result = cv2.matchTemplate(prefilter_colors_yellow(image, config), template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return round(max_val * 100, 1)


def get_max_val_blue(image, template_path, config):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    result = cv2.matchTemplate(prefilter_colors_blue(image, config), template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return round(max_val * 100, 1)