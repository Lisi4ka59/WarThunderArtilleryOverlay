import time
import re

import cv2
import numpy as np
import mss
import pytesseract

import find_enemy_respawn
import find_marker
import json

import find_user_marker

# -------------------------------------------------
# НАСТРОЙКИ: координаты миникарты на экране
# -------------------------------------------------

MINIMAP_X = 2900  # <-- ПОДГОНИ ПОД СВОЁ РАЗРЕШЕНИЕ
MINIMAP_Y = 900
MINIMAP_W = 550
MINIMAP_H = 550

# Область с текстом масштаба ("225 m")
SCALE_ROI = (500,  # y
             400,  # x
             120,  # w
             23)  # h

# -------------------------------------------------
# Шаблоны
# -------------------------------------------------

USER_TEMPLATE_PATH = "templates/tank_player.png"
ENEMY_SPAWN_TEMPLATE_PATH = "templates/enemy_spawn_tank.png"
MARKER_TEMPLATE_PATH = "templates/marker_blue.png"

TANK_MATCH_THRESHOLD = 0.60
SPAWN_MATCH_THRESHOLD = 0.5
MARKER_MATCH_THRESHOLD = 0.60

FPS = 5
FRAME_TIME = 1.0 / FPS


def load_hsv_config(path="color_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------------------------------

# --- РИСОВАНИЕ ПРЯМОУГОЛЬНИКОВ ВОКРУГ НАЙДЕННЫХ ОБЪЕКТОВ (ОТЛАДКА) ---

def draw_boxes(result_img, centers, color):
    """Рисует зелёные прямоугольники вокруг найденных объектов."""
    for center in centers:
        cv2.circle(result_img, center, 3, (color[0], color[1], color[2]), -1)


def ocr_scale(minimap, use_ocr=True, fixed_scale_m=325):
    """
    Возвращает:
        scale_meters (в метрах),
        meters_per_px
    """
    scale_m = None

    if use_ocr:
        sy, sx, sw, sh = SCALE_ROI
        roi_bgr = minimap[sy:sy + sh, sx:sx + sw]

        img2 = cv2.resize(roi_bgr, None, fx=2, fy=2)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        _, th = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        t = pytesseract.image_to_string(th, config=config).strip()
        m = re.search(r"(\d{3})", t)
        if m:
            scale_m = float(m.group(1))

    # Если OCR выключен или не сработал
    else:
        scale_m = fixed_scale_m
    if scale_m is None:
        return None, None

    # --- 1. Формируем ROI для поиска линии ---
    text_y, text_x, text_w, text_h = SCALE_ROI

    line_y = text_y + text_h
    line_h = 2  # можно подбирать, обычно хватает 15–25 px

    line_roi = minimap[line_y: line_y + line_h,
               text_x: text_x + text_w]
    cv2.imshow("Line ROI", line_roi)

    hsv = cv2.cvtColor(line_roi, cv2.COLOR_BGR2HSV)

    # --- 2. Маска "черного" ---
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.medianBlur(mask, 5)

    cv2.imshow("mask_line", mask)

    # --- 3. Поиск линии — по маске, без Hough ---
    black_cols = np.any(mask == 255, axis=0)  # True/False по столбцам
    indices = np.where(black_cols)[0]

    if len(indices) == 0:
        return scale_m, 1

    best = indices[-1] - indices[0] + 1  # длина чёрного сегмента

    meters_per_px = scale_m / best
    return scale_m, meters_per_px*2


def draw_text_with_outline(img, text, pos, color=(0,255,0),
                           scale=0.5, thickness=1):
    x, y = pos
    # Чёрная обводка (тоньше)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness, cv2.LINE_AA)
    # Основной зелёный текст (ещё тоньше)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

# -------------------------------------------------
# ГЛАВНЫЙ ЦИКЛ
# -------------------------------------------------

def main():
    config = load_hsv_config()
    monitor = {
        "left": MINIMAP_X,
        "top": MINIMAP_Y,
        "width": MINIMAP_W,
        "height": MINIMAP_H,
    }

    with mss.mss() as sct:
        while True:
            start = time.time()
            # --- 1) СКРИН МИНИКАРТЫ ---
            raw = np.array(sct.grab(monitor))
            minimap = raw[:, :, :3]
            enemy_spawns_centres = find_enemy_respawn.find_enemy_respawn_centres(
                minimap,
                ENEMY_SPAWN_TEMPLATE_PATH,
                config)
            markers_centres_yellow = find_marker.find_marker_yellow(minimap, MARKER_TEMPLATE_PATH, config)
            markers_centres_blue = find_marker.find_marker_blue(minimap, MARKER_TEMPLATE_PATH, config)
            user_marker = find_user_marker.find_user_marker_centre(minimap, USER_TEMPLATE_PATH, config)

            sy, sx, sw, sh = SCALE_ROI
            scale_m, m_per_px = ocr_scale(minimap)

            # --- 6) РИСУЕМ РЕЗУЛЬТАТ ---
            result = minimap.copy()
            cv2.rectangle(result, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

            if user_marker and m_per_px:
                tx, ty = user_marker[0]
                #cv2.circle(result, (tx, ty), 6, (0, 255, 255), 2)

                # спавны
                # --- враги (спавны) ---
                for ex, ey in enemy_spawns_centres:
                    dpx = ((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5
                    dist = dpx * m_per_px / 2
                    dist = round(dist / 10) * 10
                    cv2.line(result, (tx, ty), (ex, ey), (255, 255, 255), 1)

                    draw_text_with_outline(
                        result,
                        f"{int(dist)}",  # ← без буквы 'm'
                        (ex - 20, ey - 10),
                        color=(0, 255, 0),  # салатовый
                        scale=0.55,
                        thickness=1
                    )

                # --- жёлтые метки ---
                for mx, my in markers_centres_yellow:
                    dpx = ((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5
                    dist = dpx * m_per_px / 2
                    dist = round(dist / 10) * 10
                    cv2.line(result, (tx, ty), (mx, my), (0, 255, 0), 1)

                    draw_text_with_outline(
                        result,
                        f"{int(dist)}",
                        (mx - 20, my - 10),
                        color=(0, 255, 0),
                        scale=0.55,
                        thickness=1
                    )

                # --- голубые метки ---
                for mx, my in markers_centres_blue:
                    dpx = ((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5
                    dist = dpx * m_per_px / 2
                    dist = round(dist / 10) * 10
                    cv2.line(result, (tx, ty), (mx, my), (0, 255, 0), 1)

                    draw_text_with_outline(
                        result,
                        f"{int(dist)}",
                        (mx - 20, my - 10),
                        color=(0, 255, 0),
                        scale=0.55,
                        thickness=1
                    )
            draw_boxes(result, enemy_spawns_centres, [0, 255, 0])
            draw_boxes(result, markers_centres_yellow, [255, 0, 0])
            draw_boxes(result, markers_centres_blue, [0, 0, 255])
            draw_boxes(result, user_marker, [255, 0, 255])


            # текст отладки
            if scale_m and m_per_px:
                cv2.putText(result, f"Scale: {scale_m} m  ({m_per_px:.2f} m/px)",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

            # --- 7) ПОКАЗЫВАЕМ ОТДЕЛЬНОЕ ОКНО ---
            cv2.imshow("Minimap Analysis", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # FPS
            dt = time.time() - start
            if dt < FRAME_TIME:
                time.sleep(FRAME_TIME - dt)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
