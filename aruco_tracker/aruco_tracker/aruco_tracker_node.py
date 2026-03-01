import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import cv2
import numpy as np
import math

# ================= CAMPO (mm) =================
WIDTH_MM = 3000
HEIGHT_MM = 2000

FIELD_MARKERS = {
    20: (600, 1400),
    21: (2400, 1400),
    22: (600, 600),
    23: (2400, 600),
}

# ================= IDS =================
ROBOT_ID = 51
ENEMY_ROBOT_ID = 52

SIMA_YELLOW_ID = 60
SIMA_BLUE_ID = 61

BOX_YELLOW_IDS = {36}
BOX_BLUE_IDS = {47}
BOX_BLACK_IDS = {41}

# ================= TIPOS =================
TYPE_YELLOW = 0
TYPE_BLUE = 1
TYPE_ENEMY = 2
TYPE_US = 3

# ================= FILTRO =================
ALPHA_POS = 0.30
ALPHA_TH = 0.35

# Ajustes de orientación por ID (si el ArUco no está alineado con el “frente” real)
THETA_OFFSET_BY_ID = {
    # ROBOT_ID: 0.0,
    # 36: math.pi/2,
    # 47: -math.pi/2,
}

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def angle_ema(prev: float, new: float, alpha: float) -> float:
    d = wrap_pi(new - prev)
    return wrap_pi(prev + alpha * d)

def get_zone_name(x_mm, y_mm):
    col = "IZQUIERDA" if x_mm < 1500 else "DERECHA"
    row = "ABAJO" if y_mm < 1000 else "ARRIBA"
    return f"{row}-{col}"


class ObjectsStateNode(Node):
    def __init__(self):
        super().__init__("objects_state_node")

        self.pub = self.create_publisher(Float32MultiArray, "/objects_state", 10)
        self.timer = self.create_timer(0.05, self.loop)

        self.cap = cv2.VideoCapture(0)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.homography = None
        self.filtered = {}  # mid -> [x_mm, y_mm, theta]

    # ---------- HOMOGRAFÍA ----------
    def compute_homography(self, corners, ids):
        img_pts = []
        real_pts = []
        ids_flat = ids.flatten().tolist()

        for mid, pos_mm in FIELD_MARKERS.items():
            if mid in ids_flat:
                idx = ids_flat.index(mid)
                center_px = np.mean(corners[idx][0], axis=0)
                img_pts.append(center_px)
                real_pts.append(pos_mm)

        if len(img_pts) < 4:
            return None

        H, _ = cv2.findHomography(
            np.array(img_pts, np.float32),
            np.array(real_pts, np.float32),
            cv2.RANSAC
        )
        return H

    def transform_point(self, p_xy_px):
        v = self.homography @ np.array([p_xy_px[0], p_xy_px[1], 1.0], dtype=float)
        if abs(v[2]) < 1e-9:
            return 0.0, 0.0
        return float(v[0] / v[2]), float(v[1] / v[2])

    def transform_inverse(self, p_xy_mm):
        Hinv = np.linalg.inv(self.homography)
        v = Hinv @ np.array([p_xy_mm[0], p_xy_mm[1], 1.0], dtype=float)
        if abs(v[2]) < 1e-9:
            return 0, 0
        return int(v[0] / v[2]), int(v[1] / v[2])

    # ---------- ORIENTACIÓN GLOBAL (MAPA) ----------
    def theta_from_corners(self, c4x2_px, mid):
        # Vector TL -> TR en coordenadas del MAPA (mm)
        p0_mm = self.transform_point(c4x2_px[0])  # TL
        p1_mm = self.transform_point(c4x2_px[1])  # TR

        theta = math.atan2(p1_mm[1] - p0_mm[1], p1_mm[0] - p0_mm[0])
        theta += THETA_OFFSET_BY_ID.get(mid, 0.0)
        return wrap_pi(theta)

    # ---------- FILTRO EMA ----------
    def ema(self, mid, x_mm, y_mm, theta):
        if mid not in self.filtered:
            self.filtered[mid] = [x_mm, y_mm, theta]
        else:
            fx, fy, ft = self.filtered[mid]
            fx = ALPHA_POS * x_mm + (1.0 - ALPHA_POS) * fx
            fy = ALPHA_POS * y_mm + (1.0 - ALPHA_POS) * fy
            ft = angle_ema(ft, theta, ALPHA_TH)
            self.filtered[mid] = [fx, fy, ft]
        return self.filtered[mid]

    # ---------- CLASIFICACIÓN ----------
    def get_tipo(self, mid):
        if mid == ROBOT_ID:
            return TYPE_US
        if mid == ENEMY_ROBOT_ID:
            return TYPE_ENEMY
        if mid == SIMA_YELLOW_ID:
            return TYPE_YELLOW
        if mid == SIMA_BLUE_ID:
            return TYPE_BLUE
        if mid in BOX_YELLOW_IDS:
            return TYPE_YELLOW
        if mid in BOX_BLUE_IDS:
            return TYPE_BLUE
        if mid in BOX_BLACK_IDS:
            return None
        return None

    def get_obj_name(self, mid):
        if mid == ROBOT_ID:
            return "ROBOT_US"
        if mid == ENEMY_ROBOT_ID:
            return "ROBOT_ENEMY"
        if mid == SIMA_YELLOW_ID:
            return "SIMA_YELLOW"
        if mid == SIMA_BLUE_ID:
            return "SIMA_BLUE"
        if mid in BOX_YELLOW_IDS:
            return f"BOX_YELLOW_{mid}"
        if mid in BOX_BLUE_IDS:
            return f"BOX_BLUE_{mid}"
        if mid in BOX_BLACK_IDS:
            return f"BOX_BLACK_{mid}"
        return f"ID_{mid}"

    # ---------- LOOP ----------
    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            cv2.imshow("Eurobot Vision", frame)
            cv2.waitKey(1)
            return

        H_new = self.compute_homography(corners, ids)
        if H_new is not None:
            self.homography = H_new
        if self.homography is None:
            cv2.imshow("Eurobot Vision", frame)
            cv2.waitKey(1)
            return

        # ===== DIBUJO DEL CAMPO =====
        board = [(0, 0), (WIDTH_MM, 0), (WIDTH_MM, HEIGHT_MM), (0, HEIGHT_MM)]
        px = [self.transform_inverse(p) for p in board]
        cv2.polylines(frame, [np.array(px)], True, (200, 200, 200), 2)

        cv2.line(frame, self.transform_inverse((1500, 0)), self.transform_inverse((1500, HEIGHT_MM)), (150, 150, 150), 1)
        cv2.line(frame, self.transform_inverse((0, 1000)), self.transform_inverse((WIDTH_MM, 1000)), (150, 150, 150), 1)

        msg = Float32MultiArray()
        msg.data = []

        for i, mid_raw in enumerate(ids.flatten()):
            mid = int(mid_raw)

            # Ignorar marcadores fijos
            if mid in FIELD_MARKERS:
                continue

            tipo = self.get_tipo(mid)
            if tipo is None:
                continue

            c = corners[i][0]  # (4,2) px
            center_px = np.mean(c, axis=0)

            # Pose "raw" en mapa (mm) SOLO para dibujar encima del ArUco
            x_mm_raw, y_mm_raw = self.transform_point(center_px)
            theta_raw = self.theta_from_corners(c, mid)

            # -------- DIBUJO: SIEMPRE encima del ArUco detectado --------
            L_mm = 250.0
            end_px = self.transform_inverse((
                x_mm_raw + L_mm * math.cos(theta_raw),
                y_mm_raw + L_mm * math.sin(theta_raw)
            ))
            cv2.arrowedLine(
                frame,
                tuple(center_px.astype(int)),
                end_px,
                (0, 255, 0),
                2
            )

            # Texto cerca del ArUco detectado
            name = self.get_obj_name(mid)
            cv2.putText(
                frame,
                name,
                (int(center_px[0]) + 5, int(center_px[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # -------- PUBLICACIÓN: filtrado por ID (ojo si IDs se repiten) --------
            x_mm_f, y_mm_f, theta_f = self.ema(mid, x_mm_raw, y_mm_raw, theta_raw)
            msg.data.extend([x_mm_f / 1000.0, y_mm_f / 1000.0, theta_f, float(tipo)])

            if mid == ROBOT_ID:
                zone = get_zone_name(x_mm_raw, y_mm_raw)
                cv2.putText(frame, f"ZONA: {zone}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.pub.publish(msg)
        cv2.imshow("Eurobot Vision", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = ObjectsStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
