import cv2
import numpy as np
import math
import time
import json

# =====================================================
# 1. CONFIGURACIÓN DEL JUEGO Y MAPA
# =====================================================

WIDTH_MM = 3000
HEIGHT_MM = 2000

# IDs
ROBOT_ID = 51

# Marcadores del Campo (Esquinas)
FIELD_MARKERS = {
    20: (600, 1400),
    21: (2400, 1400),
    22: (600, 600),
    23: (2400, 600)
}

# Elementos del Juego
BOX_IDS = {
    41: "CAJA_NEGRA_VACIA",
    36: "CAJA_A_BELLOTA",
    47: "CAJA_B_BELLOTA"
}

# Configuración ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
parameters.polygonalApproxAccuracyRate = 0.05 
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
try:
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
except:
    pass

# =====================================================
# 2. MENÚ DE SELECCIÓN DE EQUIPO
# =====================================================

def select_team():
    print("\n" + "="*50)
    print("   EUROBOT 2026: VISIÓN SENIOR")
    print("="*50)
    print("Selecciona tu equipo:")
    print(" [1] AZUL (Blue Team)")
    print(" [2] AMARILLO (Yellow Team)")
    
    while True:
        try:
            choice = input(">> Ingresa 1 o 2: ")
            if choice == '1':
                return "BLUE", (255, 100, 0) # Azul (BGR)
            elif choice == '2':
                return "YELLOW", (0, 255, 255) # Amarillo (BGR)
        except:
            pass

TEAM_NAME, TEAM_COLOR = select_team()

# =====================================================
# 3. LÓGICA DE ZONAS Y MATEMÁTICAS
# =====================================================

def compute_homography(corners, ids):
    detected_pts = []
    real_pts = []
    ids_flat = ids.flatten()

    for mid, real_pos in FIELD_MARKERS.items():
        if mid in ids_flat:
            index = list(ids_flat).index(mid)
            real_pts.append(real_pos)
            detected_pts.append(np.mean(corners[index][0], axis=0))

    if len(detected_pts) < 4: return None
    
    detected_pts = np.array(detected_pts, dtype=np.float32)
    real_pts = np.array(real_pts, dtype=np.float32)
    H, _ = cv2.findHomography(detected_pts, real_pts, method=cv2.RANSAC)
    return H

def transform_point(point, H):
    p = np.array([point[0], point[1], 1], dtype=np.float32)
    res = H @ p
    if res[2] == 0: return (0, 0)
    return (res[0] / res[2], res[1] / res[2])

def transform_inverse(real_pt, H):
    try:
        Hinv = np.linalg.inv(H)
        p = np.array([real_pt[0], real_pt[1], 1], dtype=np.float32)
        img = Hinv @ p
        if img[2] == 0: return (0, 0)
        return (int(img[0] / img[2]), int(img[1] / img[2]))
    except:
        return (0,0)

def get_zone_name(x, y):
    """Determina en qué cuadrante está el robot"""
    # El campo mide 3000x2000. El centro es (1500, 1000)
    col = "IZQUIERDA" if x < 1500 else "DERECHA"
    row = "ABAJO" if y < 1000 else "ARRIBA"
    return f"{row}-{col}"

# =====================================================
# 4. INICIALIZACIÓN DE CÁMARA
# =====================================================

cap = cv2.VideoCapture(0) # <--- CAMBIA A 0 SI ES NECESARIO
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Configuración de imagen limpia
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -3.0) 
cap.set(cv2.CAP_PROP_GAIN, 200) 
cap.set(cv2.CAP_PROP_CONTRAST, 150)
cap.set(cv2.CAP_PROP_FOCUS, 0)

homography = None
last_print_time = time.time()

# =====================================================
# 5. BUCLE PRINCIPAL
# =====================================================

print(f"\nSistema Iniciado para equipo {TEAM_NAME}. Pulsa ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        new_H = compute_homography(corners, ids)
        if new_H is not None:
            homography = new_H

    current_scene_data = {
        "team": TEAM_NAME,
        "robot": None,
        "boxes": []
    }

    if homography is not None:
        
        # 1. DIBUJAR DIVISIONES (LÍNEAS)
        # Línea Vertical Central (X=1500)
        top_mid = transform_inverse((1500, HEIGHT_MM), homography)
        bot_mid = transform_inverse((1500, 0), homography)
        cv2.line(frame, top_mid, bot_mid, (180, 180, 180), 2) # Gris claro
        
        # Línea Horizontal Central (Y=1000)
        left_mid = transform_inverse((0, 1000), homography)
        right_mid = transform_inverse((WIDTH_MM, 1000), homography)
        cv2.line(frame, left_mid, right_mid, (180, 180, 180), 2) # Gris claro

        # 2. DIBUJAR LÍMITES DEL CAMPO
        board_pts = np.float32([[0, 0], [WIDTH_MM, 0], [WIDTH_MM, HEIGHT_MM], [0, HEIGHT_MM]])
        px_pts = [transform_inverse(pt, homography) for pt in board_pts]
        cv2.polylines(frame, [np.array(px_pts)], True, TEAM_COLOR, 3)

        # 3. PROCESAR OBJETOS
        if ids is not None:
            ids_flat = ids.flatten()
            for i, mid in enumerate(ids_flat):
                c = corners[i][0]
                center_px = np.mean(c, axis=0)
                pos_mm = transform_point(center_px, homography)
                
                # --- ROBOT ---
                if mid == ROBOT_ID:
                    # Angulo
                    front_px = (c[0] + c[1]) / 2
                    front_mm = transform_point(front_px, homography)
                    dx, dy = front_mm[0] - pos_mm[0], front_mm[1] - pos_mm[1]
                    angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                    
                    # Calcular Zona
                    zone_label = get_zone_name(pos_mm[0], pos_mm[1])

                    # Guardar Datos
                    current_scene_data["robot"] = {
                        "x": int(pos_mm[0]), 
                        "y": int(pos_mm[1]), 
                        "theta": int(angle),
                        "zone": zone_label
                    }
                    
                    # Dibujar
                    arrow_end = transform_inverse((
                        pos_mm[0] + 300 * math.cos(math.radians(angle)),
                        pos_mm[1] + 300 * math.sin(math.radians(angle))
                    ), homography)
                    
                    start_pt = (int(center_px[0]), int(center_px[1]))
                    cv2.arrowedLine(frame, start_pt, arrow_end, (0, 255, 0), 3, tipLength=0.3)
                    
                    # Texto informativo (Posición + Zona)
                    info_text = f"ROBOT: {zone_label} ({int(pos_mm[0])},{int(pos_mm[1])})"
                    cv2.putText(frame, info_text, (start_pt[0], start_pt[1]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- CAJAS ---
                elif mid in BOX_IDS:
                    box_type = BOX_IDS[mid]
                    current_scene_data["boxes"].append({
                        "id": int(mid),
                        "type": box_type,
                        "x": int(pos_mm[0]), 
                        "y": int(pos_mm[1])
                    })
                    
                    # Dibujar
                    color = (0, 0, 255) if "NEGRA" in box_type else (255, 255, 0)
                    cx, cy = int(center_px[0]), int(center_px[1])
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f"{box_type}", (cx-20, cy-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Output JSON cada 0.5s
    if time.time() - last_print_time > 0.5:
        if homography is not None:
            print(json.dumps(current_scene_data, indent=None))
        else:
            print(">> BUSCANDO CAMPO...")
        last_print_time = time.time()

    cv2.imshow('Eurobot 2026 Detector', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()