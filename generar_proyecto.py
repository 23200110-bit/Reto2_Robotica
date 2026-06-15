#!/usr/bin/env python3
"""
generar_proyecto.py — Script automatizado para estructurar el Repositorio Reto2_Robotica.
Universidad ESAN · Robótica 2026-I · Reto Clasificatorio 2 (RC-2)

Este script crea automáticamente todas las carpetas necesarias y escribe el contenido
de cada archivo de ROS 2 para que el paquete esté listo para ser compilado y subido a GitHub.
"""

import os

# Definir la estructura de directorios
DIRECTORIOS = [
    "scripts_vision",
    "capytown_esan_pkg",
    os.path.join("capytown_esan_pkg", "capytown_esan_pkg"),
    os.path.join("capytown_esan_pkg", "config"),
    os.path.join("capytown_esan_pkg", "launch"),
]

# Diccionario con el contenido de los archivos
ARCHIVOS = {}

# ----------------- 1. ARCHIVOS CONFIGURACIÓN ROS2 -----------------

ARCHIVOS[os.path.join("capytown_esan_pkg", "setup.py")] = """from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'capytown_esan_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ESAN_Alumnos',
    maintainer_email='alumnos@esan.edu.pe',
    description='Paquete de seguimiento de carril autónomo - Reto 2 (CapyTown)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = capytown_esan_pkg.lane_detector:main',
            'lane_controller = capytown_esan_pkg.lane_controller:main',
        ],
    },
)
"""

ARCHIVOS[os.path.join("capytown_esan_pkg", "package.xml")] = """<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>capytown_esan_pkg</name>
  <version>0.1.0</version>
  <description>Paquete de seguimiento de carril autónomo - Reto 2 (CapyTown)</description>
  <maintainer email="alumnos@esan.edu.pe">ESAN_Alumnos</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>cv_bridge</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""

ARCHIVOS[os.path.join("capytown_esan_pkg", "setup.cfg")] = """[develop]
script_dir=$base/lib/capytown_esan_pkg
[install]
install_scripts=$base/lib/capytown_esan_pkg
"""

# ----------------- 2. NODO DE VISIÓN (LANE_DETECTOR.PY) -----------------

ARCHIVOS[os.path.join("capytown_esan_pkg", "capytown_esan_pkg", "lane_detector.py")] = """#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import yaml
import json

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.get_logger().info('Iniciando nodo de detección de carril: lane_detector...')

        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('lane_width_m', 0.28)
        self.declare_parameter('px_per_meter', 450.0)
        self.declare_parameter('look_ahead_row', 380)
        self.declare_parameter('hsv_params_path', 'hsv_params.yaml')
        self.declare_parameter('ipm_params_path', 'ipm_config.json')

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.lane_width_m = self.get_parameter('lane_width_m').get_parameter_value().double_value
        self.px_per_meter = self.get_parameter('px_per_meter').get_parameter_value().double_value
        self.look_ahead_row = self.get_parameter('look_ahead_row').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.error_pub = self.create_publisher(Float32, '/lane_error', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data
        )

        self.K = None
        self.D = None
        self.M = None
        self.ipm_w = 640
        self.ipm_h = 480

        self.hsv_white_low = np.array([0, 0, 200])
        self.hsv_white_high = np.array([180, 40, 255])
        self.hsv_yellow_low = np.array([15, 80, 80])
        self.hsv_yellow_high = np.array([40, 255, 255])

        self.load_calibration_files()

    def load_calibration_files(self):
        yaml_path = 'camera_params.yaml'
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.K = np.array(data['camera_matrix']['data']).reshape((3, 3))
                    self.D = np.array(data['distortion_coefficients']['data'])
                self.get_logger().info('✓ Parámetros intrínsecos de cámara cargados.')
            except Exception as e:
                self.get_logger().error(f'Error leyendo {yaml_path}: {e}')

        hsv_yaml = self.get_parameter('hsv_params_path').get_parameter_value().string_value
        if os.path.exists(hsv_yaml):
            try:
                with open(hsv_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    self.hsv_white_low = np.array(data['white']['low'])
                    self.hsv_white_high = np.array(data['white']['high'])
                    self.hsv_yellow_low = np.array(data['yellow']['low'])
                    self.hsv_yellow_high = np.array(data['yellow']['high'])
                self.get_logger().info('✓ Límites de color HSV cargados.')
            except Exception as e:
                self.get_logger().error(f'Error leyendo {hsv_yaml}: {e}')

        ipm_json = self.get_parameter('ipm_params_path').get_parameter_value().string_value
        if os.path.exists(ipm_json):
            try:
                with open(ipm_json, 'r') as f:
                    config = json.load(f)
                    src = np.array(config['src'], dtype=np.float32)
                    dst = np.array(config['dst'], dtype=np.float32)
                    self.M = cv2.getPerspectiveTransform(src, dst)
                    self.ipm_w = config.get('size', [640, 480])[0]
                    self.ipm_h = config.get('size', [640, 480])[1]
                self.get_logger().info('✓ Matriz de homografía IPM configurada.')
            except Exception as e:
                self.get_logger().error(f'Error leyendo {ipm_json}: {e}')
        else:
            src = np.float32([[150, 280], [490, 280], [600, 460], [40, 460]])
            dst = np.float32([[120, 0], [520, 0], [520, 480], [120, 480]])
            self.M = cv2.getPerspectiveTransform(src, dst)

    def detect_line_centroid(self, mask, row_y, width_window=80):
        row_pixels = mask[row_y, :]
        active_indices = np.where(row_pixels > 0)[0]
        if len(active_indices) == 0:
            return None
        hist, bins = np.histogram(active_indices, bins=np.arange(0, mask.shape[1] + 10, 10))
        if len(hist) == 0:
            return None
        peak_bin = np.argmax(hist)
        center_estimate = int((bins[peak_bin] + bins[peak_bin + 1]) / 2)
        local_pixels = active_indices[
            (active_indices >= center_estimate - width_window) & 
            (active_indices <= center_estimate + width_window)
        ]
        if len(local_pixels) > 0:
            return float(np.mean(local_pixels))
        return None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Fallo al convertir imagen: {e}')
            return

        if self.K is not None and self.D is not None:
            frame = cv2.undistort(frame, self.K, self.D, None, self.K)

        ipm_image = cv2.warpPerspective(frame, self.M, (self.ipm_w, self.ipm_h))
        hsv = cv2.cvtColor(ipm_image, cv2.COLOR_BGR2HSV)
        
        mask_white = cv2.inRange(hsv, self.hsv_white_low, self.hsv_white_high)
        mask_yellow = cv2.inRange(hsv, self.hsv_yellow_low, self.hsv_yellow_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        x_white = self.detect_line_centroid(mask_white, self.look_ahead_row)
        x_yellow = self.detect_line_centroid(mask_yellow, self.look_ahead_row)

        lane_width_px = self.lane_width_m * self.px_per_meter
        center_of_image = self.ipm_w / 2.0
        center_px = None

        if x_white is not None and x_yellow is not None:
            center_px = (x_white + x_yellow) / 2.0
        elif x_white is not None:
            center_px = x_white - (lane_width_px / 2.0)
        elif x_yellow is not None:
            center_px = x_yellow + (lane_width_px / 2.0)

        msg_error = Float32()
        if center_px is not None:
            error_m = (center_px - center_of_image) / self.px_per_meter
            msg_error.data = float(error_m)
        else:
            msg_error.data = float('nan')

        self.error_pub.publish(msg_error)

        debug_view = ipm_image.copy()
        cv2.line(debug_view, (0, self.look_ahead_row), (self.ipm_w, self.look_ahead_row), (255, 0, 255), 1)
        if x_white is not None:
            cv2.circle(debug_view, (int(x_white), self.look_ahead_row), 8, (0, 255, 0), -1)
        if x_yellow is not None:
            cv2.circle(debug_view, (int(x_yellow), self.look_ahead_row), 8, (0, 255, 255), -1)
        if center_px is not None:
            cv2.drawMarker(debug_view, (int(center_px), self.look_ahead_row), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.line(debug_view, (int(center_of_image), 0), (int(center_of_image), self.ipm_h), (128, 128, 128), 1)
            cv2.putText(debug_view, f"Error: {msg_error.data * 100:.1f} cm", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug_view, "¡PERDIDO - SIN LINEAS!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_view, 'bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Error al publicar debug_image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Apagando nodo de visión...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

# ----------------- 3. NODO DE CONTROL (LANE_CONTROLLER.PY) -----------------

ARCHIVOS[os.path.join("capytown_esan_pkg", "capytown_esan_pkg", "lane_controller.py")] = """#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class LaneController(Node):
    def __init__(self):
        super().__init__('lane_controller')
        self.get_logger().info('Iniciando nodo de control lateral: lane_controller...')

        self.declare_parameter('kp', 2.2)
        self.declare_parameter('ki', 0.05)
        self.declare_parameter('kd', 0.18)
        self.declare_parameter('v_cruise', 0.22)
        self.declare_parameter('v_min', 0.10)
        self.declare_parameter('max_omega', 1.8)
        self.declare_parameter('error_timeout', 0.5)
        self.declare_parameter('max_windup', 0.25)

        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.v_cruise = self.get_parameter('v_cruise').get_parameter_value().double_value
        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.max_omega = self.get_parameter('max_omega').get_parameter_value().double_value
        self.error_timeout = self.get_parameter('error_timeout').get_parameter_value().double_value
        self.max_windup = self.get_parameter('max_windup').get_parameter_value().double_value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.error_sub = self.create_subscription(Float32, '/lane_error', self.error_callback, 10)

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = self.get_clock().now()
        self.last_valid_error_time = self.get_clock().now()
        self.is_lost = True

        self.timer = self.create_timer(0.05, self.watchdog_timer_callback)

    def error_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_time = current_time

        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value

        error = msg.data
        if math.isnan(error):
            self.is_lost = True
            return

        self.is_lost = False
        self.last_valid_error_time = current_time

        P = self.kp * error
        self.integral += error * dt
        self.integral = max(-self.max_windup, min(self.max_windup, self.integral))
        I = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative
        self.prev_error = error

        omega = -(P + I + D)
        omega = max(-self.max_omega, min(self.max_omega, omega))

        error_abs = abs(error)
        if error_abs > 0.15:
            v_linear = self.v_min
        else:
            v_linear = self.v_cruise - (self.v_cruise - self.v_min) * (error_abs / 0.15)

        self.publish_twist(v_linear, omega)

    def publish_twist(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

    def watchdog_timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_valid_error_time).nanoseconds / 1e9
        if self.is_lost or elapsed > self.error_timeout:
            self.publish_twist(0.0, 0.0)
            self.get_logger().warn("⚠ Watchdog activo: Deteniendo motores.", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = LaneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Apagando nodo de control...')
        node.publish_twist(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""

# ----------------- 4. ARCHIVO LAUNCH FILE -----------------

ARCHIVOS[os.path.join("capytown_esan_pkg", "launch", "lane_following.launch.py")] = """import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('capytown_esan_pkg'),
        'config'
    )
    
    detector_node = Node(
        package='capytown_esan_pkg',
        executable='lane_detector',
        name='lane_detector',
        parameters=[{
            'hsv_params_path': os.path.join(config_dir, 'hsv_params.yaml'),
            'ipm_params_path': os.path.join(config_dir, 'ipm_config.json'),
        }],
        output='screen'
    )
    
    controller_node = Node(
        package='capytown_esan_pkg',
        executable='lane_controller',
        name='lane_controller',
        output='screen'
    )
    
    return LaunchDescription([
        detector_node,
        controller_node
    ])
"""

# Inicializador de paquete vacío
ARCHIVOS[os.path.join("capytown_esan_pkg", "capytown_esan_pkg", "__init__.py")] = ""

# Archivo README descriptivo en la raíz
ARCHIVOS["README.md"] = """# 🚗 Reto 2 — CapyTown: "Las 3 Vueltas del Jirón"
Paquete de navegación autónoma y seguimiento de carril para ROS 2 (Humble/Iron) en Raspberry Pi 5.

## Estructura
- `capytown_esan_pkg/` : Paquete de ROS 2 (Nodos de visión y control).
- `scripts_vision/` : Scripts de calibración y pruebas rápidas fuera de ROS 2.
"""

def main():
    print("====================================================")
    print("   Generando Estructura de CapyTown (Reto 2)        ")
    print("====================================================\\n")
    
    # 1. Crear directorios
    for direc in DIRECTORIOS:
        if not os.path.exists(direc):
            os.makedirs(direc)
            print(f"[CARPETA] Creada con éxito: {direc}")
        else:
            print(f"[CARPETA] Ya existía: {direc}")
            
    # 2. Crear archivos y poblar código
    print("\\nEscribiendo archivos de código fuente...")
    for filepath, contenido in ARCHIVOS.items():
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(contenido)
            print(f"  [ARCHIVO] Escrito: {filepath}")
        except Exception as e:
            print(f"  [ERROR] Falló al escribir {filepath}: {e}")
            
    print("\\n====================================================")
    print("  ¡Estructura generada con éxito!                   ")
    print("  Ya puedes borrar el archivo 'generar_proyecto.py' ")
    print("====================================================")

if __name__ == "__main__":
    main()