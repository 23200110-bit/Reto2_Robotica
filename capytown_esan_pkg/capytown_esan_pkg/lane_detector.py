#!/usr/bin/env python3
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
