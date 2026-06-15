#!/usr/bin/env python3
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
