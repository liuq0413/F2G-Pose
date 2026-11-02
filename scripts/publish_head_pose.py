#!/usr/bin/python3
# 用途：收到抓取程序的 ROS 请求后调用 F2G-Pose，发布头部相机系物体 PoseStamped；本脚本不发布运动指令。
# 运行：source /home/liuqian/WBC/Apex_Deploy/apex_ros_env.sh && /usr/bin/python3 scripts/publish_head_pose.py --help
"""ROS 环境与 F2G-Pose 虚拟环境之间的按需位姿桥接。"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty


def pose_message(value, now, max_age):
    stamp = float(value['stamp'])
    position = value['position_m']
    quaternion = value['quaternion_xyzw']
    if (not math.isfinite(stamp) or not -.05 <= now - stamp <= max_age or
            len(position) != 3 or len(quaternion) != 4 or
            not all(math.isfinite(float(x)) for x in position + quaternion) or
            abs(math.sqrt(sum(float(x)**2 for x in quaternion)) - 1) > .01 or
            value['frame_id'] != 'luna_head_calibrated_optical'):
        raise ValueError('位姿时间戳、坐标系、位置或四元数无效。')
    message = PoseStamped()
    seconds = math.floor(stamp)
    nanoseconds = int(round((stamp - seconds) * 1e9))
    if nanoseconds == 1_000_000_000:
        seconds += 1
        nanoseconds = 0
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = nanoseconds
    message.header.frame_id = value['frame_id']
    message.pose.position.x, message.pose.position.y, message.pose.position.z = position
    (message.pose.orientation.x, message.pose.orientation.y,
     message.pose.orientation.z, message.pose.orientation.w) = quaternion
    return message


class HeadPosePublisher(Node):
    def __init__(self, args):
        super().__init__('f2g_head_pose_publisher')
        self.args = args
        self.started = False
        self.process = None
        self.publisher = self.create_publisher(PoseStamped, args.pose_topic, qos_profile_sensor_data)
        self.create_subscription(Empty, args.request_topic, self.on_request, qos_profile_sensor_data)
        self.get_logger().info(f'等待 {args.request_topic} 请求，发布到 {args.pose_topic}；不执行机器人运动')

    def on_request(self, _message):
        if self.started:
            return
        self.started = True
        threading.Thread(target=self.infer, daemon=True).start()

    def infer(self):
        a = self.args
        command = [str(a.worker_python), '-u', '-m', 'f2g_pose.head_pose_worker',
                   '--camera-url', a.camera_url, '--checkpoint', str(a.checkpoint),
                   '--radio-repository', str(a.radio_repository),
                   '--radio-checkpoint', str(a.radio_checkpoint),
                   '--sam3-python', str(a.sam3_python),
                   '--sam3-checkpoint', str(a.sam3_checkpoint),
                   '--prompt', a.prompt, '--duration', str(a.duration)]
        if a.instance_index is not None:
            command += ['--instance-index', str(a.instance_index)]
        try:
            self.process = subprocess.Popen(command, cwd=a.project_root, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in self.process.stdout:
                try:
                    payload = json.loads(line)
                    if 'error' in payload:
                        self.get_logger().error(f'位姿推理失败：{payload["error"]}')
                        break
                    pose = pose_message(payload['pose'], time.time(), a.max_pose_age)
                except (ValueError, KeyError, TypeError) as exc:
                    self.get_logger().warning(f'忽略无效或过期的推理结果：{exc}')
                    continue
                self.publisher.publish(pose)
                self.get_logger().info(f'已发布头部相机系物体位置：{payload["pose"]["position_m"]}')
            self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.get_logger().error(f'位姿子进程失败：{exc}')
        finally:
            self.process = None
            self.started = False
            self.get_logger().info('本次请求结束；发布器可继续处理下一次请求')

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root', type=Path, default=root)
    parser.add_argument('--worker-python', type=Path, default=root / '.venv/bin/python')
    parser.add_argument('--camera-url', default='http://6.6.7.100:8765/frame')
    parser.add_argument('--checkpoint', type=Path, default=root / 'weights/f2g-pose.pt')
    parser.add_argument('--radio-repository', type=Path, default=root / 'third_party_local/RADIO')
    parser.add_argument('--radio-checkpoint', type=Path, default=root / 'weights/radio-v2.5-l_half.pth.tar')
    parser.add_argument('--sam3-python', type=Path, default=root / 'third_party_local/.venv-sam3/bin/python')
    parser.add_argument('--sam3-checkpoint', type=Path, default=root / 'weights/sam3/sam3.pt')
    parser.add_argument('--request-topic', default='/f2g_pose/request')
    parser.add_argument('--pose-topic', default='/object_pose')
    parser.add_argument('--prompt', default='water bottle')
    parser.add_argument('--instance-index', type=int)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--max-pose-age', type=float, default=.5)
    args = parser.parse_args(argv)
    if not args.prompt.strip() or args.instance_index is not None and args.instance_index < 1 or not all(
            math.isfinite(v) and v > 0 for v in (args.duration, args.max_pose_age)):
        parser.error('请求参数无效')
    for path in (args.worker_python, args.checkpoint, args.radio_repository,
                 args.radio_checkpoint, args.sam3_python, args.sam3_checkpoint):
        if not path.exists():
            parser.error(f'缺少文件或目录：{path}')
    return args


def main():
    args = parse_args()
    rclpy.init(args=[])
    node = HeadPosePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
