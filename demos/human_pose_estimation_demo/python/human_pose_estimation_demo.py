#!/usr/bin/env python3
"""
 Copyright (C) 2020-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter, process_time_ns
import time
import datetime

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
import models
import monitors
from images_capture import open_images_capture
from pipelines import AsyncPipeline
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ae', 'openpose'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                       help='Optional. Number of frames to store in output. '
                            'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    # add
    args.add_argument('--record', default=False, action='store_true',
                      help='Record anonymously.')


    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.1, type=float,
                                   help='Optional. Probability threshold for poses filtering.')
    common_model_args.add_argument('--tsize', default=None, type=int,
                                   help='Optional. Target input size. This demo implements image pre-processing '
                                        'pipeline that is common to human pose estimation approaches. Image is first '
                                        'resized to some target size and then the network is reshaped to fit the input '
                                        'image shape. By default target image size is determined based on the input '
                                        'shape from IR. Alternatively it can be manually set via this parameter. Note '
                                        'that for OpenPose-like nets image is resized to a predefined height, which is '
                                        'the target size in this case. For Associative Embedding-like nets target size '
                                        'is the length of a short first image side.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def get_model(ie, args, aspect_ratio):
    if args.architecture_type == 'ae':
        Model = models.HpeAssociativeEmbedding
    elif args.architecture_type == 'openpose':
        Model = models.OpenPose
    else:
        raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))
    return Model(ie, args.model, target_size=args.tsize, aspect_ratio=aspect_ratio, prob_threshold=args.prob_threshold)


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))


def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton, draw_ellipses=False):
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(int).tolist()
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def draw_poses_skeleton(img, poses, point_score_threshold, video_writer, skeleton=default_skeleton, draw_ellipses=False):
    if poses.size == 0:
        print(img.shape[1])
        print(img.shape[0])
        return img
    stick_width = 4

    img_limbs = np.copy(img)

    img = np.zeros_like(img)
    img_limbs = np.zeros_like(img)
    for pose in poses:
        points = pose[:, :2].astype(int).tolist()
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    video_writer.write(img)
    return img


def print_raw_results(poses, scores):
    log.info('Poses:')
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.info('{} | {:.2f}'.format(pose_str, pose_score))


def main():
    args = build_argparser().parse_args()
    metrics = PerformanceMetrics()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    log.info('Loading network...')
    model = get_model(ie, args, frame.shape[1] / frame.shape[0])
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

    log.info('Starting inference...')
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    next_frame_id = 1
    next_frame_id_to_show = 0
    record_list = []

    dt_now = datetime.datetime.now()
    output_dir = dt_now.strftime('%Y%m%d')
    Path('output/'+output_dir).mkdir(exist_ok=True)
    if dt_now.minute <= 29:
        output_file_name = dt_now.strftime('%H') + '00'
    else:
        output_file_name = dt_now.strftime('%H') + '30'
    dt_pre = datetime.datetime.now()

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
    video_writer = cv2.VideoWriter()
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 3,
            (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    try:
        while True:
            if hpe_pipeline.callback_exceptions:
                raise hpe_pipeline.callback_exceptions[0]
            # Process all completed requests
            results = hpe_pipeline.get_result(next_frame_id_to_show)
            if results:
                dt_now = datetime.datetime.now()
                if (dt_pre.minute == 59 and dt_now.minute == 0) or (dt_pre.minute == 29 and dt_now.minute == 30):
                    if dt_pre.minute <= 29:
                        output_file_name_pre = dt_pre.strftime('%H') + '00'
                    else:
                        output_file_name_pre = dt_pre.strftime('%H') + '30'

                    print("saved: ", output_dir + output_file_name_pre)
                    np.save("output/"+output_dir+"/"+output_file_name+".npy", record_list)
                    record_list = []
                    print("フォルダを作成します。")
                    output_dir = dt_now.strftime('%Y%m%d')
                    Path('output/'+output_dir).mkdir(exist_ok=True)
                    if dt_now.minute <= 29:
                        output_file_name = dt_now.strftime('%H') + '00'
                    else:
                        output_file_name = dt_now.strftime('%H') + '30'
                        

                # pre_minute = dt_now.minute
                dt_pre = dt_now
                (poses, scores), frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

                if len(poses) and args.raw_output_message:
                    print_raw_results(poses, scores)
                if args.record:
                    record_list.append([poses, time.time()])

                presenter.drawGraphs(frame)
                frame = draw_poses(frame, poses, args.prob_threshold)
                # frame = draw_poses_skeleton(frame, poses, args.prob_threshold, video_writer)
                metrics.update(start_time, frame)
                if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                    # video_writer.write(frame)
                    pass
                if not args.no_show:
                    cv2.imshow('Pose estimation results', frame)
                    key = cv2.waitKey(1)

                    ESC_KEY = 27
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break
                    presenter.handleKey(key)
                next_frame_id_to_show += 1
                continue

            if hpe_pipeline.is_ready():
                # Get new image/frame
                start_time = perf_counter()
                frame = cap.read()
                if frame is None:
                    break

                # Submit for inference
                hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                next_frame_id += 1

            else:
                # Wait for empty request
                hpe_pipeline.await_any()
    except KeyboardInterrupt:
        pass

    hpe_pipeline.await_all()
    # Process completed requests
    while hpe_pipeline.has_completed_request():
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message:
                print_raw_results(poses, scores)

            presenter.drawGraphs(frame)
            frame = draw_poses(frame, poses, args.prob_threshold)
            metrics.update(start_time, frame)
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
            if not args.no_show:
                cv2.imshow('Pose estimation results', frame)
                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            next_frame_id_to_show += 1
        else:
            break

    if args.record:
        print("saved: ", output_dir+output_file_name)
        np.save("output/"+output_dir+"/"+output_file_name+".npy", record_list)
    metrics.print_total()
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
