from ast import parse
import numpy as np
import cv2
from argparse import ArgumentParser
from pathlib import Path


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))


def build_argparser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='Input npy file.')
    parser.add_argument('-o', '--output', default=False, action='store_true',
                        help='Output mp4 file.')

    return parser


def draw_poses_skeleton(img, poses, point_score_threshold, skeleton=default_skeleton, draw_ellipses=False):
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
    return img


def main():
    args = build_argparser().parse_args()
    img_width = 1280
    img_height = 720
    

    mp4_file_name = str(args.input.parent) + "/"+ str(args.input.stem) + ".mp4"

    # Define the codec and create VideoWriter object
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(mp4_file_name, fourcc, 5, (img_width, img_height))

    frame = np.zeros((img_height, img_width, 3))

    load_data = np.load(args.input, allow_pickle=True)
    load_poses, load_time = load_data[:,0], load_data[:, 1]

    for i, poses in enumerate(load_poses):
        frame = draw_poses_skeleton(frame, poses, 0.1)

        if args.output:
            video_writer.write(frame.astype(np.uint8))
        else:
            cv2.imshow('Pose estimation results', frame)
        
            key = cv2.waitKey(40) # FPS

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
    

if __name__ == '__main__':
    main()
