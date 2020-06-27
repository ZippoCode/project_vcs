import argparse, random
import cv2
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom importing
from parameters import *
from read_write import get_videos, read_video, write_video, save_bounding_boxes


def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    point_one = (left, top - round(1.5 * labelSize[1]))
    point_two = (left + round(1.5 * labelSize[0]), top + baseLine)
    cv2.rectangle(frame, point_one, point_two, (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs, classes):
    frameHeight, frameWidth = frame.shape[0], frame.shape[1]
    classIds, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    founded_bbox = dict()
    for i in indices:
        i = i[0]
        box = boxes[i]
        class_name = classes[classIds[i]]
        x, y, width, height = box[0], box[1], box[2], box[3]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if width > frameWidth:
            width = frameWidth
        if height > frameHeight:
            height = frameHeight
        if class_name in founded_bbox:
            founded_bbox[class_name].append((x, y, width, height))
        else:
            founded_bbox[class_name] = [(x, y, width, height)]
        drawPred(frame, classes, classIds[i], confidences[i], x, y, x + width, y + height)
    return founded_bbox


def detect_person(frames, net, classes):
    if frames is None or len(frames) == 0:
        print('Frames not found. Return ...')
        return
    frames_detected = []
    detected_object_dict = dict()
    try:
        for num, frame in enumerate(frames):
            time_start = cv2.getTickCount()
            size = inpWidth, inpHeight
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, size, [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputs_names = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(outputs_names)
            detected_object_dict[num] = postprocess(frame, outs, classes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            frames_detected.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            time_end = cv2.getTickCount()
            time_elaboration = (time_end - time_start) / cv2.getTickFrequency()
            print('\t> Elaborate {}/{} frame in {} s'.format(num + 1, len(frames), time_elaboration))

    except KeyboardInterrupt:
        print('Stop processing')
        pass

    return frames_detected, detected_object_dict


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default=PATH_YOLO_CFG,
                        help='Path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str, default=PATH_YOLO_WEIGHTS,
                        help='Path of trained model.', dest='weightfile')
    # parser.add_argument('-videofile', type=str, default='../data/videos/001/GOPR5818.MP4',
    #                     help='Path of your video file.', dest='videofile')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Configure Network
    net = cv2.dnn.readNetFromDarknet(args.cfgfile, args.weightfile)
    if torch.cuda.is_available():  # Check if we are going to use GPU
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Read classes
    with open(PATH_COCO_NAMES, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
        print("Classes {}".format(classes))

    list_videos = get_videos()
<<<<<<< HEAD
    list_videos = ['../data/videos/002/20180206_114604.mp4']
    # Video with real person 003/GOPR1940.MP4 - 002/20180206_113059.mp4
=======
    list_videos = ['../data/videos/004/IMG_3803.MOV']
    # Video with real person 003/GOPR1940.MP4 - 002/20180206_113059.mp4 - 002/20180206_114604.mp4
>>>>>>> 2f17b00ebd4ead94678850d33ab17f81bba0c32e
    list_videos = random.choices(list_videos, k=1)
    for video_name in list_videos:
        if not os.path.isfile(video_name):
            print("Input video file ", video_name, " doesn't exist")
            sys.exit(1)
        frames = read_video(video_name, reduce_size=False)
        print("Start processing")
        people_detected, frame_with_bbox_dict = detect_person(frames, net, classes)
        output_name = video_name.split('/')[-1][:-4]
        write_video(output_name + '.avi', people_detected, fps=30, path=PATH_DESTINATION_PERSON_DETECTED)
        save_bounding_boxes(frame_with_bbox_dict, output_name)
        print('Done processing')
