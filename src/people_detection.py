import argparse
import cv2
import os.path
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from parameters import *


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(
        1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
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
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        drawPred(frame, classes, classIds[i], confidences[i],
                 left, top, left + width, top + height)


def detect_person(cfgfile, weightfile, videofile):
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
        print(classes)

    net = cv2.dnn.readNetFromDarknet(cfgfile, weightfile)

    # check if we are going to use GPU
    if use_cuda:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    if videofile:
        if not os.path.isfile(videofile):
            print("Input video file ", videofile, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(videofile)
        name_video = videofile.split('/')[-1][:-4]
        if not os.path.exists(PATH_DESTINATION_PERSON_DETECTED):
            Path(PATH_DESTINATION_PERSON_DETECTED).mkdir(
                parents=True, exist_ok=True)
        outputFile = os.path.join(
            PATH_DESTINATION_PERSON_DETECTED, name_video + '.avi')
    else:
        cap = cv2.VideoCapture(0)
        outputFile = '../yolo/default_camera.avi'

    codec = cv2.VideoWriter_fourcc(*'MJPG')
    (h, w) = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter(outputFile, codec, 30, (h, w))

    try:
        while True:
            time_start = cv2.getTickCount()
            hasFrame, frame = cap.read()
            if not hasFrame:
                print("Done processing !!!\nOutput file is stored as {}.\nNot HasFrame".format(
                    outputFile))
                cap.release()
                break
            size = inpWidth, inpHeight
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255, size, [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            postprocess(frame, outs, classes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (
                t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            vid_writer.write(frame.astype(np.uint8))
            time_end = cv2.getTickCount()
            print('Elaborate frame in {} s'.format(
                (time_end - time_start) / cv2.getTickFrequency()))

    except KeyboardInterrupt:
        print('Stop processing')
        pass
    print('Done processing')


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default=PATH_YOLO_CFG,
                        help='Path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str, default=PATH_YOLO_WEIGHTS,
                        help='Path of trained model.', dest='weightfile')
    parser.add_argument('-videofile', type=str, default='../data/videos/001/GOPR5818.MP4',
                        help='Path of your video file.', dest='videofile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.videofile:
        detect_person(args.cfgfile, args.weightfile, args.videofile)
