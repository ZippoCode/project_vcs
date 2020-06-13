import argparse
import cv2
import os.path

from parameters import *
from tool.utils import *
from tool.darknet2pytorch import Darknet

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

classesFile = "../yolo/coco.names"


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(
        1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


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


def detect_yolo_v3(cfgfile, weightfile, videofile):
    # Load names of classes
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
        print(classes)

    net = cv2.dnn.readNetFromDarknet(cfgfile, weightfile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Process inputs
    if (videofile):
        # Open the video file
        if not os.path.isfile(videofile):
            print("Input video file ", videofile, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(videofile)
        outputFile = '../yolo/' + \
                     videofile.split('/')[-1][:-4] + '_yolo_output.avi'
    else:
        cap = cv2.VideoCapture(0)

    codec = cv2.VideoWriter_fourcc(*'MJPG')
    (h, w) = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter(outputFile, codec, 30, (h, w))

    # Process each frame
    try:
        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print("Done processing !!!\nOutput file is stored as {}.\nNot HasFrame".format(outputFile))
                cap.release()
                break

            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))

            # Remove the bounding boxes with low confidence
            postprocess(frame, outs, classes)

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (
                    t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            vid_writer.write(frame.astype(np.uint8))

    except KeyboardInterrupt:
        print('Stop processing')
        pass
    print('Done processing')


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str,
                        # default='../yolo/yolov4-custom.cfg',
                        default=PATH_YOLO_CFG,
                        help='Path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default=PATH_YOLO_WEIGHTS,
                        # default='../yolo/yolov4-custom_last.weights',
                        help='Path of trained model.', dest='weightfile')
    parser.add_argument('-videofile', type=str,
                        default='../data/videos/001/GOPR5818.MP4',
                        # default='../data/videos/002/20180206_113059.mp4',
                        # default='../yolo/001.png',
                        help='Path of your video file.', dest='videofile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.videofile:
        detect_yolo_v3(args.cfgfile, args.weightfile, args.videofile)
        # detect_yolo_v4(args.cfgfile, args.weightfile, args.videofile)
