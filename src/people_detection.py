import cv2
import argparse
import sys
import numpy as np
import os.path
import matplotlib.pyplot as plt

from tool.utils import *
from tool.darknet2pytorch import Darknet


# Step 1: Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

classesFile = "../yolo/coco.names"


# Step 3: Define function
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
        1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
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

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classes, classIds[i], confidences[i],
                 left, top, left + width, top + height)


def detect_yolo_v3(cfgfile, weightfile, videofile, use_cuda=False):
    # Load names of classes
    classes = None
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

    # Process inputs
    if (videofile):
        # Open the video file
        if not os.path.isfile(videofile):
            print("Input video file ", videofile, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(videofile)
        outputFile = '../yolo/' + \
            videofile.split('/')[-1][:-4]+'_yolo_output.avi'
    else:
        cap = cv2.VideoCapture(0)
    # Get the video writer initialized to save the output video
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Process each frame
    try:
        while True:

            # get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                print('Not HasFrame')
                cv2.waitKey(3000)
                # Release device
                cap.release()
                break

            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(
                frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

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


def detect_yolo_v4(cfgfile, weightfile, videofile, use_cuda=False):
    num_classes = 1
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    # Process inputs
    if (videofile):
        # Open the video file
        if not os.path.isfile(videofile):
            print("Input video file ", videofile, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(videofile)
        outputFile = '../yolo/' + \
            videofile.split('/')[-1][:-4]+'_yolo_output.avi'
    else:
        cap = cv2.VideoCapture(0)
    # Get the video writer initialized to save the output video
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Process each frame
    try:
        while True:
            # get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                print('Not HasFrame')
                cv2.waitKey(3000)
                # Release device
                cap.release()
                break

            # print(type(frame))
            img = Image.fromarray(frame).convert('RGB')
            sized = img.resize((m.width, m.height))
            # sized = cv2.resize(frame, (m.width, m.height))

            for i in range(2):
                boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)

            class_names = load_class_names(classesFile)
            result_frame = plot_boxes(img, boxes, None, class_names)
            result_frame = np.array(result_frame)
            # Write the frame with the detection boxes
            vid_writer.write(result_frame.astype(np.uint8))

    except KeyboardInterrupt:
        print('Stop processing')
        pass
    print('Done processing')


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str,
                        # default='../yolo/yolov4-custom.cfg',
                        default='../yolo/cfg/yolov3.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='../yolo/backup/yolov3_final.weights',
                        # default='../yolo/yolov4-custom_last.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-videofile', type=str,
                        default='../data/videos/004/IMG_3800.MOV',
                        # default='../data/videos/002/20180206_113059.mp4',
                        # default='../yolo/001.png',
                        help='path of your image file.', dest='videofile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.videofile:
        detect_yolo_v3(args.cfgfile, args.weightfile, args.videofile)
        # detect_yolo_v4(args.cfgfile, args.weightfile, args.videofile)
