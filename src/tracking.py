# import the necessary packages
import imutils
import cv2
# Custom importing
from painting_detection import elaborate_edge_detection


def tracking(videofile=None):
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

    # if a video path was not supplied, grab the reference to the web cam
    # vs = read_video(videofile, reduce_size=False)
    vs = cv2.VideoCapture(videofile)
    # initialize the FPS throughput estimator
    fps = None

    track = False

    if track == False:
        frame = vs.read()
        frame = frame[1]
        frame = imutils.resize(frame, width=600)
        list_boundings = elaborate_edge_detection(frame, show_images=False)

        for bounding, num in zip(list_boundings, range(len(list_boundings))):
            upper_left, upper_right, down_left, down_right = bounding
            x_min = min(upper_left[0], upper_right[0], down_left[0], down_right[0])
            y_min = min(upper_left[1], upper_right[1], down_left[1], down_right[1])
            x_max = max(upper_left[0], upper_right[0], down_left[0], down_right[0])
            y_max = max(upper_left[1], upper_right[1], down_left[1], down_right[1])
            initBB = (x_min, y_min, x_max - x_min, y_max - y_min)
            tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
            trackers.add(tracker, frame, initBB)
        track = True

    # loop over frames from the video stream
    while track:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1]

        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, boxes) = trackers.update(frame)
            # check to see if the tracking was a success
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # if we are using a webcam, release the pointer
    if videofile is None:
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
    # close all windows
    cv2.destroyAllWindows()


tracking('../data/videos/002/20180206_113059.mp4')
