import cv2
from detection import edge_detection

def capVideo(video_path, name_video):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    # it gives me 0.0 :/

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc('F','M','P','4'), fps, (width, height))
    if(not out.isOpened()):
        print("Error Output Video")
        return
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_out = edge_detection(frame)
            out.write(img_out)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()