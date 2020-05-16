import cv2
import os

from Detection import edge_detection


def capVideo(video_path, name_video):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    # it gives me 0.0 :/
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(name_video, -1, fps, (width, height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_out = edge_detection(frame)

            # out.write(img_out)

            cv2.imshow('frame', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for forder in os.listdir("./data/videos"):
        if forder == '.DS_Store':
            continue
        path = os.path.join("./data/videos", forder)
        for file in os.listdir(path):
            if file.endswith(".MOV") or file.endswith(".mp4") or file.endswith(".MP4"):
                video = os.path.join(path, file)
                capVideo(video, 'data/result/' + forder + '/' + file)
                print('Finish ...')

# capVideo('./data/videos/000/VIRB0392.mp4', 'VIRB0392.mp4')
