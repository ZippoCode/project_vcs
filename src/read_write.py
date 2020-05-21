import cv2

# Custom importing
from detection import edge_detection, get_bounding_boxes
from plotting import plt_images
from improve_quality import multiscale_retinex
from rectification import affine_transformation


def read_single_image(input_filename, output_filename='output.jpg'):
    image = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print("Error reading image")
        return
    edge_detection(image)
    cv2.imwrite(output_filename, image)


def capVideo(video_path, name_video):
    """

    :param video_path:
    :param name_video:
    :return:
    """
    print("Reading file: {}".format((video_path)))
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    # it gives me 0.0 :/

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name_video, fourcc, fps, (width, height))
    if (not out.isOpened()):
        print("Error Output Video")
        return
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            images = []
            titles = []

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(frame)
            titles.append("Original Image")

            # Apply Multiscale Retinex
            frame_retinex = multiscale_retinex(frame)
            images.append(frame_retinex)
            titles.append('Multiscale retinex')

            # Edge Detection
            edit_images, edit_titles = edge_detection(frame_retinex)
            for image in edit_images:
                images.append(image)
            for title in edit_titles:
                titles.append(title)

            list_painting = get_bounding_boxes(images[-1])
            for painting in list_painting:
                # Affine Transformation for painting
                new_image = affine_transformation(frame, painting)
                cv2.imwrite('../output/'+name_video+'.jpg', new_image)

                upper_left, upper_right, down_left, down_right = painting
                cv2.line(frame, upper_left, upper_right, (0, 255, 0), 10)
                cv2.line(frame, upper_left, down_left, (0, 255, 0), 10)
                cv2.line(frame, down_left, down_right, (0, 255, 0), 10)
                cv2.line(frame, upper_right, down_right, (0, 255, 0), 10)

                # THIS IS THE FOUR POINTS
                x, y = upper_left
                w = upper_right[0] - x
                h = down_left[1] - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 10)

                # Try to calculate w, h
                cv2.circle(frame, upper_left, 12, (0, 255, 0), -1)
                cv2.circle(frame, down_left, 12, (255, 0, 0), -1)
                cv2.circle(frame, upper_right, 12, (0, 0, 255), -1)
                cv2.circle(frame, down_right, 12, (255, 255, 255), -1)

                out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Add the drawing image with the rectangle
            images.append(frame)
            titles.append('Final result')

            # Show the images
            plt_images(images, titles)
        else:
            break

        # Show only first image from video
        break

    cap.release()
    print("Saving file: {}".format(name_video))
    out.release()
    cv2.destroyAllWindows()
