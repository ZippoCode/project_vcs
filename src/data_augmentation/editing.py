import random, numpy as np, cv2, math

from constants.parameters import *
from data_augmentation.save_image import write_output, rotating_and_saving


def get_bbox(image, text):
    H, W, C = image.shape
    labels = [label for label in text.split('\n') if len(label) > 0]
    bbox = []
    for label in labels:
        cla, x, y, w, h = [float(num) for num in label.strip('\n').split(' ')]
        x, y = int(round((x - w / 2) * W)), int(round((y - h / 2) * H))
        w, h = int(round(w * W)), int(round(h * H))
        if w > 0 and h > 0:
            bbox.append((x, y, w, h))
    return bbox


def save_edit_value_pixel(image, bboxs, name_image):
    # NORMALIZED
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    write_output(norm_image, bboxs, name_image, '_norm')


def save_filter(image, labels, name_image):
    # CONTRAST AND BRIGHTNESS ADJUSTMENT
    alpha, beta = random.uniform(1, 3), random.uniform(0, 100)
    bandc_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    write_output(bandc_image, labels, name_image, '_filter_ContrAndBrigh')

    # HISTOGRAM EQUALIZATION
    hist_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    hist_image[:, :, 0] = cv2.equalizeHist(hist_image[:, :, 0])
    write_output(cv2.cvtColor(hist_image, cv2.COLOR_YUV2BGR), labels, name_image, '_filter_histEqui')

    # MEDIAN BLUR
    ksize = 2 * (random.randint(5, 21) // 2) + 1
    mfr_image = cv2.medianBlur(image, ksize)
    write_output(mfr_image, labels, name_image, '_filter_medianBlur')

    # FILTER
    # kernel_size = random.randint(5, 15)
    # kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    # kernel /= (kernel_size * kernel_size)
    # fr_image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    # write_output(fr_image, labels, name_image, '_fr')


def save_noised_image(image, bboxs, name_image):
    """
        Taken a image and saving it with other images which adding noise

    :param image:
    :param bboxs:
    :param name_image:
    :return:
    """
    # GAUSSIAN NOISE
    mean, var = 0, random.uniform(0, 0.5)
    sigma = var ** 0.5
    H, W, C = image.shape
    gauss = np.random.normal(mean, sigma, (H, W, C))
    gauss = gauss.reshape(image.shape).astype(np.uint8)
    gauss_image = cv2.add(image, gauss)
    write_output(gauss_image, bboxs, name_image, '_noise_gauss')

    # SPECKLE NOISE
    speckle_image = image + image * gauss
    write_output(speckle_image, bboxs, name_image, '_noise_speckle')

    # POISSON NOISE
    poiss_image = np.copy(image)
    vals = len(np.unique(poiss_image))
    vals = 2 ** np.ceil(np.log2(vals))
    poiss_image = poiss_image + np.random.poisson(poiss_image * vals) / float(vals)
    write_output(poiss_image, bboxs, name_image, '_noise_poisson')

    # SALT AND PEPPER NOISE
    sp, amount = random.random(), random.uniform(0, 0.1)
    sp_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * sp)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    sp_image[coords] = 255
    num_pepper = np.ceil(amount * image.size * sp)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    sp_image[coords] = 0
    write_output(sp_image, bboxs, name_image, '_noise_s&p')


def save_rotated_image(image, labels, name_image):
    """

    :param image:
    :param name_image:
    :param path_labels:
    :return:
    """
    # ROTATE
    angle = random.randint(11, 25)
    path_rot_one = PATH_EDIT + name_image + '_rotate1'
    rotating_and_saving(image, angle, labels, path_rot_one)

    # ROTATE NEGATIVE
    angle = random.randint(-20, -5)
    path_rot_two = PATH_EDIT + name_image + '_rotate2'
    rotating_and_saving(image, angle, labels, path_rot_two)

    # ROTATE THREE
    angle = random.randint(1, 10)
    path_rot_three = PATH_EDIT + name_image + '_rotate3'
    rotating_and_saving(image, angle, labels, path_rot_three)

    # ROTATE THREE
    angle = random.randint(-10, -4)
    path_rot_three = PATH_EDIT + name_image + '_rotate4'
    rotating_and_saving(image, angle, labels, path_rot_three)

    # RESIZE AND ROTATE
    H, W, C = image.shape
    scale_percent = random.randint(10, 20) if H > 500 else random.randint(190, 200)
    new_width = int(W * scale_percent / 100)
    new_height = int(H * scale_percent / 100)
    res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    angle = random.randint(-10, 0)
    path_rae = PATH_EDIT + name_image + '_resAndrot_one'
    rotating_and_saving(res_image, angle, labels, path_rae)

    # RESIZE AND ROTATE
    H, W, C = image.shape
    scale_percent = random.randint(40, 60) if H > 500 else random.randint(140, 160)
    new_width = int(W * scale_percent / 100)
    new_height = int(H * scale_percent / 100)
    res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    angle = random.randint(0, 11)
    path_rae = PATH_EDIT + name_image + '_resAndrot_two'
    rotating_and_saving(res_image, angle, labels, path_rae)


def save_resized_image(image, bbox, name_image):
    # RESIZE
    width, height, channels = image.shape
    scale_percent = random.randint(140, 151) if width < 500 else random.randint(40, 51)
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resize_image = cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_AREA)
    write_output(resize_image, bbox, name_image, '_resize')

    resize_small_image = cv2.resize(image, (225, 225), interpolation=cv2.INTER_AREA)
    write_output(resize_small_image, bbox, name_image, '_resizeSmall')


def object_away_random_erasing(original_image, original_labels, name_image, name_result):
    re_image = np.copy(original_image)
    sl, sh, r1 = 0.02, 0.4, 0.3
    for xo, yo, wo, ho in get_bbox(re_image, original_labels):
        area = wo * ho
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < wo and h < ho:
            x = np.random.randint(0, wo - w) + xo
            y = np.random.randint(0, ho - h) + yo
            re_image[y: y + h, x: x + w, 0] = np.random.uniform(0, 255, (h, w))
            re_image[y: y + h, x: x + w, 1] = np.random.uniform(0, 255, (h, w))
            re_image[y: y + h, x: x + w, 2] = np.random.uniform(0, 255, (h, w))

    write_output(re_image, original_labels, name_image, name_result)


def edge_object_histogram(image, readed, name_image):
    eoh_image = np.copy(image)
    K = 4
    threshould = random.randint(80, 110)
    ddepth, scale, delta = cv2.CV_16S, 1, 0
    bboxs = get_bbox(image, readed)
    for x, y, w, h in bboxs:
        object = cv2.cvtColor(image[y: y + h, x: x + w, :], cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(object, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(object, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = np.where(grad >= threshould, grad, 0)
        theta = (np.arctan2(grad_y, grad_x) + np.pi) / (2 * np.pi) * 180

        psi, B = np.zeros((grad.shape[0], grad.shape[1], K)), np.zeros(K, dtype=float)
        tot = np.sum(grad)
        max, value_max = -1, -1
        for k in range(K):
            condition = np.logical_and(theta >= k * (180 / K), theta < (k + 1) * (180 / K))
            psi[:, :, k] = np.where(condition, grad, 0)
            epsilon = random.random()
            B[k] = (np.sum(psi[:, :, k]) + epsilon) / (tot + epsilon)
            if B[k] > value_max:
                value_max = B[k]
                max = k

            # plt.imshow(psi[:, :, k], cmap='gray'), plt.show()
        for c in range(eoh_image.shape[2]):
            eoh_image[y: y + h, x: x + w, c] = psi[:, :, max]

    write_output(eoh_image, readed, name_image, '_eoh')
