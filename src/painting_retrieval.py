import cv2
import os
import numpy as np
import json, codecs
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

# Custom importing
from parameters import *


def execute_descriptors(folder_database):
    descriptors = None
    if os.path.isfile(PATH_KEYPOINTS_DB):
        descriptors_text = codecs.open(PATH_KEYPOINTS_DB, 'r', encoding='utf-8').read()
        descriptors = json.loads(descriptors_text)
        descriptors = np.array(descriptors)
        print(f"Found file! Set of {descriptors.shape[0]} descriptors with {descriptors.shape[1]} features.")
    else:
        print("File with descriptors not found. It needs to compute these")
        detector = cv2.xfeatures2d.SIFT_create()
        images_name = sorted([file for file in os.listdir(folder_database)])
        print(f"Found {len(images_name)} images.")
        for name_image in images_name:
            image = cv2.imread(f"{folder_database}/{name_image}", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            _, des = detector.detectAndCompute(image, None)
            if descriptors is None:
                descriptors = des
            else:
                descriptors = np.concatenate([descriptors, des])
        with codecs.open(PATH_KEYPOINTS_DB, 'w', encoding='utf-8') as file:
            json.dump(descriptors.tolist(), file, separators=(',', ':'))
        print(f"Extract a set of {descriptors.shape[0]} descriptors with {descriptors.shape[1]} features.")
    return descriptors


def extract_dictionary(descriptors, num_cluster):
    if os.path.isfile(PATH_MEANS) and os.path.isfile(PATH_WEIGHTS) and os.path.isfile(PATH_COVARIANCES):
        print("Found trained Gaussian Mixture Model.")
        weights_vector = np.array(json.loads(codecs.open(PATH_WEIGHTS, 'r', encoding='utf-8').read()))
        means_matrix = np.array(json.loads(codecs.open(PATH_MEANS, 'r', encoding='utf-8').read()))
        covariances_matrix = np.array(json.loads(codecs.open(PATH_COVARIANCES, 'r', encoding='utf-8').read()))
        return weights_vector, means_matrix, covariances_matrix

    print("Fitted GMM not found! It needs to fit one.")
    max_iter, covariance_type = 3000, 'diag'
    gmm = GaussianMixture(n_components=num_cluster, covariance_type=covariance_type, max_iter=max_iter)
    gmm.fit(descriptors)

    covariances_vector = gmm.covariances_
    covariances_matrix = np.empty(shape=(gmm.n_components, descriptors.shape[1], descriptors.shape[1]))
    for i in range(gmm.n_components):
        covariances_matrix[i, :, :] = np.diag(covariances_vector[i, :])
    means_matrix = gmm.means_
    weights_vector = gmm.weights_

    with codecs.open(PATH_MEANS, 'w', encoding='utf-8') as file:
        json.dump(means_matrix.tolist(), file, separators=(',', ':'))
    with codecs.open(PATH_WEIGHTS, 'w', encoding='utf-8') as file:
        json.dump(weights_vector.tolist(), file, separators=(',', ':'))
    with codecs.open(PATH_COVARIANCES, 'w', encoding='utf-8') as file:
        json.dump(covariances_matrix.tolist(), file, separators=(',', ':'))
    return weights_vector, means_matrix, covariances_matrix


def computer_fisher_vector(image, pca, detector, weights, means, covariances):
    h, w = image.shape
    if h > 350 or w > 350:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, des = detector.detectAndCompute(image, None)
    des = pca.transform(des)

    num_clusters = len(weights)
    num_descriptors, num_features = des.shape
    g = []
    for i in range(num_clusters):
        g.append(multivariate_normal(mean=means[i], cov=covariances[i]))
    prob = np.empty((num_descriptors, num_clusters))
    for t in range(num_descriptors):
        for i, g_x in enumerate(g):
            prob[t, i] = g_x.pdf(des[t, :])

    # Calculate Soft Assignment
    soft_assignment = np.empty((num_descriptors, num_clusters))
    for t in range(num_descriptors):
        soft_assignment[t, :] = np.multiply(weights, prob[t, :]) / np.sum(weights * prob[t, :])

    # Dot-product
    covariances_matrix = np.empty((covariances.shape[0], covariances.shape[1]))
    for k in range(covariances_matrix.shape[0]):
        covariances_matrix[k, :] = np.diagonal(covariances[k])

    # Compute Proportion and Average of descriptors
    descriptor_proportion = np.sum(soft_assignment, axis=0) / num_descriptors
    descriptor_average = np.empty((num_clusters, num_features))

    for i in range(num_clusters):
        value = np.zeros(num_features)
        for t in range(num_descriptors):
            value += soft_assignment[t, i] * des[t, :]
        if np.sum(soft_assignment[:, i]) == 0:
            return None, None
        descriptor_average[i, :] = value / np.sum(soft_assignment[:, i])

    delta = np.empty((num_clusters, num_features))
    for i in range(num_clusters):
        delta[i] = (descriptor_average[i, :] - means[i, :]) / np.sqrt(covariances_matrix[i, :])

    # Normalize Fisher Vector
    delta = delta.ravel()
    delta = np.sqrt(abs(delta)) * np.sign(delta)
    delta = delta / np.sqrt(np.dot(delta, delta))
    delta = delta.reshape((num_clusters, num_features))
    return descriptor_proportion, delta


def match_paintings(query_painting, folder_database=SOURCE_PAINTINGS_DB):
    """
        Given a rectified painting return a ranked list of all the images in the painting DB,
        sorted by descending similarity with the detected painting.

    :param query_painting: numpy.ndarray with shape (H, W)
    :param folder_database: the folder which contains the database painting

    :return: a list of (string, similarity)
    """
    if not os.path.exists(PATH_FISHER_VECTOR_DB):
        print(f"Create folder: {PATH_FISHER_VECTOR_DB}")
        os.makedirs(PATH_FISHER_VECTOR_DB)

    num_clusters = 100
    num_components = 64
    pca = PCA(n_components=num_components)
    print('Start Retrieval ...')
    descriptors = execute_descriptors(folder_database)
    pca.fit(descriptors)
    descriptors = pca.transform(descriptors)
    weights, means, covariances = extract_dictionary(descriptors, num_cluster=num_clusters)

    # Compute Fisher Vector
    image_names = sorted([file for file in os.listdir(folder_database)])
    detector = cv2.xfeatures2d.SIFT_create()
    fisher_vectors = dict()
    for i in range(len(image_names)):
        name = os.path.splitext(image_names[i])[0]
        path_fisher_vector = f"{PATH_FISHER_VECTOR_DB}/{name}.json"
        if os.path.isfile(path_fisher_vector):  # Check if exist Fisher Vector file
            fv = np.array(json.loads(codecs.open(path_fisher_vector, 'r', encoding='utf-8').read()))
            fisher_vectors[image_names[i]] = fv[:, 0], fv[:, 1:]
        else:
            image = cv2.imread(f"{folder_database}/{image_names[i]}", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            prop, delta = computer_fisher_vector(image, pca, detector, weights, means, covariances)
            if prop is not None or delta is not None:
                fisher_vectors[image_names[i]] = prop, delta
                fv = np.concatenate((prop.reshape((prop.shape[0], 1)), delta), axis=1)
                with codecs.open(path_fisher_vector, 'w', encoding='utf-8') as file:
                    json.dump(fv.tolist(), file, separators=(',', ':'))

    # Compute Similarity
    if len(query_painting.shape) > 2:
        query_painting = cv2.cvtColor(query_painting, cv2.COLOR_BGR2GRAY)
    q_prop, q_delta = computer_fisher_vector(query_painting, pca, detector, weights, means, covariances)
    list_similarity = {}
    for name_image, (db_prop, db_delta) in fisher_vectors.items():
        similarity = 0
        for i in range(num_clusters):
            # similarity += (db_prop[i] * q_prop[i]) / weights[i] * np.dot(db_delta[i, :], q_delta[i, :])
            # similarity += (db_prop[i] * q_prop[i]) * (num_components - 2 * hamming(db_delta[i, :], q_delta[i, :]))
            similarity += 1 - cosine(db_delta[i, :], q_delta[i, :])
        if similarity > 0:
            norm = (np.sqrt(num_components * np.sum(db_prop)) * np.sqrt(num_components * np.sum(q_prop)))
            list_similarity[name_image] = similarity / norm
            # list_similarity[name_image] = similarity
    list_similarity = sorted(list_similarity.items(), key=lambda x: x[1], reverse=True)
    return list_similarity
