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
from read_write import find_image


def save_file(filename: str, array: np.ndarray):
    with codecs.open(filename, 'w', encoding='utf-8') as file:
        json.dump(array.tolist(), file, separators=(',', ':'))


class PaintingRetrieval:

    def __init__(self, num_clusters=128, num_components=64, folder_database=SOURCE_PAINTINGS_DB):
        assert (num_components > 0 and num_components > 0)
        self.num_clusters = num_clusters
        self.num_components = num_components
        self.folder_database = folder_database
        self.path_images = find_image(folder_database)
        print(f"Database contains {len(self.path_images)} images")

        self.pca = PCA(n_components=num_components)
        self.detector = cv2.xfeatures2d.SIFT_create()

        # Components
        self.descriptors = None
        self.database_fisher_vector = dict()
        self.weights_vector = None
        self.means_matrix = None
        self.covariances_matrix = None

        # Configuration
        self.execute_descriptors()
        self.extract_dictionary()
        self.compute_database_fisher_vector()

    def execute_descriptors(self):
        if os.path.isfile(PATH_KEYPOINTS_DB):
            descriptors_text = codecs.open(PATH_KEYPOINTS_DB, 'r', encoding='utf-8').read()
            self.descriptors = np.array(json.loads(descriptors_text))
            print(f"Found descriptors file! It contains {self.descriptors.shape[0]} descriptors.")
        else:
            print(f"File with descriptors not found. It needs to create one!")
            for name_image in self.path_images:
                image = cv2.imread(name_image, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                _, des = self.detector.detectAndCompute(image, None)
                if self.descriptors is None:
                    self.descriptors = des
                else:
                    self.descriptors = np.concatenate([self.descriptors, des])
            save_file(PATH_KEYPOINTS_DB, self.descriptors)
            print("File with descriptors saved!")
            print(f"Extract a set of {self.descriptors.shape[0]} descriptors.")
        self.pca.fit(self.descriptors)
        self.descriptors = self.pca.transform(self.descriptors)
        print(f"Applied Principal Component Analysis. Now the dimension of descriptor is: {self.pca.n_components_}")

    def extract_dictionary(self):
        if os.path.isfile(PATH_MEANS) and os.path.isfile(PATH_WEIGHTS) and os.path.isfile(PATH_COVARIANCES):
            self.weights_vector = np.array(json.loads(codecs.open(PATH_WEIGHTS, 'r', encoding='utf-8').read()))
            self.means_matrix = np.array(json.loads(codecs.open(PATH_MEANS, 'r', encoding='utf-8').read()))
            self.covariances_matrix = np.array(json.loads(codecs.open(PATH_COVARIANCES, 'r', encoding='utf-8').read()))
            print("Found trained Gaussian Mixture Model.")

        else:
            print("File with configurations not found. Wait because it needs to fit the GMM!")
            max_iter, covariance_type = 3000, 'diag'
            gmm = GaussianMixture(n_components=self.num_clusters, covariance_type=covariance_type, max_iter=max_iter)
            gmm.fit(self.descriptors)
            self.covariances_matrix = np.zeros(shape=(gmm.n_components, self.num_components, self.num_components))
            for i in range(gmm.n_components):
                self.covariances_matrix[i, :, :] = np.diag(gmm.covariances_[i, :])
            self.means_matrix = gmm.means_
            self.weights_vector = gmm.weights_
            print("Gaussian Mixture Model fitted!")
            # Store File
            save_file(PATH_MEANS, self.means_matrix)
            save_file(PATH_WEIGHTS, self.weights_vector)
            save_file(PATH_COVARIANCES, self.covariances_matrix)
            print("Configuration file of GMM stored.")

    def compute_fisher_vector(self, image):
        h, w = image.shape
        if h > 350 and w > 350:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, des = self.detector.detectAndCompute(image, None)
        des = self.pca.transform(des)
        num_descriptors = des.shape[0]

        # Compute MLE
        g = []
        for i in range(self.num_clusters):
            g.append(multivariate_normal(mean=self.means_matrix[i], cov=self.covariances_matrix[i]))
        prob = np.zeros((num_descriptors, self.num_clusters))
        for t in range(num_descriptors):
            for i, g_x in enumerate(g):
                prob[t, i] = g_x.pdf(des[t, :])

        # Calculate Soft Assignment
        soft_assignment = np.zeros((num_descriptors, self.num_clusters))
        for t in range(num_descriptors):
            soft_assignment[t, :] = np.multiply(self.weights_vector, prob[t, :]) / np.sum(
                self.weights_vector * prob[t, :])

        # Dot-product
        covariances_matrix = np.empty((self.covariances_matrix.shape[0], self.covariances_matrix.shape[1]))
        for k in range(covariances_matrix.shape[0]):
            covariances_matrix[k, :] = np.diagonal(self.covariances_matrix[k])

        # Compute Proportion and Average of descriptors
        descriptor_proportion = np.sum(soft_assignment, axis=0) / num_descriptors
        descriptor_average = np.empty((self.num_clusters, self.num_components))

        for i in range(self.num_clusters):
            value = np.zeros(self.num_components)
            for t in range(num_descriptors):
                value += soft_assignment[t, i] * des[t, :]
            if np.sum(soft_assignment[:, i]) == 0:
                return None, None
            descriptor_average[i, :] = value / np.sum(soft_assignment[:, i])

        delta = np.empty((self.num_clusters, self.num_components))
        for i in range(self.num_clusters):
            delta[i] = (descriptor_average[i, :] - self.means_matrix[i, :]) / np.sqrt(covariances_matrix[i, :])

        fisher_vector = None
        for i in range(self.num_clusters):
            fv = descriptor_proportion[i] / np.sqrt(self.weights_vector[i]) * delta[i, :] * num_descriptors
            if fisher_vector is None:
                fisher_vector = fv
            else:
                fisher_vector = np.concatenate([fisher_vector, fv])

        # Normalize Fisher Vector
        fisher_vector = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        fisher_vector = fisher_vector / np.sqrt(np.dot(fisher_vector, fisher_vector))
        fisher_vector = fisher_vector.reshape((self.num_clusters, self.num_components))
        return fisher_vector

    def compute_database_fisher_vector(self):
        if not os.path.exists(PATH_FISHER_VECTOR_DB):
            print(f"Create folder: {PATH_FISHER_VECTOR_DB}")
            os.makedirs(PATH_FISHER_VECTOR_DB)
        for path_image in self.path_images:
            filename = os.path.splitext(os.path.basename(path_image))[0]
            path_filename_fv = f"{PATH_FISHER_VECTOR_DB}/{filename}.json"
            if os.path.isfile(path_filename_fv):
                fisher_vector = np.array(json.loads(codecs.open(path_filename_fv, 'r', encoding='utf-8').read()))
                self.database_fisher_vector[filename] = fisher_vector
                continue
            image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
            fisher_vector = self.compute_fisher_vector(image)
            self.database_fisher_vector[filename] = fisher_vector
            save_file(path_filename_fv, fisher_vector)
        print(f"Stored Fisher Vector JSON files")

    def match_painting(self, path_image: str):
        image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        query_fisher_vector = self.compute_fisher_vector(image)
        list_similarities = {}
        for name_image, dataset_fisher_vector in self.database_fisher_vector.items():
            similarity = 1 - cosine(np.ravel(dataset_fisher_vector), np.ravel(query_fisher_vector))
            if similarity > 0:
                list_similarities[name_image] = similarity
        list_similarities = sorted(list_similarities.items(), key=lambda x: x[1], reverse=True)
        return list_similarities
