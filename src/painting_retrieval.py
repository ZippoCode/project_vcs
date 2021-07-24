import cv2
import json
import codecs
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, hamming

# Custom importing
from parameters import *
from read_write import find_image


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_file(filename: str, array: object):
    try:
        with codecs.open(filename, 'w', encoding='utf-8') as file:
            json.dump(array, file, separators=(',', ':'), cls=NumpyEncoder)
    except KeyboardInterrupt:
        os.remove(filename)


class PaintingRetrieval:

    def __init__(self, num_clusters=128, num_components=64, folder_database=SOURCE_PAINTINGS_DB):
        assert (num_clusters > 0 and num_components > 0)
        self.num_clusters = num_clusters
        self.num_components = num_components
        self.folder_database = folder_database
        self.path_images = find_image(folder_database)
        self.path_images.sort()
        print(f"Database contains {len(self.path_images)} images")

        self.pca = PCA(n_components=num_components)
        self.detector = cv2.xfeatures2d.SIFT_create()

        # Components
        self.descriptors = dict()
        self.all_descriptors = None
        self.database_fisher_vector = dict()

        # Gaussian attributes
        self.weights_vector = None
        self.means_matrix = None
        self.covariances_matrix = None
        self.variance = None
        self.gmm = GaussianMixture(n_components=self.num_clusters, covariance_type=COVARIANCE_TYPE, max_iter=MAX_ITER,
                                   verbose=2)

        # Configuration the class
        self.execute_descriptors()
        self.extract_dictionary()
        self.compute_database_fisher_vector()

    def execute_descriptors(self):
        if os.path.isfile(PATH_KEYPOINTS_DB):
            descriptors_text = codecs.open(PATH_KEYPOINTS_DB, 'r', encoding='utf-8').read()
            self.descriptors = dict(json.loads(descriptors_text))
            print(f"Found descriptors file!")
        else:
            print(f"File with descriptors not found. It needs to create one!")
            for name_image in tqdm(self.path_images):
                filename = os.path.splitext(os.path.basename(name_image))[0]
                image = cv2.imread(name_image, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                _, des = self.detector.detectAndCompute(image, None)
                self.descriptors[filename] = des
            save_file(PATH_KEYPOINTS_DB, self.descriptors)
            print("File with descriptors saved!")
        self.all_descriptors = np.concatenate(list(self.descriptors.values()))
        print(f"Extract {self.all_descriptors.shape[0]} descriptors with D: {self.all_descriptors.shape[1]}.")
        self.pca.fit(self.all_descriptors)
        print(f"Applied Principal Component Analysis. Now the dimension of descriptor is: {self.pca.n_components_}")

    def extract_dictionary(self):
        print(f"[INFO] Configuration GMM - Number of Components: {self.gmm.n_components}")

        if os.path.isfile(PATH_MEANS) and os.path.isfile(PATH_WEIGHTS) and os.path.isfile(PATH_COVARIANCES):
            self.weights_vector = np.array(json.loads(codecs.open(PATH_WEIGHTS, 'r', encoding='utf-8').read()))
            self.means_matrix = np.array(json.loads(codecs.open(PATH_MEANS, 'r', encoding='utf-8').read()))
            self.covariances_matrix = np.array(json.loads(codecs.open(PATH_COVARIANCES, 'r', encoding='utf-8').read()))
            self.variance = np.diagonal(self.covariances_matrix, axis1=1, axis2=2)

            # Configuration GMM
            self.gmm.weights_ = self.weights_vector
            self.gmm.means_ = self.means_matrix
            self.gmm.covariances_ = self.covariances_matrix
            self.gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covariances_matrix))
            print("Found trained Gaussian Mixture Model.")
        else:
            print("File with configurations not found. Wait because it needs to fit the GMM!")
            self.gmm.fit(self.pca.transform(self.all_descriptors))
            self.means_matrix = self.gmm.means_
            self.weights_vector = self.gmm.weights_
            self.covariances_matrix = self.gmm.covariances_
            self.variance = np.diagonal(self.covariances_matrix, axis1=1, axis2=2)
            print("Gaussian Mixture Model fitted!")

            # Store File
            save_file(PATH_MEANS, self.means_matrix)
            save_file(PATH_WEIGHTS, self.weights_vector)
            save_file(PATH_COVARIANCES, self.covariances_matrix)
            print("Configuration file of GMM stored.")

    def compute_database_fisher_vector(self):
        if not os.path.exists(PATH_FISHER_VECTOR_DB):
            print(f"Create folder: {PATH_FISHER_VECTOR_DB}")
            os.makedirs(PATH_FISHER_VECTOR_DB)
        print("Start the process in order to represent the image as Fisher Vector")
        for path_image in tqdm(self.path_images):
            filename = os.path.splitext(os.path.basename(path_image))[0]
            path_filename_fv = f"{PATH_FISHER_VECTOR_DB}/{filename}.json"
            if os.path.isfile(path_filename_fv):
                fisher_vector = np.array(json.loads(codecs.open(path_filename_fv, 'r', encoding='utf-8').read()))
                self.database_fisher_vector[filename] = fisher_vector
                continue
            des = np.concatenate(list([self.descriptors[filename]]))
            des = self.pca.transform(des)
            fisher_vector = self.compute_fisher_vector(des)
            self.database_fisher_vector[filename] = fisher_vector
            save_file(path_filename_fv, fisher_vector)
        print(f"Stored Fisher Vector files")

    def compute_fisher_vector(self, des: np.ndarray):
        num_descriptors = des.shape[0]
        soft_assignment = self.gmm.predict_proba(des)

        # Compute Proportion and Average of descriptors
        descriptor_proportion = np.sum(soft_assignment, axis=0) / num_descriptors
        descriptor_average = np.empty((self.num_clusters, self.num_components))

        for i in range(self.num_clusters):
            value = np.zeros(self.num_components)
            for t in range(num_descriptors):
                value += soft_assignment[t, i] * des[t, :]
            if np.sum(soft_assignment[:, i]) == 0:
                descriptor_average[i, :] = np.zeros(self.num_components)
            else:
                descriptor_average[i, :] = value / np.sum(soft_assignment[:, i])

        delta = np.empty((self.num_clusters, self.num_components))
        for i in range(self.num_clusters):
            delta[i] = (descriptor_average[i, :] - self.means_matrix[i, :]) / np.sqrt(self.variance[i, :])

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

    def match_painting(self, path_image: str):
        image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        _, des = self.detector.detectAndCompute(image, None)
        des = self.pca.transform(des)
        query_fisher_vector = self.compute_fisher_vector(des)
        list_similarities = {}
        for name_image, dataset_fisher_vector in self.database_fisher_vector.items():
            similarity = 1 - cosine(np.ravel(dataset_fisher_vector), np.ravel(query_fisher_vector))
            if similarity > 0:
                list_similarities[name_image] = similarity
        list_similarities = sorted(list_similarities.items(), key=lambda x: x[1], reverse=True)
        return list_similarities
