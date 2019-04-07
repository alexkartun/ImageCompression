import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import scipy.misc as smp
from scipy.misc import imread
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, k, max_iter=10):
        self.data = data
        self.k = k
        self.max_iter = max_iter

    def init_centroids(self):
        if self.k == 2:
            return np.asarray([[0., 0., 0.],
                               [0.07843137, 0.06666667, 0.09411765]])
        elif self.k == 4:
            return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                               [0.49019608, 0.41960784, 0.33333333],
                               [0.02745098, 0., 0.],
                               [0.17254902, 0.16862745, 0.18823529]])
        elif self.k == 8:
            return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                               [0.14509804, 0.12156863, 0.12941176],
                               [0.4745098, 0.40784314, 0.32941176],
                               [0.00784314, 0.00392157, 0.02745098],
                               [0.50588235, 0.43529412, 0.34117647],
                               [0.09411765, 0.09019608, 0.11372549],
                               [0.54509804, 0.45882353, 0.36470588],
                               [0.44705882, 0.37647059, 0.29019608]])
        elif self.k == 16:
            return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                               [0.4745098, 0.38039216, 0.33333333],
                               [0.65882353, 0.57647059, 0.49411765],
                               [0.08235294, 0.07843137, 0.10196078],
                               [0.06666667, 0.03529412, 0.02352941],
                               [0.08235294, 0.07843137, 0.09803922],
                               [0.0745098, 0.07058824, 0.09411765],
                               [0.01960784, 0.01960784, 0.02745098],
                               [0.00784314, 0.00784314, 0.01568627],
                               [0.8627451, 0.78039216, 0.69803922],
                               [0.60784314, 0.52156863, 0.42745098],
                               [0.01960784, 0.01176471, 0.02352941],
                               [0.78431373, 0.69803922, 0.60392157],
                               [0.30196078, 0.21568627, 0.1254902],
                               [0.30588235, 0.2627451, 0.24705882],
                               [0.65490196, 0.61176471, 0.50196078]])
        else:
            print('This value of K is not supported.')
            return None

    def calculate_distances(self, centroids):
        distances = np.zeros((self.data.shape[0], self.k))
        for i in range(self.k):
            # calculate euclidean distance
            distances_to_i_centroid = norm(self.data - centroids[i, :], axis=1)
            distances[:, i] = distances_to_i_centroid
        return distances

    @staticmethod
    def get_closest_clusters(distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self, clusters):
        centroids = np.zeros((self.k, self.data.shape[1]))
        for i in range(self.k):
            centroids[i, :] = np.mean(self.data[clusters == i, :], axis=0)
        return centroids

    @staticmethod
    def print_centroids(iteration, centroids):
        new_centroids = np.floor(centroids * 100) / 100
        centroids_repr = ', '.join(str(c) for c in new_centroids.tolist())
        print('iter {}: {}'.format(iteration, centroids_repr))

    def fit(self):
        iter2sse = defaultdict(float)
        centroids = self.init_centroids()
        self.print_centroids(0, centroids)
        for i in range(0, self.max_iter):
            distances = self.calculate_distances(centroids)
            clusters = self.get_closest_clusters(distances)
            centroids = self.update_centroids(clusters)
            self.print_centroids(i + 1, centroids)
            iter2sse[i] = self.calculate_sse(clusters, centroids)
        return centroids, iter2sse

    def calculate_sse(self, clusters, centroids):
        distance = np.zeros(self.data.shape[0])
        for i in range(self.k):
            distance[clusters == i] = norm(self.data[clusters == i] - centroids[i], axis=1)
        return float(np.sum(np.square(distance)))

    def predict(self, centroids):
        distance = self.calculate_distances(centroids)
        return self.get_closest_clusters(distance)

    def compress(self, clusters, centroids):
        compressed_data = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(self.k):
            compressed_data[clusters == i, :] = centroids[i]
        return compressed_data


def preprocess_data(path):
    data = imread(path)
    data_size = data.shape
    normalized_data = data.astype(float) / 255.
    return normalized_data.reshape((data_size[0] * data_size[1], data_size[2])), data_size


def postprocess_data(path, compressed_data, data_size):
    compressed_data = compressed_data.reshape((data_size[0], data_size[1], data_size[2]))
    compressed_img = smp.toimage(compressed_data)
    compressed_img.save(path)


def plot_sse(path, iter2sse):
    x, y = zip(*sorted(iter2sse.items()))
    plt.plot(x, y)
    plt.ylabel('average loss')
    plt.xlabel('iteration')
    plt.savefig(path)
    plt.clf()


def main():
    data, data_size = preprocess_data('dog.jpeg')
    for k in [2, 4, 8, 16]:
        print('k={}:'.format(k))
        km = KMeans(data, k)
        centroids, iter2sse = km.fit()      # training
        clusters = km.predict(centroids)    # testing
        plot_sse('{}_sse.jpeg'.format(k), iter2sse)
        compressed_data = km.compress(clusters, centroids)
        postprocess_data('{}_dog.jpeg'.format(k), compressed_data, data_size)


if __name__ == '__main__':
    main()
