import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import scipy.misc as smp
from scipy.misc import imread
import matplotlib.pyplot as plt


class KMeans:
    """
    KMeans.
    """

    def __init__(self, data, k, max_iter=10):
        """
        KMeans constructor
        :param data: data for clustering
        :param k: KMeans hyper parameter, number of clusters
        :param max_iter: threshold number of iterations
        """
        self.data = data
        self.k = k
        self.max_iter = max_iter

    def init_centroids(self):
        """
        get initial centroids according to k value
        :return: initial array of centroids
        """
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
        """
        calculating distances of every pixel to each of a centroids, using euclidean distance as a metric
        :param centroids: centroids to check with
        :return: closest distances
        """
        distances = np.zeros((self.data.shape[0], self.k))
        for i in range(self.k):
            # calculate euclidean distance
            distances_to_i_centroid = norm(self.data - centroids[i, :], axis=1)
            distances[:, i] = np.square(distances_to_i_centroid)
        return distances

    @staticmethod
    def get_closest_clusters(distances):
        """
        calculating closest centroid for each one of the pixels
        :param distances: distances from each of the pixels to the centroids
        :return: closest centroids for each of the points
        """
        return np.argmin(distances, axis=1)

    def update_centroids(self, clusters):
        """
        updating centroids based on KMeans algorithm
        :param clusters: closest centroid's index of each one of the pixels
        :return: new updated centroids
        """
        centroids = np.zeros((self.k, self.data.shape[1]))
        for i in range(self.k):
            centroids[i, :] = np.mean(self.data[clusters == i, :], axis=0)
        return centroids

    @staticmethod
    def print_centroids(iteration, centroids):
        """
        printing the centroids
        :param iteration: number of iteration
        :param centroids: centroids to be printed
        :return: None
        """
        if len(centroids.shape) == 1:
            centroids_repr = ' '.join(str(np.floor(100 * centroids) / 100).split()).replace('[ ', '['). \
                replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')
        else:
            centroids_repr = ' '.join(str(np.floor(100 * centroids) / 100).split()).replace('[ ', '['). \
                                 replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')[1:-1]
        print('iter {}: {}'.format(iteration, centroids_repr))

    def fit(self):
        """
        core of KMeans algorithm
        :return: None
        """
        iter2sse = defaultdict(float)
        centroids = self.init_centroids()
        for i in range(self.max_iter + 1):
            self.print_centroids(i, centroids)
            distances = self.calculate_distances(centroids)
            clusters = self.get_closest_clusters(distances)
            iter2sse[i] = self.calculate_loss(clusters, centroids)
            centroids = self.update_centroids(clusters)
        return centroids, iter2sse

    def calculate_loss(self, clusters, centroids):
        """
        calculating mean loss of model by taking distances of the pixels to their closest centroid
        :param clusters: closest centroid's index of each one of the pixels
        :param centroids: most updated KMeans centroids
        :return: loss value of the model till now
        """
        distance = np.zeros(self.data.shape[0])
        for i in range(self.k):
            distance[clusters == i] = norm(self.data[clusters == i] - centroids[i], axis=1)
        return float(np.sum(np.square(distance)) / self.data.shape[0])

    def predict(self, centroids):
        """
        predicting pixels closest centroids
        :param centroids: most updated KMeans centroids
        :return: closest centroid's index of each one of the pixels
        """
        distance = self.calculate_distances(centroids)
        return self.get_closest_clusters(distance)

    def compress(self, clusters, centroids):
        """
        compressing the pixels by changing their value to be as the value of their closest centroid
        :param clusters: closest centroid's index of each one of the pixels
        :param centroids: most updated KMeans centroids
        :return: return compressed pixels
        """
        compressed_data = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(self.k):
            compressed_data[clusters == i, :] = centroids[i]
        return compressed_data


def preprocess_data(path):
    """
    preprocess data(img) be extracting pixels, normalizing them and reshaping
    :param path: relative path to the image
    :return: extracted and preprocessed pixels
    """
    data = imread(path)
    data_size = data.shape
    normalized_data = data.astype(float) / 255.
    return normalized_data.reshape((data_size[0] * data_size[1], data_size[2])), data_size


def postprocess_data(path, compressed_data, data_size):
    """
    reshaping the pixels, converting back to image object and save
    :param path: path of the new image to be saved
    :param compressed_data: compressed pixels
    :param data_size: image dimensions
    :return: None
    """
    compressed_data = compressed_data.reshape((data_size[0], data_size[1], data_size[2]))
    compressed_img = smp.toimage(compressed_data)
    compressed_img.save(path)


def plot_sse(path, iter2sse):
    """
    saving the loss values per iteration to the plot
    :param path: path of the plot to be saved
    :param iter2sse: map of loss value per each iteration
    :return: None
    """
    x, y = zip(*sorted(iter2sse.items()))
    plt.plot(x, y, color='green', marker='o', linestyle='dashed')
    plt.xticks(x)
    plt.ylabel('average loss')
    plt.xlabel('iteration')
    plt.savefig(path)
    plt.clf()


def main():
    """
    main function that running for each k, KMeans algorithm
    :return: None
    """
    data, data_size = preprocess_data('dog.jpeg')
    for k in [2, 4, 8, 16]:
        print('k={}:'.format(k))
        km = KMeans(data, k)
        centroids, iter2sse = km.fit()  # training
        clusters = km.predict(centroids)  # testing
        plot_sse('{}_sse.jpeg'.format(k), iter2sse)
        compressed_data = km.compress(clusters, centroids)
        postprocess_data('{}_dog.jpeg'.format(k), compressed_data, data_size)


if __name__ == '__main__':
    main()
