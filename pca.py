import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

MNIST_FASHION_LABELS = {
    0:  "T-shirt/top",
    1:	"Trouser",
    2: 	"Pullover",
    3: 	"Dress",
    4: 	"Coat",
    5: 	"Sandal",
    6: 	"Shirt",
    7: 	"Sneaker",
    8: 	"Bag",
    9: 	"Ankle boot"
}

USED_DATASETS = (2, 7, 1)


def get_pictures():
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

    pictures1 = []
    pictures2 = []
    pictures3 = []

    for i in range(x_train.shape[0]):
        if y_train[i] == USED_DATASETS[0]:
            pictures1.append(x_train[i])
        elif y_train[i] == USED_DATASETS[1]:
            pictures2.append(x_train[i])
        elif y_train[i] == USED_DATASETS[2]:
            pictures3.append(x_train[i])

    return np.array(pictures1), np.array(pictures2), np.array(pictures3)


if __name__ == '__main__':
    # get data
    pictures1, pictures2, pictures3 = get_pictures()
    total_pictures = pictures1.shape[0] + pictures2.shape[0] + pictures3.shape[0]
    data = np.concatenate((pictures1, pictures2, pictures3), axis=0).reshape(total_pictures, pictures1.shape[1]*pictures1.shape[2])

    # centre the data
    mean_picture = sum(data)/total_pictures
    centered_data = data - mean_picture

    # fit the model
    pca = PCA()
    pca.fit(centered_data)

    # get initial covariance and variance
    covariance = pca.get_covariance()
    variance = [covariance[i][i] for i in range(covariance.shape[0])]

    # transform data
    transformed_data = pca.transform(centered_data)

    # get covariance and variance after pca
    pca2 = PCA()
    pca2.fit(transformed_data)
    covariance_after_pca = pca2.get_covariance()
    variance_after_pca = [covariance_after_pca[i][i] for i in range(covariance_after_pca.shape[0])]

    # plot covariance matrices
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Covariance matrices")
    fig.colorbar(axs[0].imshow(covariance), ax=axs[0])
    axs[0].set_title("Before PCA")
    fig.colorbar(axs[1].imshow(covariance_after_pca), ax=axs[1])
    axs[1].set_title("After PCA")
    fig.colorbar(axs[2].imshow([row[:20] for row in covariance_after_pca[:20]]), ax=axs[2])
    axs[2].set_title("Top left corner after PCA")

    fig.tight_layout()
    plt.savefig("out/covariance.png")
    plt.close('all')

    # plot variances
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Feature variances")
    axs[0].bar([i for i in range(28*28)], variance)
    axs[0].set_title("Before PCA")

    axs[1].bar([i for i in range(28*28)], variance_after_pca)
    axs[1].set_title("After PCA")

    axs[2].bar([i for i in range(25)], variance_after_pca[:25])
    axs[2].set_title("After PCA, first 25 features")

    fig.tight_layout()
    plt.savefig("out/variances.png")
    plt.close('all')

    # get pca primary components
    components = pca.components_

    # reshape components into picture format
    components_pictures = components.reshape(28*28, 28, 28)

    # plot primary components
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle("Primary Components")
    for i in range(25):
        im = axs[i // 5][i % 5].imshow(components_pictures[i])
        fig.colorbar(im, ax=axs[i // 5][i % 5])
        axs[i // 5][i % 5].set_title(f"component {i}")
        axs[i // 5][i % 5].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    fig.tight_layout()
    plt.savefig("out/primaryComponents.png")
    plt.close('all')

    # reduce dimension to 2D
    transformed_data_2D = [data[:2] for data in transformed_data]

    # plot 2 first features on 2D pane
    dataset1_x = [data[0] for data in transformed_data_2D[:6000]]
    dataset1_y = [data[1] for data in transformed_data_2D[:6000]]

    dataset2_x = [data[0] for data in transformed_data_2D[6000:12000]]
    dataset2_y = [data[1] for data in transformed_data_2D[6000:12000]]

    dataset3_x = [data[0] for data in transformed_data_2D[12000:]]
    dataset3_y = [data[1] for data in transformed_data_2D[12000:]]

    plt.plot(dataset1_x, dataset1_y, 'bo', label=MNIST_FASHION_LABELS[USED_DATASETS[0]], markersize=1)
    plt.plot(dataset2_x, dataset2_y, 'go', label=MNIST_FASHION_LABELS[USED_DATASETS[1]], markersize=1)
    plt.plot(dataset3_x, dataset3_y, 'ro', label=MNIST_FASHION_LABELS[USED_DATASETS[2]], markersize=1)

    plt.legend()
    plt.title("First 2 features after PCA")
    plt.savefig("out/first2features.png")
    plt.close('all')

    # function to plot reduced pictures in original base
    def plot_reduced_reduced_dimension_picture(transformed_picture, title):
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))
        dimensions = [28*28, 500, 100, 50, 10, 2]
        for i in range(len(dimensions)):
            reduced_picture = np.concatenate((transformed_picture[:dimensions[i]], np.zeros(28*28 - dimensions[i])))
            reduced_picture_in_original_base = pca.inverse_transform([reduced_picture]) + mean_picture
            axs[i // 3][i % 3].imshow(reduced_picture_in_original_base.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
            axs[i // 3][i % 3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            axs[i // 3][i % 3].set_title(f"dimension = {dimensions[i]}")
        axs[0][0].set_title(f"original picture (dim={28*28})")
        fig.suptitle(title)
        fig.tight_layout()

    # plot some examples to show how reducing dimension after pca affects data in original base
    dataset1_example = transformed_data[863]
    dataset2_example = transformed_data[6000 + 425]
    dataset3_example = transformed_data[12000 + 5112]

    plot_reduced_reduced_dimension_picture(dataset1_example, f"{MNIST_FASHION_LABELS[USED_DATASETS[0]]} picture after PCA dimension reduction")
    plt.savefig(f"out/{MNIST_FASHION_LABELS[USED_DATASETS[0]]}Example.png")
    plt.close('all')

    plot_reduced_reduced_dimension_picture(dataset2_example, f"{MNIST_FASHION_LABELS[USED_DATASETS[1]]} picture after PCA dimension reduction")
    plt.savefig(f"out/{MNIST_FASHION_LABELS[USED_DATASETS[1]]}Example.png")
    plt.close('all')

    plot_reduced_reduced_dimension_picture(dataset3_example, f"{MNIST_FASHION_LABELS[USED_DATASETS[2]]} picture after PCA dimension reduction")
    plt.savefig(f"out/{MNIST_FASHION_LABELS[USED_DATASETS[2]]}Example.png")
    plt.close('all')












