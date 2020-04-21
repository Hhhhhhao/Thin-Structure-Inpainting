import glob
import torch
import cv2
import random
import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import rotate
from skimage.util import random_noise
from matplotlib import pyplot as plt
plt.switch_backend('agg')


DEFAULT_SEED = 1997


def get_files(directory, format='png'):
    """
    To get a list of file names in one directory, especially images
    :param directory: a path to the directory of the image files
    :return: a list of all the file names in that directory
    """
    if format is 'png':
        file_list = glob.glob(directory + "*.png")
    elif format is 'tif':
        file_list = glob.glob(directory + "*.tif")
    else:
        raise ValueError("dataset do not support")

    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_list

def get_blob_masks(mask_dir):
    masks = []
    files = get_files(mask_dir)
    for file in files:
        mask = cv2.imread(file, 0)
        masks.append(mask)
    return masks


def skeletonize_image(image):
    """
    Skeletonize a 2D binary root array
    :param image: a 2D binary image array to be skeletonized
    :return: a skeletonized 2D binary image array
    """

    # dilate kernel for pre-dilation before skeletonize
    kernel = np.ones((1,1), np.uint8)

    # first dilate the image a little bit for better skeletonize
    image = cv2.dilate(image, kernel, iterations=1)

    # skeletonize the image
    skeleton = skeletonize(image)
    return skeleton.astype(np.uint8)


def dilate_image(image, min_iter=2, max_iter=5, rng=None):
    """
    Dilate a 2D binary root skeleton using iteration decided by iteration
    :param image: 2D binary root skeleton
    :param min_iter: minimum dilation iteraion
    :param max_iter: maximum dilation iteration
    :param rng: numpy random state
    :return:
    """
    if rng is None:
        rng = np.random.RandomState(DEFAULT_SEED)

    # dilate kernel for image dilation
    kernel = np.ones((3,3), np.uint8)

    # random dilation iteration decided by rng random state
    dilate_iter = rng.randint(min_iter, max_iter)

    # dilate image skeleton using kernel and iteration
    dilation = cv2.dilate(image, kernel, dilate_iter)
    return dilation.astype(np.uint8)


def rotate_image(image, angle=5, rng=None):
    """
    Rotate a 2D binary image in terms of angle
    :param image: a 2D binary image array
    :param angle: rotation angle
    :param rng: numpy random state
    :return: a rotated image
    """
    if rng is None:
        rng = np.random.RandomState(DEFAULT_SEED)

    angle = rng.uniform(-1., 1.) * angle
    image = rotate(image, angle, resize=False, order=0, preserve_range=True)
    return image


def add_salt_noise_to_skeleton(skeleton, seed=1997):
    """
    Add salt noise to an image, used to add noisy edges to a 2D binary root skeleton
    :param skeleton: 2D binary root skeleton array
    :param seed: numpy random seed
    :return: a noisy 2D binary root skeleton array
    """
    # create a noise mask
    noise_mask = np.zeros_like(skeleton)
    noise_mask = random_noise(noise_mask, "salt", seed=seed, amount=0.05)

    # add noise to skeleton
    noise_image = skeleton.copy()
    noise_image[noise_mask == 1] = 1

    # find the largest connected component
    noise_image = find_largest_component(noise_image)
    return noise_image.astype(np.uint8)


def find_largest_component(image):
    """
    Find the largest fully connected component in an image
    :param image: 2D binary image array
    :return: 2D binary image array containing only largest connected component in the original image
    """
    new_image = np.zeros_like(image)
    mask = np.uint8(image == 1)
    labels, stats = cv2.connectedComponentsWithStats(mask, 8)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_image[labels == largest_label] = 1
    return new_image


def randomization_image(image,
                        dilation=True,
                        noisy_texture=True,
                        rotation=True,
                        training=True,
                        seed=None):
    """
    Get a randomized image from randomization
    :param image: 2D binary image
    :param dilation: whether to apply dilation transform
    :param noisy_texture: whether to add noise to root texture
    :param rotation: whether to rotate the images
    :param training: training phase or not
    :param seed: random seed
    :return: a transformed 2D image
    """
    if training is True:
        rand_seed = seed
    else:
        rand_seed = DEFAULT_SEED

    # if dilation is True
    # skeletonize the root structure
    # if noisy texture is True, add noise to the structure
    # then dilate the root
    if dilation:
        # first skeletonize an image
        image = skeletonize_image(image)

        # add noise to image
        if noisy_texture:
            image = add_salt_noise_to_skeleton(image, rand_seed)

        image = dilate_image(image, rng=np.random.RandomState(rand_seed))

    # if rotate is True and it's in training
    # rotate the dilated image
    if rotation and training:
        rotation_rng = np.random.RandomState(rand_seed)
        p = rotation_rng.uniform(0, 1.)
        if p >= 0.5:
            image = rotate_image(image, angle=5)
    return image


def resize_image(image):
    """
    Resize the image to be dividable by 16 so that the network can handle it
    :param image: 2D binary image array
    :return:
    """
    height = image.shape[0]
    width = image.shape[1]

    desire_width = int(np.floor(width / 16)) * 16
    desire_height = int(np.floor(height / 16)) * 16
    resized_image = cv2.resize(image, (desire_width, desire_height), interpolation=cv2.INTER_NEAREST)
    return resized_image


def get_root_pixels(image, boarder_height, boarder_width, root_pixel_value=1):
    """
    Get the root pixels from an 2D image, with boarder removed defined by boarder height and boarder width
    :param image: 2D image array
    :param boarder_height: boarder height to be removed
    :param boarder_width: boarder width to be removed
    :param root_pixel_value: root pixel value
    :return: points
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    masked_image = image.copy()
    mask = np.zeros_like(image)
    mask[boarder_height:image_height-boarder_height, boarder_width:image_width-boarder_width] = 1
    masked_image[mask == 0] = 0
    points = np.where(masked_image == root_pixel_value)
    return points


def get_sliding_windows(image,
                        size=256):
    """
    Get sliding windows from a whole image
    :param image: an 2D binary root image to be sliced
    :param size: size of windows to be extracted
    :return: windows: a list of patches of desire size
             locations: a list of coordinates of location where original pathches are extracted
    """
    windows = []
    locations = []
    # set pixel threshold
    # pixel_threshold = int(0.01 * size ** 2)
    # image width
    image_width = image.shape[1]
    image_height = image.shape[0]

    # check image width size
    if 100 < image_width < size:

        for h_idx in range(0, (image_height - image_width), image_width):
            # a patch of size (image_width, image_width)
            window = image[h_idx:h_idx+image_width, :]
            # location of format (y_0, x_0, y_1, x_1)
            location = (h_idx, 0, h_idx+image_width, image_width)

            # if window.shape[0] != desire_size:
            # window = cv2.resize(window, (desire_size, desire_size), cv2.INTER_LINEAR)
            # _, window = cv2.threshold(window, 0.5, 1, cv2.THRESH_BINARY)

            points = get_root_pixels(window, 0, 0, 1)
            num_pixels = len(points[0])

            if num_pixels >= 15:
                windows.append(window)
                locations.append(location)
            else:
                continue

    elif image_width >= size:
        for h_idx in range(0, (image_height-size), size):
            for w_idx in range(0, (image_width-size), size):
                # a patch of size (extract_size, extract_size)
                window = image[h_idx:h_idx+size, w_idx:w_idx+size]
                # patch location of format (y_0, x_0, y_1, x_1)
                location = (h_idx, w_idx, h_idx+size, w_idx+size)

                # if window.shape[0] != desire_size:
                # window = cv2.resize(window, (desire_size, desire_size), cv2.INTER_LINEAR)
                # _, window = cv2.threshold(window, 0.5, 1, cv2.THRESH_BINARY)

                points = get_root_pixels(window, 0, 0, 1)
                num_pixels = len(points[0])

                if num_pixels >= 50:
                    windows.append(window)
                    locations.append(location)
                else:
                    continue

    return windows, locations


def check_image_value(image, min=0, max=1):
    """
    Check whether an 2D image is binary
    :param image: 2D image array
    :param min: minimum image value
    :param max: maximum image value
    :return: assertation
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            assert image[i][j] == min or image[i][j] == max, "wrong value {} at ({}, {})".format(image[i][j], i, j)


def get_image(image_id,
              dilation=True,
              noisy_texture=True,
              rotation=True,
              training=True,
              seed=DEFAULT_SEED):
    """
    Augmentation of a synthetic root to be more similar to chickpea root
    :param dilation: True/False whether to dilate the root
    :param noisy_texture: True/False whether to add noise into texture of the root
    :param rotation: True/False whether to rotate the root
    :param training: True/False whether at training stage
    :return: an transformed root image
    """
    # read image
    image = cv2.imread(image_id, 0)

    # normalize image into [0,1]
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = (image/255.).astype(np.uint8)

    # decide whether to randomize image or not
    p = np.random.uniform(0., 1.)
    if p >= 0.25:
        image = randomization_image(image,
                                    dilation=dilation,
                                    noisy_texture=noisy_texture,
                                    rotation=rotation,
                                    training=training,
                                    seed=seed)
    else:
        image = randomization_image(image,
                                    dilation=False,
                                    noisy_texture=False,
                                    rotation=False,
                                    training=False,
                                    seed=seed)
    return image


def get_patches(image, size=256):
    """
    Get patches from a root image
    :return: windows and locations
    """
    windows, locations = get_sliding_windows(image, size)
    return windows, locations


def connected_component(graph, connectivity=4):
    """
    Find the connected components of a graph
    :param graph: a binary graph
    :param connectivity: the connenectivity
    :return: num of labels (substract 1 to get the num of components)
             labels
             stats
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(graph, connectivity, cv2.CV_32S)
    return num_labels, labels, stats


def inpaint_full_image(img, model, threshold=50):
    """
    Inpaint a full root
    :param img: a root image
    :param model: a trained model (generator)
    :param threshold: threshold value for binary thresholding the generator output
    """
    img = img / 255.
    height = img.shape[0]
    width = img.shape[1]
    desire_height = int(np.floor(height / 16.)) * 16
    desire_width = int(np.floor(width / 16.)) * 16

    resized_img = cv2.resize(img, (desire_width, desire_height), interpolation=cv2.INTER_LINEAR)
    _, resized_img = cv2.threshold(resized_img, 0.5, 1, cv2.THRESH_BINARY)

    inputs = np.expand_dims(resized_img, axis=-1)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = inputs.transpose((0, 3, 1, 2))

    if torch.cuda.is_available():
        inputs = torch.cuda.FloatTensor(inputs)
    else:
        inputs = torch.FloatTensor(inputs)

    prediction = model.inference(inputs)
    prediction = prediction.cpu().numpy()
    prediction = prediction.transpose((0, 2, 3, 1))
    mask = (resized_img == 0)
    predict_image = resized_img.copy()
    predict_image[mask] = prediction[0, :, :, 1][mask]
    inpainted = predict_image.copy()
    # binary_inpainted = inpainted.copy()
    # inpainted = prediction.copy()
    # inpainted[prediction[:, :] >= 0.5] = 1 # prediction[:, :, 1]
    unthreshed_inpainted = (inpainted * 255).astype(np.uint8)

    binary_inpainted = unthreshed_inpainted.copy()
    binary_inpainted = binary_inpainted.astype(np.uint8)

    _, binary_inpainted = cv2.threshold(binary_inpainted, threshold, 255, cv2.THRESH_BINARY)

    rgb_inpainted = cv2.cvtColor((resized_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    rgb_inpainted[:, :, -1] = binary_inpainted

    return resized_img*255, binary_inpainted, rgb_inpainted, unthreshed_inpainted


def remove_artifacts(img, threshold=10):
    # grayscale image as input
    num_labels, labels, stats = connected_component((img/255).astype(np.uint8), 8)
    new_img = img.copy()

    for i, area in enumerate(stats[1:, cv2.CC_STAT_AREA]):
        if area < threshold:
            index = i + 1
            new_img[labels == index] = 0

    return new_img


def mask_with_gaps(image,
                   min_num_small=5, max_num_small=10,
                   min_num_large=1, max_num_large=1,
                   min_small=8, max_small=15,
                   min_large=25, max_large=35,
                   training=True):
    """
    Introduce square gaos
    :param image: an 2D image to be corrupted by square masks
    :param min_num_small: minimum number of small square gaps
    :param num_small: maximum number of small sqaure gaps
    :param min_num_large: minimum number of large square gaps
    :param num_large: maximum number of large square gaps
    :param min_small: minimum size of small square gaps
    :param max_small: maximum size of small square gaps
    :param min_large: minimum size of large square gaps
    :param max_large: maximum size of large square gaps
    :param training: whether is training process; if during training, the size and number are random
    :return: masked_img: the resulting masked 2D image
    :return: masks: a list of masks, used for computing the MSE within gaps
    """
    if not training:
        np.random.seed(DEFAULT_SEED)

    num_small_gaps = np.random.randint(min_num_small, max_num_small)
    num_large_gaps = np.random.randint(min_num_large, max_num_large)
    img_height = image.shape[0]
    img_width = image.shape[1]
    masked_img = image.copy()
    mask = np.zeros((img_height, img_width))

    for i in range(num_large_gaps):

        if not training:
            np.random.seed(i)

        h = random.randint(min_large, max_large)
        w = random.randint(min_large, max_large)
        points = get_root_pixels(masked_img, h, w)
        num_points = len(points[0]) - 1
        if num_points < 5:
            break
        rand_idx = random.randint(0, num_points)
        y1 = int(points[0][rand_idx] - h / 2)
        x1 = int(points[1][rand_idx] - w / 2)
        mask[y1:y1 + h, x1:x1 + w] = 1

    for i in range(num_small_gaps):
        if not training:
            np.random.seed(i)

        h = random.randint(min_small, max_small)
        w = random.randint(min_small, max_small)
        points = get_root_pixels(masked_img, h, w)
        num_points = len(points[0]) - 1
        if num_points < 5:
            break
        rand_idx = random.randint(0, num_points)
        y1 = int(points[0][rand_idx] - h / 2)
        x1 = int(points[1][rand_idx] - w / 2)
        mask[y1:y1 + h, x1:x1 + w] = 1

    masked_img[mask == 1] = 0

    return masked_img, mask


def mask_with_blobs(image,
                    total_blob_masks,
                    min_num_blobs=5,
                    max_num_blobs=15,
                    min_mask_size=32,
                    max_mask_size=64,
                    training=True):
    """
    Introduce blob gaos
    :param image: an 2D image to be corrupted by blob masks
    :param total_blob_masks: blob masks list
    :param min_num_blobs: minimum number of blob gaps
    :param max_num_blobs: maximum number of blob gaps
    :param min_mask_size: minimum mask size
    :param max_mask_size: maximum mask size
    :param training: whether is training process; if during training, the size and number are random
    :return: masked_img: the resulting masked 2D image
    :return: masks: a list of masks, used for computing the MSE within gaps
    """
    masked_img = image.copy()
    mask = np.zeros_like(image)

    if not training:
        np.random.seed(DEFAULT_SEED)

    num_blobs = np.random.randint(min_num_blobs, max_num_blobs)
    mask_size = np.random.randint(min_mask_size, max_mask_size)
    # sample masks from BLOB_MASKS LIST
    np.random.shuffle(total_blob_masks)
    blob_masks = total_blob_masks[:num_blobs]

    for i, blob_mask in enumerate(blob_masks):
        blob_mask = singleimage_random_rotation(blob_mask, angle=180)
        blob_mask = cv2.resize(blob_mask, (mask_size, mask_size))
        points = get_root_pixels(masked_img, mask_size, mask_size, 1)

        if len(points[0]) <= 30:
            break

        if not training:
            np.random.seed(DEFAULT_SEED)

        rand_idx = np.random.randint(0, len(points[0]))
        x = points[0][rand_idx]
        y = points[1][rand_idx]

        mask[y - int(np.floor(mask_size / 2.)):y + (mask_size - int(np.floor(mask_size / 2.))),
        x - int(np.floor(mask_size / 2.)):x + (mask_size - int(np.floor(mask_size / 2.)))][blob_mask == 255] = 1
        masked_img[y - int(np.floor(mask_size / 2.)):y + (mask_size - int(np.floor(mask_size / 2.))),
        x - int(np.floor(mask_size / 2.)):x + (mask_size - int(np.floor(mask_size / 2.)))][blob_mask == 255] = 0

    return masked_img, mask


def mask_with_brush(image,
                    minVertex=10, maxVertex=20,
                    minLength=20, maxLength=25,
                    minBrushWidth=20, maxBrughWidth=25,
                    maxAngle=45,
                    training=True):
    """
    Introduce brush gaps
    """

    # print(image.shape)
    mask = np.zeros((image.shape[0], image.shape[1], 3))
    # mask = np.zeros((64, 64, 3))
    masked_img = image.copy()

    if not training:
        np.random.seed(DEFAULT_SEED)

    numVertex = np.random.randint(minVertex, maxVertex)

    points = np.where(masked_img != 0)

    if len(points[0]) >= 5:
        idx = np.random.randint(0, len(points[0]))
        startX = points[0][idx]
        startY = points[1][idx]
    else:
        startX = 0
        startY = 0

    for i in range(numVertex):
        if not training:
            np.random.seed(i)

        angle = np.random.randint(maxAngle)

        if i%2 == 0:
            angle = 2 * np.pi - angle  #reverse mode

        length = np.random.randint(minLength, maxLength)
        brushWidth = np.random.randint(minBrushWidth, maxBrughWidth)

        endX = int(startX + length * np.sin(angle))
        endY = int(startY + length * np.cos(angle))
        mask = cv2.line(mask, (startX, startY), (endX, endY), color=(255, 255, 255), thickness=brushWidth)
        startX = endX
        startY = endY
        mask = cv2.circle(mask, (startX, startY), radius=int(brushWidth/2), color=(255, 255, 255), thickness=-1)

        p = np.random.uniform(0, 1)

        if p > 0.5:
            mask = cv2.flip(mask, 1)

        p = np.random.uniform(0, 1)

        if p > 0.5:
            mask = cv2.flip(mask, 0)

    mask = mask[..., 0] / 255.

    masked_img[mask == 1] = 0

    return masked_img, mask


def preprocessing(patch_list, mask_type='square', total_blob_masks=None, training=True):
    """
    Make a complete patch list corrupted
    :param patch_list: a list containining complte patches
    :param mask_type: type of gaps to be introduced
    :param total_blob_masks: blob masks list
    :param training: whether is training process
    :return a complete patch list as ground truth, a corrupted patch list as input, and the masks indicating the missing pixels
    """
    batch_x = []
    batch_y = []
    batch_masks = []

    for img in patch_list:
        if mask_type == 'square':
            masked_img, mask = mask_with_gaps(img,
                                              min_num_small=5, max_num_small=15,
                                              min_num_large=1, max_num_large=2,
                                              min_small=5, max_small=15,
                                              min_large=25, max_large=45,
                                              training=training)
        elif mask_type == 'brush':
            masked_img, mask = mask_with_brush(img,
                                         minVertex=5, maxVertex=15,
                                         minLength=10, maxLength=30,
                                         minBrushWidth=10, maxBrughWidth=30,
                                         maxAngle=45,
                                         training=training)
        elif mask_type == 'blob':
            masked_img, mask = mask_with_blobs(img,
                                               total_blob_masks,
                                               min_num_blobs=5,
                                               max_num_blobs=20,
                                               min_mask_size=32,
                                               max_mask_size=64,
                                               training=training)
        elif mask_type == 'mix':
            mask = np.zeros_like(img)
            masked_img = img.copy()
            _, brush_mask = mask_with_brush(img,
                                         minVertex=5, maxVertex=10,
                                         minLength=5, maxLength=15,
                                         minBrushWidth=5, maxBrughWidth=15,
                                         maxAngle=60,
                                         training=training)
            _, square_mask = mask_with_gaps(img,
                                    min_num_small=5, max_num_small=15,
                                    min_num_large=1, max_num_large=2,
                                    min_small=5, max_small=15,
                                    min_large=25, max_large=35,
                                    training=training)
            _, blob_mask = mask_with_blobs(img,
                                    total_blob_masks,
                                    min_num_blobs=5,
                                    max_num_blobs=10,
                                    min_mask_size=32,
                                    max_mask_size=64,
                                    training=training)

            mask[brush_mask == 1] = 1
            mask[square_mask == 1] = 1
            mask[blob_mask == 1] = 1
            masked_img[mask == 1] = 0
        else:
            raise ValueError('mask type: {}'.format(mask_type))
        _, masked_img = cv2.threshold(masked_img, 0.5, 1, cv2.THRESH_BINARY)
        batch_x.append(masked_img)

        img = img
        img = np.expand_dims(img, axis=-1)
        reverse_img = np.ones(img.shape)
        reverse_img[img == 1.0] = 0
        batch_y.append(np.concatenate((img, reverse_img), axis=-1))
        batch_masks.append(mask)

    batch_x = np.stack(batch_x)
    batch_x = np.expand_dims(batch_x, axis=-1)
    batch_y = np.stack(batch_y)
    batch_masks = np.stack(batch_masks)
    batch_masks = np.expand_dims(batch_masks, axis=-1)
    return batch_x, batch_y, batch_masks


def singleimage_random_rotation(image, angle=5):
    """
    Rotate an image
    :param image: 2D binary image;
    :param angle: rotation angle
    """
    angle = np.random.uniform(-1., 1.) * angle
    new_image = rotate(image, angle, resize=False, order=0, preserve_range=True)
    return new_image



def convert_labels_to_rgb(labels):
    """
    convert graph components labels into RGB image
    """
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


def extract_patch_from_tensor(tensor_X, patch_size):
    """
    Function for patch discriminator
    """
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(tensor_X.size(2) // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(tensor_X.size(3) // patch_size[1])]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            patches = tensor_X[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
            patches = torch.where(patches >= 0.5, torch.ones_like(patches), torch.zeros_like(patches))
            list_X.append(patches)
    return list_X