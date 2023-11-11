import os
import cv2
import numpy as np
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_processing import get_files, get_image
synthetic_path = os.path.join(dirname, 'data/root/synthetic')
chickpea_patch_path = os.path.join(dirname, 'data/root/real/patch/')
chickpea_full_path = os.path.join(dirname, 'data/root/real')
# test_chickpea_path = os.path.join(dirname, 'data/root/final_test/chickpea-patch/')
retinal_full_path = os.path.join(dirname, 'data/retinal/')
test_synthetic_path = os.path.join(dirname, 'data/root/synthetic/')
# test_synthetic_path = os.path.join(dirname, 'saved/Root-Baseline/0503_095454/testing/synthetic-patch/')
# test_synthetic_path = os.path.join(dirname, 'saved/Root-Baseline/0503_202933/testing/synthetic-patch/')
test_chickpea_path = os.path.join(dirname, 'saved/Root-Baseline/0502_083309/testing/chickpea-patch/')
# test_chickpea_path = os.path.join(dirname, 'saved/Root-Baseline/0503_095454/testing/chickpea-patch/')
# test_chickpea_path = os.path.join(dirname, 'saved/Root-Baseline/0503_202933/testing/chickpea-patch/')


class SyntheticRootDataset(Dataset):
    """
    Dataset for synthetic root
    """
    def __init__(self,
                 which_set='train',
                 dilation=True,
                 noisy_texture=True,
                 rotation=True):
        """
        Synthetic root dataset initialization
        :param which_set: str: 'train', 'valid', 'test'
        :param dilation: flag of root dilation
        :param noisy_texture: flag of noisy texture
        :param rotation: flag of rotation augmentation
        """

        assert which_set in ['train', 'valid', 'test', 'test/total' 'p', 't'], "wrong set:{}".format(which_set)
        if which_set in ['train', 't']:
            self.training = True
        else:
            self.training = False

        if which_set == 'test':
            which_set += '/total'

        self.which_set = which_set
        self.dilation = dilation
        self.noisy_texture = noisy_texture
        self.rotation = rotation

        # synthetic root data path
        synthetic_path = os.path.join(dirname, 'data/root/synthetic')
        # get file list as image ids
        path_to_files = os.path.join(synthetic_path, which_set+'/')
        if os.path.exists(path_to_files):
            ids = os.listdir(path_to_files)
            # remove root images that are too thin
            for i in range(len(ids)):
                ids[i] = path_to_files + ids[i]
            self.ids = self.select_images(ids)
        else:
            raise ValueError("data path do not exist:{}".format(path_to_files))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]

        # load image file and then random transform images
        image = get_image(image_id=image_id,
                          dilation=self.dilation,
                          noisy_texture=self.noisy_texture,
                          rotation=self.rotation,
                          training=self.training,
                          seed=index)

        return image

    @staticmethod
    def select_images(ids, width=256):
        """
        Pre-selection of the root images to filter out segmentation masks that are too thin
        :param ids: id list
        :param width: minimum segmentation width
        :return: a filtered id list
        """
        new_ids = []
        for id in tqdm(ids):
            
            if 'dicot' in id:
                image = cv2.imread(id, 0)
                image_width = image.shape[1]
                if image_width >= width:
                    new_ids.append(id)
        return new_ids


class ChickpeaPatchRootDataset(Dataset):
    """
    Chickpea selected patch dataset
    """
    def __init__(self, which_set='train'):
        """
        :param which_set: 'train', 'valid', or 'test'
        """

        assert which_set in ['train', 'valid', 'test'], "wrong set:{}".format(which_set)
        if which_set in ['train']:
            self.training = True
        else:
            self.training = False

        self.which_set = which_set
        chickpea_patch_path = os.path.join(dirname, 'data/root/real/patch/')
        # get file list as image ids
        path_to_files = os.path.join(chickpea_patch_path, which_set+'/')
        if os.path.exists(path_to_files):
            self.ids = get_files(path_to_files)
        else:
            raise ValueError("data path do not exist:{}".format(path_to_files))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        # load image file
        image = cv2.imread(image_id, 0)
        image = image/255.
        return image


class ChickpeaFullRootDataset(Dataset):
    """
    Chickpea Full Root Dataset
    """
    def __init__(self, which_set='train'):
        """
        :param which_set: 'train', 'valid', or 'test'
        """

        assert which_set in ['train', 'valid', 'test'], "wrong set:{}".format(which_set)
        if which_set in ['train', 't']:
            self.training = True
        else:
            self.training = False

        self.which_set = which_set
        chickpea_full_path = os.path.join(dirname, 'data/root/real')
        # get file list as image ids
        path_to_files = os.path.join(chickpea_full_path, which_set+'/')
        if os.path.exists(path_to_files):
            self.ids = get_files(path_to_files)
        else:
            raise ValueError("data path do not exist:{}".format(path_to_files))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]

        # load image file and then random transform images
        image = get_image(image_id=image_id,
                          dilation=False,
                          noisy_texture=False,
                          rotation=False,
                          training=self.training,
                          seed=index)
        return image


class RoadDataset(Dataset):
    """
    Road segmentation image dataset
    """
    def __init__(self, which_set):
        assert which_set in ['train', 'valid', 'test'], "wrong set:{}".format(which_set)
        if which_set == 'train':
            self.training = True
        else:
            self.training = False
        self.base_path = os.path.join(dirname, 'data/road')
        self.image_ids = get_files(os.path.join(self.base_path, which_set+'/'), format='tif')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        # load image
        image = cv2.imread(image_id, 0)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = image/255.

        return image


class LineDataset(Dataset):
    """
    Line drawings sketch dataset
    """
    def __init__(self, which_set):
        assert which_set in ['train', 'valid', 'test'], "wrong set:{}".format(which_set)
        if which_set == 'train':
            self.training = True
        else:
            self.training = False
        self.base_path = os.path.join(dirname, 'data/line')
        self.image_ids = get_files(os.path.join(self.base_path, which_set+'/'), format='png')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        # load image
        image = cv2.imread(image_id, 0)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = image/255.

        return image


class RetinalDataset(Dataset):
    """
    Retinal vessel segmentation dataset
    """
    def __init__(self, which_set='train'):
        assert which_set in ['train', 'valid', 'test'], "wrong set:{}".format(which_set)
        if which_set == 'train':
            self.training = True
        else:
            self.training = False

        if which_set == 'test':
            file_format = 'tif'
        else:
            file_format = 'png'

        self.which_set = which_set
        # get file list as image ids
        retinal_full_path = os.path.join(dirname, 'data/retinal/')
        path_to_files = os.path.join(retinal_full_path, which_set+'/')
        if os.path.exists(path_to_files):
            self.ids = get_files(path_to_files, format=file_format)
        else:
            raise ValueError("data path do not exist:{}".format(path_to_files))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        # load image
        image = cv2.imread(image_id, 0)
        image = image/255.
        return image


class TestRootDataset(Dataset):
    def __init__(self,
                 name='synthetic'):

        if name == 'synthetic':
            data_path = test_synthetic_path
        elif name == 'chickpea':
            data_path = test_chickpea_path
        else:
            raise ValueError('wrong name')

        if os.path.exists(data_path):
            self.input_ids = get_files(data_path + 'binary_input/')
            self.target_ids = get_files(data_path + 'binary_target/')
        else:
            raise ValueError("data path do not exist:{}".format(data_path))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        target_id = self.target_ids[index]

        # load image file
        input = cv2.imread(input_id, 0)
        input = input/255.
        input = np.expand_dims(input, -1)

        # load image file
        target = cv2.imread(target_id, 0)
        target = target/255.
        target = np.expand_dims(target, -1)

        data ={'input':input, 'target':target}

        return data
