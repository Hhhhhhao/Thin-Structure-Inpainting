import os
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
blob_masks_path = os.path.join(dirname, 'data/root/gaps/small/')
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader.datasets import SyntheticRootDataset, ChickpeaPatchRootDataset, ChickpeaFullRootDataset, RoadDataset, LineDataset, RetinalDataset
from functools import partial
from utils.data_processing import get_patches, preprocessing, get_blob_masks


def full_seg_collate_fn(data,
                         batch_size,
                         mask_type,
                         image_size,
                         total_blob_masks,
                         training):
    """
    full segmentation mask collate function

    :param data: minibatch data
    :param batch_size: number of patches
    :param mask_type: gaps type 'square'|'brush'|'blob'|'mix'
    :param image_size: patch size
    :param total_blob_masks: blob masks list
    :param training: flag of training
    :return: a training minibatch
    """
    image = data[0]
    # split the single segmentation masks into patches
    windows, locations = get_patches(
        image=image,
        size=image_size)
    orig_length = len(windows)

    # if number of patches is greater than batch size, randomly select some patches for training
    if orig_length >= batch_size:
        if training:
            permutation = np.random.permutation(orig_length)
            windows = np.array(windows)
            locations = np.array(locations)
            windows = windows[permutation]
            locations = locations[permutation]
        windows = windows[:batch_size]
        locations = locations[:batch_size]
    # else replicate the patches
    elif 0 < orig_length <= batch_size:
        windows = np.tile(windows, (batch_size, 1, 1))
        locations = locations * batch_size
        windows = windows[:batch_size]
        locations = locations[:batch_size]

    inputs, targets, batch_masks = preprocessing(windows, mask_type, total_blob_masks, training)
    inputs = inputs.transpose((0, 3, 1, 2))
    targets = targets.transpose((0, 3, 1, 2))
    targets = targets[:, 0, :, :]
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=1)

    if torch.cuda.is_available():
        image = torch.cuda.FloatTensor(image)
        batch_x = torch.cuda.FloatTensor(inputs)
        batch_y = torch.cuda.LongTensor(targets)
    else:
        image = torch.FloatTensor(image)
        batch_x = torch.FloatTensor(inputs)
        batch_y = torch.LongTensor(targets)

    return batch_x, batch_y, batch_masks, locations, orig_length, image


def chickpea_patch_collate_fn(data, mask_type, total_blob_masks, training):
    """
    chickpea patch collate function
    :param data: data list
    :param mask_type: gap type 'square'|'brush'|'blob'|'mix'
    :param total_blob_masks: blob masks list
    :param training: flag of training
    :return:
    """
    windows = data
    inputs, targets, batch_masks = preprocessing(windows, mask_type, total_blob_masks, training)
    inputs = inputs.transpose((0, 3, 1, 2))
    targets = targets.transpose((0, 3, 1, 2))
    targets = targets[:, 0, :, :]

    if torch.cuda.is_available():
        batch_x = torch.cuda.FloatTensor(inputs)
        batch_y = torch.cuda.LongTensor(targets)
    else:
        batch_x = torch.FloatTensor(inputs)
        batch_y = torch.LongTensor(targets)

    return batch_x, batch_y, 0, 0, 0


def test_collate_fn(data):

    input = data[0]["input"]
    target = data[0]["target"]
    input = np.expand_dims(input, 0)
    input = input.transpose((0, 3, 1, 2))
    batch_y = np.expand_dims(target, 0)

    if torch.cuda.is_available():
        batch_x = torch.cuda.FloatTensor(input)
    else:
        batch_x = torch.FloatTensor(input)

    return batch_x, batch_y


class SyntheticRootDataLoader(DataLoader):
    """
    Synthetic root data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 mask_type='mix',
                 dilation=True,
                 noisy_texture=True,
                 rotation=True,
                 image_size=256,
                 num_workers=0
                 ):
        """
        Initialization of synthetic root data loader
        :param which_set: 'train', 'valid', 'test'
        :param batch_size: batch size (how many small patches to be extracted from a single segmentation mask)
        :param mask_type: gap type 'square'|'blob'|'brush'|'mix'
        :param dilation: root dilation
        :param noisy_texture: noisy texture
        :param rotation: root rotation
        :param image_size: patch size
        :param num_workers: number of workers, normally set to 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        # synthetic root dataset
        self.dataset = SyntheticRootDataset(
            which_set=which_set,
            dilation=dilation,
            noisy_texture=noisy_texture,
            rotation=rotation)
        self.n_samples = len(self.dataset)
        self.image_size = image_size

        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(SyntheticRootDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, #batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(full_seg_collate_fn,
                batch_size=self.batch_size,
                mask_type=mask_type,
                image_size=image_size,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training
        ))


class ChickpeaPatchRootDataLoader(DataLoader):
    """
    Chickpea patch root dataloader
    """
    def __init__(self, which_set='train', batch_size=16, mask_type='mix', num_workers=0):
        """
        Initialization of chickpea patch data loader
        :param which_set: 'train'|'test'|'valid'
        :param batch_size: number of patches to be extracted from a single segmentation mask
        :param mask_type: 'square'|'brush'|'blob'|'mix'
        :param num_workers: 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = ChickpeaPatchRootDataset(
            which_set=which_set
        )
        self.n_samples = len(self.dataset)
        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(ChickpeaPatchRootDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size, #batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(chickpea_patch_collate_fn,
                mask_type=mask_type,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training
        ))


class ChickpeaFullRootDataLoader(DataLoader):
    """
    Chickpea Full Root data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 mask_type='mix',
                 image_size=256,
                 num_workers=0
                 ):
        """

        :param which_set: 'train'|'test'|'valid'
        :param batch_size: number of patches to be extracted from a single segmentation mask
        :param mask_type: 'square'|'brush'|'blob'|'mix'
        :param image_size: patch size, normally (256, 256)
        :param num_workers: 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = ChickpeaFullRootDataset(
            which_set=which_set
        )
        self.n_samples = len(self.dataset)

        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(ChickpeaFullRootDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, # batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(full_seg_collate_fn,
                batch_size=self.batch_size,
                mask_type=mask_type,
                image_size=image_size,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training))


class RoadDataLoader(DataLoader):
    """
    Satellite road segmentation data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 mask_type='mix',
                 image_size=256,
                 num_workers=0
                 ):
        """
        Initialization of road data loader
        :param which_set: 'train'|'valid'|'test'
        :param batch_size: number of patches
        :param mask_type: 'square'|'brush'|'blob'|'mix'
        :param image_size: patch size
        :param num_workers: normally set to 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = RoadDataset(
            which_set=which_set
        )
        self.n_samples = len(self.dataset)
        self.image_size = image_size

        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(RoadDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, # batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(full_seg_collate_fn,
                batch_size=self.batch_size,
                mask_type=mask_type,
                image_size=image_size,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training
        ))


class LineDataLoader(DataLoader):
    """
    Line drawings sketch data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 mask_type='mix',
                 image_size=256,
                 num_workers=0
                 ):
        """
        Initialization of line data loader
        :param which_set: 'train'|'valid'|'test'
        :param batch_size: number of patches
        :param mask_type: 'square'|'brush'|'blob'|'mix'
        :param image_size: patch size
        :param num_workers: normally set to 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = LineDataset(
            which_set=which_set
        )
        self.n_samples = len(self.dataset)
        self.image_size = image_size

        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(LineDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, # batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(full_seg_collate_fn,
                batch_size=self.batch_size,
                mask_type=mask_type,
                image_size=image_size,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training
        ))


class RetinalDataLoader(DataLoader):
    """
    Retinal vessel segmentation data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 mask_type='mix',
                 image_size=256,
                 num_workers=0
                 ):
        """
        Initialization of retinal data loader
        :param which_set: 'train'|'valid'|'test'
        :param batch_size: number of patches
        :param mask_type: 'square'|'brush'|'blob'|'mix'
        :param image_size: patch size
        :param num_workers: normally set to 0
        """

        assert mask_type in ['square', 'blob', 'brush', 'mix']
        if mask_type in ['blob', 'mix']:
            self.total_blob_masks = get_blob_masks(blob_masks_path)
        else:
            self.total_blob_masks = None

        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = RetinalDataset(
            which_set=which_set
        )
        self.n_samples = len(self.dataset)
        self.image_size = image_size

        # set shuffle of dataset
        if self.dataset.training:
            self.shuffle = True
        else:
            self.shuffle = False

        super(RetinalDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, # batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(full_seg_collate_fn,
                batch_size=self.batch_size,
                mask_type=mask_type,
                image_size=image_size,
                total_blob_masks=self.total_blob_masks,
                training=self.dataset.training
        ))

