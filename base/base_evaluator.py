import os
import pandas as pd
import numpy as np
import cv2
from utils.util import ensure_dir
from utils.data_processing import convert_labels_to_rgb
from skimage.measure import label


class BaseEvaluator(object):
    """
    Base class for all evaluators
    """
    def __init__(self, model, config):
        self.model = model
        self.model.eval()
        self.config = config

        # create validation and testing directory in model checkpoints directory
        self.validation_dir = os.path.join(self.config["checkpoint_dir"], 'validation')
        ensure_dir(self.validation_dir)
        self.testing_dir = os.path.join(self.config["checkpoint_dir"], 'testing')
        ensure_dir(self.testing_dir)

    def get_patch_example_dir_names(self):
        """
        Obtain the patch example saving folder names
        :return: a list of the folder names
        """
        return ['binary_input', 'binary_target', 'binary_prediction',
                'labeled_input', 'labeled_target', 'labeled_prediction',
                'unthresh_prediction']

    def get_full_image_example_dir_names(self):
        """
        Obtain the whole mask example saving folder names
        :return: a list of folder names
        """
        return ['binary_input', 'binary_target', 'binary_prediction',
                'labeled_input', 'labeled_target', 'labeled_prediction',
                'unthresh_prediction', 'rgb_prediction']

    def get_patch_metric_names(self):
        """
        Obtain the metric names for patch-level evaluation
        :return: a list of metric names
        """
        return ['mse_overall_input', 'mse_within_gaps_input', 'mse_overall_pred', 'mse_within_gaps_pred', 'diff_num_pix_input_target', 'diff_num_pix_pred_target',
                'diff_num_labels_input_target', 'diff_num_labels_pred_target']

    def save_dict_2_csvs(self, dict, save_dir, save_name, contain_inf=True):
        """
        Save dictionary into csv files via pandas
        :param dict: a dictionary contains the raw evaluation
        :param save_dir: saving directory
        :param save_name: saving name
        :param contain_inf: flag of whether or not to remove the inf and nan values
        :return: the dataframe version of the dictionary
        """
        df = pd.DataFrame(dict, columns=dict.keys())

        if contain_inf:
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

        df.to_csv(save_dir + '/' + save_name + '.csv')
        df.describe().to_csv(save_dir + '/' + save_name + '-stats.csv')

        return df

    def save_patch_examples(self, example_dict, save_dir, save_name):
        """
        Save patch examples saved in example dictionary
        :param example_dict: a dictionary contains patch examples
        :param save_dir: saving directory
        :param save_name: saving name
        :return: None
        """

        # create sub folders
        sub_dirs = self.get_patch_example_dir_names()
        dirs = []
        for d in sub_dirs:
            path = os.path.join(save_dir, save_name, d)
            ensure_dir(path)
            dirs.append(path)

        # obtain image lists
        input_images = example_dict["input_images"]
        target_images = example_dict["target_images"]
        pred_images = example_dict["pred_images"]
        unthresh_pred_images = example_dict["unthresh_pred_images"]

        i = 0
        for input, target, unthresh_pred, pred in zip(input_images, target_images, unthresh_pred_images, pred_images):
            # convert binary images into colourful labels
            labeled_input = label(input.astype(np.uint8), neighbors=8, background=0)
            labeled_input = convert_labels_to_rgb(labeled_input)
            labeled_target = label(target.astype(np.uint8), neighbors=8, background=0)
            labeled_target = convert_labels_to_rgb(labeled_target)
            labeled_pred = label(pred.astype(np.uint8), neighbors=8, background=0)
            labeled_pred = convert_labels_to_rgb(labeled_pred)

            images = [input, target, pred, labeled_input, labeled_target, labeled_pred, unthresh_pred]

            # save images
            for save_path, image in zip(dirs, images):
                cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), (image*255).astype(np.uint8))
            i+=1

    def evaluate_patches(self, dataloader, save_dir, save_name):
        raise NotImplementedError

    def evaluate_full_image_list(self, file_list, save_dir, save_name):
        raise NotImplementedError

    def evaluate_valid(self):
        pass

    def evaluate_test(self):
        pass

