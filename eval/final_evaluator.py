from base import BaseEvaluator
import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

dirname = os.path.dirname(__file__)
main_dirname = os.path.dirname(dirname)
chickpea_valid_path = os.path.join(main_dirname, 'output/Unet-randomization/0312_075622/testing/chickpea-full-image/binary_input/')

import data_loader.data_loaders as module_data
from utils.util import ensure_dir
from utils.data_processing import convert_labels_to_rgb, remove_artifacts, inpaint_full_image, get_files
from skimage.measure import compare_mse, label


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class UnetEvaluator(BaseEvaluator):
    def __init__(self, model, config):
        super(UnetEvaluator, self).__init__(model, config)

        self.syn_test_dataloader = module_data.TestRootDataLoader(name='synthetic')
        self.real_test_dataloader = module_data.TestRootDataLoader(name='chickpea')
        self.chickpea_test_file_list = get_files(chickpea_valid_path)

        self.testing_dir = os.path.join(self.config["checkpoint_dir"], 'testing')
        ensure_dir(self.testing_dir)

    def evaluate_valid(self):
        print("synthetic valid patches")
        self.evaluate_patches(
            dataloader=self.syn_valid_dataloader,
            save_dir=self.validation_dir,
            save_name='synthetic-patch')

        print("real valid patches")
        self.evaluate_patches(
            dataloader=self.real_valid_dataloader,
            save_dir=self.validation_dir,
            save_name='chickpea-patch')

        print("real valid full images")
        self.evaluate_full_image_list(
            file_list=self.chickpea_valid_file_list[:40],
            save_dir=self.validation_dir,
            save_name='chickpea-full-image')

    def evaluate_test(self):
        print("synthetic test patches")
        self.evaluate_patches(
            dataloader=self.syn_test_dataloader,
            save_dir=self.testing_dir,
            save_name='synthetic-patch')

        print("real test patches")
        self.evaluate_patches(
            dataloader=self.real_test_dataloader,
            save_dir=self.testing_dir,
            save_name='chickpea-patch')

        print("real test full images")
        self.evaluate_full_image_list(
            file_list=self.chickpea_test_file_list,
            save_dir=self.testing_dir,
            save_name='chickpea-full-image')

    def evaluate(self):
        # self.evaluate_valid()
        self.evaluate_test()

    def get_full_image_example_dir_names(self):
        return ['binary_input', 'binary_prediction', 'binary_prediction_rm',
                'labeled_input', 'labeled_prediction',
                'unthresh_prediction', 'rgb_prediction', 'labeled_prediction_rm']

    def get_full_image_metric_names(self):
        return ['num_labels_input', 'num_labels_pred', 'num_labels_pred_rm']

    def evaluate_patches(self, dataloader, save_dir, save_name):
        patch_metric_names = self.get_patch_metric_names()
        metrics = {n: [] for n in patch_metric_names}
        examples = {'input_images': [], 'target_images': [], 'unthresh_pred_images': [], 'pred_images': []}
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):

                outputs = self.model.inference(inputs)

                inputs = inputs.cpu().numpy()
                inputs = inputs.transpose((0, 2, 3, 1))
                outputs = outputs.cpu().numpy()
                outputs = outputs.transpose((0, 2, 3, 1))

                # squeeze batch size dimension
                input_image = np.squeeze(inputs, axis=0)
                target_image = np.squeeze(targets, axis=0)
                outputs = np.squeeze(outputs, axis=0)
                unthresh_predict_image = outputs.copy()

                # get mask
                masks = np.subtract(target_image, input_image).astype(np.uint8)

                # threshold prediction
                # predict_image = unthresh_predict_image.copy()
                mask = (masks != 0)
                predict_image = input_image.copy()
                predict_image[mask] = unthresh_predict_image[:, :, 1:][mask]
                predict_image = np.concatenate((predict_image, predict_image), axis=-1)
                predict_image[predict_image >= 0.5] = 1
                predict_image[predict_image < 0.5] = 0

                examples["input_images"].append(input_image[..., 0])
                examples["target_images"].append(target_image[..., 0])
                examples["unthresh_pred_images"].append(unthresh_predict_image[..., 1])
                examples["pred_images"].append(predict_image[..., 1])

                if np.sum(mask) == 0:
                    continue
                    # compute overall mse
                metrics["mse_overall_pred"].append(
                    compare_mse(predict_image[..., 1:].astype(np.uint8), target_image.astype(np.uint8)))
                # compute mse within gaps
                metrics["mse_within_gaps_pred"].append(np.sum(np.square(
                    np.subtract(np.multiply(predict_image[..., :1].astype(np.uint8), mask),
                                np.multiply(target_image[..., :1].astype(np.uint8), mask)))) / np.sum(mask))

                # compute overall mse
                mse_overall_input = compare_mse(input_image[..., :1].astype(np.uint8), target_image.astype(np.uint8))
                metrics["mse_overall_input"].append(mse_overall_input)
                # compute mse within gaps
                metrics["mse_within_gaps_input"].append(np.sum(np.square(
                    np.subtract(input_image[..., :1].astype(np.uint8), target_image.astype(np.uint8)))) / np.sum(mask))

                # calculate number of pixels
                diff_num_pix_input_target = np.sum(np.abs(mask))
                metrics["diff_num_pix_input_target"].append(diff_num_pix_input_target)
                diff_num_pix_pred_target = np.sum(np.abs(np.subtract(target_image, predict_image[..., 1:])))
                metrics["diff_num_pix_pred_target"].append(diff_num_pix_pred_target)

                # calculate number of fully connected components
                _, num_labels_input = label(input_image[..., 0].astype(np.uint8), neighbors=8, background=0, return_num=True)
                _, num_labels_target = label(target_image.astype(np.uint8), neighbors=8, background=0, return_num=True)
                _, num_labels_pred = label(predict_image[..., 1].astype(np.uint8), neighbors=8, background=0,
                                           return_num=True)
                metrics["diff_num_labels_input_target"].append(np.abs(num_labels_input - num_labels_target))
                metrics["diff_num_labels_pred_target"].append(np.abs(num_labels_pred - num_labels_target))

            _ = self.save_dict_2_csvs(metrics, save_dir, save_name)
            _ = self.save_patch_examples(examples, save_dir, save_name)

    def evaluate_full_image_list(self, file_list, save_dir, save_name):
        sub_dirs = self.get_full_image_example_dir_names()
        dirs = []
        for d in sub_dirs:
            if 'target' not in d:
                path = os.path.join(save_dir, 'chickpea-full-image', d)
                ensure_dir(path)
                dirs.append(path)

        metric_names = self.get_full_image_metric_names()
        metrics = {n:[] for n in metric_names}

        with torch.no_grad():
            for i, file in enumerate(tqdm(file_list)):
                image = cv2.imread(file, 0)
                if image.shape[0] > 3200 or image.shape[1] > 3200:
                   continue
                image = remove_artifacts(image, 10)

                resized_img, binary_inpainted, rgb_inpainted, unthresh_inpainted = inpaint_full_image(image, self.model, 50)

                remove_binary_inptined = remove_artifacts(binary_inpainted, 10)

                labeled_input, num_labels_input = label((resized_img/255.).astype(np.uint8), neighbors=8, background=0, return_num=True)
                labeled_input = convert_labels_to_rgb(labeled_input)
                labeled_pred, num_labels_pred = label((binary_inpainted/255.).astype(np.uint8), neighbors=8, background=0, return_num=True)
                labeled_pred = convert_labels_to_rgb(labeled_pred)
                labeled_pred_rm, num_labels_pred = label((remove_binary_inptined/255.).astype(np.uint8), neighbors=8, background=0, return_num=True)
                labeled_pred_rm = convert_labels_to_rgb(labeled_pred_rm)

                images = [resized_img, binary_inpainted, remove_binary_inptined, labeled_input, labeled_pred, unthresh_inpainted, rgb_inpainted, labeled_pred_rm]
                metrics["num_labels_input"].append(num_labels_input)
                metrics["num_labels_pred"].append(num_labels_pred)
                metrics["num_labels_pred_rm"].append(num_labels_pred)

                for save_path, image in zip(dirs, images):
                    cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), image.astype(np.uint8))


        df = pd.DataFrame(metrics, columns=metrics.keys())
        df.to_csv(save_dir + '/' + save_name + '.csv')
        df.describe().to_csv(save_dir + '/' + save_name + '-stats.csv')
