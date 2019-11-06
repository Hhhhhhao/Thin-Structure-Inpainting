import os
import numpy as np
import torch
from base import BaseGANTrainer
from matplotlib import pyplot as plt
from utils.util import ensure_dir
from utils.data_processing import extract_patch_from_tensor
from data_loader.data_loaders import full_seg_collate_fn


plt.switch_backend('agg')


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class GANTrainer(BaseGANTrainer):
    """
    GAN Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, models, optimizers, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):

        super(GANTrainer, self).__init__(
            models,
            optimizers,
            loss=loss,
            metrics=metrics,
            resume=resume,
            config=config,
            train_logger=train_logger)

        self.config = config
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 5

        # loss weight for generator cross entropy loss
        self.lambda_1 = self.config["trainer"]["lambda_1"]
        # loss weight for generator roll out loss
        self.lambda_2 = self.config["trainer"]["lambda_2"]
        self.lambda_3 = self.config["trainer"]["lambda_3"]

    def _eval_metrics(self, outputs, targets):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(outputs, targets)
            # self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_generator_epoch(self, epoch):
        """
        Pre training logic for an epoch
        :param epoch: Current training epoch
        :return: A log that contrains all information you want to save
        Note:
        If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The m
            etrics in log must have the key 'metrics'.
        """

        self.generator.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (inputs, targets, _, _, _, _) in enumerate(self.train_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.generator_optimizer.zero_grad()
            outputs, _ = self.generator(inputs)
            loss = self.loss["ce"](outputs, targets)
            loss.backward()
            self.generator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    loss.item()))
                # self.writer.add_image('inputs', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'Generator_CrossEntropyLoss': total_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }

        return log

    def _train_discriminator_epoch(self, epoch):
        """
        Pre training logic for an epoch
        :param epoch: Current training epoch
        :return: A log that contrains all information you want to save
        Note:
        If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.discriminator.train()
        self.generator.eval()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (inputs, targets, _, _, _) in enumerate(self.train_data_loader):

            if batch_idx == int(len(self.train_data_loader) / 4):
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            true_labels = torch.ones((batch_size, 1)).to(self.device)
            fake_labels = -torch.ones((batch_size, 1)).to(self.device)

            self.discriminator_optimizer.zero_grad()
            outputs = self.generator(inputs)
            list_X = extract_patch_from_tensor(outputs[:, :1, :, :].detach(), patch_size=(128, 128))
            list_Y = extract_patch_from_tensor(targets.unsqueeze(1).type(torch.cuda.FloatTensor), patch_size=(128, 128))

            discriminator_loss = 0
            for X, Y in zip(list_X, list_Y):
                generated_labels = self.discriminator(X)
                generated_labels = generated_labels.view(batch_size, 1)
                generated_loss = self.loss["lsgan"](generated_labels, fake_labels)
                target_labels = self.discriminator(Y)
                target_labels = target_labels.view(batch_size, 1)
                target_loss = self.loss["lsgan"](target_labels, true_labels)
                discriminator_loss += generated_loss + target_loss

            discriminator_loss = discriminator_loss / len(list_X)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('loss', discriminator_loss.item())
            total_loss += discriminator_loss.item()
            total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Discriminator Pre-Train Epoch: {} [{}/{} ({:.0f}%)] Discriminator Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        self.train_data_loader.n_samples,
                        100.0 * batch_idx / len(self.train_data_loader),
                        discriminator_loss.item()))

        log = {
            'Discriminator_Loss': total_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }

        return log

    def forward(self, batch_x):
        output, prob = self.generator(batch_x)
        return output, prob

    def backward_local_D(self, generator_outputs, ground_truths, real_labels, fake_labels):
        list_X = extract_patch_from_tensor(generator_outputs.detach(), patch_size=(128, 128))
        # todo add CUDA
        list_Y = extract_patch_from_tensor(ground_truths.unsqueeze(1).type(torch.cuda.FloatTensor),
                                           patch_size=(128, 128))

        discriminator_loss = 0
        for X, Y in zip(list_X, list_Y):
            generated_labels = self.local_discriminator(X)
            generated_labels = generated_labels.view(generated_labels.size(0), 1)
            generated_loss = self.loss["lsgan"](generated_labels, fake_labels)
            target_labels = self.local_discriminator(Y)
            target_labels = target_labels.view(target_labels.size(0), 1)
            target_loss = self.loss["lsgan"](target_labels, real_labels)
            discriminator_loss += generated_loss + target_loss

        discriminator_loss = discriminator_loss / len(list_X)
        discriminator_loss.backward()
        return discriminator_loss.item()

    def backward_global_D(self, other_inputs, outputs, targets, other_targets, real_labels, fake_labels):
        # batch_size = outputs.size(0)
        # other_inputs, other_targets, _, _, _, _ = self.prepare_patches(batch_size)
        outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
        dissimilarity_1 = self.global_discriminator(other_inputs, targets.unsqueeze(1).type(torch.cuda.FloatTensor))
        dissimilarity_2 = self.global_discriminator(outputs, targets.unsqueeze(1).type(torch.cuda.FloatTensor))
        similarity = self.global_discriminator(other_targets.unsqueeze(1).type(torch.cuda.FloatTensor),
                                                   targets.unsqueeze(1).type(torch.cuda.FloatTensor))
        dissimilarity_loss_1 = self.loss["vanilla_gan"](dissimilarity_1, fake_labels)
        dissimilarity_loss_2 = self.loss["vanilla_gan"](dissimilarity_2, fake_labels)
        similarity_loss = self.loss["vanilla_gan"](similarity, real_labels)
        discriminator_loss = 0.5 * (dissimilarity_loss_1 + dissimilarity_loss_2) + similarity_loss
        discriminator_loss.backward()

        return discriminator_loss.item()

    def backward_G(self, outputs, probs, targets, locations, orig_window_length, full_image, other_full_image,
                   real_labels):
        log_probs, rewards = self.global_discriminator.reward_forward(probs, locations, orig_window_length, full_image,
                                                           other_full_image)
        generator_global_loss = self.loss["pg"](rewards, log_probs)

        # outputs = self.generator(inputs)
        list_G = extract_patch_from_tensor(probs, patch_size=(128, 128))
        generator_local_loss = 0
        for G in list_G:
            generated_labels = self.local_discriminator(G)
            generator_local_loss += self.loss["lsgan"](generated_labels, real_labels)
        generator_local_loss = generator_local_loss / len(list_G)
        generator_cce_loss = self.loss["ce"](outputs, targets)

        generator_loss = self.lambda_1 * generator_cce_loss + self.lambda_2 * generator_local_loss + self.lambda_3 * generator_global_loss
        generator_loss.backward()

        return generator_loss.item(), generator_cce_loss.item(), generator_local_loss.item(), generator_global_loss.item()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.generator.train()
        self.local_discriminator.train()
        self.global_discriminator.train()

        total_generator_cce_loss = 0
        total_generator_mask_cce_loss = 0
        total_generator_local_loss = 0
        total_generator_global_loss = 0
        total_generator_loss = 0
        total_discriminator_local_loss = 0
        total_discriminator_global_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (inputs, targets, masks, locations, orig_window_length, full_image) in enumerate(
                self.train_data_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            real_labels = torch.ones((batch_size, 1)).to(self.device)
            fake_labels = torch.zeros((batch_size, 1)).to(self.device)
            other_inputs, other_targets, _, _, _, other_full_image = self.prepare_patches(batch_size)

            # forward G
            outputs, probs = self.forward(inputs)
            probs = probs[:, 1:, :, :]
            # probs = self.fill_inputs(probs[:, 1: , :, :], inputs, masks)

            # fix weigths for training discriminator
            for p in self.local_discriminator.parameters():
                p.require_grad = True
            # fix weigths for training discriminator
            for p in self.global_discriminator.parameters():
                p.require_grad = True

            # train the discriminator
            self.local_discriminator_optimizer.zero_grad()
            self.global_discriminator_optimizer.zero_grad()
            local_discriminator_loss = self.backward_local_D(probs.detach(), other_targets, real_labels,
                                                             fake_labels)
            global_discriminator_loss = self.backward_global_D(other_inputs, probs.detach(), targets, other_targets,
                                                               real_labels, fake_labels)
            self.local_discriminator_optimizer.step()
            self.global_discriminator_optimizer.step()

            # fix weigths for training discriminator
            for p in self.local_discriminator.parameters():
                p.require_grad = False
            # fix weigths for training discriminator
            for p in self.global_discriminator.parameters():
                p.require_grad = False

            # backward G
            self.generator_optimizer.zero_grad()
            generator_total_loss, generator_cce_loss, generator_local_loss, generator_global_loss = self.backward_G(
                outputs, probs, targets, locations, orig_window_length, full_image, full_image, real_labels)
            self.generator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('generator_total_loss', generator_total_loss)
            self.writer.add_scalar('generator_crossentropy_loss', generator_cce_loss)
            self.writer.add_scalar('generator_local_loss', generator_local_loss)
            self.writer.add_scalar('generator_global_loss', generator_global_loss)
            self.writer.add_scalar('local_discriminator_loss', local_discriminator_loss)
            self.writer.add_scalar('global_discriminator_loss', global_discriminator_loss)
            total_generator_loss += generator_total_loss
            total_generator_cce_loss += generator_cce_loss
            total_generator_local_loss += generator_local_loss
            total_generator_global_loss += generator_global_loss
            total_discriminator_local_loss += local_discriminator_loss
            total_discriminator_global_loss += global_discriminator_loss
            total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'Generator: [CCE:{:.6f}, Local:{:.6f}, Global:{:.6f}, Total:{:.6f}] '
                                 'Discriminator: [Local:{:.6f}, Global:{:.6f}]'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    generator_cce_loss,
                    generator_local_loss,
                    generator_global_loss,
                    generator_total_loss,
                    local_discriminator_loss,
                    global_discriminator_loss
                ))

        log = {
            'generator_crossentropy_loss': total_generator_cce_loss / len(self.train_data_loader),
            'generator_mask_crossentropy_loss':total_generator_mask_cce_loss / len(self.train_data_loader),
            'generator_local_loss': total_generator_local_loss / len(self.train_data_loader),
            'generator_global_loss': total_generator_global_loss / len(self.train_data_loader),
            'generator_total_loss': total_generator_loss / len(self.train_data_loader),
            'discriminator_local_loss': total_discriminator_local_loss / len(self.train_data_loader),
            'discriminator_global_loss': total_discriminator_global_loss / len(self.train_data_loader),
            'metrics': (total_metrics / len(self.train_data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.generator.eval()
        self.local_discriminator.eval()
        self.global_discriminator.eval()
        total_generator_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, (inputs, targets, _, _, _, _) in enumerate(self.valid_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, probs = self.generator(inputs)
                generator_cce_loss = self.loss["ce"](outputs, targets)

                self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
                self.writer.add_scalar('Generator_CrossEntropyLoss', generator_cce_loss.item())
                total_generator_val_loss += generator_cce_loss.item()
                total_val_metrics += self._eval_metrics(outputs, targets)

        return {
            'generator_val_loss': total_generator_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def save_images(self, generator_outputs, targets, generator_labels, target_labels, epoch, batch_idx, r=1, c=2):
        generator_outputs = generator_outputs.cpu().numpy()
        generator_outputs = generator_outputs.transpose((0, 2, 3, 1))
        generator_outputs = generator_outputs[..., 1]
        generator_outputs[generator_outputs >= 0.5] = 1
        generator_outputs[generator_outputs < 0.5] = 0
        targets = targets.cpu().numpy()
        generator_labels = generator_labels.cpu().numpy()
        target_labels = target_labels.cpu().numpy()

        fig, axs = plt.subplots(r, c)

        if r == 1:
            axs[0].set_title('Fake Disc:{:.2f}'.format(generator_labels[0, 0]))
            axs[0].imshow(generator_outputs[0], cmap='gray')
            axs[0].axis('off')

            axs[1].set_title('Target Disc:{:.2f}'.format(target_labels[0, 0]))
            axs[1].imshow(targets[0], cmap='gray')
            axs[1].axis('off')
        else:
            count = 0
            for row in range(r):
                axs[row, 0].set_title('Fake Disc:{:.1f}'.format(generator_labels[count, 0]))
                axs[row, 0].imshow(generator_outputs[count])
                axs[row, 0].axis('off')

                axs[row, 1].set_title('Target Disc:{:.1f}'.format(target_labels[count, 0]))
                axs[row, 1].imshow(targets[count])
                axs[row, 1].axis('off')
                count += 1

        ensure_dir(os.path.join(self.checkpoint_dir, 'results', 'epoch_{}').format(epoch))
        fig.savefig('{0}/results/epoch_{1}/{2}.jpg'.format(self.checkpoint_dir, epoch, batch_idx))
        plt.close(fig)

    def prepare_patches(self, batch_size):
        index = np.random.randint(0, 5000)
        image = self.train_data_loader.dataset.__getitem__(index)
        batch_x, batch_y, batch_masks, locations, orig_window_length, image = full_seg_collate_fn(
            [image],
            batch_size=batch_size,
            mask_type=self.train_data_loader.mask_type,
            image_size=self.train_data_loader.image_size,
            total_blob_masks=self.train_data_loader.total_blob_masks,
            training=True
        )

        return batch_x, batch_y, batch_masks, locations, orig_window_length, image

    def fill_inputs(self, probs, inputs, batch_masks):
        probability = probs.detach().cpu().numpy()
        corrupted_inputs = inputs.detach().cpu().numpy()
        masks = batch_masks.detach().cpu().numpy()
        masks = masks[:, :1, :, :]
        mask = (masks != 0)
        predictions = corrupted_inputs.copy()
        predictions[mask] = probability[mask]
        # corrupted_inputs[probability >= 0.5] = 1
        predictions = torch.FloatTensor(predictions)
        predictions = predictions.to(self.device)
        return predictions
