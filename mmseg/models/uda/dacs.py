# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, generate_class_mask,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg, target_night_img,
                      target_night_img_metas, target_day_img,
                      target_day_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True, use_l1Loss=False)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label for target_night
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits_night = self.get_ema_model().encode_decode(
            target_night_img, target_night_img_metas)

        ema_softmax_night = torch.softmax(ema_logits_night.detach(), dim=1)
        pseudo_prob_night, pseudo_label_night = torch.max(ema_softmax_night, dim=1)
        ps_large_p_night = pseudo_prob_night.ge(self.pseudo_threshold).long() == 1
        ps_size_night = np.size(np.array(pseudo_label_night.cpu()))
        pseudo_weight_night = torch.sum(ps_large_p_night).item() / ps_size_night
        pseudo_weight_night = pseudo_weight_night * torch.ones(
            pseudo_prob_night.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight_night[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight_night[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight_night = torch.ones((pseudo_weight_night.shape), device=dev)


        # Generate pseudo-label for target_day
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits_day = self.get_ema_model().encode_decode(
            target_day_img, target_day_img_metas)

        ema_softmax_day = torch.softmax(ema_logits_day.detach(), dim=1)
        pseudo_prob_day, pseudo_label_day = torch.max(ema_softmax_day, dim=1)
        ps_large_p_day = pseudo_prob_day.ge(self.pseudo_threshold).long() == 1
        ps_size_day = np.size(np.array(pseudo_label_day.cpu()))
        pseudo_weight_day = torch.sum(ps_large_p_day).item() / ps_size_day
        pseudo_weight_day = pseudo_weight_day * torch.ones(
            pseudo_prob_day.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight_day[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight_day[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight_day = torch.ones((pseudo_weight_day.shape), device=dev)

        # Apply mixing (source + night)
        mixed_img_night, mixed_lbl_night = [None] * batch_size, [None] * batch_size
        mix_masks_night = get_class_masks(gt_semantic_seg)

        #img[i].shape: ([3,512,512]) gt_semantic_seg[i].shape:([1,512,512]) gt_semantic_seg[i][0].shape:([512,512]) 
        #pseudo_label_night[i].shape:([512,512]) pseudo_weight_night[i].shape:([512,512])
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks_night[i]
            mixed_img_night[i], mixed_lbl_night[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_night_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label_night[i])))
            _, pseudo_weight_night[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight_night[i], pseudo_weight_night[i])))
        mixed_img_night = torch.cat(mixed_img_night)
        mixed_lbl_night = torch.cat(mixed_lbl_night)


        # Apply mixing (day + night)
        # 对应文档3.5.6里的bi-mix的步骤1
        logits_T_d = self.get_ema_model().encode_decode(
            target_day_img, target_day_img_metas)
        #logits_T_d.shape: ([2,19,512,512])
        pseudo_label_d = torch.softmax(logits_T_d.detach(), dim=1)
        #pseudo_label_d: ([2,19,512,512])
        mixed_img_day, mixed_lbl_day = [None] * batch_size, [None] * batch_size
        mix_masks_day = get_class_masks(gt_semantic_seg)
        max_probs_d, pred_d = torch.max(pseudo_label_d, dim=1) #得到了伪标签

        for i in range(batch_size):
            #对应文档3.5.6里的bi-mix的步骤2
            classes = torch.unique(pred_d[i])
            #pred_d[i]: ([512,512])
            ##dynamic pick up
            index1=30>classes
            index2=classes>11
            index=index1&index2
            classes = (classes[index]).cuda()
            #对应文档3.5.6里的bi-mix的步骤3
            if i == 0: #对每个batch的第一张图
                #MixMask0_d2n: ([1,1,512,512])
                MixMask0_d2n = generate_class_mask(pred_d[i], classes).unsqueeze(0).cuda()
            else:
                MixMask1_d2n = generate_class_mask(pred_d[i], classes).unsqueeze(0).cuda()

        strong_parameters['mix'] = MixMask0_d2n
        image_M_d2n0, _ = strong_transform(strong_parameters,
                                              data=torch.cat((target_day_img[0].unsqueeze(0), target_night_img[0].unsqueeze(0))))
        #target_day_img[0].unsqueeze(0).shape:([1,3,512,512])
        strong_parameters["Mix"] = MixMask1_d2n
        image_M_d2n1, _ = strong_transform(strong_parameters,
                                              data=torch.cat((target_day_img[1].unsqueeze(0), target_night_img[1].unsqueeze(0))))
        mixed_img_dayWithNight = torch.cat((image_M_d2n0, image_M_d2n1))
        #mixed_img_dayWithNight.shape:([2,3,512,512])


        # Train on mixed images (source + night)
        mix_losses_night = self.get_model().forward_train(
            mixed_img_night, img_metas, mixed_lbl_night, target_day_img, pseudo_weight_night, return_feat=True, use_l1Loss=False)
        mix_losses_night.pop('features')
        mix_losses_night = add_prefix(mix_losses_night, 'mix')
        mix_loss_night, mix_log_vars_night = self._parse_losses(mix_losses_night)
        log_vars.update(mix_log_vars_night)
        mix_loss_night.backward()

        # Train on mixed images (day + night)
        mix_losses_dayWithNight = self.get_model().forward_train(
            mixed_img_night, img_metas, mixed_lbl_night, target_day_img , return_feat=True, use_l1Loss=True)
        # mix_losses_dayWithNight.pop('features')
        # loss_l1 = nn.L1Loss()
        # mix_losses_dayWithNight = loss_l1(r_mix, r)
        mix_losses_dayWithNight = 0.001*mix_losses_dayWithNight
        # mix_losses_dayWithNight = add_prefix(mix_losses_dayWithNight, 'mix')
        # mix_losses_dayWithNight, mix_log_vars_dayWithNight = self._parse_losses(mix_losses_dayWithNight)
        # log_vars.update(mix_losses_dayWithNight)
        mix_losses_dayWithNight.backward()
        # mix_losses_day = self.get_model().forward_train(
        #     mixed_img_day, img_metas, mixed_lbl_day, pseudo_weight_day, return_feat=True)
        # mix_losses_day.pop('features')
        # mix_losses_day = add_prefix(mix_losses_day, 'mix')
        # mix_loss_day, mix_log_vars_day = self._parse_losses(mix_losses_day)
        # log_vars.update(mix_log_vars_day)
        # mix_loss_day.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_night_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img_night, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label_night[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks_night[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl_night[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight_night[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
