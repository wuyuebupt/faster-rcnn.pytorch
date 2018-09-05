from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # print (gt_boxes.size())
        gt_boxes_append = gt_boxes[:,:,:5].new(gt_boxes[:,:,:5].size()).zero_()
        # print (gt_boxes_append.size())
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights, tracking_bbox_targets, tracking_bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        # outside * loss(inside * bbox) 
        # outside is kind of bool
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        tracking_bbox_outside_weights = (tracking_bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, tracking_bbox_targets, tracking_bbox_inside_weights, tracking_bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    # have to match the label and have tracking target
    def _get_tracking_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes, tracking_gt_box_flag):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
         
            # inds = torch.nonzero(clss[b] > 0).view(-1)
            # tracking_inds = torch.nonzero(tracking_gt_box_flag[b] > 0).view(-1)
            # print (inds, tracking_inds)
            ## clss[b] = 128 * [[0:30]] for 31 classes
            ## tracking_gt_box_flag = 128 * [(0 or 1)] for if the tracking target exists
            both_cls_tracking = clss[b] * tracking_gt_box_flag[b]    
            inds_cls_tracking = torch.nonzero(both_cls_tracking).view(-1)
            # print (inds_cls_tracking)
            # for i in range(inds.numel()):
            for i in range(inds_cls_tracking.numel()):
                ind = inds_cls_tracking[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            # inds -> all positive rois
            inds = torch.nonzero(clss[b] > 0).view(-1)
            # be more strict 

            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
          
        # print (overlaps)
        # print (max_overlaps)
        # print (gt_assignment)
        # exit()

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)
        # print (batch_size, num_proposal, num_boxes_per_img)

        # print (gt_boxes.size(1)) 
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1).index(offset.view(-1))\
                                                            .view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        labels_batch_tracking = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        tracking_gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            ## have to do three parts
            ## [tracking : 1]
            ## [0.5      : tracking]
            ## [0.0      : 0.1]
            assert (cfg.TRAIN.FG_THRESH_TRACKING >= cfg.TRAIN.FG_THRESH)

            fg_tracking_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH_TRACKING).view(-1)
            fg_tracking_num_rois = fg_tracking_inds.numel()
            # print(fg_tracking_inds)
            # print(fg_tracking_num_rois)
            # exit()


            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()
            # print(fg_tracking_inds)
            # print(fg_num_rois)
            # exit()


            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            # print (bg_inds)
            bg_num_rois = bg_inds.numel()
            # print(bg_num_rois)
            # exit()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # this should be the most case 
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                # print (fg_inds)
                # print (fg_tracking_inds)
                # exit()
                # have to further selected the tracking target with higher threshold later 
                # keep labels the same, keep sampling the same, no change here

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # print (keep_inds)
            # print (keep_inds.shape)

            # for j in range(keep_inds.shape[0]):
            #    print ((keep_inds[j] == fg_tracking_inds).nonzero())

            # for fg_tracking_ind in range(fg_tracking_num_rois):
            #     print ((keep_inds == fg_tracking_inds[fg_tracking_ind]).nonzero())

            # print ((keep_inds == fg_tracking_inds[fg_tracking_ind]))
            ## find positive samples for tracking
            # exit()

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            # print (labels_batch)
            # labels_batch_tracking[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
                # labels_batch_tracking[i][fg_rois_per_this_image:] = 0

            # change labels for tracking target
            for fg_tracking_ind in range(fg_tracking_num_rois):
                # print ((keep_inds == fg_tracking_inds[fg_tracking_ind]).nonzero())
                tracking_ind = torch.nonzero(keep_inds == fg_tracking_inds[fg_tracking_ind]).view(-1)
                tracking_number = tracking_ind.numel()
                assert(tracking_number <= 1)
                for j in range(tracking_number):
                    labels_batch_tracking[i][tracking_ind[j]] = labels_batch[i][tracking_ind[j]]


            # print (tracking_ind)
            # print (labels_batch)
            # print (labels_batch_tracking)
            # exit()

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            # the first 5 are for the original gt
            # print (gt_boxes[i][gt_assignment[i][keep_inds]].shape)
            # 128*10
            # detection gt is from 0 to 5
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]][:,:5]
            # gt_assign for the next
            # tracking target if from 5 to 10
            tracking_gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]][:,5:]

        # print(tracking_gt_rois_batch)
        # print(tracking_gt_rois_batch.shape)
        # compute target data and bbox tartget
        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        # similarly, we have to generate the [tracking]_traget, leave it for tomorrow....
        # 
        tracking_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], tracking_gt_rois_batch[:,:,:4])
        # print(tracking_target_data)
        # print(tracking_target_data.shape)
        # print(labels_batch)
        # print(labels_batch.shape)
        # exit()

        # print (gt_rois_batch[:,:,4])
        # print (tracking_gt_rois_batch[:,:,4])
        tracking_bbox_targets, tracking_bbox_inside_weights = \
                self._get_tracking_bbox_regression_labels_pytorch(tracking_target_data, labels_batch_tracking, num_classes, tracking_gt_rois_batch[:,:,4])
        # tracking_bbox_targets, tracking_bbox_inside_weights = \
                # self._get_tracking_bbox_regression_labels_pytorch(tracking_target_data, labels_batch, num_classes, tracking_gt_rois_batch[:,:,4])
        # print (tracking_bbox_inside_weights)
        # print (tracking_bbox_inside_weights.shape)
        # exit()


        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights, tracking_bbox_targets, tracking_bbox_inside_weights
        # return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
