import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from torch.nn.parameter import Parameter
import math
import sys

from model.rpn.bbox_transform import bbox_overlaps_batch

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.reduce_dimension = cfg.REDUCE_DIMENSION
        # new linear prediction
        # self.attention_regression = RelationUnit(2048, 512, 32, self.n_classes) 
        # self.attention_regression = RelationUnit(2048, 256, 32, self.n_classes) 
        self.boundary_scale       = cfg.NEIGHBOR_MOVE 
        self.circle_neighbor      = cfg.CIRCLE
        self.cls_neighbor         = cfg.CLS_NEIGHBOR
        self.cls_reduce_d         = cfg.CLS_REDUCE_D
        self.reg_neighbor         = cfg.REG_NEIGHBOR
        self.reg_reduce_d         = cfg.REG_REDUCE_D

        self.alpha_same_with_beta = cfg.ALPHA_SAME_WITH_BETA
        self.sigma_geometry       = cfg.SIGMA_GEOMETRY
        self.cls_alpha_option     = cfg.CLS_ALPHA_OPTION


        # print ("boundary move      : ",self.boundary_scale)
        # print ("circle             : ",self.circle_neighbor)
        # print ("reduce_dimension   : ",self.reduce_dimension)

        self.attention_regression = RelationUnit(2048, self.reduce_dimension, 32, self.n_classes, self.cls_reduce_d, self.reg_reduce_d, self.alpha_same_with_beta) 

        # self.attention_regression = RelationUnit(2048, 32) 
        # self.attention_regression = RelationUnit(512, 32) 
        # self.attention_regression = RelationUnit(2048, 64) 
        # self.attention_regression = RelationUnit(256, 64) 
        # self.attention_regression = RelationUnit(128, 16) 
        # self.attention_regression = RelationUnit(1024, 64) 

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    # def forward(self, im_data, im_info, gt_boxes, num_boxes):
    def forward(self, im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        proposal_boxes = proposal_boxes.data
        num_proposals = num_proposals.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        # rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, gt_rois = roi_data
            # print (gt_boxes.shape)
            # print (rois.shape)
            # exit()
            # print (rois.shape)
            # print (rois_target.shape)
            # # print (gt_rois)
            # print (gt_rois.shape)
            # exit()

            ## v0.5
            # print (rois_target.size(2))
            rois_target     = rois_target.view(-1, rois_target.size(2))
            rois_inside_ws  = rois_inside_ws.view(-1, rois_inside_ws.size(2))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))           
            neighbor_num = 9
            neighbor_rois_target     = rois_target.expand(neighbor_num, rois_target.shape[0], rois_target.shape[1])
            neighbor_rois_inside_ws  = rois_inside_ws.expand(neighbor_num, rois_inside_ws.shape[0], rois_inside_ws.shape[1])
            neighbor_rois_outside_ws = rois_outside_ws.expand(neighbor_num, rois_outside_ws.shape[0], rois_outside_ws.shape[1] )
            # print (neighbor_target)
            neighbor_rois_target = Variable(neighbor_rois_target)
            neighbor_rois_inside_ws = Variable(neighbor_rois_inside_ws)
            neighbor_rois_outside_ws = Variable(neighbor_rois_outside_ws)
            ## end v0.5:
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target)
            rois_inside_ws = Variable(rois_inside_ws)
            rois_outside_ws = Variable(rois_outside_ws)
            # gt_rois = Variable(gt_rois.view(-1, gt_rois.size(2)))

            # print (gt_rois.shape)
            # exit()
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # print (rois_target.shape)
        ## rois -> to roi neighbors
        ## 
        # print (im_info)
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.5)
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.1)
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.3)
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.1)
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.3)
        # boundary_move_scale = 0.3
        boundary_move_scale = self.boundary_scale
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.5, boundary_move_scale)

        rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info, boundary_move_scale, self.circle_neighbor)
        ## square false
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info, boundary_move_scale, False)

        ## circle true
        # rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info, boundary_move_scale, True)

        # print (rois_attention_candidates.shape)
        # print (rois_attention_candidates)
        # print (delta_rois.shape)
        # print (delta_rois)
        # exit()

        ##### calculate all ious for rois_neighbor and gt_boxes
        ### rois_attention_candidates 9x1x128x5
        ### gt_rois 1x128x5
        ### 
        # print (gt_rois.shape) 
        # print (rois_attention_candidates.shape) 
        ###
        if self.training:
            overlaps_iou_candidates_gt = self._iou_rois_candidates_to_gt(rois_attention_candidates, gt_rois)
        else:
            overlaps_iou_candidates_gt = None
 
        
        # print (overlaps_iou_candidates_gt(:,:,0))
        # print (overlaps_iou_candidates_gt[:,:,0])
        # exit()


        rois_attention_candidates = Variable(rois_attention_candidates)
        # delta_rois = Variable(delta_rois)
        rois = Variable(rois)

        ## gt rois
        if self.training:
            # gt_attention_candidates = self._gt_to_candidates(gt_rois, im_info, boundary_move_scale)
            gt_attention_candidates = self._gt_to_candidates(gt_rois, im_info)
            gt_attention_candidates = Variable(gt_attention_candidates) 
            gt_rois = Variable(gt_rois.view(-1, gt_rois.size(2)))
        else:
            gt_attention_candidates = None
            delta_gt_attention = None

        # do roi pooling based on predicted rois

        # print (base_feat.shape) # 4x1024x38x50


        ### for proposals
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        # print (pooled_feat.shape) # 512x1024x7x7
        
        pooled_feat = self._head_to_tail(pooled_feat)

        ### for neighbors

        # print (pooled_feat.shape)
        # rois_attention_pooled_feat = pooled_feat.data.new(9, pooled_feat.size(0), pooled_feat.size(1)).zero_()
        # rois_attention_pooled_feat = pooled_feat.view(1, pooled_feat.size(0), pooled_feat.size(1)).expand(9, pooled_feat.size(0), pooled_feat.size(1))
        rois_attention_pooled_feat = []
        # print (rois_attention_pooled_feat.shape)

        for i in range(9):
            # print (i)
            # only do the roi align now
            assert(cfg.POOLING_MODE == 'align')
            # print (self.RCNN_roi_align(base_feat, rois_attention_candidates[i,:,:,:].view(-1, 5)).shape)
            pooled_feat_tmp = self.RCNN_roi_align(base_feat, rois_attention_candidates[i, :, :, :].view(-1, 5))
            pooled_feat_tmp = self._head_to_tail(pooled_feat_tmp)
            # v1
            # rois_attention_pooled_feat.append(pooled_feat_tmp)
            # v2
            # pooled_attention_feat = self.RCNN_attention_feat(pooled_feat_tmp)
            # pooled_attention_feat = self.relu(pooled_attention_feat)
            rois_attention_pooled_feat.append(pooled_feat_tmp)
        # print (len(rois_attention_pooled_feat))
        # exit()

        gt_attention_pooled_feat = []

        # import pdb
        # pdb.set_trace()

        ### for gt rois
        if self.training:
            # for i in range(4):
            for i in range(1):
                assert(cfg.POOLING_MODE == 'align')
                # pooled_feat_gt = self.RCNN_roi_align(base_feat, gt_rois.view(-1, 5))
                pooled_feat_gt = self.RCNN_roi_align(base_feat, gt_attention_candidates[i,:,:,:].view(-1, 5))

                # feed pooled features to top model
                # print (pooled_feat.shape) # 512x1024x7x7
                pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                # pooled_attention_feat_gt = self.RCNN_attention_feat(pooled_feat_gt)
                # pooled_attention_feat_gt = self.relu(pooled_feat_gt)

                gt_attention_pooled_feat.append(pooled_feat_gt)
        else:
            pooled_attention_feat_gt=None

        ## do the attention prediction
        # bbox_pred = attention_regression(rois_attention_candidates, rois_attention_pooled_feat)
        # print (self.attention_regression)
        # print (rois_attention_candidates.is_cuda)
        # print (rois_attention_pooled_feat.is_cuda)

        # bbox_pred, wx1, wy1, wx2, wy2, dx1, dy1, dx2, dy2, ox1, oy1, ox2, oy2 = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat) 
        # bbox_pred, bbox_pred_beta, alpha_softmax, beta_softmax = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat, pooled_attention_feat_gt) 
        # bbox_pred, bbox_pred_beta, alpha_softmax, beta_softmax = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat, gt_attention_pooled_feat) 
        bbox_pred, bbox_pred_beta, alpha_softmax, beta_softmax, bbox_pred_offset, bbox_pred_offset_beta, alpha, beta, \
        bbox_cls, alpha_cls_softmax, beta_cls_softmax, alpha_cls, beta_cls = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat, gt_attention_pooled_feat, overlaps_iou_candidates_gt) 
        # print (bbox_cls.shape)
        # exit()


        # print (bbox_pred)
        # print (bbox_pred.shape)
        # print (bbox_pred_offset.shape)
        # print (alpha_softmax.shape)

        # print (wx1.shape)
        # exit()
        ## 
        ## 
        
        # compute bbox offset
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # print (bbox_pred.shape)
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)
        # print (bbox_pred.shape)
        # exit()

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob_proposal = F.softmax(cls_score)

        if self.cls_neighbor: 
            ## neighbor 
            if self.cls_alpha_option == 0:
                cls_score = bbox_cls * alpha_cls_softmax
                cls_prob = torch.nn.Softmax(dim=2)(cls_score)
                cls_prob = torch.sum(cls_prob, 0)
            else:
                cls_score_softmax = torch.nn.Softmax(dim=2)(bbox_cls)
                # if self.circle_neighbor:
                #     cls_proposal = cls_score_softmax[8, :, :]
                # else:
                #     cls_proposal = cls_score_softmax[4, :, :]
                # print (cls_proposal.shape)
                # cls_proposal = cls_score_softmax[0, :, :]
                
                cls_proposal = cls_prob_proposal
                argmax = cls_proposal.max(1)[1]
                # print (argmax)
                mask = torch.nonzero(argmax)
                # print (mask)
                cls_prob_alpha = cls_score_softmax * alpha_cls_softmax
                cls_prob_neighbor = torch.sum(cls_prob_alpha, 0)

                cls_weights = torch.cuda.FloatTensor(cls_proposal.size(0), 1).zero_()
                # print (cls_weights.shape)

                # print (cls_prob_alpha.shape)
                for i in range(mask.numel()):
                    # print (mask[i]) 
                    ind = int(mask[i])
                    cls_weights[ind, 0] = 1.0
                # print (cls_weights)
                cls_weights = Variable(cls_weights)

                prob_neg = 1 - cls_proposal[:,0]
                
                # print (prob_neg.shape)
                # print (prob_neg.shape)
                # print (prob_neg)
                prob_neg = prob_neg.view(300, 1)
                # exit()

                cls_prob_neighbor_ = cls_prob_neighbor * prob_neg
                # print (cls_prob_neighbor_.shape)

                cls_prob = cls_weights * cls_prob_neighbor_ + (1 - cls_weights) * cls_proposal 


                # cls_prob = cls_proposal 
                # exit()
                # cls_prob_alpha = cls_score_softmax * alpha_cls_softmax

                # print (alpha_cls_softmax.shape)
                # print (alpha_cls_softmax[:,2,0])
                # print (cls_score_softmax.shape)
                # print (cls_score_softmax[4,2,:])
                # exit()

                # print (alpha_cls_softmax[:,200,0])
                # print (indices)
                # print (value)

                # print (indices.shape)
                # print (cls_score_softmax.shape)
                # argmax = alpha_cls_softmax.max(0)[1]
                # print (argmax.shape)
                # print (argmax)

                # indices = indices.view(indices.size(1))
                # print (y.shape)
                # print (y.shape)
                # print (y[:,0])
                

                # cls_prob = cls_score_softmax[argmax]
                
                # cls_prob = cls_score_softmax[indices, :, :]
                # print (cls_prob.shape)


                # print ( alpha_softmax[:,0,0])
                # value2, indices2 = torch.max(alpha_softmax, 0)
                # print (value2[:,0])
                # print (value2.shape)
                # print (indices2.shape)
                # print (indices2[:,0])
                # one_hot = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1), alpha_cls_softmax.size(2))
                # one_hot = one_hot.scatter_(0, indices.data, 1)

                # y.scatter_(0, indices.data, 1)
                # value, indices = torch.max(alpha_cls_softmax, 0)
                # indices = indices.permute(1, 0)
                # y = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1))
                # y.zero_()
                # y = y.view(y.size(0), y.size(1), 1)
                # y = Variable(y)
                # cls_prob = cls_score_softmax * y
                # exit()
        else:
            ## proposal
            cls_prob = cls_prob_proposal
       

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_bbox_beta = 0
        RCNN_loss_cls_beta = 0
        KL_loss = 0
        KL_loss_cls = 0
        

        if self.training:
            # classification loss
            # RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            if self.cls_neighbor:
                RCNN_loss_cls = _cross_entropy_neighbor(bbox_cls, alpha_cls_softmax, rois_label, self.cls_alpha_option)
                RCNN_loss_cls_beta  = _cross_entropy_neighbor(bbox_cls, beta_cls_softmax, rois_label, self.cls_alpha_option)
            else:
                RCNN_loss_cls = _cross_entropy_proposal(bbox_cls, rois_label, self.circle_neighbor)
                RCNN_loss_cls_beta  = None

            # print (RCNN_loss_cls_alpha.shape)
            # print (RCNN_loss_cls_beta.shape)
            # sys.exit(0)
            # bounding box regression L1 loss
            # print (bbox_pred)
            # print (rois_target)
            # RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            ## v0.4
            # RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            # print (RCNN_loss_bbox.shape)
            ## v0.5
            if self.reg_neighbor:
                RCNN_loss_bbox = _smooth_l1_loss_neighbor(bbox_pred, neighbor_rois_target, neighbor_rois_inside_ws, neighbor_rois_outside_ws, bbox_pred_offset, alpha_softmax)
                RCNN_loss_bbox_beta = _smooth_l1_loss_neighbor(bbox_pred_beta, neighbor_rois_target, neighbor_rois_inside_ws, neighbor_rois_outside_ws, bbox_pred_offset_beta, beta_softmax)
                # print (RCNN_loss_bbox.shape)
                # exit()

                ## from gt training for beta
                ## v0.2
                # RCNN_loss_bbox_beta = _smooth_l1_loss(bbox_pred_beta, rois_target, rois_inside_ws, rois_outside_ws)
            else:
                ## proposal
                # RCNN_loss_bbox = _smooth_l1_loss_proposal(bbox_pred, neighbor_rois_target, neighbor_rois_inside_ws, neighbor_rois_outside_ws, bbox_pred_offset, self.circle_neighbor)
                if self.circle_neighbor:
                    RCNN_loss_bbox = _smooth_l1_loss(bbox_pred_offset[8, :, :], rois_target, rois_inside_ws, rois_outside_ws)
                else:
                    RCNN_loss_bbox = _smooth_l1_loss(bbox_pred_offset[4, :, :], rois_target, rois_inside_ws, rois_outside_ws)
                # RCNN_loss_bbox = _smooth_l1_loss_proposal(bbox_pred, neighbor_rois_target, neighbor_rois_inside_ws, neighbor_rois_outside_ws, bbox_pred_offset, self.circle_neighbor)

                RCNN_loss_bbox_beta = None
                # raise("Not implement")
                
            ## KL loss between beta and alpha
            # KL_loss = F.kl_div(alpha_softmax, beta_softmax)
            # KL_distance = torch.distributions.kl.kl_divergence(alpha_softmax, beta_softmax)
            # print (KL_distance)
            # KL_loss = _kl_divergence_loss(alpha_softmax, beta_softmax)
            if self.cls_neighbor:
                ## neighbor
                KL_loss_cls = _kl_divergence_loss(alpha_cls, beta_cls)
                # KL_loss_cls = F.mse_loss(alpha_cls_softmax, beta_cls_softmax)
            else:
                ## proposal
                KL_loss_cls = None

            if self.reg_neighbor:
                KL_loss = _kl_divergence_loss(alpha, beta)
            else:
                KL_loss = None


            # print (RCNN_loss_bbox_beta, KL_loss)
            # exit()
            



        # print (RCNN_loss_bbox)
        # exit()

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, wx1, dx1, ox1
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, wx1, wy1, wx2, wy2, dx1, dy1, dx2, dy2, ox1, oy1, ox2, oy2
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

        ## adding beta, and kl divergency
        # return rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label, RCNN_loss_bbox_beta, KL_loss
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, RCNN_loss_bbox_beta, KL_loss
        return rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label, RCNN_loss_bbox_beta, KL_loss, \
                 RCNN_loss_cls_beta, KL_loss_cls

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

        # load pretrained model
        
        # init and fix

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        # init the attention module
        # normal_init(self.attention_regression, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.attention_regression._init_weights()


    def _iou_rois_candidates_to_gt(self, rois_neighbors, gt_rois):
        # rois_neighbors: (K, N, 4) ndarray of float
        # gt_rois       :    (N, 4) ndarray of float
        # ious          : (K, N) ndarray of overlap between boxes and query_boxes
        # print (rois_neighbors.shape)
        # print (gt_rois.shape)


        if self.training:
            ious = np.zeros((rois_neighbors.size(0), rois_neighbors.size(2)), dtype=np.float32)
            # rois_neighbors_reshape = rois_neighbors.view(rois_neighbors.size(0)*rois_neighbors.size(2), rois_neighbors.size(3))
            ## calculate iou
            gt_box_area = (gt_rois[:,:,2] - gt_rois[:,:,0] + 1) * (gt_rois[:,:,3] - gt_rois[:,:,1] + 1)
            # print (gt_box_area.shape)
            rois_area =  (rois_neighbors[:,:,:,2] - rois_neighbors[:,:,:,0] + 1) * (rois_neighbors[:,:,:,3] - rois_neighbors[:,:,:,1] + 1)
            # print (rois_area.shape)

            gt_expand = gt_rois.expand(rois_neighbors.size(0), gt_rois.size(0),  gt_rois.size(1), gt_rois.size(2))
            # print (gt_expand.shape)
            iw = (torch.min( rois_neighbors[ :,:,:,2], gt_expand[:,:,:,2] ) - torch.max(rois_neighbors[ :,:,:,0], gt_expand[:,:,:,0]) + 1 )
            iw[iw < 0] = 0
            ih = (torch.min( rois_neighbors[ :,:,:,3], gt_expand[:,:,:,3] ) - torch.max(rois_neighbors[ :,:,:,1], gt_expand[:,:,:,1]) + 1 )
            ih[ih < 0] = 0

            area = iw * ih
            # print (area.shape)
            gt_box_area_expand = gt_box_area.expand(rois_neighbors.size(0), gt_box_area.size(0), gt_box_area.size(1))
            # print(gt_box_area_expand.shape)
            ua = rois_area + gt_box_area_expand - area
            ious = area / ua
            # print (ious.shape)
            # print (gt_expand.size(0))
            # print (gt_expand.size(2))
            # print (gt_expand.size(3))
            # gt_expand_reshape = gt_expand.view(gt_expand.size(0)*gt_expand.size(2), gt_expand.size(3))
             
            # print(rois_neighbors_reshape.shape)
            # print(gt_expand_reshape.shape)
            # print (rois_neighbors[0, 0, 0, :])
            # print (rois_neighbors_reshape[0, :])
            ## iou to gaussian
            ### old 0.3
            ## ious_sigma = (ious - 1) / 0.3
            ### new 
            ious_sigma = (ious - 1) / self.sigma_geometry
 
            ious_gaussion = (ious_sigma * ious_sigma) / 2 

        else:
            # ious = None
            ious_gaussion = None
        return ious_gaussion

    def _gt_to_candidates(self, rois, im_info):
        ## rois : gt boxes

        img_h = im_info[0,0]
        img_w = im_info[0,1]
        # print (img_w, img_h)
        # print (rois.shape)
        # exit()
        batchsize = rois.size(0)
        number_rois = rois.size(1)
        # print (rois)
        # exit()
        attention_candidates = rois.new(1, batchsize, number_rois, 5).zero_()

        # print (attention_candidates.shape)
        # exit()

        for i in range(1):
            # print (i,j)
            index = i 
            # print (i_transform, j_transform)
            attention_candidates[index, :, :, 0] = rois[:, :, 0]  
            attention_candidates[index, :, :, 1] = rois[:, :, 1] 
            attention_candidates[index, :, :, 2] = rois[:, :, 2] 
            attention_candidates[index, :, :, 3] = rois[:, :, 3] 
            attention_candidates[index, :, :, 4] = rois[:, :, 4] 

        ## v1: boundary clamp
        # attention_candidates[:,:,:,1].clamp_(0, img_w)
        # attention_candidates[:,:,:,3].clamp_(0, img_w)
        # attention_candidates[:,:,:,2].clamp_(0, img_h)
        # attention_candidates[:,:,:,4].clamp_(0, img_h)

        ## v2: boundary offset move inside if the boxes is out
        # calcualte delta
        offset_x1 = 0 - attention_candidates[:,:,:,1] 
        offset_x1 = offset_x1.clamp_(0, img_w)
        attention_candidates[:,:,:,1] = attention_candidates[:,:,:,1] + offset_x1

        offset_y1 = 0 - attention_candidates[:,:,:,2] 
        offset_y1 = offset_y1.clamp_(0, img_h)
        attention_candidates[:,:,:,2] = attention_candidates[:,:,:,2] + offset_y1


        offset_x2 = attention_candidates[:,:,:,3] - img_w
        offset_x2 = offset_x2.clamp_(0, img_w)
        attention_candidates[:,:,:,3] = attention_candidates[:,:,:,3] - offset_x2

        offset_y2 = attention_candidates[:,:,:,4] - img_h 
        offset_y2 = offset_y2.clamp_(0, img_h)
        attention_candidates[:,:,:,4] = attention_candidates[:,:,:,4] - offset_y2

  
        ## get the delta_x 
        # query_roi = rois[:,:,:]
        # # print (query_roi.shape)       
        # # print (query_roi)       
        # query_roi = query_roi.view(-1,5)
        # # print (query_roi.shape)       
        # delta_rois = rois.new(4, batchsize * number_rois, 5).zero_()
        # # print (delta_rois.shape)

        # ## process the rois to generate the delta_x, y
        # for i in range(4):
        #     roi_tmp = attention_candidates[i,:,:,:]
        #     roi_tmp = roi_tmp.view(-1,5)
        #     # print (roi_tmp.shape)
        #     delta_rois[i, :, 0] = query_roi[:, 0]
        #     delta_rois[i, :, 1] = (roi_tmp[:, 1] - query_roi[:, 1]) / (query_roi[:, 3] - query_roi[:, 1] +1)
        #     delta_rois[i, :, 2] = (roi_tmp[:, 2] - query_roi[:, 2]) / (query_roi[:, 4] - query_roi[:, 2] +1)
        #     delta_rois[i, :, 3] = (roi_tmp[:, 3] - query_roi[:, 3]) / (query_roi[:, 3] - query_roi[:, 1] +1)
        #     delta_rois[i, :, 4] = (roi_tmp[:, 4] - query_roi[:, 4]) / (query_roi[:, 4] - query_roi[:, 2] +1)
        # print (delta_rois)
        # print (attention_candidates)
        # exit()
        # print (attention_candidates[:,0,0,:])

        # exit()
        return attention_candidates
 
    def _rois_to_candidates(self, rois, im_info, scale, CN=False):
        # rois [batchsize x 128 x5]
        ##
        ##
        img_h = im_info[0,0]
        img_w = im_info[0,1]
        # print (img_w, img_h)
        # print (rois.shape)
        # exit()
        batchsize = rois.size(0)
        number_rois = rois.size(1)
        # print (rois)
        # exit()
        ## [0, 1, 2, 3] left, top, bottom, right -> original [4] original
        ##  
        attention_candidates = rois.new(9, batchsize, number_rois, 5).zero_()
        # print (attention_candidates.shape)

# w = box(1,3) - box(1,1);
# h = box(1,4) - box(1,2);
# newbox = zeros(9,size(box,2));
# 
#%  scale = 0.5;
# scale = 1.0;
# 
# for i = 1:3
#     for j = 1:3
#         newbox((i-1)*3+j, 1)  = box(1,1) + scale * w * (i-2);
#         newbox((i-1)*3+j, 2)  = box(1,2) + scale * h * (j-2);
#         newbox((i-1)*3+j, 3)  = box(1,3) + scale * w * (i-2);
#         newbox((i-1)*3+j, 4)  = box(1,4) + scale * h * (j-2);
#         newbox((i-1)*3+j, 5)  = box(1,5) ;
#     end
# end
        if CN:
            ## circle neighbor
            pi = math.pi
            for i in range(8):
                delta_x = scale * math.cos(i * pi / 4)
                delta_y = scale * math.sin(i * pi / 4)
                # print (i,j)
                index = i
                attention_candidates[index, :, :, 0] = rois[:, :, 0]  
                attention_candidates[index, :, :, 1] = rois[:, :, 1] + (rois[:, :, 3] - rois[:, :, 1]) * delta_x 
                attention_candidates[index, :, :, 2] = rois[:, :, 2] + (rois[:, :, 4] - rois[:, :, 2]) * delta_y
                attention_candidates[index, :, :, 3] = rois[:, :, 3] + (rois[:, :, 3] - rois[:, :, 1]) * delta_x
                attention_candidates[index, :, :, 4] = rois[:, :, 4] + (rois[:, :, 4] - rois[:, :, 2]) * delta_y
            ## put original center in
            index = 8
            attention_candidates[index, :, :, 0] = rois[:, :, 0]  
            attention_candidates[index, :, :, 1] = rois[:, :, 1] 
            attention_candidates[index, :, :, 2] = rois[:, :, 2] 
            attention_candidates[index, :, :, 3] = rois[:, :, 3] 
            attention_candidates[index, :, :, 4] = rois[:, :, 4] 
        else:
            for i in range(3):
                for j in range(3):
                    # print (i,j)
                    index = i * 3 + j 
                    attention_candidates[index, :, :, 0] = rois[:, :, 0]  
                    attention_candidates[index, :, :, 1] = rois[:, :, 1] + (rois[:, :, 3] - rois[:, :, 1]) * scale * (i - 1)
                    attention_candidates[index, :, :, 2] = rois[:, :, 2] + (rois[:, :, 4] - rois[:, :, 2]) * scale * (j - 1)
                    attention_candidates[index, :, :, 3] = rois[:, :, 3] + (rois[:, :, 3] - rois[:, :, 1]) * scale * (i - 1)
                    attention_candidates[index, :, :, 4] = rois[:, :, 4] + (rois[:, :, 4] - rois[:, :, 2]) * scale * (j - 1)

        ## v1: boundary clamp
        # attention_candidates[:,:,:,1].clamp_(0, img_w)
        # attention_candidates[:,:,:,3].clamp_(0, img_w)
        # attention_candidates[:,:,:,2].clamp_(0, img_h)
        # attention_candidates[:,:,:,4].clamp_(0, img_h)

        ## v2: boundary offset move inside if the boxes is out
        # calcualte delta
        offset_x1 = 0 - attention_candidates[:,:,:,1] 
        offset_x1 = offset_x1.clamp_(0, img_w)
        attention_candidates[:,:,:,1] = attention_candidates[:,:,:,1] + offset_x1

        offset_y1 = 0 - attention_candidates[:,:,:,2] 
        offset_y1 = offset_y1.clamp_(0, img_h)
        attention_candidates[:,:,:,2] = attention_candidates[:,:,:,2] + offset_y1


        offset_x2 = attention_candidates[:,:,:,3] - img_w
        offset_x2 = offset_x2.clamp_(0, img_w)
        attention_candidates[:,:,:,3] = attention_candidates[:,:,:,3] - offset_x2

        offset_y2 = attention_candidates[:,:,:,4] - img_h 
        offset_y2 = offset_y2.clamp_(0, img_h)
        attention_candidates[:,:,:,4] = attention_candidates[:,:,:,4] - offset_y2

        ## get the delta_x 
        ## center proposal
        if CN:
            query_roi = attention_candidates[8,:,:,:]
        else:
            query_roi = attention_candidates[4,:,:,:]
            
        # print (query_roi.shape)       
        query_roi = query_roi.view(-1,5)
        # print (query_roi.shape)       
        delta_rois = rois.new(9, batchsize * number_rois, 5).zero_()
        # print (delta_rois.shape)

        ## process the rois to generate the delta_x, y
        for i in range(9):
            roi_tmp = attention_candidates[i,:,:,:]
            roi_tmp = roi_tmp.view(-1,5)
            # print (roi_tmp.shape)
            delta_rois[i, :, 0] = query_roi[:, 0]
            delta_rois[i, :, 1] = (roi_tmp[:, 1] - query_roi[:, 1]) / (query_roi[:, 3] - query_roi[:, 1] +1)
            delta_rois[i, :, 2] = (roi_tmp[:, 2] - query_roi[:, 2]) / (query_roi[:, 4] - query_roi[:, 2] +1)
            delta_rois[i, :, 3] = (roi_tmp[:, 3] - query_roi[:, 3]) / (query_roi[:, 3] - query_roi[:, 1] +1)
            delta_rois[i, :, 4] = (roi_tmp[:, 4] - query_roi[:, 4]) / (query_roi[:, 4] - query_roi[:, 2] +1)


        # print (delta_rois)
        # print (attention_candidates)
        # exit()
        # print (attention_candidates[:,0,0,:])
        # exit()



        return attention_candidates, delta_rois
                
        
    def create_architecture(self):
        self._init_modules()
        self._init_weights()




def _kl_divergence_loss(distribution_p, distribution_q):
        ## kl distance of two input 
        # print (distribution_p.shape)
        # print (distribution_q.shape)
        log_p = F.log_softmax(distribution_p, dim=0)
        log_q = F.log_softmax(distribution_q, dim=0)
        softmax_p = F.softmax(distribution_p, dim=0)
        # print (log_p.shape)
        # print (log_q.shape)
       
        kl_distance = softmax_p * (log_p - log_q)
        # print (kl_distance.shape)
        kl_distance_mean = torch.sum(kl_distance, 0)
        # print (kl_distance_mean.shape)
        loss_kl = kl_distance.mean()
        # print (loss_kl.shape)
        # exit()
        return loss_kl 




class RelationUnit(nn.Module):
    def __init__(self, original_feature_dim=2048, appearance_feature_dim=512, key_feature_dim = 32, n_classes=1, cls_reduce_d=False, reg_reduce_d=False, alpha_same_with_beta=False):
        super(RelationUnit, self).__init__()
        self.dim_k = key_feature_dim
        self.dim_ori_feat = original_feature_dim
        self.dim_feat = appearance_feature_dim
        self.n_classes = n_classes
        self.cls_reduce_d = cls_reduce_d
        self.reg_reduce_d = reg_reduce_d
        self.alpha_same_with_beta = alpha_same_with_beta

        # bias = True
        self.attention_feat = Parameter(torch.Tensor(1, 1, original_feature_dim, appearance_feature_dim)) 

        #################### for gt attention
        self.w_k = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.dim_k, 4))
        self.w_q = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.dim_k, 4))
        # self.w_k = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.dim_k, 1))
        # self.w_q = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.dim_k, 1))

        #################### for predicted alplha
        ## neighbors number: 9
        # self.alpha_w = Parameter(torch.Tensor(9, key_feature_dim, appearance_feature_dim, 4))
        # self.alpha_w = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))


        if reg_reduce_d:
            self.alpha_w = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))
            self.bbox_regress = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))
        else:
            if alpha_same_with_beta: 
                self.alpha_w = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))
            else:
                self.alpha_w = Parameter(torch.Tensor(9, 1, original_feature_dim, 4))
                
            self.bbox_regress = Parameter(torch.Tensor(9, 1, original_feature_dim, 4))

        ################### for classification
        ## 
        self.w_k_cls = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.dim_k, 1)) 
        self.w_q_cls = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.dim_k, 1)) 
        ## shared beta
        # self.w_q_cls = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.dim_k, 1)) 

        if cls_reduce_d:
            self.alpha_w_cls = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 1))
            # self.cls_score = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.n_classes))
            # self.cls_score = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.n_classes))
            self.cls_score = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.n_classes))
            # self.cls_score = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.n_classes))
            ## shared alpha
            # self.alpha_w_cls = Parameter(torch.Tensor(1, 1, appearance_feature_dim, 1))
            # self.cls_score = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.n_classes))
        else:
            if alpha_same_with_beta: 
                self.alpha_w_cls = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 1))
            else:
                self.alpha_w_cls = Parameter(torch.Tensor(9, 1, original_feature_dim, 1))
            # self.cls_score = Parameter(torch.Tensor(9, 1, original_feature_dim, self.n_classes))
            # self.cls_score = Parameter(torch.Tensor(9, 1, original_feature_dim, self.n_classes))
            self.cls_score = Parameter(torch.Tensor(9, 1, original_feature_dim, self.n_classes))
            # self.cls_score = Parameter(torch.Tensor(1, 1, original_feature_dim, self.n_classes))
            ## shared alpha
            # self.alpha_w_cls = Parameter(torch.Tensor(9, 1, original_feature_dim, 1))
            # self.cls_score = Parameter(torch.Tensor(9, 1, original_feature_dim, self.n_classes))
            # self.cls_score = Parameter(torch.Tensor(9, 1, original_feature_dim, self.n_classes))
            # self.alpha_w_cls = Parameter(torch.Tensor(9, 1, original_feature_dim, 1))
            # self.cls_score = nn.Linear(2048, self.n_classes)

        self.relu = nn.ReLU(inplace=True)
    def _init_weights(self):
         def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()

         mean = 0
         stddev = 0.01
         stddev_cls = 0.01
         stddev_reg = 0.001
         # stddev_reg = 0.001
         if cfg.TRAIN.TRUNCATED:
             self.bbox_regress.data.normal_().fmod_(2).mul_(stddev_reg).add_(mean)
             self.cls_score.data.normal_().fmod_(2).mul_(stddev_cls).add_(mean)
             # self.cls_score.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.attention_feat.normal_().fmod_(2).mul_(stddev).add_(mean)

             self.w_k.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.w_q.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.alpha_w.data.normal_().fmod_(2).mul_(stddev).add_(mean)

             self.w_k_cls.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.w_q_cls.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.alpha_w_cls.data.normal_().fmod_(2).mul_(stddev).add_(mean)
         else:
             self.bbox_regress.data.normal_(mean, stddev_reg)
             self.cls_score.data.normal_(mean, stddev_cls)
             # self.cls_score.weight.data.normal_(mean, stddev)
             self.attention_feat.data.normal_(mean, stddev)

             self.w_k.data.normal_(mean, stddev)
             self.w_q.data.normal_(mean, stddev)
             self.alpha_w.data.normal_(mean, stddev)

             self.w_k_cls.data.normal_(mean, stddev)
             self.w_q_cls.data.normal_(mean, stddev)
             self.alpha_w_cls.data.normal_(mean, stddev)


    def forward(self, rois, delta_rois, features, gt_features, ious):
        # input:
        #     rois           : bs x 128 x 4
        #     delta_rois     : 9 x [bsx128] x4
        #     features       : 9 [ [bsx128] x 2048]
        #     gt_features    : bs x 128 x 512??
        # output:
        #  bs x 128 x 4
        tmp_feat = features[0]
        N = tmp_feat.size(0)

        # print (len(features))
        all_features = torch.stack(features)
        all_features_offset_pre = all_features.view(9, N, self.dim_ori_feat, 1)
        all_features_offset_pre_nogradient = Variable(all_features_offset_pre.data, requires_grad=False)
        
        # print (all_features_offset_pre.shape)
        # print (self.attention_feat.shape)
        all_features_offset = all_features_offset_pre * self.attention_feat
        all_features_offset = torch.sum(all_features_offset, -2)
        all_features_offset = self.relu(all_features_offset)
        all_features_offset = all_features_offset.view(9, N, self.dim_feat, 1)
        all_features_offset_nogradient = Variable(all_features_offset.data, requires_grad=False)
        # print (all_features_offset.shape)
        ## v0.2

        all_features_attention = all_features_offset.view(9, N, self.dim_feat, 1, 1)
        all_features_attention_nogradient = Variable(all_features_attention.data, requires_grad=False)

        # all_features_attention =  all_features_offset_pre
        # print (all_features_attention.shape)
        # all_features_attention = all_features_attention.permute([1, 2, 3, 4, 0])
        # print (all_features_attention.shape)
        
        ## v0.3
        # all_features_attention = all_features.view(9, N, self.dim_feat, 1)

        # print (all_features_offset.shape)
        ################## predict weight for alpha
        ## new version
        # print (len(features))
        # print (features[0])
        # print (features[0].shape)
        # print (all_features_offset_pre.shape)
        # print (self.alpha_w.shape)
        if self.reg_reduce_d:
            alpha_dot  = all_features_offset * self.alpha_w
            # alpha_dot  = all_features_offset_nogradient * self.alpha_w
            offset_dot = all_features_offset * self.bbox_regress
        else:
            if self.alpha_same_with_beta:
                alpha_dot  = all_features_offset * self.alpha_w
            else:
                alpha_dot  = all_features_offset_pre * self.alpha_w
            # alpha_dot  = all_features_offset_pre_nogradient * self.alpha_w
            offset_dot = all_features_offset_pre * self.bbox_regress
            
        alpha = torch.sum(alpha_dot, -2)
        offset = torch.sum(offset_dot, -2)
        alpha_softmax = torch.nn.Softmax(dim=0)(alpha)

        # print (alpha_dot.shape) 
        # exit()
        # print (alpha)
        # print (alpha.shape)
        # print ("start offset")

        # wk_x1 = F.linear(self.WK_x1 
        ## bbox regression
        ## end bbox regression

        # print (self.bbox_regress.shape)           
        # all_features_offset = all_features.view(9, N, self.dim_feat, 1)
        # print (all_features_offset.shape)           
        


        # offset_dot = all_features_offset * self.bbox_regress
        # print (offset_dot.shape)
        # print (offset.shape)
        # print (alpha_softmax.shape)
        ## end new predict weight from features
       
        # print (w_x1.shape)
        # exit()
        # print (w_x1)

        # generate the results
        rois = rois.view(-1, 5)
        # print (rois.shape)
        # print (delta_rois.shape)
        ###
        delta_rois_8 = delta_rois[:,:,1:5]
        delta_rois_8 = Variable(delta_rois_8)
        # delta_rois_8[0:4, :,:] = delta_rois[0:4,:,:]
        # delta_rois_8[4:8, :,:] = delta_rois[5:9,:,:]
        
        # print (delta_rois_8.shape)

        ## v0.4 offset first then alpha
        delta_pred = (delta_rois_8 + offset) * alpha_softmax
        # print (delta_pred.shape)
        output = torch.sum(delta_pred, 0)
        # print (output.shape)

        ## v0.5 
        delta_pred_offset = delta_rois_8 + offset
        # print (delta_pred_offset.shape)
        # print (delta_pred_offset)
        ##
        # output = torch.sum(delta_pred, 0)

        ################# cls alpha

        if self.cls_reduce_d:
            # cls_score_dot = all_features_offset * self.cls_score
            # cls_score_dot = all_features_offset_nogradient * self.cls_score
            # alpha_dot_cls = all_features_offset * self.alpha_w_cls
            cls_score_dot = all_features_offset * self.cls_score
            # alpha_dot_cls = all_features_offset_nogradient * self.alpha_w_cls
            alpha_dot_cls = all_features_offset * self.alpha_w_cls
        else:
            if self.alpha_same_with_beta:
                alpha_dot_cls = all_features_offset * self.alpha_w_cls
            else:
                alpha_dot_cls = all_features_offset_pre * self.alpha_w_cls
            cls_score_dot = all_features_offset_pre * self.cls_score
            # alpha_dot_cls = all_features_offset_pre_nogradient * self.alpha_w_cls

            # cls_score_dot = all_features_offset_pre * self.cls_score
            # cls_score_dot = all_features_offset_pre_nogradient * self.cls_score
            # alpha_dot_cls = all_features_offset_pre * self.alpha_w_cls
            # print (all_features_offset_pre.shape)
            # all_features_offset_pre_cls = all_features_offset_pre.view(9, N, self.dim_ori_feat)
            # print (all_features_offset_pre_cls.shape)
            # cls_score_dot = self.cls_score(all_features_offset_pre_cls)
            # print (cls_score_dot.shape)

        cls_score = torch.sum(cls_score_dot, -2)
        # cls_score = cls_score_dot
        cls_score_softmax = torch.nn.Softmax(dim=2)(cls_score)
        alpha_cls = torch.sum(alpha_dot_cls, -2)
        alpha_softmax_cls = torch.nn.Softmax(dim=0)(alpha_cls)

        # print (offset_dot.shape)
        # print (cls_score.shape)
        ### [9, 128, 21]


        # print (alpha_dot_cls.shape)
        # print (alpha_cls.shape)
        # print (alpha_softmax_cls.shape)
        # sys.exit(0)

        ################## gt attention for beta
        # print (gt_features.shape)
        if self.training:
            # v0.2
            # print (self.w_q.shape)
            # print (all_features_attention.shape)
            query_dot =  all_features_attention * self.w_q
            # query_dot =  all_features_attention_nogradient * self.w_q
            query_out = torch.sum(query_dot, -3)

            query_dot_cls =  all_features_attention * self.w_q_cls
            # query_dot_cls =  all_features_attention_nogradient * self.w_q_cls
            query_out_cls = torch.sum(query_dot_cls, -3)
            # v0.3
            # query_out = all_features_attention

            # print (query_dot.shape)
            # print (query_out.shape)
            ## v0.1, 0.2
            # gt_features_attention = gt_features.view(1, N, self.dim_feat, 1, 1)
            
            gt_all_features = torch.stack(gt_features)
            # print (gt_all_features.shape)
            # gt_features_attention = gt_all_features.view(1, N, self.dim_feat, 1, 1)
            gt_features_offset_pre = gt_all_features.view(1, N, self.dim_ori_feat, 1)
            gt_features_offset     = gt_features_offset_pre * self.attention_feat
            gt_features_offset     = torch.sum(gt_features_offset, -2)
            gt_features_offset     = self.relu(gt_features_offset)
            # print (gt_features_offset.shape)
            gt_features_attention = gt_features_offset.view(1, N, self.dim_feat, 1, 1)
            gt_features_attention_nogradient = Variable(gt_features_attention.data, requires_grad=False)
            # print (gt_features_attention.shape)
            # exit()
            ## v0.3
            # gt_features_attention = gt_features.view(1, N, self.dim_feat, 1)

            # v0.2
            # print (self.w_k.shape)
            key_dot = gt_features_attention * self.w_k 
            # key_dot = gt_features_attention_nogradient * self.w_k 
            key_out = torch.sum(key_dot, -3)

            key_dot_cls = gt_features_attention * self.w_k_cls
            # key_dot_cls = gt_features_attention_nogradient * self.w_k_cls
            key_out_cls = torch.sum(key_dot_cls, -3)

            # key_out = key_out.permute([3,1,2,0])
            # v0.3
            # key_out = gt_features_attention
            # print (key_dot.shape)
            # print (key_out.shape)
            # print (query_out.shape)
            # exit()

            # v0.2
            beta_dot = query_out * key_out
            beta_out = torch.sum(beta_dot, -2)
            beta_out = beta_out / np.sqrt(self.dim_k)
            ious_view = ious.view(ious.size(0), ious.size(2))
            
            ious_expand = ious_view.expand(beta_out.size(2), ious_view.size(0), ious_view.size(1))
            
            # print (beta_dot.shape)
            # print (beta_out.shape)
            # print (ious.shape)
            # print (ious_view.shape)
            # print (ious_expand.shape)
            ious_permute = ious_expand.permute(1,2,0)
            # print (ious_permute.shape)
            ious_permute = Variable(ious_permute)
            beta_out = beta_out - ious_permute 
            ### beta_out = beta_out - ious_permute 
            # print (beta_out.shape)
            
            beta_softmax = torch.nn.Softmax(dim=0) (beta_out)
            # print (beta_softmax.shape)

            ious_view_cls = ious.view(ious.size(0), ious.size(2), 1)
            beta_dot_cls = query_out_cls * key_out_cls
            beta_out_cls = torch.sum(beta_dot_cls, -2)
            beta_out_cls = beta_out_cls / np.sqrt(self.dim_k)
            # print (beta_out_cls.shape)
            # print (ious_view_cls.shape)
            ious_view_cls = Variable(ious_view_cls)
            beta_out_cls = beta_out_cls - ious_view_cls
            ### beta_out_cls = beta_out_cls - ious_view_cls
            # exit()
            beta_softmax_cls = torch.nn.Softmax(dim=0) (beta_out_cls)
            # print (beta_softmax_cls.shape)
            # sys.exit(0)

            
            ## v0.1 : same input with alpha
            # beta_delta_pred = (delta_rois_8 + offset) * beta_softmax
            ## v0.2 : using original delta
            # delta_pred_offset_beta = delta_rois_8

            ## b4-att0
            # beta_delta_pred = delta_rois_8 * beta_softmax
            ## b4-att0-reg
            delta_pred_offset_beta = delta_pred_offset 
            beta_delta_pred = delta_pred_offset_beta * beta_softmax
            ## v0.3 
            # beta_delta_pred = None

            ## v0.1, v0.2
            output_beta = torch.sum(beta_delta_pred, 0)
            # v0.3
            # output_beta = None
            # print (output_beta.shape)
        else:
            beta_softmax = None
            output_beta = None
            delta_pred_offset_beta = None
            beta_out = None
            beta_softmax_cls = None
            beta_out_cls = None

        # return output, w_x1, w_y1, w_x2, w_y2, delta_x1, delta_y1, delta_x2, delta_y2, output_x1_before,  output_y1_before, output_x2_before, output_y2_before
        return output, output_beta, alpha_softmax, beta_softmax, delta_pred_offset, delta_pred_offset_beta, alpha, beta_out, \
               cls_score, alpha_softmax_cls, beta_softmax_cls, alpha_cls, beta_out_cls
               # cls_score_softmax, alpha_softmax_cls, beta_softmax_cls, alpha_cls, beta_out_cls
               # cls_score_softmax, alpha_softmax_cls, beta_softmax_cls, alpha_cls, beta_out_cls
               
def _cross_entropy_proposal(cls, labels, circle):
    ## 
    # print (cls.shape)
    # print (cls_weights.shape)
    # print (labels.shape)
    # print (circle)

    loss_cls = []
    if circle:
        i = 8
    else:
        i = 4
    loss_cls_tmp = F.cross_entropy(cls[i,:,:], labels, reduce=False) 
    # print (loss_cls_tmp.shape)
    loss_cls.append(loss_cls_tmp)
    loss_cls_tensor = torch.stack(loss_cls)
    # print (loss_cls_tensor.shape)
    loss = loss_cls_tensor 
    # print (loss.shape)
    loss = torch.sum(loss, 0)
    # print (loss.shape)
    loss = torch.mean(loss)
    # print (loss.shape)
    # print (loss_cls.shape)
    return loss

# bbox_cls, alpha_cls_softmax, beta_cls_softmax, alpha_cls, beta_cls
def _cross_entropy_neighbor(cls, cls_weights, labels, cls_option):
    ## 
    # print (cls.shape)
    # print (cls_weights.shape)
    # print (labels.shape)
    # cls_permute = cls.permute(1, 2, 0)
    # key_out = key_out.permute([3,1,2,0])
    # print (cls_permute.shape)

    # labels_extend = labels.expand(cls.shape[0], labels.shape[0])
    # # print (labels_extend.shape)
    # labels_extend_permute = labels_extend.permute(1,0)
    # loss = F.cross_entropy(cls_permute, labels_extend_permute, reduce=False)
    # print (loss.shape)
    # print (labels_extend_permute.shape)
    # neighbor_rois_target     = rois_target.expand(neighbor_num, rois_target.shape[0], rois_target.shape[1])
 


    ####### option 1 logits 
    if cls_option == 0:
        cls_dot = cls * cls_weights
        cls_sum = torch.sum(cls_dot, 0)
        loss = F.cross_entropy(cls_sum, labels)
        return loss
    ####### end option 1 logits 
    # cls_prob = torch.nn.Softmax(dim=1)(cls)


    # print (cls_prob.shape)
    # print (cls_dot.shape)
    # cls_sum.clamp_(1e-6, 1.1)


    ####### option 2 2048
    if cls_option == 1:
        cls_prob = torch.nn.Softmax(dim=2)(cls)

        cls_dot = cls_prob * cls_weights
        cls_sum = torch.sum(cls_dot, 0)
        cls_sum = F.threshold(cls_sum, 1e-6, 1e-6)
        
        cls_sum_log = torch.log(cls_sum)
        loss = F.nll_loss(cls_sum_log, labels)
        return loss
    ######## end option 2
    # print (cls_sum.shape)

    # loss = F.cross_entropy(cls_sum, labels)

    ## nll_loss(input, target)
    # print (cls_sum.shape)
    # print (labels.shape)
   
    # print (loss.shape)
    # exit()

    # exit()
    ######## cross entropy
    if cls_option == 2:
        loss_cls = []
        for i in range(9):
            loss_cls_tmp = F.cross_entropy(cls[i,:,:], labels, reduce=False) 
            # print (loss_cls_tmp.shape)
            loss_cls.append(loss_cls_tmp)
        loss_cls_tensor = torch.stack(loss_cls)
        # print (loss_cls_tensor.shape)
        cls_weights = cls_weights.view(cls_weights.shape[0], cls_weights.shape[1])

        loss = loss_cls_tensor * cls_weights
        # print (loss.shape)
        loss = torch.sum(loss, 0)
        # print (loss.shape)
        loss = torch.mean(loss)
        # print (loss.shape)
        # print (loss_cls.shape)
        return loss

    raise("should not happen")


def _smooth_l1_loss_proposal(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, neighbor_pred, circle, sigma=1.0, dim=[1]):
    # print(bbox_pred.shape)
    # print(bbox_targets.shape)
    # print(bbox_inside_weights.shape)
    # print(neighbor_pred.shape)
    
    # print(alpha.shape)

    ## this is the key part, if the predict changed or unchanged

    sigma_2 = sigma ** 2
    # box_diff = bbox_pred - bbox_targets
    if circle:
        box_diff = neighbor_pred[8,:,:] - bbox_targets[8,:,:]
        in_box_diff = bbox_inside_weights[8,:,:] * box_diff
    else:
        box_diff = neighbor_pred[4,:,:] - bbox_targets[4,:,:]
        in_box_diff = bbox_inside_weights[4,:,:] * box_diff
    # print (box_diff.shape)
    # print (box_diff)
    # print (in_box_diff)
    abs_in_box_diff = torch.abs(in_box_diff)
    # print (abs_in_box_diff.shape)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    # print (smoothL1_sign)
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    # print (in_loss_box)

    if circle:
        out_loss_box = bbox_outside_weights[8,:,:] * in_loss_box
    else:
        out_loss_box = bbox_outside_weights[4,:,:] * in_loss_box
        
    # print (out_loss_box)

    ## x alpha 
    # loss_box_alpha = out_loss_box
    # loss_box = torch.sum(loss_box_alpha, 0)
    loss_box = out_loss_box
    # print (loss_box.shape)
    # print (dim)
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    # print (loss_box.shape)
    loss_box = loss_box.mean()
    # print (loss_box) 
    return loss_box

def _smooth_l1_loss_neighbor(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, neighbor_pred, alpha, sigma=1.0, dim=[1]):
    # print(bbox_pred.shape)
    # print(bbox_targets.shape)
    # print(bbox_inside_weights.shape)
    # print(neighbor_pred.shape)
    # print(alpha.shape)

    ## this is the key part, if the predict changed or unchanged
    neighbor_num = neighbor_pred.shape[0]

    ## print (neighbor_num)

    sigma_2 = sigma ** 2
    # box_diff = bbox_pred - bbox_targets
    box_diff = neighbor_pred - bbox_targets
    # print (box_diff)
    in_box_diff = bbox_inside_weights * box_diff
    # print (in_box_diff)
    abs_in_box_diff = torch.abs(in_box_diff)
    # print (abs_in_box_diff.shape)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    # print (smoothL1_sign)
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    # print (in_loss_box)
    out_loss_box = bbox_outside_weights * in_loss_box
    # print (out_loss_box)

    ## x alpha 
    loss_box_alpha = alpha * out_loss_box
    loss_box = torch.sum(loss_box_alpha, 0)
    # print (loss_box.shape)
    # print (dim)
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    # print (loss_box.shape)
    loss_box = loss_box.mean()
    # print (loss_box) 
    return loss_box
