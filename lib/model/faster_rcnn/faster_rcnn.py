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

        # new linear prediction
        self.attention_regression = RelationUnit(512, 32) 

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
        rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.5)
        # print (rois_attention_candidates.shape)

        rois_attention_candidates = Variable(rois_attention_candidates)
        # delta_rois = Variable(delta_rois)
        rois = Variable(rois)

        ## gt rois
        if self.training:
            gt_attention_candidates = self._gt_to_candidates(gt_rois, im_info, 0.5)
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
            pooled_feat_tmp = self.RCNN_roi_align(base_feat, rois_attention_candidates[i,:,:,:].view(-1, 5))
            pooled_feat_tmp = self._head_to_tail(pooled_feat_tmp)
            # v1
            # rois_attention_pooled_feat.append(pooled_feat_tmp)

            # v2
            pooled_attention_feat = self.RCNN_attention_feat(pooled_feat_tmp)
            pooled_attention_feat = self.relu(pooled_attention_feat)
            rois_attention_pooled_feat.append(pooled_attention_feat)
        # print (rois_attention_pooled_feat)
        # exit()

        gt_attention_pooled_feat = []
        ### for gt rois
        if self.training:
            for i in range(4):
                assert(cfg.POOLING_MODE == 'align')
                # pooled_feat_gt = self.RCNN_roi_align(base_feat, gt_rois.view(-1, 5))
                pooled_feat_gt = self.RCNN_roi_align(base_feat, gt_attention_candidates[i,:,:,:].view(-1, 5))

                # feed pooled features to top model
                # print (pooled_feat.shape) # 512x1024x7x7
                pooled_feat_gt = self._head_to_tail(pooled_feat_gt)
                pooled_attention_feat_gt = self.RCNN_attention_feat(pooled_feat_gt)
                pooled_attention_feat_gt = self.relu(pooled_attention_feat_gt)

                gt_attention_pooled_feat.append(pooled_attention_feat_gt)
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
        bbox_pred, bbox_pred_beta, alpha_softmax, beta_softmax, bbox_pred_offset = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat, gt_attention_pooled_feat) 

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
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_bbox_beta = 0
        KL_loss = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            # print (bbox_pred)
            # print (rois_target)
            # RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            ## v0.4
            # RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            # print (RCNN_loss_bbox.shape)
            ## v0.5
            RCNN_loss_bbox = _smooth_l1_loss_alpha(bbox_pred, neighbor_rois_target, neighbor_rois_inside_ws, neighbor_rois_outside_ws, bbox_pred_offset, alpha_softmax)
            # print (RCNN_loss_bbox.shape)
            # exit()

            ## from gt training for beta
            ## v0.2
            RCNN_loss_bbox_beta = _smooth_l1_loss(bbox_pred_beta, rois_target, rois_inside_ws, rois_outside_ws)
            ## v0.3
            # RCNN_loss_bbox_beta = None

            ## KL loss between beta and alpha
            # KL_loss = F.kl_div(alpha_softmax, beta_softmax)
            # KL_distance = torch.distributions.kl.kl_divergence(alpha_softmax, beta_softmax)
            # print (KL_distance)
            KL_loss = _kl_divergence_loss(alpha_softmax, beta_softmax)
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
        return rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label, RCNN_loss_bbox_beta, KL_loss
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, RCNN_loss_bbox_beta, KL_loss

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
        normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        # init the attention module
        # normal_init(self.attention_regression, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.attention_regression._init_weights()


    def _gt_to_candidates(self, rois, im_info, scale):
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
	attention_candidates = rois.new(4, batchsize, number_rois, 5).zero_()

        # print (attention_candidates.shape)
        # exit()

        for i in range(4):
            # print (i,j)
            tar_index = i*2 + 1
            index = i 
            i_transform = tar_index / 3
            j_transform = tar_index % 3 
            # print (i_transform, j_transform)
            attention_candidates[index, :, :, 0] = rois[:, :, 0]  
            attention_candidates[index, :, :, 1] = rois[:, :, 1] + (rois[:, :, 3] - rois[:, :, 1]) * scale * (i_transform - 1)
            attention_candidates[index, :, :, 2] = rois[:, :, 2] + (rois[:, :, 4] - rois[:, :, 2]) * scale * (j_transform - 1)
            attention_candidates[index, :, :, 3] = rois[:, :, 3] + (rois[:, :, 3] - rois[:, :, 1]) * scale * (i_transform - 1)
            attention_candidates[index, :, :, 4] = rois[:, :, 4] + (rois[:, :, 4] - rois[:, :, 2]) * scale * (j_transform - 1)

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
 
    def _rois_to_candidates(self, rois, im_info, scale):
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
        log_p = torch.log(distribution_p)
        log_q = torch.log(distribution_q)
        # print (log_p.shape)
        # print (log_q.shape)
       
        kl_distance = distribution_p * (log_p - log_q)
        # print (kl_distance.shape)
        kl_distance_mean = torch.sum(kl_distance, 0)
        # print (kl_distance_mean.shape)
        loss_kl = kl_distance.mean()
        # print (loss_kl.shape)
        # exit()
        return loss_kl 




class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=2048, key_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_k = key_feature_dim
        self.dim_feat = appearance_feature_dim

        # bias = True
        bias = False
        
        #################### for gt attention
        # self.w_k = Parameter(torch.Tensor(1, 1, appearance_feature_dim, self.dim_k, 4))
        self.w_k = Parameter(torch.Tensor(4, 1, appearance_feature_dim, self.dim_k, 1))
        self.w_q = Parameter(torch.Tensor(9, 1, appearance_feature_dim, self.dim_k, 4))

        #################### for predicted alplha
        ## neighbors number: 9
        # self.alpha_w = Parameter(torch.Tensor(9, key_feature_dim, appearance_feature_dim, 4))
        self.alpha_w = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))

        # print (self.alpha_w.shape)
        # exit()
        ## init weigths 
        # stdv = 1. / math.sqrt(self.WK_x1.size(1))
        # self.WK_x1.data.uniform_(-stdv, stdv)


        ## bbox regression
        ## 
        # self.bbox_regress = Parameter(torch.Tensor(appearance_feature_dim, 4, 9))
        self.bbox_regress = Parameter(torch.Tensor(9, 1, appearance_feature_dim, 4))
        #  print (self.bbox_regress.shape)
        #  exit()

        # self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=False)
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
         if cfg.TRAIN.TRUNCATED:
             self.alpha_w.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.bbox_regress.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.w_k.data.normal_().fmod_(2).mul_(stddev).add_(mean)
             self.w_q.data.normal_().fmod_(2).mul_(stddev).add_(mean)
         else:
             self.alpha_w.data.normal_(mean, stddev)
             self.bbox_regress.data.normal_(mean, stddev)
             self.w_k.data.normal_(mean, stddev)
             self.w_q.data.normal_(mean, stddev)



    def forward(self, rois, delta_rois, features, gt_features):
        # input:
        #     rois           : bs x 128 x 4
        #     delta_rois     : 9 x [bsx128] x4
        #     features       : 9 [ [bsx128] x 2048]
        #     gt_features    : bs x 128 x 512??
        # output:
        #  bs x 128 x 4
        
        # wq * q: 128

        q_feat = features[4]
        N = q_feat.size(0)

        all_features = torch.stack(features)
        all_features_offset = all_features.view(9, N, self.dim_feat, 1)
        ## v0.2
        all_features_attention = all_features.view(9, N, self.dim_feat, 1, 1)
        ## v0.3
        # all_features_attention = all_features.view(9, N, self.dim_feat, 1)

        # print (all_features_offset.shape)
        ################## predict weight for alpha
        ## new version
        # print (len(features))
        # print (features[0])
        # print (features[0].shape)
        # print (self.alpha_w.shape)
        alpha_dot = all_features_offset * self.alpha_w
        # print (alpha_dot.shape)
        alpha = torch.sum(alpha_dot, -2)
        # print (alpha)
        # print (alpha.shape)
        # print ("start offset")

        # wk_x1 = F.linear(self.WK_x1 
        ## bbox regression
        ## end bbox regression

        # print (self.bbox_regress.shape)           
        # all_features_offset = all_features.view(9, N, self.dim_feat, 1)
        # print (all_features_offset.shape)           
        offset_dot = all_features_offset * self.bbox_regress
        # print (offset_dot.shape)
        offset = torch.sum(offset_dot, -2)
        # print (offset.shape)
        alpha_softmax = torch.nn.Softmax(dim=0)(alpha)
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


        ################## gt attention for beta
        # print (gt_features.shape)
        if self.training:
            # v0.2
            query_dot =  all_features_attention * self.w_q
            query_out = torch.sum(query_dot, -3)
            # v0.3
            # query_out = all_features_attention

            # print (query_dot.shape)
            # print (query_out.shape)
            ## v0.1, 0.2
            # gt_features_attention = gt_features.view(1, N, self.dim_feat, 1, 1)
            
            gt_all_features = torch.stack(gt_features)
            # print (gt_all_features.shape)
            gt_features_attention = gt_all_features.view(4, N, self.dim_feat, 1, 1)
            # print (gt_features_attention.shape)
            ## v0.3
            # gt_features_attention = gt_features.view(1, N, self.dim_feat, 1)

            # v0.2
            # print (self.w_k.shape)
            key_dot = gt_features_attention * self.w_k 
            key_out = torch.sum(key_dot, -3)
            key_out = key_out.permute([3,1,2,0])
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
            # print (beta_dot.shape)
            # print (beta_out.shape)
            
            beta_softmax = torch.nn.Softmax(dim=0) (beta_out)
            # print (beta_softmax.shape)
            
            ## v0.1 : same input with alpha
            # beta_delta_pred = (delta_rois_8 + offset) * beta_softmax
            ## v0.2 : using original delta
            beta_delta_pred = delta_rois_8 * beta_softmax
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

        # return output, w_x1, w_y1, w_x2, w_y2, delta_x1, delta_y1, delta_x2, delta_y2, output_x1_before,  output_y1_before, output_x2_before, output_y2_before
        return output, output_beta, alpha_softmax, beta_softmax, delta_pred_offset


def _smooth_l1_loss_alpha(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, neighbor_pred, alpha, sigma=1.0, dim=[1]):
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
