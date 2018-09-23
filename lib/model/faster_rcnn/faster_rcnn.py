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

        # self.attention_regression = RelationUnit(2048, 32) 
        self.attention_regression = RelationUnit(512, 32) 
        # self.attention_regression = RelationUnit(2048, 64) 
        # self.attention_regression = RelationUnit(256, 64) 
        # self.attention_regression = RelationUnit(128, 16) 
        # self.attention_regression = RelationUnit(1024, 64) 

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            # print (rois_target.shape)
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
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
        rois_attention_candidates, delta_rois = self._rois_to_candidates(rois, im_info,  0.5)
        # print (rois_attention_candidates.shape)

        rois_attention_candidates = Variable(rois_attention_candidates)
        # delta_rois = Variable(delta_rois)
        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # print (base_feat.shape) # 4x1024x38x50


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


        ## do the attention prediction
        # bbox_pred = attention_regression(rois_attention_candidates, rois_attention_pooled_feat)
        # print (self.attention_regression)
        # print (rois_attention_candidates.is_cuda)
        # print (rois_attention_pooled_feat.is_cuda)

        bbox_pred, wx1, wy1, wx2, wy2, dx1, dy1, dx2, dy2, ox1, oy1, ox2, oy2 = self.attention_regression(rois, delta_rois, rois_attention_pooled_feat) 

        # print (bbox_pred)
        # print (bbox_pred.shape)
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

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            # print (bbox_pred)
            # print (rois_target)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        # print (RCNN_loss_bbox)
        # exit()

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, wx1, dx1, ox1
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, wx1, wy1, wx2, wy2, dx1, dy1, dx2, dy2, ox1, oy1, ox2, oy2
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


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
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_attention_feat, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        # init the attention module
        # normal_init(self.attention_regression, 0, 0.001, cfg.TRAIN.TRUNCATED)
        self.attention_regression._init_weights()

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

        ## boundary 
        attention_candidates[:,:,:,1].clamp_(0, img_w)
        attention_candidates[:,:,:,3].clamp_(0, img_w)

        attention_candidates[:,:,:,2].clamp_(0, img_h)
        attention_candidates[:,:,:,4].clamp_(0, img_h)

  
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

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=2048, key_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_k = key_feature_dim

        # bias = True
        bias = False
        self.WK_x1_0 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_0 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_0 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_0 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_3 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_3 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_3 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_3 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_4 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_4 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_4 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_4 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_5 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_5 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_5 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_5 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_6 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_6 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_6 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_6 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_7 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_7 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_7 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_7 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_8 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_8 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_8 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_8 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 

        self.WK_x1_9 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y1_9 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_x2_9 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 
        self.WK_y2_9 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias) 


        # self.WK_x1 = []
        # self.WK_y1 = []
        # self.WK_x2 = []
        # self.WK_y2 = []
        # for i in range(9):
        #     self.WK_x1.append(nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias))
        #     self.WK_y1.append(nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias))
        #     self.WK_x2.append(nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias))
        #     self.WK_y2.append(nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias))

        self.WQ_x1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias)
        self.WQ_y1 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias)
        self.WQ_x2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias)
        self.WQ_y2 = nn.Linear(appearance_feature_dim, key_feature_dim, bias=bias)

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

         normal_init(self.WK_x1_0, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_0, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_0, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_0, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_1, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_1, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_2, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_2, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_3, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_3, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_4, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_4, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_4, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_4, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_5, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_5, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_5, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_5, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_6, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_6, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_6, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_6, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_7, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_7, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_7, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_7, 0, 0.01, cfg.TRAIN.TRUNCATED)

         normal_init(self.WK_x1_8, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y1_8, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_x2_8, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WK_y2_8, 0, 0.01, cfg.TRAIN.TRUNCATED)
         # for i in range(9):
         #     normal_init(self.WK_x1[i], 0, 0.001, cfg.TRAIN.TRUNCATED)
         #     normal_init(self.WK_y1[i], 0, 0.001, cfg.TRAIN.TRUNCATED)
         #     normal_init(self.WK_x2[i], 0, 0.001, cfg.TRAIN.TRUNCATED)
         #     normal_init(self.WK_y2[i], 0, 0.001, cfg.TRAIN.TRUNCATED)

        
         normal_init(self.WQ_x1, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WQ_y1, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WQ_x2, 0, 0.01, cfg.TRAIN.TRUNCATED)
         normal_init(self.WQ_y2, 0, 0.01, cfg.TRAIN.TRUNCATED)



    def forward(self, rois, delta_rois, features):
        # input:
        #     rois           : bs x 128 x 4
        #     delta_rois     : 9 x [bsx128] x4
        #     features       : 9 [ [bsx128] x 2048]
        # output:
        #  bs x 128 x 4
        
        # wq * q: 128

        q_feat = features[4]
        N = q_feat.size(0)
        # print (N)
        wq_x1 = self.WQ_x1(q_feat)
        wq_y1 = self.WQ_y1(q_feat)
        wq_x2 = self.WQ_x2(q_feat)
        wq_y2 = self.WQ_y2(q_feat)
        # print (wq_y2.shape)

        # wk_i * f
        wk_x1_0 = self.WK_x1_0(features[0])
        wk_y1_0 = self.WK_y1_0(features[0])
        wk_x2_0 = self.WK_x2_0(features[0])
        wk_y2_0 = self.WK_y2_0(features[0])

        wk_x1_1 = self.WK_x1_1(features[1])
        wk_y1_1 = self.WK_y1_1(features[1])
        wk_x2_1 = self.WK_x2_1(features[1])
        wk_y2_1 = self.WK_y2_1(features[1])

        wk_x1_2 = self.WK_x1_2(features[2])
        wk_y1_2 = self.WK_y1_2(features[2])
        wk_x2_2 = self.WK_x2_2(features[2])
        wk_y2_2 = self.WK_y2_2(features[2])

        wk_x1_3 = self.WK_x1_3(features[3])
        wk_y1_3 = self.WK_y1_3(features[3])
        wk_x2_3 = self.WK_x2_3(features[3])
        wk_y2_3 = self.WK_y2_3(features[3])

        wk_x1_4 = self.WK_x1_4(features[4])
        wk_y1_4 = self.WK_y1_4(features[4])
        wk_x2_4 = self.WK_x2_4(features[4])
        wk_y2_4 = self.WK_y2_4(features[4])

        wk_x1_5 = self.WK_x1_5(features[5])
        wk_y1_5 = self.WK_y1_5(features[5])
        wk_x2_5 = self.WK_x2_5(features[5])
        wk_y2_5 = self.WK_y2_5(features[5])

        wk_x1_6 = self.WK_x1_6(features[6])
        wk_y1_6 = self.WK_y1_6(features[6])
        wk_x2_6 = self.WK_x2_6(features[6])
        wk_y2_6 = self.WK_y2_6(features[6])

        wk_x1_7 = self.WK_x1_7(features[7])
        wk_y1_7 = self.WK_y1_7(features[7])
        wk_x2_7 = self.WK_x2_7(features[7])
        wk_y2_7 = self.WK_y2_7(features[7])

        wk_x1_8 = self.WK_x1_8(features[8])
        wk_y1_8 = self.WK_y1_8(features[8])
        wk_x2_8 = self.WK_x2_8(features[8])
        wk_y2_8 = self.WK_y2_8(features[8])
        # print (wk_y2_8.shape)


        wk_x1_0 = wk_x1_0.view(N, 1, self.dim_k) 
        wk_x1_1 = wk_x1_1.view(N, 1, self.dim_k) 
        wk_x1_2 = wk_x1_2.view(N, 1, self.dim_k) 
        wk_x1_3 = wk_x1_3.view(N, 1, self.dim_k) 
        wk_x1_4 = wk_x1_4.view(N, 1, self.dim_k) 
        wk_x1_5 = wk_x1_5.view(N, 1, self.dim_k) 
        wk_x1_6 = wk_x1_6.view(N, 1, self.dim_k) 
        wk_x1_7 = wk_x1_7.view(N, 1, self.dim_k) 
        wk_x1_8 = wk_x1_8.view(N, 1, self.dim_k) 

        wk_y1_0 = wk_y1_0.view(N, 1, self.dim_k) 
        wk_y1_1 = wk_y1_1.view(N, 1, self.dim_k) 
        wk_y1_2 = wk_y1_2.view(N, 1, self.dim_k) 
        wk_y1_3 = wk_y1_3.view(N, 1, self.dim_k) 
        wk_y1_4 = wk_y1_4.view(N, 1, self.dim_k) 
        wk_y1_5 = wk_y1_5.view(N, 1, self.dim_k) 
        wk_y1_6 = wk_y1_6.view(N, 1, self.dim_k) 
        wk_y1_7 = wk_y1_7.view(N, 1, self.dim_k) 
        wk_y1_8 = wk_y1_8.view(N, 1, self.dim_k) 

        wk_x2_0 = wk_x2_0.view(N, 1, self.dim_k) 
        wk_x2_1 = wk_x2_1.view(N, 1, self.dim_k) 
        wk_x2_2 = wk_x2_2.view(N, 1, self.dim_k) 
        wk_x2_3 = wk_x2_3.view(N, 1, self.dim_k) 
        wk_x2_4 = wk_x2_4.view(N, 1, self.dim_k) 
        wk_x2_5 = wk_x2_5.view(N, 1, self.dim_k) 
        wk_x2_6 = wk_x2_6.view(N, 1, self.dim_k) 
        wk_x2_7 = wk_x2_7.view(N, 1, self.dim_k) 
        wk_x2_8 = wk_x2_8.view(N, 1, self.dim_k) 

        wk_y2_0 = wk_y2_0.view(N, 1, self.dim_k) 
        wk_y2_1 = wk_y2_1.view(N, 1, self.dim_k) 
        wk_y2_2 = wk_y2_2.view(N, 1, self.dim_k) 
        wk_y2_3 = wk_y2_3.view(N, 1, self.dim_k) 
        wk_y2_4 = wk_y2_4.view(N, 1, self.dim_k) 
        wk_y2_5 = wk_y2_5.view(N, 1, self.dim_k) 
        wk_y2_6 = wk_y2_6.view(N, 1, self.dim_k) 
        wk_y2_7 = wk_y2_7.view(N, 1, self.dim_k) 
        wk_y2_8 = wk_y2_8.view(N, 1, self.dim_k) 

        # with self
        # wk_x1= torch.cat((wk_x1_0, wk_x1_1, wk_x1_2, wk_x1_3, wk_x1_4, wk_x1_5, wk_x1_6, wk_x1_7, wk_x1_8), dim=1)

        ## without self
        # wk_x1 = torch.cat((wk_x1_0, wk_x1_1, wk_x1_2, wk_x1_3,  wk_x1_5, wk_x1_6, wk_x1_7, wk_x1_8), dim=1)
        # wk_y1 = torch.cat((wk_y1_0, wk_y1_1, wk_y1_2, wk_y1_3,  wk_y1_5, wk_y1_6, wk_y1_7, wk_y1_8), dim=1)
        # wk_x2 = torch.cat((wk_x2_0, wk_x2_1, wk_x2_2, wk_x2_3,  wk_x2_5, wk_x2_6, wk_x2_7, wk_x2_8), dim=1)
        # wk_y2 = torch.cat((wk_y2_0, wk_y2_1, wk_y2_2, wk_y2_3,  wk_y2_5, wk_y2_6, wk_y2_7, wk_y2_8), dim=1)
        wk_x1 = torch.cat((wk_x1_0, wk_x1_1, wk_x1_2, wk_x1_3, wk_x1_4,  wk_x1_5, wk_x1_6, wk_x1_7, wk_x1_8), dim=1)
        wk_y1 = torch.cat((wk_y1_0, wk_y1_1, wk_y1_2, wk_y1_3, wk_y1_4,  wk_y1_5, wk_y1_6, wk_y1_7, wk_y1_8), dim=1)
        wk_x2 = torch.cat((wk_x2_0, wk_x2_1, wk_x2_2, wk_x2_3, wk_x2_4,  wk_x2_5, wk_x2_6, wk_x2_7, wk_x2_8), dim=1)
        wk_y2 = torch.cat((wk_y2_0, wk_y2_1, wk_y2_2, wk_y2_3, wk_y2_4,  wk_y2_5, wk_y2_6, wk_y2_7, wk_y2_8), dim=1)

        # print (wk_x1.shape)

        wq_x1 = wq_x1.view(N, 1, self.dim_k) 
        wq_y1 = wq_y1.view(N, 1, self.dim_k) 
        wq_x2 = wq_x2.view(N, 1, self.dim_k) 
        wq_y2 = wq_y2.view(N, 1, self.dim_k) 

        scale_dot_x1 = torch.sum((wk_x1 * wq_x1), -1)
        scale_dot_x1 = scale_dot_x1 / np.sqrt(self.dim_k)

        scale_dot_y1 = torch.sum((wk_y1 * wq_y1), -1)
        scale_dot_y1 = scale_dot_y1 / np.sqrt(self.dim_k)

        scale_dot_x2 = torch.sum((wk_x2 * wq_x2), -1)
        scale_dot_x2 = scale_dot_x2 / np.sqrt(self.dim_k)

        scale_dot_y2 = torch.sum((wk_y2 * wq_y2), -1)
        scale_dot_y2 = scale_dot_y2 / np.sqrt(self.dim_k)
        # print (scale_dot_x1.shape)
        # print (scale_dot_y1.shape)
        # print (scale_dot_x2.shape)
        # print (scale_dot_y2.shape)

        w_x1 = torch.nn.Softmax(dim=1)(scale_dot_x1)
        w_y1 = torch.nn.Softmax(dim=1)(scale_dot_y1)
        w_x2 = torch.nn.Softmax(dim=1)(scale_dot_x2)
        w_y2 = torch.nn.Softmax(dim=1)(scale_dot_y2)
        # print (w_x1.shape)
        # print (w_x1)

        # generate the results
        rois = rois.view(-1, 5)
        # print (rois.shape)
        # print (delta_rois.shape)
        ###
        # delta_rois_8 = delta_rois.new(8, delta_rois.size(1), delta_rois.size(2)).zero_()
        # delta_rois_8 = delta_rois.new(9, delta_rois.size(1), delta_rois.size(2)).zero_()
        delta_rois_8 = delta_rois
        # delta_rois_8[0:4, :,:] = delta_rois[0:4,:,:]
        # delta_rois_8[4:8, :,:] = delta_rois[5:9,:,:]
        
        # print (delta_rois_8.shape)
        # print (delta_rois_8)
        
        # output the target
        delta_x1 = torch.t(delta_rois_8[:,:,1])
        delta_y1 = torch.t(delta_rois_8[:,:,2])
        delta_x2 = torch.t(delta_rois_8[:,:,3])
        delta_y2 = torch.t(delta_rois_8[:,:,4])
        # print (delta_x1.shape)
        # print (delta_x1)
        delta_x1 = Variable(delta_x1)
        delta_y1 = Variable(delta_y1)
        delta_x2 = Variable(delta_x2)
        delta_y2 = Variable(delta_y2)


        output_x1_before =  w_x1 * delta_x1
        output_y1_before =  w_y1 * delta_y1
        output_x2_before =  w_x2 * delta_x2
        output_y2_before =  w_y2 * delta_y2
        # output_x1_before = 3 * w_x1 * delta_x1
        # output_y1_before = 3 * w_y1 * delta_y1
        # output_x2_before = 3 * w_x2 * delta_x2
        # output_y2_before = 3 * w_y2 * delta_y2

        # print (output_x1.shape)        
        output_x1 = torch.sum(output_x1_before, -1)
        output_y1 = torch.sum(output_y1_before, -1)
        output_x2 = torch.sum(output_x2_before, -1)
        output_y2 = torch.sum(output_y2_before, -1)

        output_x1 = output_x1.view(output_x1.size(0), 1)
        output_y1 = output_y1.view(output_y1.size(0), 1)
        output_x2 = output_x2.view(output_x2.size(0), 1)
        output_y2 = output_y2.view(output_y2.size(0), 1)
        # print (output_y2[0,:])
        # exit()

        # print (output_x1.shape)        
        output = torch.cat((output_x1, output_y1,output_x2, output_y2), dim=1)
        # print (output)        

        # exit()
        # N,_ = f_a.size()

        # position_embedding = position_embedding.view(-1,self.dim_g)

        # w_g = self.relu(self.WG(position_embedding))
        # w_k = self.WK(f_a)
        # w_k = w_k.view(N,1,self.dim_k)

        # w_q = self.WQ(f_a)
        # w_q = w_q.view(1,N,self.dim_k)

        # scaled_dot = torch.sum((w_k*w_q),-1 )
        # scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        # w_g = w_g.view(N,N)
        # w_a = scaled_dot.view(N,N)

        # w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        # w_mn = torch.nn.Softmax(dim=1)(w_mn)

        # w_v = self.WV(f_a)

        # w_mn = w_mn.view(N,N,1)
        # w_v = w_v.view(N,1,-1)

        # output = w_mn*w_v

        # output = torch.sum(output,-2)
        # return output
        return output, w_x1, w_y1, w_x2, w_y2, delta_x1, delta_y1, delta_x2, delta_y2, output_x1_before,  output_y1_before, output_x2_before, output_y2_before


