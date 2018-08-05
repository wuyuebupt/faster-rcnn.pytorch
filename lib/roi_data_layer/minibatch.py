# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob, flow_list_to_blob
import pdb
import scipy.io as sio


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # print (num_images)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales, im_offline_proposals, im_flows = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}
  blobs['flow'] = im_flows

  # print (im_scales)
  # print (len(im_scales))
  # print (len(roidb))
  # assert len(im_scales) == 1, "Single batch only"
  assert len(im_scales) == 2, "two means one pair input"
  assert len(roidb) == 1, "Single batch only"
  
  # friday 07/27/18 5pm
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    # print (roidb[0]['gt_classes'])
    gt_inds = np.where(roidb[0]['gt_classes'][0] != 0)[0]
    gt_inds_1 = np.where(roidb[0]['gt_classes'][1] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'][0] != 0) & np.all(roidb[0]['gt_overlaps'][0].toarray() > -1.0, axis=1))[0]
    gt_inds_1 = np.where((roidb[0]['gt_classes'][1] != 0) & np.all(roidb[0]['gt_overlaps'][1].toarray() > -1.0, axis=1))[0]

  # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  # boxes 4*2, cls, flag for tracking
  # 4*2 + 1 + 1 = 10
  # format [box_0, cls, box_1, tracking_flag]
  gt_boxes = np.empty((len(gt_inds), 10), dtype=np.float32)

  # load the offline proposals
  
  # print (len(im_offline_proposals[0][0]))
  offline_proposal_boxes = np.empty((len(im_offline_proposals[0][0]), 5), dtype=np.float32)
  offline_proposal_boxes[:, 0:4] = im_offline_proposals[0][0][:, 0:4] * im_scales[0][0]
  # offline_proposal_boxes[:, 4] = im_offline_proposals[0][0][:, 4] 

  # gt_boxes is 
  # two boxes have the same im_clale
  assert(im_scales[0][0] == im_scales[1][0])
  gt_boxes[:, 0:4] = roidb[0]['boxes'][0][gt_inds, 0:4] * im_scales[0][0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][0][gt_inds]
  gt_boxes[:, 5:9] = roidb[0]['boxes'][0][gt_inds, 4:8] * im_scales[1][0]
  gt_boxes[:, 9] = roidb[0]['boxes'][0][gt_inds, 8]


  # the second part
  # gt_boxes_1 = np.empty((len(gt_inds_1), 5), dtype=np.float32)
  gt_boxes_1 = np.empty((len(gt_inds_1), 10), dtype=np.float32)
  offline_proposal_boxes_1 = np.empty((len(im_offline_proposals[1][0]), 5), dtype=np.float32)
  offline_proposal_boxes_1[:, 0:4] = im_offline_proposals[1][0][:, 0:4] * im_scales[1][0]
  # offline_proposal_boxes_1[:, 4] = im_offline_proposals[1][0][:, 4] 

  # gt_boxes is 
  gt_boxes_1[:, 0:4] = roidb[0]['boxes'][1][gt_inds_1, 0:4] * im_scales[1][0]
  gt_boxes_1[:, 4] = roidb[0]['gt_classes'][1][gt_inds_1]
  gt_boxes_1[:, 5:9] = roidb[0]['boxes'][1][gt_inds_1, 4:8] * im_scales[0][0]
  gt_boxes_1[:, 9] = roidb[0]['boxes'][1][gt_inds_1, 8]


  pair_gt_boxes = (gt_boxes, gt_boxes_1)
  pair_offline_proposal_boxes = (offline_proposal_boxes, offline_proposal_boxes_1)
  pair_info = (np.array([[im_blob[0].shape[1], im_blob[0].shape[2], im_scales[0][0]]],dtype=np.float32), 
		np.array([[im_blob[1].shape[1], im_blob[1].shape[2], im_scales[1][0]]],dtype=np.float32))

  blobs['gt_boxes'] = pair_gt_boxes
  blobs['offline_proposals'] = pair_offline_proposal_boxes
  blobs['im_info'] = pair_info

  blobs['img_id'] = roidb[0]['img_id']


  # blobs['gt_boxes'] = gt_boxes
  # blobs['offline_proposals'] = offline_proposal_boxes
  # blobs['im_info'] = np.array(
  #   [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
  #   dtype=np.float32)

  # blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  # print (num_images)
  # exit()

  processed_ims = []
  im_scales = []
  offline_proposals = []
  processed_flows = []

  processed_ims_1 = []
  im_scales_1 = []
  offline_proposals_1 = []
  processed_flows_1 = []

  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'][0])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    # load the boxes
    # boxes = np.zeros((num_objs, 4), dtype=np.float)
    offline_proposal_bbox = sio.loadmat(roidb[i]['offline_proposal'][0])['boxes']
    # print (offline_proposal_bbox.shape)
    # print (im.shape)
    
    # load flow
    # print (roidb[i]['flow'][0])

    flow_x  = imread(roidb[i]['flow'][0] + '.x.png')
    flow_y  = imread(roidb[i]['flow'][0] + '.y.png')
    flow_x  = flow_x.astype(np.float32)
    flow_x  = flow_x / 255 * 80 - 40
    flow_y  = flow_y.astype(np.float32)
    flow_y  = flow_y / 255 * 80 - 40
    flow    = np.stack((flow_x, flow_y), axis=2)

    # print (im.shape)
    # print (flow_x.shape, flow_y.shape, flow.shape)
    # print (flow_x[0:10, 0:10])
    # exit()    

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      im_width = im.shape[1]
      oldx1 = offline_proposal_bbox[:, 0].copy()
      oldx2 = offline_proposal_bbox[:, 2].copy()
      offline_proposal_bbox[:, 0] = im_width - oldx2 - 1
      offline_proposal_bbox[:, 2] = im_width - oldx1 - 1

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)
    offline_proposals.append(offline_proposal_bbox)
    # print (flow.shape)
    flow, flow_scale = prep_im_for_blob(flow, np.array([[[0., 0.]]]), target_size,
                    cfg.TRAIN.MAX_SIZE)
    # print (flow.shape)
    processed_flows.append(flow)
    # exit()



    # print (roidb[i]['image'])
    im_1 = imread(roidb[i]['image'][1])

    if len(im_1.shape) == 2:
      im_1 = im_1[:,:,np.newaxis]
      im_1 = np.concatenate((im_1,im_1,im_1), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im_1 = im_1[:,:,::-1]

    # load the boxes
    # boxes = np.zeros((num_objs, 4), dtype=np.float)
    offline_proposal_bbox_1 = sio.loadmat(roidb[i]['offline_proposal'][1])['boxes']
    # print (offline_proposal_bbox.shape)
    # print (im.shape)
    flow_x_2  = imread(roidb[i]['flow'][1] + '.x.png')
    flow_y_2  = imread(roidb[i]['flow'][1] + '.y.png')
    flow_x_2  = flow_x_2.astype(np.float32)
    flow_x_2  = flow_x_2 / 255 * 80 - 40
    flow_y_2  = flow_y_2.astype(np.float32)
    flow_y_2  = flow_y_2 / 255 * 80 - 40
    flow_2    = np.stack((flow_x_2, flow_y_2), axis=2)

 
    
    if roidb[i]['flipped']:
      im_1 = im_1[:, ::-1, :]
      im_width = im_1.shape[1]
      oldx1 = offline_proposal_bbox_1[:, 0].copy()
      oldx2 = offline_proposal_bbox_1[:, 2].copy()
      offline_proposal_bbox_1[:, 0] = im_width - oldx2 - 1
      offline_proposal_bbox_1[:, 2] = im_width - oldx1 - 1

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im_1, im_scale_1 = prep_im_for_blob(im_1, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales_1.append(im_scale_1)
    processed_ims_1.append(im_1)
    offline_proposals_1.append(offline_proposal_bbox_1)

    flow_2, flow_scale_2 = prep_im_for_blob(flow_2, np.array([[[0., 0.]]]), target_size,
                    cfg.TRAIN.MAX_SIZE)
    # print (flow.shape)
    processed_flows_1.append(flow_2)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  blob_1 = im_list_to_blob(processed_ims_1)
  
  # 
  flow_blob   = flow_list_to_blob(processed_flows)
  flow_blob_1 = flow_list_to_blob(processed_flows_1)
  # exit()

  # paired return
  blob_pair = (blob, blob_1)
  im_scales_pair = (im_scales, im_scales_1)
  offline_proposals_pair = (offline_proposals, offline_proposals_1)
  flow_pair = (flow_blob, flow_blob_1)
  return blob_pair, im_scales_pair, offline_proposals_pair, flow_pair
  # return blob, im_scales, offline_proposals
