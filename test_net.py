# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="/srv/share/jyang375/models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet_vid":
      args.imdb_name = "imagenet_vid_train+imagenet_det_train"
      args.imdbval_name = "imagenet_vid_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  # fasterRCNN.load_state_dict(checkpoint['model'])
  fasterRCNN.load_state_dict({k:v for k,v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
  # print (checkpoint['model'].keys())
  # print (checkpoint['model']['RCNN_bbox_pred.weight'])
  # print (checkpoint['model']['RCNN_bbox_pred.bias'])
  # print (fasterRCNN.RCNN_bbox_pred.weight)
  # print (fasterRCNN.RCNN_bbox_pred.bias)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  num_proposals = torch.LongTensor(1)
  proposal_boxes = torch.FloatTensor(1)

  im_data_2 = torch.FloatTensor(1)
  im_info_2 = torch.FloatTensor(1)
  num_boxes_2 = torch.LongTensor(1)
  gt_boxes_2 = torch.FloatTensor(1)
  num_proposals_2 = torch.LongTensor(1)
  proposal_boxes_2 = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    num_proposals = num_proposals.cuda()
    proposal_boxes = proposal_boxes.cuda()

    im_data_2 = im_data_2.cuda()
    im_info_2 = im_info_2.cuda()
    num_boxes_2 = num_boxes_2.cuda()
    gt_boxes_2 = gt_boxes_2.cuda()
    num_proposals_2 = num_proposals_2.cuda()
    proposal_boxes_2 = proposal_boxes_2.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)
  num_proposals = Variable(num_proposals, volatile=True)
  proposal_boxes = Variable(proposal_boxes, volatile=True)

  im_data_2 = Variable(im_data_2, volatile=True)
  im_info_2 = Variable(im_info_2, volatile=True)
  num_boxes_2 = Variable(num_boxes_2, volatile=True)
  gt_boxes_2 = Variable(gt_boxes_2, volatile=True)
  num_proposals_2 = Variable(num_proposals_2, volatile=True)
  proposal_boxes_2 = Variable(proposal_boxes_2, volatile=True)


  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)

  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_boxes_2 = [[[] for _ in xrange(num_images+1)]
               for _ in xrange(imdb.num_classes)]
  all_tracking_boxes = [[[] for _ in xrange(num_images+1)]
               for _ in xrange(imdb.num_classes)]
  all_tracking_boxes_2 = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')
  det_file_2 = os.path.join(output_dir, 'detections_2.pkl')
  det_file_tracking = os.path.join(output_dir, 'detections_tracking.pkl')
  det_file_tracking_2 = os.path.join(output_dir, 'detections_tracking_2.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      proposal_boxes.data.resize_(data[4].size()).copy_(data[4])
      num_proposals.data.resize_(data[5].size()).copy_(data[5])

      # the second part
      im_data_2.data.resize_(data[6].size()).copy_(data[6])
      im_info_2.data.resize_(data[7].size()).copy_(data[7])
      gt_boxes_2.data.resize_(data[8].size()).copy_(data[8])
      num_boxes_2.data.resize_(data[9].size()).copy_(data[9])
      proposal_boxes_2.data.resize_(data[10].size()).copy_(data[10])
      num_proposals_2.data.resize_(data[11].size()).copy_(data[11])
      # print (im_data.shape)
      # # print (im_data_2.shape)
      # print (gt_boxes.shape)
      # print (proposal_boxes.shape)
      # print (proposal_boxes)
      # exit()


      det_tic = time.time()
      rois, cls_prob, bbox_pred, tracking_cls_prob, tracking_bbox_pred, \
      rois_2, cls_prob_2, bbox_pred_2, tracking_cls_prob_2, tracking_bbox_pred_2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals, im_data_2, im_info_2, gt_boxes_2, num_boxes_2, proposal_boxes_2, num_proposals_2)
      # print (rois.shape)
      # print (cls_prob.shape, bbox_pred.shape)
      # print (tracking_cls_prob.shape, tracking_bbox_pred.shape)



      # rois, cls_prob, bbox_pred, \
      # RCNN_loss_cls, RCNN_loss_bbox, \
      # rois_label, \
      # RCNN_loss_cls_2, RCNN_loss_bbox_2, \
      # RCNN_loss_tracking_cls, RCNN_loss_tracking_bbox, \
      # RCNN_loss_tracking_cls_2, RCNN_loss_tracking_bbox_2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals, im_data_2, im_info_2, gt_boxes_2, num_boxes_2, proposal_boxes_2, num_proposals_2)


      # rois, cls_prob, bbox_pred, \
      # rpn_loss_cls, rpn_loss_box, \
      # RCNN_loss_cls, RCNN_loss_bbox, \
      # rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      
      tracking_scores = tracking_cls_prob.data
      tracking_boxes  = rois.data[:, :, 1:5]

      scores_2 = cls_prob_2.data
      boxes_2 = rois_2.data[:, :, 1:5]
      
      tracking_scores_2 = tracking_cls_prob_2.data
      tracking_boxes_2  = rois_2.data[:, :, 1:5]



      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          tracking_box_deltas = tracking_bbox_pred.data

          box_deltas_2 = bbox_pred_2.data
          tracking_box_deltas_2 = tracking_bbox_pred_2.data

          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                tracking_box_deltas = tracking_box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                tracking_box_deltas = tracking_box_deltas.view(1, -1, 4 * len(imdb.classes))

                box_deltas_2 = box_deltas_2.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas_2 = box_deltas_2.view(1, -1, 4 * len(imdb.classes))

                tracking_box_deltas_2 = tracking_box_deltas_2.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                tracking_box_deltas_2 = tracking_box_deltas_2.view(1, -1, 4 * len(imdb.classes))




          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

          pred_tracking_boxes = bbox_transform_inv(tracking_boxes, tracking_box_deltas, 1)
          pred_tracking_boxes = clip_boxes(pred_tracking_boxes, im_info.data, 1)




          pred_boxes_2 = bbox_transform_inv(boxes_2, box_deltas_2, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

          pred_tracking_boxes_2 = bbox_transform_inv(tracking_boxes_2, tracking_box_deltas_2, 1)
          pred_tracking_boxes_2 = clip_boxes(pred_tracking_boxes_2, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      # the scale
      pred_boxes /= data[1][0][2]
      pred_tracking_boxes /= data[1][0][2]
      pred_boxes_2 /= data[1][0][2]
      pred_tracking_boxes_2 /= data[1][0][2]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      tracking_scores = tracking_scores.squeeze()
      pred_tracking_boxes = pred_tracking_boxes.squeeze()
      scores_2 = scores_2.squeeze()
      pred_boxes_2 = pred_boxes_2.squeeze()
      tracking_scores_2 = tracking_scores_2.squeeze()
      pred_tracking_boxes_2 = pred_tracking_boxes_2.squeeze()

      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          # im = cv2.imread(imdb.image_path_at(i))
          im = cv2.imread(imdb.image_path_at(i)[0])
          im_2 = cv2.imread(imdb.image_path_at(i)[1])
          im2show = np.copy(im)
          tracking_im2show = np.copy(im_2)
          im2show_2 = np.copy(im_2)
          tracking_im2show_2 = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

          ## tracking res
          tracking_inds = torch.nonzero(tracking_scores[:,j]>thresh).view(-1)
          # if there is det
          if tracking_inds.numel() > 0:
            tracking_cls_scores = tracking_scores[:,j][tracking_inds]
            _, tracking_order = torch.sort(tracking_cls_scores, 0, True)
            if args.class_agnostic:
              cls_tracking_boxes = pred_tracking_boxes[tracking_inds, :]
            else:
              cls_tracking_boxes = pred_tracking_boxes[tracking_inds][:, j * 4:(j + 1) * 4]
            
            cls_tracking_dets = torch.cat((cls_tracking_boxes, tracking_cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_tracking_dets = cls_tracking_dets[tracking_order]
            tracking_keep = nms(cls_tracking_dets, cfg.TEST.NMS)
            cls_tracking_dets = cls_tracking_dets[tracking_keep.view(-1).long()]
            if vis:
              tracking_im2show = vis_detections(tracking_im2show, imdb.classes[j], cls_tracking_dets.cpu().numpy(), 0.3)
            all_tracking_boxes[j][i+1] = cls_tracking_dets.cpu().numpy()
          else:
            all_tracking_boxes[j][i+1] = empty_array


          ## for the second image
          inds_2 = torch.nonzero(scores_2[:,j]>thresh).view(-1)
          # if there is det
          if inds_2.numel() > 0:
            cls_scores_2 = scores_2[:,j][inds_2]
            _, order_2 = torch.sort(cls_scores_2, 0, True)
            if args.class_agnostic:
              cls_boxes_2 = pred_boxes_2[inds_2, :]
            else:
              cls_boxes_2 = pred_boxes_2[inds_2][:, j * 4:(j + 1) * 4]
            
            cls_dets_2 = torch.cat((cls_boxes_2, cls_scores_2.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets_2 = cls_dets_2[order_2]
            keep_2 = nms(cls_dets_2, cfg.TEST.NMS)
            cls_dets_2 = cls_dets_2[keep_2.view(-1).long()]
            if vis:
              im2show_2 = vis_detections(im2show_2, imdb.classes[j], cls_dets_2.cpu().numpy(), 0.3)
            all_boxes_2[j][i+1] = cls_dets_2.cpu().numpy()
          else:
            all_boxes_2[j][i+1] = empty_array

          ## tracking res
          tracking_inds_2 = torch.nonzero(tracking_scores_2[:,j]>thresh).view(-1)
          # if there is det
          if tracking_inds_2.numel() > 0:
            tracking_cls_scores_2 = tracking_scores_2[:,j][tracking_inds_2]
            _, tracking_order_2 = torch.sort(tracking_cls_scores_2, 0, True)
            if args.class_agnostic:
              cls_tracking_boxes_2 = pred_tracking_boxes_2[tracking_inds_2, :]
            else:
              cls_tracking_boxes_2 = pred_tracking_boxes_2[tracking_inds_2][:, j * 4:(j + 1) * 4]
            
            cls_tracking_dets_2 = torch.cat((cls_tracking_boxes_2, tracking_cls_scores_2.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_tracking_dets_2 = cls_tracking_dets_2[tracking_order_2]
            tracking_keep_2 = nms(cls_tracking_dets_2, cfg.TEST.NMS)
            cls_tracking_dets_2 = cls_tracking_dets_2[tracking_keep_2.view(-1).long()]
            if vis:
              tracking_im2show_2 = vis_detections(tracking_im2show_2, imdb.classes[j], cls_tracking_dets_2.cpu().numpy(), 0.3)
            all_tracking_boxes_2[j][i] = cls_tracking_dets_2.cpu().numpy()
          else:
            all_tracking_boxes_2[j][i] = empty_array



      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]
 
          image_scores = np.hstack([all_boxes_2[j][i+1][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes_2[j][i+1][:, -1] >= image_thresh)[0]
                  all_boxes_2[j][i+1] = all_boxes_2[j][i+1][keep, :]


          image_scores = np.hstack([all_tracking_boxes[j][i+1][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_tracking_boxes[j][i+1][:, -1] >= image_thresh)[0]
                  all_tracking_boxes[j][i+1] = all_tracking_boxes[j][i+1][keep, :]

          image_scores = np.hstack([all_tracking_boxes_2[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_tracking_boxes_2[j][i][:, -1] >= image_thresh)[0]
                  all_tracking_boxes_2[j][i] = all_tracking_boxes_2[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()
      frame1 = 'outImages/frame1'
      frame2 = 'outImages/frame2'
      frame1to2 = 'outImages/frame1to2'
      frame2to1 = 'outImages/frame2to1'

      if vis:
          path1 = os.path.join(frame1, '{:06d}'.format(i))
          path2 = os.path.join(frame2, '{:06d}'.format(i+1))
          path1to2 = os.path.join(frame1to2, '{:06d}'.format(i+1))
          path2to1 = os.path.join(frame2to1, '{:06d}'.format(i))
          cv2.imwrite(path1 +'.png', im2show)
          cv2.imwrite(path2 +'.png', im2show_2)
          cv2.imwrite(path1to2 + '.png', tracking_im2show)
          cv2.imwrite(path2to1 + '.png', tracking_im2show_2)
          # pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  ## all nms
  # print ('start')
  # all_boxes_with_tracking = [[[] for _ in xrange(num_images)]
  #              for _ in xrange(imdb.num_classes)]

  # for i in range(num_images):
  #     for j in xrange(1, imdb.num_classes):
  #       ## one more nms with max per image over all classes
  #         print (len(all_boxes[j][i]))
  #         print ((all_boxes[j][i]))
  #         print (len(all_boxes_2[j][i]))
  #         print (len(all_tracking_boxes[j][i]))
  #         print (len(all_tracking_boxes_2[j][i]))
  #         print (i,j)
  #         # current_boxes = all_boxes[j][i] + all_boxes_2[j][i] + all_tracking_boxes[j][i] +all_tracking_boxes_2[j][i]
  #         current_boxes = []
  #         current_boxes.extend(all_boxes[j][i])
  #         current_boxes.extend(all_boxes_2[j][i])
  #         current_boxes.extend(all_tracking_boxes[j][i])
  #         current_boxes.extend(all_tracking_boxes_2[j][i])
  #         # if len(all_boxes[j][i])>0:
  #         #     current_boxes.extend(all_boxes[j][i])
  #         # if len(all_boxes_2[j][i])>0:
  #         #     current_boxes = current_boxes + all_boxes_2[j][i]
  #         # if len(all_tracking_boxes[j][i])>0:
  #         #     current_boxes = current_boxes + all_tracking_boxes[j][i]
  #         # if len(all_tracking_boxes_2[j][i])>0:
  #         #     current_boxes = current_boxes + all_tracking_boxes_2[j][i]
  #         print (len(current_boxes))
  #         # all
  #         scores = np.asarray(current_boxes)[:,4]
  #         print (scores)
  #         # if there is det
  #         if len(current_boxes) >  0:
  #           cls_scores = torch.from_numpy(scores)
  #           _, order = torch.sort(cls_scores, 0, True)
  #           cls_dets = torch.from_numpy(np.asarray(current_boxes))
  #           cls_dets = cls_dets[order]
  #           print (cls_dets)
  #           keep = nms(cls_dets, cfg.TEST.NMS)
  #           cls_dets = cls_dets[keep.view(-1).long()]
  #           all_boxes_with_tracking[j][i] = cls_dets.cpu().numpy()
  #         else:
  #           all_boxes_with_tracking[j][i] = empty_array
  #         print (len(all_boxes_with_tracking[j][i]))



  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  with open(det_file_2, 'wb') as f:
      pickle.dump(all_boxes_2, f, pickle.HIGHEST_PROTOCOL)
  with open(det_file_tracking, 'wb') as f:
      pickle.dump(all_tracking_boxes, f, pickle.HIGHEST_PROTOCOL)
  with open(det_file_tracking_2, 'wb') as f:
      pickle.dump(all_tracking_boxes_2, f, pickle.HIGHEST_PROTOCOL)

  # print('Evaluating detections')
  # imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
