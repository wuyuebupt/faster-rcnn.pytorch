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
from model.rpn.bbox_transform import bbox_transform_inv, bbox_transform_inv_attention
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
## 
from model.rpn.bbox_transform import bbox_overlaps_batch

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

  parser.add_argument('--neighbor_move', dest='neighbor_move',
                      help='confg like 0.3',
                      default="0.0", type=float)
  parser.add_argument('--circle', dest='circle',
                      help='True of False',
                      action='store_true')


## for usage of proposal, neighbor, 2048 or 512
  parser.add_argument('--cls_neighbor', dest='cls_neighbor',
                      help='True of False',
                      action='store_true')
  parser.add_argument('--cls_reduce_d', dest='cls_reduce_d',
                      help='True of False',
                      action='store_true')
  parser.add_argument('--reg_neighbor', dest='reg_neighbor',
                      help='True of False',
                      action='store_true')
  parser.add_argument('--reg_reduce_d', dest='reg_reduce_d',
                      help='True of False',
                      action='store_true')

  parser.add_argument('--reduce_dimension', dest='reduce_dimension',
                      help='reduce_dimension for attention',
                      default=128, type=int)


  parser.add_argument('--alpha_same_with_beta', dest='alpha_same_with_beta',
                      help='True of False',
                      action='store_true')

  parser.add_argument('--sigma_geometry', dest='sigma_geometry',
                      help='float',
                      default='0.3', type=float)

  parser.add_argument('--cls_alpha_option', dest='cls_alpha_option',
                      help='options {0: logits, 1: softmax , 2: cross entropy}',
                      default=2, type=int)



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

  cfg.NEIGHBOR_MOVE = args.neighbor_move
  cfg.CIRCLE = args.circle
  print ("neighbor_move     : ", args.neighbor_move)
  print ("circle            : ", args.circle)


  cfg.CLS_NEIGHBOR = args.cls_neighbor
  cfg.CLS_REDUCE_D = args.cls_reduce_d
  cfg.REG_NEIGHBOR = args.reg_neighbor
  cfg.REG_REDUCE_D = args.reg_reduce_d
  cfg.REDUCE_DIMENSION = args.reduce_dimension
  cfg.ALPHA_SAME_WITH_BETA = args.alpha_same_with_beta
  cfg.SIGMA_GEOMETRY = args.sigma_geometry
  cfg.CLS_ALPHA_OPTION = args.cls_alpha_option


  print ("cls_neighbor      : ", args.cls_neighbor)
  print ("cls_reduce_d      : ", args.cls_reduce_d)
  print ("reg_neighbor      : ", args.reg_neighbor)
  print ("reg_reduce_d      : ", args.reg_reduce_d)
  print ("reduce_dimension  : ", args.reduce_dimension)

  print ("alpha_same_with_beta : ", args.alpha_same_with_beta)
  print ("sigma_geometry       : ", args.sigma_geometry)
  print ("cls_alpha_option     : ", args.cls_alpha_option)



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
  # print (checkpoint['model'].keys())
  for key in checkpoint['model'].keys():
    if key.split('.')[0] == 'attention_regression':
        print (key, checkpoint['model'][key].shape)
  fasterRCNN.load_state_dict(checkpoint['model'])
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

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    num_proposals = num_proposals.cuda()
    proposal_boxes = proposal_boxes.cuda()


  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)
  num_proposals = Variable(num_proposals)
  proposal_boxes = Variable(proposal_boxes)

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

  all_boxes_rois = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]



  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')
  det_file_rois = os.path.join(output_dir, 'detections_rois.pkl')
  det_file_ious = os.path.join(output_dir, 'detections_ious.pkl')

  det_file_pred = os.path.join(output_dir, 'detections_cls_alpha.pkl')

  print ("Loading results")
  with open(det_file_pred, 'rb') as f:
      rois_cls_alpha = pickle.load(f)
  print ("Finish loading results")

  ### evaluate results
  # with open(det_file, 'rb') as f:
  #     all_boxes = pickle.load(f)
  # print('Evaluating detections')
  # imdb.evaluate_detections(all_boxes, output_dir)
  # exit()
  #


  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  iou_pairs = []
  # cls_prob_alpha = []


  # for i in range(10):
  # for i in range(5):
  for i in range(1):
  # for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      proposal_boxes.data.resize_(data[4].size()).copy_(data[4])
      num_proposals.data.resize_(data[5].size()).copy_(data[5])
      # rpn_loss_cls, rpn_loss_box, \

      det_tic = time.time()
      # rois, cls_prob, bbox_pred, \
      # RCNN_loss_cls, RCNN_loss_bbox, \
      # rois_label, \
      # RCNN_loss_bbox_beta, kl_loss,  \
      # RCNN_loss_cls_beta, kl_loss_cls = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals)

      # rois, cls_prob, bbox_pred, rois_label, \
      # RCNN_loss_cls_proposal, \
      # RCNN_loss_cls_alpha_positive, RCNN_loss_cls_alpha_negative, RCNN_loss_cls_beta_positive, \
      # RCNN_loss_bbox, RCNN_loss_bbox_beta, \
      # kl_loss, kl_loss_cls, \
      # alpha_cls_softmax, cls_score_softmax = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals)
      
      rois      = rois_cls_alpha[i][0]
      bbox_pred = rois_cls_alpha[i][1]
      alpha     = rois_cls_alpha[i][2]
      score     = rois_cls_alpha[i][3]
      print (rois.dtype)
      # print (rois)
      rois = Variable(torch.from_numpy(rois))
      # print (rois.dtype)
      # exit()
      bbox_pred = Variable(torch.from_numpy(bbox_pred))
      alpha = Variable(torch.from_numpy(alpha))
      score = Variable(torch.from_numpy(score))
      rois = rois.cuda()
      bbox_pred = bbox_pred.cuda()
      alpha     = alpha.cuda()
      score     = score.cuda()
      # print (rois.shape)

      print (rois[0,0,:])
      print (bbox_pred[0,0,:])
      print (alpha[:,0,0])
      print (score[:,0,0])
      # if i == 4:
      #     print (rois[0,0,:])
      #     print (bbox_pred[0,0,:])
      #     print (alpha[:,0,0])
      #     print (score[:,0,0])

      # print (rois.shape)
      # print (bbox_pred.shape)
      # rois      = rois_cls_alpha[i+1][0]
      # bbox_pred = rois_cls_alpha[i+1][1]
      # alpha     = rois_cls_alpha[i+1][2]
      # score     = rois_cls_alpha[i+1][3]
      # print (rois[0,0,:])
      # print (bbox_pred[0,0,:])
      # print (alpha[:,0,0])
      # print (score[:,0,0])
      # print (rois.shape)
      # print (bbox_pred.shape)

      # exit()
      # print (alpha.shape)
      # print (score.shape)
      alpha_cls_softmax = alpha
      bbox_cls = score

      if args.cls_neighbor:
          ## neighbor 
          ## if args.cls_alpha_option == 0:
          ##     cls_score = bbox_cls * alpha_cls_softmax
          ##     cls_prob = torch.nn.Softmax(dim=2)(cls_score)
          ##     cls_prob = torch.sum(cls_prob, 0)
          ## else:
          ##     cls_score_softmax = torch.nn.Softmax(dim=2)(bbox_cls)
          ##     if args.circle:
          ##         cls_proposal = cls_score_softmax[8, :, :]
          ##     else:
          ##         cls_proposal = cls_score_softmax[4, :, :]
          ##     # print (cls_proposal.shape)
          ##     # cls_proposal = cls_score_softmax[0, :, :]

          ##     value, indices = torch.max(alpha_cls_softmax, 0)
          ##     # print (indices.shape)
          ##     # one_hot = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1), alpha_cls_softmax.size(2))
          ##     y = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1))
          ##     y.zero_()
          ##     # indices = indices.view(indices.size(0))
          ##     indices = indices.permute(1, 0)
          ##     # print (indices.shape)
          ##     # print (y.shape)
          ##     # y = y.scatter_(0, indices.data, 1)
          ##     y = y.scatter_(0, indices.data, 1)
          ##     y = y.view(y.size(0), y.size(1), 1)
          ##     y = Variable(y)
          ##     cls_prob_alpha = cls_score_softmax * y

          ##     # cls_prob_alpha = cls_prob_alpha * y

          ##     ## new select prob
          ##     # cls_prob_pre = torch.sum(cls_prob_alpha, 0)
          ##     cls_prob_pre = torch.sum(cls_prob_alpha, 0)
          ##     # cls_prob = cls_score_softmax[4,:,:]

          ##     ## mask for neg
          ##     # mask = torch.nonzero(cls_score_softmax[4,:,0] > cls_prob_pre[:,0])
          ##     mask = torch.nonzero(cls_score_softmax[4,:,0] > cls_prob_pre[:,0])
          ##     # mask = torch.
          ##     # print (mask)

          ##     cls_weights = torch.cuda.FloatTensor(cls_prob_pre.size(0), 1).zero_()
          ##     # # print (cls_weights.shape)

          ##     # # print (cls_prob_alpha.shape)
          ##     # for i in range(mask.numel()):
          ##     for j in range(mask.numel()):
          ##         # print (mask[i]) 
          ##         ind = int(mask[j])
          ##         cls_weights[ind, 0] = 1.0
          ##     # print (cls_weights)
          ##     cls_weights = Variable(cls_weights)
          ##     # print (cls_weights.shape)
          ##     # print (cls_prob_pre.shape)
          ##     # cls_weights = cls_weights.view()
          ##     # cls_prob = cls_weights * cls_score_softmax[4,:,:] + (1 - cls_weights) * cls_prob_pre
          ##     # cls_prob =  cls_prob_pre
          ##     cls_prob =  cls_score_softmax[4,:,:]
              if args.cls_alpha_option == 0:
                  cls_score = bbox_cls * alpha_cls_softmax
                  cls_prob = torch.nn.Softmax(dim=2)(cls_score)
                  cls_prob = torch.sum(cls_prob, 0)
              else:
                  cls_score_softmax = torch.nn.Softmax(dim=2)(bbox_cls)
                  if args.circle:
                      cls_proposal = cls_score_softmax[8, :, :]

                  else:
                      cls_proposal = cls_score_softmax[4, :, :]
                  # print (cls_proposal.shape)
                  # cls_proposal = cls_score_softmax[0, :, :]

                  #### proposal bg if max
                  value_bg_p, indices_bg_p = torch.max(cls_proposal, 1)
                  # print (indices.shape)
                  # print (indices)
                  mask_bg_p = torch.nonzero( indices_bg_p )

                  cls_weights_bg_p = torch.cuda.FloatTensor(cls_proposal.size(0), 1).zero_()
                  for h in range(mask_bg_p.numel()):
                      # print (mask[i]) 
                      ind = int(mask_bg_p[h])
                      cls_weights_bg_p[ind, 0] = 1.0

                  # print (cls_weights)
                  cls_weights_bg_p = Variable(cls_weights_bg_p)

                  # print (mask)

                  #### neighbor alpha if max
                  value_alpha_n, indices_alpha_n = torch.max(alpha_cls_softmax, 0)
                  # print (indices.shape)
                  # one_hot = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1), alpha_cls_softmax.size(2))
                  y = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1))
                  y.zero_()
                  # indices = indices.view(indices.size(0))
                  indices_alpha_n = indices_alpha_n.permute(1, 0)
                  # print (indices.shape)
                  # print (y.shape)
                  y = y.scatter_(0, indices_alpha_n.data, 1)
                  y = y.view(y.size(0), y.size(1), 1)
                  y = Variable(y)
                  # print (y[:,0,0])
                  # print (y[:,2,0])
                  # value, indices = torch.max(alpha_cls_softmax, 0)
                  # indices = indices.permute(1, 0)
                  # y = torch.cuda.FloatTensor(alpha_cls_softmax.size(0), alpha_cls_softmax.size(1))
                  # y.zero_()
                  # y = y.view(y.size(0), y.size(1), 1)
                  # y = Variable(y)
                  # cls_prob = cls_score_softmax * y
                  # print (mask)
                  # exit()
                  # cls_proposal = cls_prob_proposal
                  # argmax = cls_proposal.max(1)[1]
                  # print (argmax)
                  # mask = torch.nonzero(argmax)
                  # print (mask)



                  cls_prob_alpha_ave = cls_score_softmax * alpha_cls_softmax

                  ##########
                  cls_prob_pre_ave = torch.sum(cls_prob_alpha_ave, 0)
                  mask_bg_p_n = torch.nonzero(cls_proposal[:,0] > cls_prob_pre_ave[:,0])
                  cls_weights_bg_p_n = torch.cuda.FloatTensor(cls_proposal.size(0), 1).zero_()
                  # # print (cls_weights.shape)

                  # # print (cls_prob_alpha.shape)
                  for h in range(mask_bg_p_n.numel()):
                      # print (mask[i]) 
                      ind = int(mask_bg_p_n[h])
                      cls_weights_bg_p_n[ind, 0] = 1.0
                  # print (cls_weights)
                  cls_weights_bg_p_n = Variable(cls_weights_bg_p_n)
                  # print (cls_weights.shape)
                  # print (cls_prob_pre.shape)
                  # cls_weights = cls_weights.view()

                  cls_prob_alpha = cls_score_softmax * y
                  cls_prob_max_alpha_pre = torch.sum(cls_prob_alpha, 0)

                  #### neighbor bg if max
                  value_bg_n, indices_bg_n = torch.max(cls_prob_pre_ave, 1)
                  # print (indices.shape)
                  # print (indices)
                  mask_bg_n = torch.nonzero( indices_bg_n )

                  cls_weights_bg_n = torch.cuda.FloatTensor(cls_proposal.size(0), 1).zero_()
                  for h in range(mask_bg_n.numel()):
                      # print (mask[i]) 
                      ind = int(mask_bg_n[h])
                      cls_weights_bg_n[ind, 0] = 1.0

                  # print (cls_weights)
                  cls_weights_bg_n = Variable(cls_weights_bg_n)

                  ## combine
                  ## cls_prob = cls_weights_bg_p * (cls_weights_bg_p_n * cls_proposal + (1 - cls_weights_bg_p_n) * cls_prob_pre_ave ) + (1 - cls_weights_bg_p) * cls_prob_pre_ave

                  ## cls_weights_bg_p : proposal pos weight
                  ## cls_weights_bg_n : neighjbor pos weight
                  ## cls_weights_bg_p_n : proposal bg > neighbor bg

                  mask_bg_p_or_n = 1 - cls_weights_bg_p * cls_weights_bg_n

                  # cls_prob =  mask_bg_p_or_n * (cls_weights_bg_p_n * cls_proposal + (1 - cls_weights_bg_p_n) * cls_prob_pre_ave ) + (1-mask_bg_p_or_n) * cls_prob_max_alpha_pre 
                  # cls_prob = cls_prob_max_alpha_pre
                  # cls_prob = cls_prob_pre_ave
                  cls_prob = cls_proposal


                  #######################################################

      # print (rois.size(1))
      # print (rois.shape)
      cls_prob = cls_prob.view(1, rois.size(1) , -1)
      # print (cls_prob.shape)
      # exit()

      scores = cls_prob.data
      # print (scores.shape)
      # print (scores[0,0,:])
      # exit()
      # scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      # cls_prob_alpha.append([boxes, alpha_cls_softmax.data, cls_score_softmax])

      # wx1, wy1, wx2, wy2, \
      # dx1, dy1, dx2, dy2, \
      # ox1, oy1, ox2, oy2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      # print (oy2[0,:])
      # print (ox2[0,:])
      # print (bbox_pred[0,0,:])
      # wx1, dx1, ox1 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      # rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)
            # if args.class_agnostic:
            #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
            #                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            #     box_deltas = box_deltas.view(1, -1, 4)
            # else:
            #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
            #                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            #     box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          # pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = bbox_transform_inv_attention(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      roi_boxes = boxes / data[1][0][2]
      pred_boxes /= data[1][0][2]
      gt_boxes[:, :, :4] = gt_boxes[:, :, :4] / data[1][0][2]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      roi_boxes = roi_boxes.squeeze()
      box_deltas = box_deltas.squeeze()

      # print (pred_boxes.shape)
      # print (pred_boxes.cpu().numpy())
      # print (scores.shape)
      # print (scores.cpu().numpy())
      # print (pred_boxes.cpu().numpy())

      # wx1 = wx1.squeeze()
      # dx1 = dx1.squeeze()
      # ox1 = ox1.view(ox1.size(0),1)
      #print (ox1.shape)
      # ox1 = ox1.squeeze()
      # print (ox1.shape)

      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)

            cls_boxes = pred_boxes[inds, :]
            cls_roi_boxes = roi_boxes[inds, :]
            if vis:
                roi_wx1 = wx1[inds, :]
                roi_wy1 = wy1[inds, :]
                roi_wx2 = wx2[inds, :]
                roi_wy2 = wy2[inds, :]

                roi_dx1 = dx1[inds, :]
                roi_dy1 = dy1[inds, :]
                roi_dx2 = dx2[inds, :]
                roi_dy2 = dy2[inds, :]

                roi_ox1 = ox1[inds, :]
                roi_oy1 = oy1[inds, :]
                roi_ox2 = ox2[inds, :]
                roi_oy2 = oy2[inds, :]

                # roi_dx1 = dx1[inds, :]
                # roi_ox1 = ox1[inds, :]
                roi_box_deltas = box_deltas[inds, :]

            # if args.class_agnostic:
            #   cls_boxes = pred_boxes[inds, :]
            # else:
            #   cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_roi_dets = torch.cat((cls_roi_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            cls_roi_dets = cls_roi_dets[order]
            if vis:
                roi_wx1 = roi_wx1[order]
                roi_wy1 = roi_wy1[order]
                roi_wx2 = roi_wx2[order]
                roi_wy2 = roi_wy2[order]

                roi_dx1 = roi_dx1[order]
                roi_dy1 = roi_dy1[order]
                roi_dx2 = roi_dx2[order]
                roi_dy2 = roi_dy2[order]

                roi_ox1 = roi_ox1[order]
                roi_oy1 = roi_oy1[order]
                roi_ox2 = roi_ox2[order]
                roi_oy2 = roi_oy2[order]
                # roi_dx1 = roi_dx1[order]
                # roi_ox1 = roi_ox1[order]
                roi_box_deltas = roi_box_deltas[order]

            keep = nms(cls_dets, cfg.TEST.NMS)

            cls_dets = cls_dets[keep.view(-1).long()]
            cls_roi_dets = cls_roi_dets[keep.view(-1).long()]
            if vis:
                roi_wx1 = roi_wx1[keep.view(-1).long()]
                roi_wy1 = roi_wy1[keep.view(-1).long()]
                roi_wx2 = roi_wx2[keep.view(-1).long()]
                roi_wy2 = roi_wy2[keep.view(-1).long()]

                roi_dx1 = roi_dx1[keep.view(-1).long()]
                roi_dy1 = roi_dy1[keep.view(-1).long()]
                roi_dx2 = roi_dx2[keep.view(-1).long()]
                roi_dy2 = roi_dy2[keep.view(-1).long()]

                roi_ox1 = roi_ox1[keep.view(-1).long()]
                roi_oy1 = roi_oy1[keep.view(-1).long()]
                roi_ox2 = roi_ox2[keep.view(-1).long()]
                roi_oy2 = roi_oy2[keep.view(-1).long()]
                # roi_dx1 = roi_dx1[keep.view(-1).long()]
                # roi_ox1 = roi_ox1[keep.view(-1).long()]
                roi_box_deltas = roi_box_deltas[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3, (0,204,0))
              im2show = vis_detections(im2show, imdb.classes[j], cls_roi_dets.cpu().numpy(), 0.3, ( 0,255, 255))
              
              # print (j)
              # print ('x1', roi_wx1)
              # print ('y1', roi_wy1)
              # print ('x2', roi_wx2)
              # print ('y2', roi_wy2)

              # # print ('dx1', roi_dx1)
              # # print ('dy1', roi_dy1)
              # # print ('dx2', roi_dx2)
              # # print ('dy2', roi_dy2)

              # print ('ox1', roi_ox1)
              # print ('oy1', roi_oy1)
              # print ('ox2', roi_ox2)
              # print ('oy2', roi_oy2)
              # print (cls_dets)

              # print (j)
              # print (roi_wx1)
              # print (roi_dx1)
              # print (roi_ox1)
              # print (roi_box_deltas)
              # print (cls_dets)
              # print (cls_roi_dets)
            all_boxes[j][i] = cls_dets.cpu().numpy()
            all_boxes_rois[j][i] = cls_roi_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array
            all_boxes_rois[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]
                  all_boxes_rois[j][i] = all_boxes_rois[j][i][keep, :]
      # print (i)
      # print (len(all_boxes))
      # obj = [all_boxes[j][i] for j in xrange(1, imdb.num_classes)]
      # obj =np.hstack( [all_boxes[j][i] for j in xrange(1, imdb.num_classes)])
      # print (obj)
      # print (all_boxes[15][i])

      # print (all_boxes[1][i])
      ### analyze
      # load gt boxes
      # print (gt_boxes)
      # get overlaps
      bboxes_all_classes = []
      bboxes_rois_all_classes = []
      classes_pred = []
      for j in xrange(1, imdb.num_classes):
          bboxes_all_classes.extend(all_boxes[j][i])
          bboxes_rois_all_classes.extend(all_boxes_rois[j][i])
          classes_pred.extend([j for tmp in range(len(all_boxes[j][i]))])
      # print (classes_pred)
      # exit()

      # print (bboxes_all_classes)
      bboxes_all = np.array(bboxes_all_classes)
      bboxes_rois_all = np.array(bboxes_rois_all_classes)
      # print (bboxes_all.shape)
      # exit()
          
      # overlap_boxes = torch.from_numpy(all_boxes[j][i][:,:4])
      overlap_boxes = torch.from_numpy(bboxes_all[:,:4])
      overlap_boxes = Variable(overlap_boxes)
      overlap_boxes = overlap_boxes.contiguous().cuda()

      overlap_boxes_rois = torch.from_numpy(bboxes_rois_all[:,:4])
      overlap_boxes_rois = Variable(overlap_boxes_rois)
      overlap_boxes_rois = overlap_boxes_rois.contiguous().cuda()


      # print (overlap_boxes)
      # print (overlap_boxes.shape)

      # print (all_boxes[j][i])
      # print (all_boxes[j][i].shape)

      ## overlap_boxes N x 4
      ## gt boxes 1 x K x 4
      overlaps = bbox_overlaps_batch(overlap_boxes, gt_boxes)
      overlaps_rois = bbox_overlaps_batch(overlap_boxes_rois, gt_boxes)
      # print (gt_boxes)

      ## overlaps N x K
      # print (overlaps)
      # print (overlaps_rois)
      max_overlaps, gt_assignment = torch.max(overlaps, 2)
      # print (max_overlaps)
      # print (max_overlaps[0])
      # print (max_overlaps.shape)
      # print (gt_assignment)
      ## 
      threshold = 0.0
      above_thres_inds = torch.nonzero(max_overlaps[0] >= threshold).view(-1)
      above_thres_inds = torch.nonzero(max_overlaps[0] >= threshold).view(-1)
      # print (above_thres_inds.shape)
      # print (above_thres_inds.size())
      # print (len(above_thres_inds.shape))
      # exit()

      if len(above_thres_inds.shape) == 0:
          continue
      keep_inds = above_thres_inds
      # print (keep_inds)
     
      # number_boxes = overlap_boxes.shape[0]
      # number_boxes = overlap_boxes.shape[0]
      number_boxes = above_thres_inds.shape[0]
      # print (number_boxes)
      # print (overlaps.shape)
      # print (overlaps[0].shape)
      for k in range(number_boxes):
          ## two iou
          # print (gt_assignment[0][keep_inds])
          # print (gt_assignment[0][keep_inds][i])
          # print (keep_inds[i])
          # print (overlaps[0].shape)
          # print (overlaps[0][keep_inds[i]])
          # print (overlaps[0][keep_inds[i]][0, gt_assignment[0][keep_inds][i].data[0]])
          # print (overlaps_rois[0][keep_inds[i]][0, gt_assignment[0][keep_inds][i].data[0]])
          pred_iou = overlaps[0][keep_inds[k]][0, gt_assignment[0][keep_inds][k].data[0]].data[0]
          roi_iou  = overlaps_rois[0][keep_inds[k]][0, gt_assignment[0][keep_inds][k].data[0]].data[0]
          pred_score = bboxes_all[int(keep_inds[k])][4]
          # pred_score = bboxes_all[keep_inds[k]][4]
          pred_class = classes_pred[int(keep_inds[k].data[0])]

          # print (pred_iou, roi_iou)
          # print (roi_iou, pred_iou, pred_class, ) 
          k_bbox = overlap_boxes[keep_inds[k]].data.cpu().numpy()
          k_roi  = overlap_boxes_rois[keep_inds[k]].data.cpu().numpy()
          k_gt   = gt_boxes[0, gt_assignment[0][keep_inds][k].data[0]].data.cpu().numpy()
          # print (k_bbox, k_roi, k_gt)
          iou_pairs.append([roi_iou, pred_iou, pred_score, pred_class, k_bbox, k_roi, k_gt]) 
          # over_i = overlaps[0][keep# _inds[i]][0, gt_assignment[0][keep_inds][i].data[0]]
          # print (over_i[0, gt_assignment[0][keep_inds][i].data[0]])

          # exit()

          # print (overlaps[0][keep_inds[i]])
          # print (overlaps_rois[0][gt_assignment[0][keep_inds][i]])
          # exit()
          
          # print (overlaps[0][gt_assignment[0][keep_inds]])
          # print (overlaps_rois[0][keep_inds])

      # gt_boxes[i][gt_assignment[i][keep_inds]]
      # exit()

      ### analyze
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          output_file = 'logs/testOutput/{:05d}.png'.format(i)
          # cv2.imwrite(output_file, im2show)

          # cv2.imwrite('result.png', im2show)
          # pdb.set_trace()

          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  # with open(det_file_rois, 'wb') as f:
  #     pickle.dump( all_boxes_rois, f, pickle.HIGHEST_PROTOCOL)

  #     # pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  # # print (iou_pairs)
  # with open(det_file_ious, 'wb') as f:
  #     pickle.dump( iou_pairs, f, pickle.HIGHEST_PROTOCOL)

  # with open(det_file_pred, 'wb') as f:
  #     pickle.dump( cls_prob_alpha, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
  # imdb.evaluate_detections(all_boxes_rois, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
