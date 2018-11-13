# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import math

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      # help='directory to save models', default="/srv/share/jyang375/models",
                      help='directory to save models', default="./save_dir/",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)


# new for absolut paht for loading any file
  parser.add_argument('--config', dest='config_file',
                      help='confg like res101.yml',
                      default="", type=str)
  
  parser.add_argument('--data_dir', dest='data_folder',
                      help='confg like data/h5data_gt/',
                      default="", type=str)

  parser.add_argument('--pretrained_model', dest='pretrained_model',
                      help='confg like model_dir/resnet-101.pth',
                      default="", type=str)


#       loss = args.bbox_weight         * RCNN_loss_bbox.mean()
# 
#       if args.cls_neighbor:
#           loss = loss + \
#              args.kl_weight              * kl_loss_cls.mean() +\
#              args.cls_weights_alpha_pos  * RCNN_loss_cls_proposal.mean() +\
#              args.cls_weights_alpha_neg  * RCNN_loss_cls_alpha_negative.mean() +\
#              args.cls_weights_beta_pos   * RCNN_loss_cls_beta.mean()
#       else:
#           loss = loss + \
#              args.cls_weights_alpha_pos  * RCNN_loss_cls_proposal.mean() + \
#              args.cls_weights_alpha_neg  * RCNN_loss_cls_alpha_negative.mean()
# 
# 
#       if args.reg_neighbor:
#           loss = loss + \
#              args.bbox_beta_weight    * RCNN_loss_bbox_beta.mean() + \
#              args.kl_weight           * kl_loss.mean()


# codes for satan on philly
  parser.add_argument('--neighbor_move', dest='neighbor_move',
                      help='confg like 0.3',
                      default="0.0", type=float)

  parser.add_argument('--cls_weights_alpha_pos', dest='cls_weights_alpha_pos',
                      help='confg like 1.0',
                      default="1.0", type=float)

  parser.add_argument('--cls_weights_alpha_neg', dest='cls_weights_alpha_neg',
                      help='confg like 1.0',
                      default="1.0", type=float)

  parser.add_argument('--cls_weights_beta_pos', dest='cls_weights_beta_pos',
                      help='confg like 1.0',
                      default="1.0", type=float)

  parser.add_argument('--cls_weights_proposal', dest='cls_weights_proposal',
                      help='confg like 1.0',
                      default="1.0", type=float)

  parser.add_argument('--bbox_weight', dest='bbox_weight',
                      help='confg like 10.0',
                      default="10.0", type=float)

  parser.add_argument('--bbox_beta_weight', dest='bbox_beta_weight',
                      help='confg like 1.0',
                      default="50.0", type=float)
  parser.add_argument('--kl_weight', dest='kl_weight',
                      help='confg like 1.0',
                      default="1.0", type=float)

  parser.add_argument('--circle', dest='circle',
                      help='True of False',
                      action='store_true')
                      # default=False, type=bool)

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

  parser.add_argument('--att_weight', dest='att_weight',
                      help='float',
                      default='1', type=float)

  parser.add_argument('--sigma_geometry', dest='sigma_geometry',
                      help='float',
                      default='0.3', type=float)

  parser.add_argument('--cls_alpha_option', dest='cls_alpha_option',
                      help='options {0: logits, 1: softmax , 2: cross entropy}',
                      default=2, type=int)


  parser.add_argument('--resume_dir', dest='resume_dir',
                      # help='directory to save models', default="/srv/share/jyang375/models",
                      help='directory to save models', default="/",
                      type=str)


#      loss = RCNN_loss_cls.mean() + 10 * RCNN_loss_bbox.mean() \
#           + 50 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "imagenet_vid":
      args.imdb_name = "imagenet_vid_train+imagenet_det_train"
      args.imdbval_name = "imagenet_vid_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

## some files to put into args

  args.cfg_file = args.config_file
  # args.data_folder = args.data_folder
  print (args.cfg_file)
  print (args.data_folder)


  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  # fix blocks
  cfg.RESNET.FIXED_BLOCKS = 1
  cfg.RESNET.FIXED_TOPS = False
  
  cfg.DATA_DIR = args.data_folder
  print (cfg.DATA_DIR)
  cfg.MODEL_PATH = args.pretrained_model
  print (cfg.MODEL_PATH)

  ## satan
  cfg.NEIGHBOR_MOVE = args.neighbor_move
  cfg.CIRCLE = args.circle

  # parser.add_argument('--cls_neighbor', dest='cle_neighbor',
  # parser.add_argument('--cls_2048', dest='cls_2048',
  # parser.add_argument('--reg_neighbor', dest='reg_neighbor',
  # parser.add_argument('--reg_2048', dest='reg_2048',

  cfg.CLS_NEIGHBOR = args.cls_neighbor
  cfg.CLS_REDUCE_D = args.cls_reduce_d
  cfg.REG_NEIGHBOR = args.reg_neighbor
  cfg.REG_REDUCE_D = args.reg_reduce_d
  cfg.REDUCE_DIMENSION = args.reduce_dimension
  cfg.ALPHA_SAME_WITH_BETA = args.alpha_same_with_beta
  cfg.SIGMA_GEOMETRY = args.sigma_geometry
  cfg.CLS_ALPHA_OPTION = args.cls_alpha_option



  print ("neighbor_move        : ", args.neighbor_move)
  # print ("cls_weight           : ", args.cls_weight)
  # print ("cls_weight_proposal  : ", args.cls_weight_proposal)
  # print ("cls_beta_weight      : ", args.cls_beta_weight)
  print ("cls_weights_alpha_pos: ", args.cls_weights_alpha_pos)
  print ("cls_weights_alpha_neg: ", args.cls_weights_alpha_neg)
  print ("cls_weights_beta_pos : ", args.cls_weights_beta_pos)
  print ("cls_weight_proposal  : ", args.cls_weights_proposal)
  print ("bbox_weight          : ", args.bbox_weight)
  print ("bbox_beta_weight     : ", args.bbox_beta_weight)
  print ("kl_weight            : ", args.kl_weight)
  print ("circle               : ", args.circle)

  print ("cls_neighbor         : ", args.cls_neighbor)
  print ("cls_reduce_d         : ", args.cls_reduce_d)
  print ("reg_neighbor         : ", args.reg_neighbor)
  print ("reg_reduce_d         : ", args.reg_reduce_d)
  print ("reduce_dimension     : ", args.reduce_dimension)

  print ("alpha_same_with_beta : ", args.alpha_same_with_beta)
  print ("sigma_geometry       : ", args.sigma_geometry)
  print ("cls_alpha_option     : ", args.cls_alpha_option)
  print ("att_weight           : ", args.att_weight)
  

  # parser.add_argument('--neighbor_move', dest='neighbor_move',
  #                     help='confg like 0.3',
  #                     default="0.0", type=float)

  # parser.add_argument('--cls_weight', dest='cls_weight',
  #                     help='confg like 1.0',
  #                     default="1.0", type=float)
  # parser.add_argument('--bbox_alpha_weight', dest='bbox_alpha_weight',
  #                     help='confg like 10.0',
  #                     default="10.0", type=float)

  # parser.add_argument('--bbox_beta_weight', dest='bbox_beta_weight',
  #                     help='confg like 1.0',
  #                     default="50.0", type=float)
  # parser.add_argument('--kl_weight', dest='kl_weight'  



  
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if args.resume_dir != '/':
      output_resume_dir = args.resume_dir + "/" + args.net + "/" + args.dataset
      if not os.path.exists(output_resume_dir):
        os.makedirs(output_resume_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

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
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  num_proposals = Variable(num_proposals)
  proposal_boxes = Variable(proposal_boxes)


  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      elif 'attention_regression' in key:
        params += [{'params':[value],'lr': lr * args.att_weight, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        print (key, lr * args.att_weight)
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  # print (len(params))
  # print (params[0].keys())
  # print (len(params[0]['params']))
  # print (params[0]['params'][0].shape)
  # print (params[0]['lr'])
  # print (params[0]['weight_decay'])

  # exit()


  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)



  ## auto resume
  from os import listdir
  from os.path import isfile, join

  if args.resume_dir != '/':
      mypath = output_resume_dir
      onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  else:
      onlyfiles = []

  print (onlyfiles)
  if len(onlyfiles) > 0:
      ## resume from last
      sessions = [int(arr.split('_')[2]) for arr in onlyfiles]
      checkepochs = [int(arr.split('_')[3]) for arr in onlyfiles]
      checkpoints = [int(arr.split('_')[4].split('.')[0]) for arr in onlyfiles]
      print (sessions)
      print (checkepochs)
      print (checkpoints)
      lastepoch = max(checkepochs)
      index = checkepochs.index(lastepoch)
      lastsession = sessions[index]
      lastpoints = checkpoints[index]
      print (lastepoch)
      print (lastsession)
      print (lastpoints)

      # load_name = os.path.join(output_dir,
      load_name = os.path.join(output_resume_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(lastsession, lastepoch, lastpoints))
      print("loading checkpoint %s" % (load_name))
      checkpoint = torch.load(load_name)
      args.session = checkpoint['session']
      args.start_epoch = checkpoint['epoch']
      fasterRCNN.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr = optimizer.param_groups[0]['lr']
      if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
      print("loaded checkpoint %s" % (load_name))


  ## checkout folder ??
  ## if nan -> 

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))



  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)

  nan_flag = False



  # for epoch in range(args.start_epoch, args.max_epochs + 1):
  epoch = args.start_epoch - 1
  print (epoch)
  
  # for epoch in range(args.start_epoch, args.max_epochs + 1):

  while True:

    ## if from nan
    if nan_flag:
        ## back to the current epoch
        epoch = epoch - 1  
        ## reset the flag
        nan_flag = False

        ## load last model, as we continue, nan model should not be saved
        mypath = output_dir
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        print (onlyfiles)
        if len(onlyfiles) > 0:
            ## resume from last
            sessions = [int(arr.split('_')[2]) for arr in onlyfiles]
            checkepochs = [int(arr.split('_')[3]) for arr in onlyfiles]
            checkpoints = [int(arr.split('_')[4].split('.')[0]) for arr in onlyfiles]
            print (sessions)
            print (checkepochs)
            print (checkpoints)
            lastepoch = max(checkepochs)
            index = checkepochs.index(lastepoch)
            lastsession = sessions[index]
            lastpoints = checkpoints[index]
            print (lastepoch)
            print (lastsession)
            print (lastpoints)

            load_name = os.path.join(output_dir,
              'faster_rcnn_{}_{}_{}.pth'.format(lastsession, lastepoch, lastpoints))
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name)
            # print (checkpoint['model'].keys())
            # print (fasterRCNN.state_dict().keys())
            # args.session = checkpoint['session']
            # args.start_epoch = checkpoint['epoch']
            # fasterRCNN.load_state_dict(checkpoint['model'])
            fasterRCNN.load_state_dict({'module.'+k:v for k,v in checkpoint['model'].items()})
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
              cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))

    if epoch == args.max_epochs:
        break
    epoch = epoch + 1
   

    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      proposal_boxes.data.resize_(data[4].size()).copy_(data[4])
      num_proposals.data.resize_(data[5].size()).copy_(data[5])

      # rpn_loss_cls, rpn_loss_box, \
      # print (gt_boxes)
      # print (proposal_boxes)
      fasterRCNN.zero_grad()

      rois, cls_prob, bbox_pred, rois_label, \
      RCNN_loss_cls_proposal, \
      RCNN_loss_cls_alpha_positive, RCNN_loss_cls_alpha_negative, RCNN_loss_cls_beta_positive, \
      RCNN_loss_bbox, RCNN_loss_bbox_beta, \
      kl_loss, kl_loss_cls = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, proposal_boxes, num_proposals)
      # RCNN_loss_bbox_beta, kl_loss  = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      # wx1, wy1, wx2, wy2, \
      # dx1, dy1, dx2, dy2, \
      # ox1, oy1, ox2, oy2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
      #      + RCNN_loss_cls.mean() + 10 * RCNN_loss_bbox.mean() \

      # loss = RCNN_loss_cls.mean() + 10 * RCNN_loss_bbox.mean() \
      #      + 50 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()

      # loss = args.cls_weight        * RCNN_loss_cls.mean() + \
      #        args.bbox_weight       * RCNN_loss_bbox.mean()

      # loss = args.cls_weight_proposal *  RCNN_loss_cls_proposal.mean() + \
      #        args.bbox_weight         * RCNN_loss_bbox.mean()

      loss = args.bbox_weight         * RCNN_loss_bbox.mean()

      if args.cls_neighbor:
          loss = loss + \
             args.kl_weight              * kl_loss_cls.mean() +\
             args.cls_weights_alpha_pos  * RCNN_loss_cls_alpha_positive.mean() +\
             args.cls_weights_alpha_neg  * RCNN_loss_cls_alpha_negative.mean() +\
             args.cls_weights_beta_pos   * RCNN_loss_cls_beta_positive.mean()
      else:
          loss = loss + \
             args.cls_weights_proposal   * RCNN_loss_cls_proposal.mean() 
          

      if args.reg_neighbor:
          loss = loss + \
             args.bbox_beta_weight    * RCNN_loss_bbox_beta.mean() + \
             args.kl_weight           * kl_loss.mean() 
      
      
      loss_temp += loss.data[0]

      # loss = args.cls_weight        * RCNN_loss_cls.mean() + \
      #        args.bbox_alpha_weight * RCNN_loss_bbox.mean() + \
      #        args.bbox_beta_weight  * RCNN_loss_bbox_beta.mean() + \
      #        args.kl_weight         * kl_loss.mean() + \
      #        args.cls_weight        * RCNN_loss_cls_beta.mean() + \
      #        args.kl_weight         * kl_loss_cls.mean()


      #     + 10 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()
      #      + kl_loss.mean()
      #      + 10 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()
      #      + RCNN_loss_cls.mean() + 15 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 10 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() +  RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 5 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 3 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

      # loss = RCNN_loss_bbox.mean()

      # loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

      # loss_temp += loss.data.item()

      # backward


      #     + 10 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()
      #      + kl_loss.mean()
      #      + 10 * RCNN_loss_bbox_beta.mean() + kl_loss.mean()
      #      + RCNN_loss_cls.mean() + 15 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 10 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() +  RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 5 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + 3 * RCNN_loss_bbox.mean()
      #      + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

      # loss = RCNN_loss_bbox.mean()

      # loss = RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

      # loss_temp += loss.data[0]
      # loss_temp += loss.data.item()

      # backward
      #pdb.set_trace()
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        if args.mGPUs:
          # loss_rpn_cls = rpn_loss_cls.mean().data[0]
          # loss_rpn_box = rpn_loss_box.mean().data[0]
          loss_rpn_cls = 0
          loss_rpn_box = 0
          # loss_rcnn_cls = RCNN_loss_cls.mean().data[0]

          
          # loss_rcnn_cls = RCNN_loss_cls_proposal.mean().data[0]
          loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
          # loss_rcnn_cls = RCNN_loss_cls.mean().data.item()
          # loss_rcnn_box = RCNN_loss_bbox.mean().data.item()
          ## v0.2

          if args.reg_neighbor:
              loss_rcnn_box_beta = RCNN_loss_bbox_beta.mean().data[0]
              loss_kl = kl_loss.mean().data[0]
          else:
              loss_rcnn_box_beta = 0 
              loss_kl = 0
          
          if args.cls_neighbor:
              # loss_rcnn_cls_beta = RCNN_loss_cls_beta.mean().data[0]
              # loss_rcnn_cls_alpha = RCNN_loss_cls.mean().data[0]
              loss_rcnn_alpha_positive =  RCNN_loss_cls_alpha_positive.mean().data[0]
              loss_rcnn_alpha_negative =  RCNN_loss_cls_alpha_negative.mean().data[0] 
              loss_rcnn_beta_positive  =  RCNN_loss_cls_beta_positive.mean().data[0]
              loss_rcnn_cls_proposal = 0 
              loss_kl_cls = kl_loss_cls.mean().data[0]
          else:
              loss_kl_cls = 0
              loss_rcnn_alpha_positive = 0 
              loss_rcnn_alpha_negative = 0  
              loss_rcnn_beta_positive  = 0 
              loss_rcnn_cls_proposal = RCNN_loss_cls_proposal.mean().data[0]

          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          # loss_rpn_cls = rpn_loss_cls.data[0]
          # loss_rpn_box = rpn_loss_box.data[0]
          loss_rpn_cls = 0
          loss_rpn_box = 0
          # loss_rcnn_cls = RCNN_loss_cls_proposal.data[0]
          loss_rcnn_box = RCNN_loss_bbox.data[0]
          # loss_rcnn_cls = RCNN_loss_cls.data.item()
          # loss_rcnn_box = RCNN_loss_bbox.data.item()
          ## v0.2

          if args.reg_neighbor:
              loss_rcnn_box_beta = RCNN_loss_bbox_beta.data[0]
              loss_kl = kl_loss.data[0]
          else:
              loss_rcnn_box_beta = 0 
              loss_kl = 0
          
          if args.cls_neighbor:
              # loss_rcnn_cls_beta = RCNN_loss_cls_beta.data[0]
              # loss_kl_cls = kl_loss_cls.data[0]
              # loss_rcnn_cls_alpha = RCNN_loss_cls.data[0]
              loss_rcnn_alpha_positive =  RCNN_loss_cls_alpha_positive.data[0]
              loss_rcnn_alpha_negative =  RCNN_loss_cls_alpha_negative.data[0] 
              loss_rcnn_beta_positive  =  RCNN_loss_cls_beta_positive.data[0]
              loss_rcnn_cls_proposal = 0 
              loss_kl_cls = kl_loss_cls.data[0]
          else:
              # loss_rcnn_cls_beta = 0 
              # loss_rcnn_cls_alpha = 0
              # loss_kl_cls = 0
              loss_kl_cls = 0
              loss_rcnn_alpha_positive = 0 
              loss_rcnn_alpha_negative = 0  
              loss_rcnn_beta_positive  = 0 
              loss_rcnn_cls_proposal = RCNN_loss_cls_proposal.data[0]


          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr), flush=True)
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start), flush=True)
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls_proposal: %.4f, rcnn_cls_alpha_pos: %.4f, rcnn_cls_alpha_neg: %.4f, rcn_cls_beta_pos: %.4f, rcnn_box %.4f, bbox_beta %.4f, kl %.4f, kl_cls %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls_proposal, loss_rcnn_alpha_positive, loss_rcnn_alpha_negative, loss_rcnn_beta_positive,  loss_rcnn_box, loss_rcnn_box_beta, loss_kl, loss_kl_cls), flush=True)
        # print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls_proposal: %.4f, rcnn_box %.4f, bbox_beta %.4f, kl %.4f, cls_alpha %.4f, cls_beta %.4f, kl_cls %.4f" \
        #              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_box_beta, loss_kl, loss_rcnn_cls_alpha, loss_rcnn_cls_beta, loss_kl_cls), flush=True)
        #              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, 0.0, loss_kl))

        ## 
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        ## loss temp is nan
        loss_temp = float('nan') 
        print (loss_temp)
        if math.isnan(loss_temp):
            nan_flag = True
            print ("################NAN in epoch " +str(epoch)+ "########")
            break
        loss_temp = 0
        start = time.time()

    ## if nan, no save, continue
    if nan_flag:
        continue

    if args.mGPUs:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
    else:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
    print('save model: {}'.format(save_name),flush=True)

    end = time.time()
    print(end - start)
