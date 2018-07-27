from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet_vid
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class imagenet_vid(imdb):
    def __init__(self, det_vid, image_set, devkit_path, data_path):
        imdb.__init__(self, image_set)
        self._det_vid = det_vid
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = data_path
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))

        self._classes = ('__background__',)
        self._wnid = (0,)

        for i in xrange(30):
            self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)

        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        self._image_ext = ['.JPEG']
        self._proposal_ext = ['.mat']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)


    def image_offline_proposal_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_offline_proposal_from_index(self._image_index[i])

    def image_offline_proposal_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        # image_path = os.path.join(self._data_path, 'Data', 'DET', self._image_set, index + self._image_ext[0])
        if self._det_vid == 'det':
            proposal_path = os.path.join(self._data_path, 'RPNs', 'DET', index[0] + self._proposal_ext[0])
            proposal_path_1 = os.path.join(self._data_path, 'RPNs', 'DET', index[1] + self._proposal_ext[0])
        else:
            proposal_path = os.path.join(self._data_path, 'RPNs', 'VID', index[0] + self._proposal_ext[0])
            proposal_path_1 = os.path.join(self._data_path, 'RPNs', 'VID', index[1] + self._proposal_ext[0])
            
        # image_path = os.path.join(self._data_path, 'Data', 'DET', self._image_set, index + self._image_ext[0])
        # NOTE: the proposal file might not exit in training 
        # 1084113 .mat  in offline proposals files
        # 1122397 .JPEG in all training images
        # assert os.path.exists(proposal_path), 'path does not exist: {}'.format(proposal_path)
        # return proposal_path
        return (proposal_path, proposal_path_1)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        print (index)
        # image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        # image_path = os.path.join(self._data_path, 'Data', 'DET', self._image_set, index + self._image_ext[0])
        if self._det_vid == 'det':
            image_path_first  = os.path.join(self._data_path, 'Data', 'DET', index[0] + self._image_ext[0])
            image_path_second = os.path.join(self._data_path, 'Data', 'DET', index[1] + self._image_ext[0])
        else:
            image_path_first  = os.path.join(self._data_path, 'Data', 'VID', index[0] + self._image_ext[0])
            image_path_second = os.path.join(self._data_path, 'Data', 'VID', index[1] + self._image_ext[0])
            if not os.path.exists(image_path_second):
                image_path_second = image_path_first
        
        assert os.path.exists(image_path_first), 'path does not exist: {}'.format(image_path_first)
        assert os.path.exists(image_path_second), 'path does not exist: {}'.format(image_path_second)
        image_path = (image_path_first, image_path_second)
            
        # image_path = os.path.join(self._data_path, 'Data', 'DET', self._image_set, index + self._image_ext[0])
        # assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # two datasets
       # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        print ("loading ", self._det_vid, " ", self._image_set)

        if self._image_set == 'train':
            if self._det_vid == 'det':
                # image_set_index_file = os.path.join(self._data_path, 'ImageSets', 'DET_VID', 'DET_train_30classes.txt')
                image_set_index_file = os.path.join(self._data_path, 'ImageSets', 'DET_VID', 'DET_train_rpn.txt')

            elif self._det_vid == 'vid':
                image_set_index_file = os.path.join(self._data_path, 'ImageSets', 'DET_VID', 'VID_train_15frames.txt')
            else:
                print ("should not happen")
    
            assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
            with open(image_set_index_file) as f:
                lines = [x.strip().split(' ') for x in f.readlines()]

            # if len(lines[0]) == 2: # det
            if self._det_vid == 'det':
                image_index = [(x[0], x[0]) for x in lines]
            else:
                image_index = [('%s/%06d' % (x[0], int(x[2])), '%s/%06d' % (x[0], int(x[2]) + 1)) for x in lines]
                # TODO: filter unexist pairs

                # self.image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
                # self._pattern = [x[0]+'/%06d' for x in lines]


            return image_index
            # for i in range(1,200):
            #     print(i)
            #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET', 'train_' + str(i) + '.txt')
            #     with open(image_set_file) as f:
            #         tmp_index = [x.strip() for x in f.readlines()]
            #         # print (tmp_index)
            #         vtmp_index = []
            #         for line in tmp_index:
            #             line = line.split(' ')
            #             # vtmp_index.append(self._data_path + '/Data/DET/train/' + line[0])
            #             if int(line[1]) == 1:
            #                 vtmp_index.append(line[0])
            #             # image_list = os.popen('ls ' + self._data_path + '/Data/DET/train/' + line[0] + '/*.JPEG').read().split()
            #             # image_list = os.popen('ls ' + self._data_path + '/Data/DET/train/' + line[0] + '*.JPEG').read().split()
            #             # tmp_list = []
            #             # for imgs in image_list:
            #             #     tmp_list.append(imgs[:-5])
            #             # vtmp_index = vtmp_index + tmp_list
            #             # print (len(vtmp_index))
            #     

            #     num_lines = len(vtmp_index)
            #     # print (num_lines)
            #     # exit()
            #     ids = np.random.permutation(num_lines)
            #     count = 0
            #     while count < 2000:
            #         image_index.append(vtmp_index[ids[count % num_lines]])
            #         count = count + 1
            # # for i in range(1,201):
            # # for i in range(1,201):
            # #     if self._valid_image_flag[i] == 1:
            # #         # image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
            # #         image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
            # #         with open(image_set_file) as f:
            # #             tmp_index = [x.strip() for x in f.readlines()]
            # #         num_lines = len(tmp_index)
            # #         ids = np.random.permutation(num_lines)
            # #         count = 0
            # #         while count < 2000:
            # #             image_index.append(tmp_index[ids[count % num_lines]])
            # #             count = count + 1
            # image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
            # f = open(image_set_file, 'w')
            # for lines in image_index:
            #     f.write(lines + '\n')
            # f.close()
        else:
            if self._det_vid == 'det':
                print ("sorry, not for det")
            elif self._det_vid == 'vid':
                image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET_VID', 'val.txt')

            # image_set_file = os.path.join(self._data_path, 'ImageSets', 'val.txt')
            with open(image_set_file) as f:
                image_index = [x.strip().split()[0] for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        cache_file = os.path.join(self.cache_path, self._det_vid + '_' + self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        # filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')
        # filename = os.path.join(self._data_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        if self._det_vid == 'det':
            filename_0 = os.path.join(self._data_path, 'Annotations', 'DET', index[0] + '.xml')
            filename_1 = os.path.join(self._data_path, 'Annotations', 'DET', index[1] + '.xml')
        else:
            filename_0 = os.path.join(self._data_path, 'Annotations', 'VID', index[0] + '.xml')
            filename_1 = os.path.join(self._data_path, 'Annotations', 'VID', index[1] + '.xml')
            if not os.path.exists(filename_1):
                filename_1 =  filename_0
 
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        # the first image label
        with open(filename_0) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)

        class_to_index = valid_objs = np.zeros((num_objs), dtype=np.bool)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            # cls = self._wnid_to_ind_image[
            #         str(get_data_from_tag(obj, "name")).lower().strip()]
            if not self._wnid_to_ind.has_key(str(get_data_from_tag(obj, "name")).lower().strip() ):
                continue
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        assert (boxes[:, 2] >= boxes[:, 0]).all()
        overlaps = scipy.sparse.csr_matrix(overlaps)

        # read the second image label
        with open(filename_1) as f:
            data_1 = minidom.parseString(f.read())

        objs_1 = data_1.getElementsByTagName('object')
        num_objs_1 = len(objs_1)

        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes_1 = np.zeros((num_objs_1, 4), dtype=np.int32)
        gt_classes_1 = np.zeros((num_objs_1), dtype=np.int32)
        overlaps_1 = np.zeros((num_objs_1, self.num_classes), dtype=np.float32)
        valid_objs_1 = np.zeros((num_objs_1), dtype=np.bool)

        class_to_index_1 = valid_objs_1 = np.zeros((num_objs_1), dtype=np.bool)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs_1):
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            # cls = self._wnid_to_ind_image[
            #         str(get_data_from_tag(obj, "name")).lower().strip()]
            if not self._wnid_to_ind.has_key(str(get_data_from_tag(obj, "name")).lower().strip() ):
                continue
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes_1[ix, :] = [x1, y1, x2, y2]
            gt_classes_1[ix] = cls
            overlaps_1[ix, cls] = 1.0

        assert (boxes_1[:, 2] >= boxes_1[:, 0]).all()
        overlaps_1 = scipy.sparse.csr_matrix(overlaps_1)

        # boxes returned
        boxes_pair = (boxes, boxes_1)
        gt_classes_pair = (gt_classes, gt_classes_1)
        overlaps_pair   = (overlaps, overlaps_1)
        return {'boxes' : boxes_pair,
                'gt_classes': gt_classes_pair,
                'gt_overlaps' : overlaps_pair,
                'flipped' : False}
        # return {'boxes' : boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps' : overlaps,
        #         'flipped' : False}


if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
