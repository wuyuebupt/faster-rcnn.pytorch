import h5py
import numpy as np
import cv2
import argparse
import pdb

######################################################################
# write/read a simple matrix
######################################################################
def write_matrix():
    print('=== write 10x10 matrix ===')
    with h5py.File('demo.h5', 'w') as f:
        dset = f.create_dataset('mat', data=np.ones((10, 10)), dtype='i4')

def read_matrix():
    print('=== read images ===')
    f = h5py.File('demo.h5', 'r')
    dset = f['mat']        
    print(np.array(dset))

######################################################################
# write/read train and val images, gt cls and bbox, and proposals
######################################################################
def write_images():
    print('=== write train/val images ===')
    # load imge paths
    with open('train.txt') as f:    
        tr_img_paths = [s for s in f.read().split('\n') if s]
    with open('val.txt') as f:
        val_img_paths = [s for s in f.read().split('\n') if s]

    # gen h5 file
    # 2 groups: train and val
    # each image is a dataset
    with h5py.File('trainval.h5', 'w') as f:
        # create two groups: train, val
        tr_group = f.create_group('train')
        val_group = f.create_group('val')

        # train group
        # each image is a dataset, including data (HxWx3 nparray) and multiple attributes
        # we can use attribute to encode object ground truth class, bounding boxes, and proposals
        #
        # NOTE1: the image path is like train/xxxxxx.jpg
        #        this is not valid for h5, '/' should be replaced, e.g. '-'
        # NOTE2: each attribute should be either scalar or NumPy array
        #        each attribute should be small (<64k)
        for f in tr_img_paths:
            img = cv2.imread(f)  
            H, W, _ = img.shape
            key = f.replace('/', '-')
            dset = tr_group.create_dataset(key, data=img, chunks=None)
            dset.attrs['img_height'] = H
            dset.attrs['img_width'] = W
            dset.attrs['gt-cls'] = np.array([0, 1])
            dset.attrs['gt-bbox'] = np.array([
                [100, 100, 300, 300],
                [200, 200, 500, 500]
            ])
            dset.attrs['proposals'] = np.ones((300, 4))

        # val group
        # the same format to train group
        for f in val_img_paths:
            img = cv2.imread(f)
            H, W, _ = img.shape
            key = f.replace('/', '-')
            dset = val_group.create_dataset(key, data=img, chunks=None)
            dset.attrs['img_height'] = H
            dset.attrs['img_width'] = W
            dset.attrs['gt-cls'] = np.array([0, 1])
            dset.attrs['gt-bbox'] = np.array([
                [100, 100, 300, 300],
                [200, 200, 500, 500]
            ])
            dset.attrs['proposals'] = np.ones((300, 4))


def read_images():
    f = h5py.File('trainval.h5', 'r')
    
    # train group
    print('--- train group ---')
    tr_group = f['train']
    for key in tr_group.keys():
        dset = tr_group[key]
        print('{}: {}x{}, {} objects, {} proposals'.format(
            key,
            dset.attrs['img_height'],
            dset.attrs['img_width'],
            dset.attrs['gt-cls'].shape[0],
            dset.attrs['proposals'].shape[0]
        ))    

    # val group
    print('--- val group ---')
    val_group = f['val']
    for key in val_group.keys():
        dset = val_group[key]
        print('{}: {}x{}, {} objects, {} proposals'.format(
            key,
            dset.attrs['img_height'],
            dset.attrs['img_width'],
            dset.attrs['gt-cls'].shape[0],
            dset.attrs['proposals'].shape[0]
        ))    

  
###############################################################
# main function
###############################################################
if __name__ == '__main__':
    '''
    example command:
    write images into h5: python h5_demo.py --write --image
    read images from h5: python h5_demo.py --read --image
    '''
    parser = argparse.ArgumentParser(description='h5 demo')
    parser.add_argument('--write', action='store_true', help='write h5 file')
    parser.add_argument('--read', action='store_true', help='read h5 file')
    parser.add_argument('--matrix', action='store_true', help='read or write matrix')
    parser.add_argument('--image', action='store_true', help='read or write image')

    args = parser.parse_args()

    if args.matrix:
        if args.write:
            write_matrix()

        if args.read:
            read_matrix()

    if args.image:
        if args.write:
            write_images()
        if args.read:
            read_images()

