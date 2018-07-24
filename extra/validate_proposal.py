import os,sys
import cv2
import scipy.io as sio



image_prefix = '/home/yue/project/vid/code/detection_tracking/faster-rcnn.pytorch/data/imagenet/ILSVRC/Data/'
proposal_prefix = '/home/yue/project/vid/code/detection_tracking/faster-rcnn.pytorch/data/imagenet/ILSVRC/RPNs/'


if __name__ == '__main__':
	lines = open(sys.argv[1])
	for line in lines:
		print (line)
		arr = line.strip().split()
		if 1:
			SET = 'DET'
			image_path = os.path.join(image_prefix, SET, arr[0] + '.JPEG')
			proposal_path = os.path.join(proposal_prefix, SET, arr[0] + '.mat')
		else:
			SET = 'VID'
			image_path = os.path.join(image_prefix, SET, '%s/%06d.JPEG' % (arr[0], int(arr[2])))
			proposal_path = os.path.join(proposal_prefix, SET, '%s/%06d.mat' % (arr[0], int(arr[2])))
		img = cv2.imread(image_path)
		bbox = sio.loadmat(proposal_path)['boxes']
		print image_path
		print proposal_path
		print img.shape
		print bbox.shape
		print len(bbox)
		for box in bbox:
			# print box
			pt1 = (int(box[0]),int(box[1]))
			pt2 = (int(box[2]),int(box[3]))
			cv2.rectangle(img, pt1, pt2, [0,0,255])
			if True:
				print (box[4])
				cv2.imshow('abc', img)
				cv2.waitKey()
			
		cv2.imshow('abc', img)
		cv2.waitKey()
		
		


