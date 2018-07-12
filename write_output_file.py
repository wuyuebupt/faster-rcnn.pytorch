import os,sys
import pickle


if __name__ == '__main__':
	det_file = sys.argv[1]
	boxes = pickle.load(open(det_file, 'rb'))
	# print len(boxes)
	# print boxes[27]
	# print len(boxes[27])
	# print len(boxes[27][1])
	# print boxes[27][1]
	for imgid in range(len(boxes[0])):
		for class_id in range(1,len(boxes)):
			objs = boxes[class_id][imgid]
			for obj in objs:
				# print obj
				# print "bbox: {:.2f} {:.2f} {:.2f} {:.2f} score: {:.6f}".format(obj[0], obj[1],obj[2],obj[3],obj[4])
				# imgid cls_inde score, bbox
				print "{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}".format(imgid + 1, class_id, obj[4], obj[0], obj[1],obj[2],obj[3])
