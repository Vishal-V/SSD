"""Layer utilities"""

import numpy as np

def anchor_boxes(feature_shape, image_shape, index=0, n_layers=4, aspect_ratios=(1, 2, 0.5)):
    """ Compute the anchor boxes for a given feature map"""

	s = np.linspace(0.2, 0.9, n_layers + 1)
	size = [[s[i], np.sqrt(s[i]*s[i+1])] for i in range(len(s) - 1)]
	sizes = size[index]
	
	n_boxes = len(aspect_ratios) + 1
	image_height, image_width, _ = image_shape
	feature_height, feature_width, _ = feature_shape
	norm_height = image_height * sizes[0]
	norm_width = image_width * sizes[0]
	
	width_height = []
	for ar in aspect_ratios:
	    box_width = norm_width * np.sqrt(ar)
	    box_height = norm_height / np.sqrt(ar)
	    width_height.append((box_width, box_height))
	
	box_width = image_width * sizes[1]
	box_height = image_height * sizes[1]
	width_height.append((box_width, box_height))

	width_height = np.array(width_height)

	grid_width = image_width / feature_width
	grid_height = image_height / feature_height

	start = grid_width * 0.5 
	end = (feature_width - 0.5) * grid_width
	cx = np.linspace(start, end, feature_width)

	start = grid_height * 0.5
	end = (feature_height - 0.5) * grid_height
	cy = np.linspace(start, end, feature_height)

	# grid of box centers
	cx_grid, cy_grid = np.meshgrid(cx, cy)
	cx_grid = np.expand_dims(cx_grid, -1) 
	cy_grid = np.expand_dims(cy_grid, -1)

	# tensor = (feature_map_height, feature_map_width, n_boxes, 4)
	boxes = np.zeros((feature_height, feature_width, n_boxes, 4))

	boxes[..., 0] = np.tile(cx_grid, (1, 1, n_boxes))
	boxes[..., 1] = np.tile(cy_grid, (1, 1, n_boxes))
	boxes[..., 2] = width_height[:, 0]
	boxes[..., 3] = width_height[:, 1]

	# convert (cx, cy, w, h) to (xmin, xmax, ymin, ymax)
	boxes = centroid2minmax(boxes)
	boxes = np.expand_dims(boxes, axis=0)
	return boxes


def centroid2minmax(boxes):
    """(cx, cy, w, h) to (xmin, xmax, ymin, ymax)"""
    
    minmax= np.copy(boxes).astype(np.float)
    minmax[..., 0] = boxes[..., 0] - (0.5 * boxes[..., 2])
    minmax[..., 1] = boxes[..., 0] + (0.5 * boxes[..., 2])
    minmax[..., 2] = boxes[..., 1] - (0.5 * boxes[..., 3])
    minmax[..., 3] = boxes[..., 1] + (0.5 * boxes[..., 3])
    return minmax


def intersection(boxes1, boxes2):
	m = boxes1.shape[0]
	n = boxes2.shape[0]

	xmin = 0
	xmax = 1
	ymin = 2
	ymax = 3

	boxes1_min = np.expand_dims(boxes1[:, [xmin, ymin]], axis=1)
	boxes1_min = np.tile(boxes1_min, reps=(1, n, 1))
	boxes2_min = np.expand_dims(boxes2[:, [xmin, ymin]], axis=0)
	boxes2_min = np.tile(boxes2_min, reps=(m, 1, 1))
	min_xy = np.maximum(boxes1_min, boxes2_min)

	boxes1_max = np.expand_dims(boxes1[:, [xmax, ymax]], axis=1)
	boxes1_max = np.tile(boxes1_max, reps=(1, n, 1))
	boxes2_max = np.expand_dims(boxes2[:, [xmax, ymax]], axis=0)
	boxes2_max = np.tile(boxes2_max, reps=(m, 1, 1))
	max_xy = np.minimum(boxes1_max, boxes2_max)

	side_lengths = np.maximum(0, max_xy - min_xy)

	intersection_areas = side_lengths[:, :, 0] * side_lengths[:, :, 1]
	return intersection_areas


def union(boxes1, boxes2, intersection_areas):

	m = boxes1.shape[0] # number of boxes in boxes1
	n = boxes2.shape[0] # number of boxes in boxes2

	xmin = 0
	xmax = 1
	ymin = 2
	ymax = 3

	width = (boxes1[:, xmax] - boxes1[:, xmin])
	height = (boxes1[:, ymax] - boxes1[:, ymin])
	areas = width * height
	boxes1_areas = np.tile(np.expand_dims(areas, axis=1), reps=(1,n))
	width = (boxes2[:,xmax] - boxes2[:,xmin])
	height = (boxes2[:,ymax] - boxes2[:,ymin])
	areas = width * height
	boxes2_areas = np.tile(np.expand_dims(areas, axis=0), reps=(m,1))

	union_areas = boxes1_areas + boxes2_areas - intersection_areas
	return union_areas


def iou(boxes1, boxes2):
	intersection_areas = intersection(boxes1, boxes2)
	union_areas = union(boxes1, boxes2, intersection_areas)
	return intersection_areas / union_areas


def get_gt_data(iou, n_classes=4, anchors=None, labels=None, normalize=False, threshold=0.6):
    """Retrieve ground truth class, bbox offset, and mask"""

	maxiou_per_gt = np.argmax(iou, axis=0)

	if threshold < 1.0:
	    iou_gt_thresh = np.argwhere(iou>threshold)
	    if iou_gt_thresh.size > 0:
	        extra_anchors = iou_gt_thresh[:,0]
	        extra_classes = iou_gt_thresh[:,1]
	        extra_labels = labels[extra_classes]
	        indexes = [maxiou_per_gt, extra_anchors]
	        maxiou_per_gt = np.concatenate(indexes, axis=0)
	        labels = np.concatenate([labels, extra_labels], axis=0)

	# mask generation
	gt_mask = np.zeros((iou.shape[0], 4))
	gt_mask[maxiou_per_gt] = 1.0

	# class generation
	gt_class = np.zeros((iou.shape[0], n_classes))
	gt_class[:, 0] = 1
	gt_class[maxiou_per_gt, 0] = 0
	maxiou_col = np.reshape(maxiou_per_gt, (maxiou_per_gt.shape[0], 1))
	label_col = np.reshape(labels[:,4], (labels.shape[0], 1)).astype(int)
	row_col = np.append(maxiou_col, label_col, axis=1)
	
	gt_class[row_col[:,0], row_col[:,1]]  = 1.0

	gt_offset = np.zeros((iou.shape[0], 4))

	if normalize:
	    anchors = minmax2centroid(anchors)
	    labels = minmax2centroid(labels)
	    offsets1 = labels[:, 0:2] - anchors[maxiou_per_gt, 0:2]
	    offsets1 /= anchors[maxiou_per_gt, 2:4]
	    offsets1 /= 0.1

	    offsets2 = np.log(labels[:, 2:4]/anchors[maxiou_per_gt, 2:4])
	    offsets2 /= 0.2  

	    offsets = np.concatenate([offsets1, offsets2], axis=-1)
	else:
	    offsets = labels[:, 0:4] - anchors[maxiou_per_gt]

	gt_offset[maxiou_per_gt] = offsets
	return gt_class, gt_offset, gt_mask
