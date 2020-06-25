import numpy as np
from scipy.optimize import linear_sum_assignment

def format_MOTChallenge(frame_num, trackers):
    trk_fmt = np.zeros((len(trackers), 10), dtype="int")
    for i, (trk, trk_num) in enumerate(trackers):
        x1 = int(trk.tlwh[0])
        y1 = int(trk.tlwh[1])
        w = int(trk.tlwh[2])
        h = int(trk.tlwh[3])

        trk_fmt[i] = (frame_num, trk_num, x1, y1, w, h, 1, -1, -1, -1)

    return trk_fmt

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes/detections
    """
    bb_test = bb_test.to_tlbr()
    bb_gt = bb_gt.to_tlbr()

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

def hungarian_matching(detections, trackers, cost_threshold):
    # Ensure correct data format
    detections = list(detections)
    trackers = list(trackers)
    if(len(trackers) == 0):
        return set(), set(detections), set()

    # Compute Cost Matrix
    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = iou(det, trk.get_state())

    # Run Hungarian Algorithm
    matched_indices = linear_sum_assignment(-cost_matrix)
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    # Find detections and trackers which were not matched
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(det)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(trk)

    # Filter out matched with high cost
    matches = []
    for d, t in matched_indices:
        if(cost_matrix[d, t] < cost_threshold):
            unmatched_detections.append(detections[d])
            unmatched_trackers.append(trackers[t])
        else:
            matches.append((detections[d], trackers[t]))
    if(len(matches) == 0):
        matches = set()
    else:
        matches = set(matches)

    return matches, set(unmatched_detections), set(unmatched_trackers)
