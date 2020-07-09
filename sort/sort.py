import numpy as np
from operator import itemgetter
from .utils import hungarian_matching, iou
from .tracklet import Tracklet


class SORT(object):
    def __init__(self, max_age=1, min_hits=3, matching="basic", round1_iou=0.3, round2_iou=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracklet_counter = 1
        self.frames = 0

        self.round1_iou = round1_iou
        self.round2_iou = round2_iou
        if matching=="basic":
            self._match = self._basic_matching
        elif matching=="cascade":
            self._match = self._matching_cascade
        else:
            raise RuntimeError("Specified matching not supported.")

        self.trackers = []
        self.lost_trackers = []

    def get_trackers(self):
        return self.trackers, self.lost_trackers

    def _init_tracklet(self, dets):
        new_trackers = []
        for det in dets:
            trk = Tracklet(self.tracklet_counter, det, self.min_hits)
            new_trackers.append(trk)
            self.tracklet_counter += 1

        self.trackers += new_trackers

    def predict(self):
        self.frames += 1
        for trk in self.trackers:
            trk.predict()
        
    def update(self, dets):
        # Associate predictions to observations
        matched, unmatched_dets, unmatched_trks = self._match(dets)
        new_trackers = []

        # Update matched trackers with assigned detections
        for det, trk in matched:
            trk.update(det)
            new_trackers.append(trk)

        # Delete trackers with too high age
        for trk in unmatched_trks:
            if trk.time_since_update > self.max_age:
                self.lost_trackers.append(trk)
            else:
                new_trackers.append(trk)

        # Update the active trackers
        self.trackers = new_trackers

        # Create new trackers for unmatched detections
        self._init_tracklet(unmatched_dets)

        # Format output
        ret = []
        for trk in self.trackers:
            # If alive and not in probation, add it to the return
            if not trk.in_probation or self.frames < self.min_hits:
                d = trk.get_state()
                ret.append((d, trk.id))

        return ret

    def _matching_cascade(self, dets):
        unmatched_dets = dets.copy()
        matched_trks = set()
        unmatched_trks = set()

        # First round matching with cascade by age for confirmed trackers
        for l in range(1, self.max_age+1):
            trk_l = [trk for trk in self.trackers if trk.time_since_update == l and not trk.in_probation]
            matches_l, unmatched_dets, unmatched_trks_l = hungarian_matching(unmatched_dets, trk_l, self.round1_iou)
            matched_trks.update(matches_l)
            unmatched_trks.update(unmatched_trks_l)

        # Second round matching on remaining detections, trackers in probation,
        # and still unmatched trackers
        unconfirmed_trks = set([trk for trk in self.trackers if trk.in_probation])
        unmatched_trks.update(unconfirmed_trks)

        matches, unmatched_dets, unmatched_trks = hungarian_matching(unmatched_dets, unmatched_trks, self.round2_iou)
        matched_trks.update(matches)

        return matched_trks, unmatched_dets, unmatched_trks
        
    def _basic_matching(self, dets):
        return hungarian_matching(dets, self.trackers, self.round1_iou)