import cv2
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, \
    argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time

import numpy as np
import util


class CMT:
    def __init__(self):
        self.DETECTOR = 'BRISK'
        self.DESCRIPTOR = 'BRISK'
        self.DESC_LENGTH = 512
        self.MATCHER = 'BruteForce-Hamming'
        self.THR_OUTLIER = 20
        self.THR_CONF = 0.75
        self.THR_RATIO = 0.8

        self.estimate_scale = True
        self.estimate_rotation = True


def initialise(CMTobject, im_gray0, tl, br):
    CMTobject.DETECTOR = 'BRISK'
    CMTobject.DESCRIPTOR = 'BRISK'
    CMTobject.DESC_LENGTH = 512
    CMTobject.MATCHER = 'BruteForce-Hamming'
    CMTobject.THR_OUTLIER = 20
    CMTobject.THR_CONF = 0.75
    CMTobject.THR_RATIO = 0.8

    CMTobject.estimate_scale = True
    CMTobject.estimate_rotation = True

    # Initialise detector, descriptor, matcher
    if cv2.__version__ == '3.0.0':
        CMTobject.detector = cv2.BRISK_create()
        CMTobject.descriptor = cv2.BRISK_create()
    else:
        CMTobject.detector = cv2.FeatureDetector_create(CMTobject.DETECTOR)
        CMTobject.descriptor = cv2.DescriptorExtractor_create(CMTobject.DESCRIPTOR)
    CMTobject.matcher = cv2.DescriptorMatcher_create(CMTobject.MATCHER)

    # Get initial keypoints in whole image
    keypoints_cv = CMTobject.detector.detect(im_gray0)

    # Remember keypoints that are in the rectangle as selected keypoints
    ind = util.in_rect(keypoints_cv, tl, br)
    selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
    selected_keypoints_cv, CMTobject.selected_features = CMTobject.descriptor.compute(im_gray0, selected_keypoints_cv)
    selected_keypoints = util.keypoints_cv_to_np(selected_keypoints_cv)
    num_selected_keypoints = len(selected_keypoints_cv)

    if num_selected_keypoints == 0:
        raise Exception('No keypoints found in selection')

    # Remember keypoints that are not in the rectangle as background keypoints
    background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
    background_keypoints_cv, background_features = CMTobject.descriptor.compute(im_gray0, background_keypoints_cv)
    _ = util.keypoints_cv_to_np(background_keypoints_cv)

    # Assign each keypoint a class starting from 1, background is 0
    CMTobject.selected_classes = array(range(num_selected_keypoints)) + 1
    background_classes = zeros(len(background_keypoints_cv))

    # Stack background features and selected features into database
    CMTobject.features_database = vstack((background_features, CMTobject.selected_features))

    # Same for classes
    CMTobject.database_classes = hstack((background_classes, CMTobject.selected_classes))

    # Get all distances between selected keypoints in squareform
    pdist = scipy.spatial.distance.pdist(selected_keypoints)
    CMTobject.squareform = scipy.spatial.distance.squareform(pdist)

    # Get all angles between selected keypoints
    angles = np.empty((num_selected_keypoints, num_selected_keypoints))
    for k1, i1 in zip(selected_keypoints, range(num_selected_keypoints)):
        for k2, i2 in zip(selected_keypoints, range(num_selected_keypoints)):
            # Compute vector from k1 to k2
            v = k2 - k1

            # Compute angle of this vector with respect to x axis
            angle = math.atan2(v[1], v[0])

            # Store angle
            angles[i1, i2] = angle

    CMTobject.angles = angles

    # Find the center of selected keypoints
    center = np.mean(selected_keypoints, axis=0)

    # Remember the rectangle coordinates relative to the center
    CMTobject.center_to_tl = np.array(tl) - center
    CMTobject.center_to_tr = np.array([br[0], tl[1]]) - center
    CMTobject.center_to_br = np.array(br) - center
    CMTobject.center_to_bl = np.array([tl[0], br[1]]) - center

    # Calculate springs of each keypoint
    CMTobject.springs = selected_keypoints - center

    # Set start image for tracking
    CMTobject.im_prev = im_gray0

    # Make keypoints 'active' keypoints
    CMTobject.active_keypoints = np.copy(selected_keypoints)

    # Attach class information to active keypoints
    CMTobject.active_keypoints = hstack((selected_keypoints, CMTobject.selected_classes[:, None]))

    # Remember number of initial keypoints
    CMTobject.num_initial_keypoints = len(selected_keypoints_cv)

    return CMTobject


def estimate(CMTobject, keypoints):
    center = array((nan, nan))
    scale_estimate = nan
    med_rot = nan

    # At least 2 keypoints are needed for scale
    if keypoints.size > 1:

        # Extract the keypoint classes
        keypoint_classes = keypoints[:, 2].squeeze().astype(np.int)

        # Retain singular dimension
        if keypoint_classes.size == 1:
            keypoint_classes = keypoint_classes[None]

        # Sort
        ind_sort = argsort(keypoint_classes)
        keypoints = keypoints[ind_sort]
        keypoint_classes = keypoint_classes[ind_sort]

        # Get all combinations of keypoints
        all_combs = array([val for val in itertools.product(range(keypoints.shape[0]), repeat=2)])

        # But exclude comparison with itCMTobject
        all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]

        # Measure distance between allcombs[0] and allcombs[1]
        ind1 = all_combs[:, 0]
        ind2 = all_combs[:, 1]

        class_ind1 = keypoint_classes[ind1] - 1
        class_ind2 = keypoint_classes[ind2] - 1

        duplicate_classes = class_ind1 == class_ind2

        if not all(duplicate_classes):
            ind1 = ind1[~duplicate_classes]
            ind2 = ind2[~duplicate_classes]

            class_ind1 = class_ind1[~duplicate_classes]
            class_ind2 = class_ind2[~duplicate_classes]

            pts_allcombs0 = keypoints[ind1, :2]
            pts_allcombs1 = keypoints[ind2, :2]

            # This distance might be 0 for some combinations,
            # as it can happen that there is more than one keypoint at a single location
            dists = util.L2norm(pts_allcombs0 - pts_allcombs1)

            original_dists = CMTobject.squareform[class_ind1, class_ind2]

            scalechange = dists / original_dists

            # Compute angles
            angles = np.empty((pts_allcombs0.shape[0]))

            v = pts_allcombs1 - pts_allcombs0
            angles = np.arctan2(v[:, 1], v[:, 0])

            original_angles = CMTobject.angles[class_ind1, class_ind2]

            angle_diffs = angles - original_angles

            # Fix long way angles
            long_way_angles = np.abs(angle_diffs) > math.pi

            angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(
                angle_diffs[long_way_angles]) * 2 * math.pi

            scale_estimate = median(scalechange)
            if not CMTobject.estimate_scale:
                scale_estimate = 1;

            med_rot = median(angle_diffs)
            if not CMTobject.estimate_rotation:
                med_rot = 0;

            keypoint_class = keypoints[:, 2].astype(np.int)
            votes = keypoints[:, :2] - scale_estimate * (util.rotate(CMTobject.springs[keypoint_class - 1], med_rot))

            # Remember all votes including outliers
            CMTobject.votes = votes

            # Compute pairwise distance between votes
            pdist = scipy.spatial.distance.pdist(votes)

            # Compute linkage between pairwise distances
            linkage = scipy.cluster.hierarchy.linkage(pdist)

            # Perform hierarchical distance-based clustering
            T = scipy.cluster.hierarchy.fcluster(linkage, CMTobject.THR_OUTLIER, criterion='distance')

            # Count votes for each cluster
            cnt = np.bincount(T)  # Dummy 0 label remains

            # Get largest class
            Cmax = argmax(cnt)

            # Identify inliers (=members of largest class)
            inliers = T == Cmax
            # inliers = med_dists < THR_OUTLIER

            # Remember outliers
            CMTobject.outliers = keypoints[~inliers, :]

            # Stop tracking outliers
            keypoints = keypoints[inliers, :]

            # Remove outlier votes
            votes = votes[inliers, :]

            # Compute object center
            center = np.mean(votes, axis=0)

    return (center, scale_estimate, med_rot, keypoints)


def process_frame(CMTobject, im_gray):
    tracked_keypoints, _ = util.track(CMTobject.im_prev, im_gray, CMTobject.active_keypoints)
    (center, scale_estimate, rotation_estimate, tracked_keypoints) = estimate(CMTobject, tracked_keypoints)

    # Detect keypoints, compute descriptors
    keypoints_cv = CMTobject.detector.detect(im_gray)
    keypoints_cv, features = CMTobject.descriptor.compute(im_gray, keypoints_cv)

    # Create list of active keypoints
    active_keypoints = zeros((0, 3))

    # Get the best two matches for each feature
    matches_all = CMTobject.matcher.knnMatch(features, CMTobject.features_database, 2)
    # Get all matches for selected features
    if not any(isnan(center)):
        selected_matches_all = CMTobject.matcher.knnMatch(features, CMTobject.selected_features,
                                                          len(CMTobject.selected_features))

    # For each keypoint and its descriptor
    if len(keypoints_cv) > 0:
        transformed_springs = scale_estimate * util.rotate(CMTobject.springs, -rotation_estimate)
        for i in range(len(keypoints_cv)):

            # Retrieve keypoint location
            location = np.array(keypoints_cv[i].pt)

            # First: Match over whole image
            # Compute distances to all descriptors
            matches = matches_all[i]
            distances = np.array([m.distance for m in matches])

            # Convert distances to confidences, do not weight
            combined = 1 - distances / CMTobject.DESC_LENGTH

            classes = CMTobject.database_classes

            # Get best and second best index
            bestInd = matches[0].trainIdx
            secondBestInd = matches[1].trainIdx

            # Compute distance ratio according to Lowe
            ratio = (1 - combined[0]) / (1 - combined[1])

            # Extract class of best match
            keypoint_class = classes[bestInd]

            # If distance ratio is ok and absolute distance is ok and keypoint class is not background
            if ratio < CMTobject.THR_RATIO and combined[0] > CMTobject.THR_CONF and keypoint_class != 0:
                # Add keypoint to active keypoints
                new_kpt = append(location, keypoint_class)
                active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

            # In a second step, try to match difficult keypoints
            # If structural constraints are applicable
            if not any(isnan(center)):

                # Compute distances to initial descriptors
                matches = selected_matches_all[i]
                distances = np.array([m.distance for m in matches])
                # Re-order the distances based on indexing
                idxs = np.argsort(np.array([m.trainIdx for m in matches]))
                distances = distances[idxs]

                # Convert distances to confidences
                confidences = 1 - distances / CMTobject.DESC_LENGTH

                # Compute the keypoint location relative to the object center
                relative_location = location - center

                # Compute the distances to all springs
                displacements = util.L2norm(transformed_springs - relative_location)

                # For each spring, calculate weight
                weight = displacements < CMTobject.THR_OUTLIER  # Could be smooth function

                combined = weight * confidences

                classes = CMTobject.selected_classes

                # Sort in descending order
                sorted_conf = argsort(combined)[::-1]  # reverse

                # Get best and second best index
                bestInd = sorted_conf[0]
                secondBestInd = sorted_conf[1]

                # Compute distance ratio according to Lowe
                ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd])

                # Extract class of best match
                keypoint_class = classes[bestInd]

                # If distance ratio is ok and absolute distance is ok and keypoint class is not background
                if ratio < CMTobject.THR_RATIO and combined[bestInd] > CMTobject.THR_CONF and keypoint_class != 0:

                    # Add keypoint to active keypoints
                    new_kpt = append(location, keypoint_class)

                    # Check whether same class already exists
                    if active_keypoints.size > 0:
                        same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
                        active_keypoints = np.delete(active_keypoints, same_class, axis=0)

                    active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

    # If some keypoints have been tracked
    if tracked_keypoints.size > 0:

        # Extract the keypoint classes
        tracked_classes = tracked_keypoints[:, 2]

        # If there already are some active keypoints
        if active_keypoints.size > 0:

            # Add all tracked keypoints that have not been matched
            associated_classes = active_keypoints[:, 2]
            missing = ~np.in1d(tracked_classes, associated_classes)
            active_keypoints = append(active_keypoints, tracked_keypoints[missing, :], axis=0)

        # Else use all tracked keypoints
        else:
            active_keypoints = tracked_keypoints

    # Update object state estimate
    _ = active_keypoints
    CMTobject.center = center
    CMTobject.scale_estimate = scale_estimate
    CMTobject.rotation_estimate = rotation_estimate
    CMTobject.tracked_keypoints = tracked_keypoints
    CMTobject.active_keypoints = active_keypoints
    CMTobject.im_prev = im_gray
    CMTobject.keypoints_cv = keypoints_cv
    _ = time.time()

    CMTobject.tl = (nan, nan)
    CMTobject.tr = (nan, nan)
    CMTobject.br = (nan, nan)
    CMTobject.bl = (nan, nan)

    CMTobject.bb = array([nan, nan, nan, nan])

    CMTobject.has_result = False
    if not any(isnan(CMTobject.center)) and CMTobject.active_keypoints.shape[0] > CMTobject.num_initial_keypoints / 10:
        CMTobject.has_result = True

        tl = util.array_to_int_tuple(
            center + scale_estimate * util.rotate(CMTobject.center_to_tl[None, :], rotation_estimate).squeeze())
        tr = util.array_to_int_tuple(
            center + scale_estimate * util.rotate(CMTobject.center_to_tr[None, :], rotation_estimate).squeeze())
        br = util.array_to_int_tuple(
            center + scale_estimate * util.rotate(CMTobject.center_to_br[None, :], rotation_estimate).squeeze())
        bl = util.array_to_int_tuple(
            center + scale_estimate * util.rotate(CMTobject.center_to_bl[None, :], rotation_estimate).squeeze())

        min_x = min((tl[0], tr[0], br[0], bl[0]))
        min_y = min((tl[1], tr[1], br[1], bl[1]))
        max_x = max((tl[0], tr[0], br[0], bl[0]))
        max_y = max((tl[1], tr[1], br[1], bl[1]))

        CMTobject.tl = tl
        CMTobject.tr = tr
        CMTobject.bl = bl
        CMTobject.br = br

        CMTobject.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])

    return CMTobject