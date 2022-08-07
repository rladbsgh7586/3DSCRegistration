import open3d as o3d
import os
import logging
import numpy as np
import time

from util.trajectory import CameraPose
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy, get_matching_indices
from pygcn.train import get_predict_points
from lib.eval import find_nn_cpu
import math
import multiprocessing
from itertools import repeat
from functools import partial


def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
      xyz0, xyz1, feat0, feat1, distance_threshold,
      o3d.registration.TransformationEstimationPointToPoint(False), 4, [
          o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      ]
    , o3d.registration.RANSACConvergenceCriteria(400000, 500)
  # , o3d.registration.RANSACConvergenceCriteria(50000, 500)
  )
  return result_ransac.transformation


def gather_results(results):
  traj = []
  for r in results:
    success = r[0]
    if success:
      traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
  return traj


def gen_matching_pair(pts_num):
  matching_pairs = []
  for i in range(pts_num):
    for j in range(i + 1, pts_num):
      matching_pairs.append([i, j, pts_num])
  return matching_pairs


def read_data(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  return data['points'], data['xyz'], data['feature']
  # xyz = make_open3d_point_cloud(data['xyz'])
  # feat = make_open3d_feature_from_numpy(data['feature'])
  # return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  # logging.info("matching %s %s" % (name_i, name_j))
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)

  # num_rand_keypoints = 5000
  # if num_rand_keypoints > 0:
  #   # Randomly subsample N points
  #   Ni, Nj = len(xyz_i), len(xyz_j)
  #   inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
  #   inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)
  #
  #   sample_i, sample_j = xyz_i[inds_i], xyz_j[inds_j]
  #   feat_i, feat_j = feat_i[inds_i], feat_j[inds_j]
  #
  # xyz_i = make_open3d_point_cloud(sample_i)
  # xyz_j = make_open3d_point_cloud(sample_j)
  xyz_i = make_open3d_point_cloud(xyz_i)
  xyz_j = make_open3d_point_cloud(xyz_j)
  feat_i = make_open3d_feature_from_numpy(feat_i)
  feat_j = make_open3d_feature_from_numpy(feat_j)

  trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  return [True, i, j, s, np.linalg.inv(trans)]


def do_single_pair_evaluation_with_similarity(feature_path, set_name, m, voxel_size, similarity_path, threshold, gt_pose):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  if not os.path.exists("%s%s/%03d_%03d.npy" % (similarity_path, set_name, i, j)):
    return [False, i, j, s]
  matching_pair = np.load("%s%s/%03d_%03d.npy" % (similarity_path, set_name, i, j))
  samples = matching_pair[:, 0]
  matching_pair = matching_pair[matching_pair[:, 2] > threshold]
  if len(matching_pair) == 0:
    trans = np.identity(4)
    # return [True, i, j, s, np.linalg.inv(trans)], len(matching_pair)

  # logging.info("matching with similarity %s %s" % (name_i, name_j))
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  src = make_open3d_point_cloud(data_i['xyz'])
  tgt = make_open3d_point_cloud(data_j['xyz'])
  matches = get_matching_indices(src, tgt, np.linalg.inv(gt_pose), voxel_size*1.5)
  if len(matches) < 2:
    coord_i_gt = []
  else:
    coord_i_gt = np.array(matches)[:, 0]
  xyz_i, xyz_j = data_i['xyz'][matching_pair[:, 0]], data_j['xyz'][matching_pair[:, 1]]
  feat_i, feat_j = data_i['feature'][matching_pair[:, 0]], data_j['feature'][matching_pair[:, 1]]

  TP, TN, FP, FN = 0, 0, 0, 0
  for i in samples:
    if i in coord_i_gt:
      if i in matching_pair[:, 0]:
        TP+=1
      else:
        FN+=1
    else:
      if i in matching_pair[:, 0]:
        FP += 1
      else:
        TN += 1
  return TP, TN, FP, FN

def do_single_pair_matching_with_similarity(feature_path, set_name, m, voxel_size, similarity_path, threshold):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  if not os.path.exists("%s%s/%03d_%03d.npy" % (similarity_path, set_name, i, j)):
    return [False, i, j, s]
  matching_pair = np.load("%s%s/%03d_%03d.npy" % (similarity_path, set_name, i, j))
  matching_pair = matching_pair[matching_pair[:, 2] > threshold]
  if len(matching_pair) == 0:
    trans = np.identity(4)
    return [True, i, j, s, np.linalg.inv(trans)], len(matching_pair)

  # logging.info("matching with similarity %s %s" % (name_i, name_j))
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))

  xyz_i, xyz_j = data_i['xyz'][matching_pair[:, 0]], data_j['xyz'][matching_pair[:, 1]]
  feat_i, feat_j = data_i['feature'][matching_pair[:, 0]], data_j['feature'][matching_pair[:, 1]]
  xyz_i = make_open3d_point_cloud(xyz_i)
  xyz_j = make_open3d_point_cloud(xyz_j)
  feat_i = make_open3d_feature_from_numpy(feat_i)
  feat_j = make_open3d_feature_from_numpy(feat_j)

  trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  return [True, i, j, s, np.linalg.inv(trans)], len(matching_pair)

def do_single_pair_matching_with_gcn(feature_path, set_name, m, voxel_size, graph_path, model):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)

  logging.info("matching with gcn %s %s" % (name_i, name_j))
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  xyz_i, xyz_j = data_i['xyz'], data_j['xyz']
  feat_i, feat_j = data_i['feature'], data_j['feature']
  content_path = graph_path + 'test_content_%s_%03d_%03d.npy' % (set_name, i, j)
  pairs_path = graph_path + 'test_pairs_%s_%03d_%03d.npy'% (set_name, i, j)
  feature_match_path = graph_path + 'test_feature_matching_%s_%03d_%03d.npy' % (set_name, i, j)
  predicted_points, acc = get_predict_points(content_path, pairs_path, model)
  feature_match = np.load(feature_match_path)
  match_dict = {}
  for idx in range(len(feature_match)):
    match_dict[feature_match[idx][0]] = feature_match[idx][1]
  pairs = []
  predicted_target_points = []
  for idx in predicted_points:
    predicted_target_points.append(match_dict[idx])
    pairs.append([idx, match_dict[idx]])

  if len(predicted_points) == 0:
    trans = np.identity(4)
    return [True, i, j, s, np.linalg.inv(trans)], len(predicted_points), acc
  xyz_i, xyz_j = xyz_i[predicted_points], xyz_j[predicted_target_points]
  feat_i, feat_j = feat_i[predicted_points], feat_j[predicted_target_points]

  xyz_i = make_open3d_point_cloud(xyz_i)
  xyz_j = make_open3d_point_cloud(xyz_j)
  feat_i = make_open3d_feature_from_numpy(feat_i)
  feat_j = make_open3d_feature_from_numpy(feat_j)

  # tmp = []
  # for idx in range(len(pairs)):
  #   tmp.append([idx,idx])

  correspondence = o3d.utility.Vector2iVector(tmp)
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
    xyz_i, xyz_j, correspondence, distance_threshold,
    o3d.registration.TransformationEstimationPointToPoint(False), 4
    , o3d.registration.RANSACConvergenceCriteria(50000, 1000))
  trans = result_ransac.transformation

  trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)

  return [True, i, j, s, np.linalg.inv(trans)], len(predicted_points), acc

def my_func(i):
  A = match01_vector[i[0]]
  count = 0
  count2 = 0
  for j in range(nn - 1):
    B = match01_vector[i[j + 1]]
    cos_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    if cos_similarity > 0.95:
      count += 1
    if cos_similarity > 0.99:
      count2 += 1
  return i[0], inds[i[0]], count, count2

def do_single_pair_matching_with_sim(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  # logging.info("matching %s %s" % (name_i, name_j))
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)
  global inds
  start = time.time()
  inds = find_nn_cpu(feat_i, feat_j, 1)
  matching_time = time.time() - start

  # get similarity
  start = time.time()
  global match01_vector
  match01_vector = []
  match01_distance = []
  for idx in range(len(inds)):
    src_point = xyz_i[idx]
    tgt_point = xyz_j[inds[idx]]
    vector = [tgt_point[0] - src_point[0], tgt_point[1] - src_point[1], tgt_point[2] - src_point[2]]
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if distance == 0:
      match01_vector.append([0, 0, 0])
    else:
      match01_vector.append([vector[0] / distance, vector[1] / distance, vector[2] / distance])
    match01_distance.append(distance)

  global nn
  nn = 50
  threshold = 45
  idxs = find_nn_cpu(xyz_i, xyz_i, nn)
  pool = multiprocessing.Pool()
  result = pool.map(my_func, idxs)
  src_idx = []
  tgt_idx = []
  for tmp in result:
    if tmp[2] > threshold:
      src_idx.append(tmp[0])
      tgt_idx.append(tmp[1])
  pool.close()
  outlier_rejection_time = time.time() - start

  # start = time.time()
  # sample_i = make_open3d_point_cloud(xyz_i)
  # sample_j = make_open3d_point_cloud(xyz_j)
  # corres = np.stack((src_idx, tgt_idx), axis=1)
  # corres = o3d.utility.Vector2iVector(corres)
  #
  # distance_threshold = voxel_size * 1.5
  # result = o3d.registration.registration_ransac_based_on_correspondence(
  #     sample_i, sample_j, corres, distance_threshold,
  #     o3d.registration.TransformationEstimationPointToPoint(False), 4,
  #     o3d.registration.RANSACConvergenceCriteria(4000000, 1000))
  # trans = result.transformation
  # ransac_time = time.time() - start

  sample_i = xyz_i[src_idx]
  sample_j = xyz_j[tgt_idx]
  feat_i = feat_i[src_idx]
  feat_j = feat_j[tgt_idx]
  sample_i = make_open3d_point_cloud(sample_i)
  sample_j = make_open3d_point_cloud(sample_j)
  feat_i = make_open3d_feature_from_numpy(feat_i)
  feat_j = make_open3d_feature_from_numpy(feat_j)

  trans = run_ransac(sample_i, sample_j, feat_i, feat_j, voxel_size)
  ransac_time = time.time() - start
  return [True, i, j, s, np.linalg.inv(trans)], [matching_time, outlier_rejection_time, ransac_time]

# def do_single_pair_matching_with_sim(feature_path, set_name, m, voxel_size, tran):
#   i, j, s = m
#   name_i = "%s_%03d" % (set_name, i)
#   name_j = "%s_%03d" % (set_name, j)
#   # logging.info("matching %s %s" % (name_i, name_j))
#   points_i, xyz_i, feat_i = read_data(feature_path, name_i)
#   points_j, xyz_j, feat_j = read_data(feature_path, name_j)
#   src = make_open3d_point_cloud(xyz_i)
#   tgt = make_open3d_point_cloud(xyz_j)
#   original_num = len(xyz_i)
#   matches = get_matching_indices(src, tgt, np.linalg.inv(tran), voxel_size * 1.5)
#   coord_i_gt = np.array(matches)[:, 0]
#   global inds
#   inds = find_nn_cpu(feat_i, feat_j, 1)
#
#   # get similarity
#   global match01_vector
#   match01_vector = []
#   match01_distance = []
#   for idx in range(len(inds)):
#     src_point = xyz_i[idx]
#     tgt_point = xyz_j[inds[idx]]
#     vector = [tgt_point[0] - src_point[0], tgt_point[1] - src_point[1], tgt_point[2] - src_point[2]]
#     distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
#     match01_vector.append([vector[0] / distance, vector[1] / distance, vector[2] / distance])
#     match01_distance.append(distance)
#
#   global nn
#   nn = 50
#   threshold = 45
#   idxs = find_nn_cpu(xyz_i, xyz_i, nn)
#   pool = multiprocessing.Pool()
#   result = pool.map(my_func, idxs)
#   src_idx1 = []
#   src_idx2 = []
#   TP1 = 0
#   TP2 = 0
#   precision1 = []
#   precision2 = []
#   sample_num1 = []
#   sample_num2 = []
#   for c in range(25):
#     threshold = nn - c - 1
#     for tmp in result:
#       if tmp[2] == threshold:
#         src_idx1.append(tmp[0])
#         if tmp[0] in coord_i_gt:
#           TP1 += 1
#       if tmp[3] == threshold:
#         src_idx2.append(tmp[0])
#         if tmp[0] in coord_i_gt:
#           TP2 += 1
#     sample_num1.append(len(src_idx1))
#     sample_num2.append(len(src_idx2))
#     if len(src_idx1) == 0:
#       precision1.append(0)
#     else:
#       precision1.append(TP1 / len(src_idx1))
#     if len(src_idx2) == 0:
#       precision2.append(0)
#     else:
#       precision2.append(TP2 / len(src_idx2))
#
#   pool.close()
#
#   trans = np.identity(4)
#   return [True, i, j, s, np.linalg.inv(trans)], [0, 0, 0], original_num, [precision1, sample_num1, precision2, sample_num2]