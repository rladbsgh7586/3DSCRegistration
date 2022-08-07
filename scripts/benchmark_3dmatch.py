"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d
import glob

from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, \
    gather_results, do_single_pair_matching_with_similarity, do_single_pair_evaluation_with_similarity, do_single_pair_matching_with_gcn, do_single_pair_matching_with_sim
from pygcn.train import set_model
from lib.data_loaders import sample_random_trans, M
import time
import random

import torch

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def extract_features_batch(model, config, source_path, target_path, voxel_size, device):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  logging.info(folders)
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")
      pcd = o3d.io.read_point_cloud(fi)
      timer.tic()
      print(np.array(pcd.points))
      xyz_down, feature = extract_features(
          model,
          xyz=np.array(pcd.points),
          rgb=None,
          normal=None,
          voxel_size=voxel_size,
          device=device,
          skip_check=True)

      t = timer.toc()
      if i > 0:
        tmeter.update(t)
        num_feat += len(xyz_down)

      # np.savez_compressed(
      #     os.path.join(target_path, save_fn),
      #     points=np.array(pcd.points),
      #     xyz=xyz_down,
      #     feature=feature.detach().cpu().numpy())
      if i % 20 == 0 and i > 0:
        logging.info(
            f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
        )

  f.close()

def extract_features_from_train(model, voxel_size, device):
    source_path = './dataset/threedmatch/'
    target_path = './dataset/threedmatch/features/'
    subset_names = open('./config/train_3dmatch.txt').read().split()
    files = []
    for name in subset_names:
        fname = name + "*.txt"
        fnames_txt = glob.glob(source_path + "/" + fname)
        assert len(fnames_txt) > 0, f"Make sure that the path {source_path} has data {fname}"
        for fname_txt in fnames_txt:
            with open(fname_txt) as f:
                content = f.readlines()
            fnames = [x.strip().split() for x in content]
            for fname in fnames:
                if fname[0] not in files:
                    files.append(fname[0])
                if fname[1] not in files:
                    files.append(fname[1])


    for file_name in files:
        data = np.load(os.path.join(source_path, file_name))
        points = data['pcd']

        randg = np.random.RandomState()
        trans = sample_random_trans(points, randg, 360)

        R = trans[:3, :3]
        T = trans[:3, 3]
        points = points @ R.T + T

        pcd = make_open3d_point_cloud(points)

        xyz_down, feature = extract_features(
            model,
            xyz=np.array(pcd.points),
            rgb=None,
            normal=None,
            voxel_size=voxel_size,
            device=device,
            skip_check=True)

        np.savez_compressed(
            os.path.join(target_path, file_name),
            points=np.array(pcd.points),
            xyz=xyz_down,
            T=trans,
            feature=feature.detach().cpu().numpy())

    f.close()

def registration(feature_path, voxel_size):
  """
  Gather .log files produced in --target folder and run this Matlab script
  https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
  (see Geometric Registration Benchmark section in
  http://3dmatch.cs.princeton.edu/)
  """
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  for s in sets:
    set_name = s[0]
    pts_num = int(s[1])
    matching_pairs = gen_matching_pair(pts_num)
    results = []
    times = []
    for m in matching_pairs:
      start = time.time()
      result = do_single_pair_matching(feature_path, set_name, m, voxel_size)
      times.append(time.time() - start)
      results.append(result)
    traj = gather_results(results)
    logging.info(f"Writing the trajectory to {feature_path}/{set_name}.log")
    print(np.mean(times))
    write_trajectory(traj, "%s.log" % (os.path.join(feature_path, set_name)), 4)

def registration_with_cosine_similarity(feature_path, voxel_size, similarity_path, threshold):
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  f.close()
  for s in sets:
      set_name = s[0]
      with open("./dataset/threedmatch_test/" + set_name + "-evaluation/LoMatch_gt_overlap.log") as f:
          lines = f.read().splitlines()
      pts_num = int(s[1])
      matching_pairs = gen_matching_pair(pts_num)
      results = []
      interest_point_num = []
      interest_point_num_LoMatch = []
      gt_idx = 0

      gts = []
      gt_path = "./dataset/threedmatch_test/"
      normal_gt = read_trajectory(os.path.join(gt_path, set_name + "-evaluation/gt.log"), 4)
      low_gt = read_trajectory(os.path.join(gt_path, set_name + "-evaluation/LoMatch_gt.log"), 4)
      for obj in normal_gt:
          gts.append([obj.metadata[0], obj.metadata[1], obj.pose])
      for obj in low_gt:
          for i in range(len(gts)):
              if gts[i][0] < obj.metadata[0]:
                  continue
              if gts[i][0] == obj.metadata[0] and gts[i][1] < obj.metadata[1]:
                  continue
              if gts[i][0] == obj.metadata[0]:
                  if gts[i][1] == obj.metadata[1]:
                      gts[i][2] = obj.pose
                      break
                  else:
                      gts.insert(i, [obj.metadata[0], obj.metadata[1], obj.pose])
                      break
              else:
                  gts.insert(i, [obj.metadata[0], obj.metadata[1], obj.pose])
                  break

      def get_trans(source_idx, target_idx, gt):
          for i in gt:
              if source_idx == i[0] and target_idx == i[1]:
                  return [True, i[2]]
              if source_idx < i[0]:
                  return [False, np.identity(4)]
              elif source_idx == i[0] and target_idx < i[1]:
                  return [False, np.identity(4)]
          return [False, np.identity(4)]
      accuracy = []
      precision = []
      ratios = []
      prediction_num = []
      times = []
      matching = []
      for m in matching_pairs:
          ratio_success=0
          while ratio_success!=1:
              overlap_gt = lines[gt_idx].split(',')
              if m[0] == int(overlap_gt[0]) and m[1] == int(overlap_gt[1]):
                  ratio = float(overlap_gt[2])
                  ratio_success = 1
              gt_idx+=1
          gt_pose = get_trans(m[0], m[1], gts)
          TP, TN, FP, FN = do_single_pair_evaluation_with_similarity(feature_path, set_name, m, voxel_size,
                                                                       similarity_path,
                                                                       threshold, gt_pose[1])
          try:
              acc = (TP + TN) / (TP + TN + FP + FN)
              precise = TP / (TP + FP)
              predict_num = TP + FP
          except:
              continue
          accuracy.append(acc)
          if gt_pose[0] == True:
              ratios.append(ratio)
              precision.append(precise)
              prediction_num.append(predict_num)

          start = time.time()
          result, sample_num = do_single_pair_matching_with_similarity(feature_path, set_name, m, voxel_size,
                                                                       similarity_path,
                                                                       threshold)
          times.append(time.time() - start)
          results.append(result)
          if ratio > 0.3:
              interest_point_num.append(sample_num)
          if ratio > 0.1 and ratio < 0.3:
              interest_point_num_LoMatch.append(sample_num)

      traj = gather_results(results)
      feature_path_tmp = "./result/similarity_cosine/registration/"
      logging.info(f"Writing the trajectory to {feature_path_tmp}/{set_name}_%d.log" % threshold)
      write_trajectory(traj, "%s_%d.log" % (os.path.join(feature_path_tmp, set_name), threshold))

      print("mean sample num: ", np.mean(prediction_num))
      print("mean time: ", np.mean(times))
      print("accuracy: ", np.mean(accuracy))
      print("mean overlap: ", np.mean(ratios))
      print("precision: ", np.mean(precision))

def registration_with_gcn(feature_path, voxel_size):
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  model = set_model()
  graph_path = "./dataset/threedmatch_graph/test_v_dd/"
  for s in sets:
    set_name = s[0]
    with open("./dataset/threedmatch_test/" + set_name + "-evaluation/LoMatch_gt_overlap.log") as f:
        lines = f.read().splitlines()
    pts_num = int(s[1])
    matching_pairs = gen_matching_pair(pts_num)
    results = []
    interest_point_num = []
    interest_point_num_LoMatch = []
    accuracy = []
    src_num = []
    tgt_num = []
    gt_idx = 0
    for m in matching_pairs:
        ratio_success = 0
        while ratio_success != 1:
            overlap_gt = lines[gt_idx].split(',')
            if m[0] == int(overlap_gt[0]) and m[1] == int(overlap_gt[1]):
                ratio = float(overlap_gt[2])
                ratio_success = 1
            gt_idx += 1
        if ratio < 0.1:
            continue
        result, predict_num, acc = do_single_pair_matching_with_gcn(feature_path, set_name, m, voxel_size, graph_path, model)
        results.append(result)
        if ratio > 0.3:
            interest_point_num.append(predict_num)
            accuracy.append(acc)
            src_num.append(predict_num)
        if ratio > 0.1 and ratio < 0.3:
            interest_point_num_LoMatch.append(predict_num)
            accuracy.append(acc)
            src_num.append(predict_num)
        # print(np.mean(src_num))
        print(np.mean(accuracy))
        print(np.mean(interest_point_num))
        print(np.mean(interest_point_num_LoMatch))
    traj = gather_results(results)
    feature_path_tmp = "./result/gcn/registration/"
    logging.info(f"Writing the trajectory to {feature_path_tmp}/{set_name}.log")
    write_trajectory(traj, "%s_test4.log" % os.path.join(feature_path_tmp, set_name))
    # np.save("%s" % os.path.join(feature_path_tmp, set_name), predict_numbers)

def registration_with_sim_ete(feature_path, voxel_size):
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  for s in sets:
    set_name = s[0]
    print(set_name)
    pts_num = int(s[1])
    matching_pairs = gen_matching_pair(pts_num)
    results = []
    times = []
    for m in matching_pairs:
      start = time.time()
      result, partial_times = do_single_pair_matching_with_sim(feature_path, set_name, m, voxel_size)
      times.append([time.time() - start] + partial_times)
      results.append(result)
    traj = gather_results(results)
    logging.info(f"Writing the trajectory to {feature_path}/{set_name}.log")
    print("time: ", np.mean(times, axis=0))
    write_trajectory(traj, "%s_sim.log" % (os.path.join(feature_path, set_name)), 4)
  # List file from the extract_features_batch function
  # with open(os.path.join(feature_path, "list.txt")) as f:
  #     sets = f.readlines()
  #     sets = [x.strip().split() for x in sets]
  #
  # for s in sets:
  #     set_name = s[0]
  #     with open("./dataset/threedmatch_test/" + set_name + "-evaluation/LoMatch_gt_overlap.log") as f:
  #         lines = f.read().splitlines()
  #     pts_num = int(s[1])
  #     matching_pairs = gen_matching_pair(pts_num)
  #     normal_gt = read_trajectory(os.path.join("./dataset/threedmatch_test/", set_name + "-evaluation/gt.log"), 4)
  #     low_gt = read_trajectory(os.path.join("./dataset/threedmatch_test/", set_name + "-evaluation/LoMatch_gt.log"), 4)
  #     gt_trans = []
  #     for obj in normal_gt:
  #         gt_trans.append([obj.metadata[0], obj.metadata[1], obj.pose])
  #     for obj in low_gt:
  #         for i in range(len(gt_trans)):
  #             if gt_trans[i][0] < obj.metadata[0]:
  #                 continue
  #             if gt_trans[i][0] == obj.metadata[0] and gt_trans[i][1] < obj.metadata[1]:
  #                 continue
  #             if gt_trans[i][0] == obj.metadata[0]:
  #                 if gt_trans[i][1] == obj.metadata[1]:
  #                     gt_trans[i][2] = obj.pose
  #                     break
  #                 else:
  #                     gt_trans.insert(i, [obj.metadata[0], obj.metadata[1], obj.pose])
  #                     break
  #             else:
  #                 gt_trans.insert(i, [obj.metadata[0], obj.metadata[1], obj.pose])
  #                 break
  #
  #     def get_trans(source_idx, target_idx, gt):
  #         for i in gt:
  #             if source_idx == i[0] and target_idx == i[1]:
  #                 return [True, i[2]]
  #             if source_idx < i[0]:
  #                 return [False, np.identity(4)]
  #             elif source_idx == i[0] and target_idx < i[1]:
  #                 return [False, np.identity(4)]
  #         return [False, np.identity(4)]
  #
  #     results = []
  #     precisions1 = []
  #     precisions2 = []
  #     sample_nums1 = []
  #     sample_nums2 = []
  #     original_nums = []
  #     ratios = []
  #     times = []
  #     gt_idx = 0
  #     for m in matching_pairs:
  #         ratio_success = 0
  #         while ratio_success != 1:
  #             overlap_gt = lines[gt_idx].split(',')
  #             if m[0] == int(overlap_gt[0]) and m[1] == int(overlap_gt[1]):
  #                 ratio = float(overlap_gt[2])
  #                 ratio_success = 1
  #             gt_idx += 1
  #         if ratio < 0.1:
  #             continue
  #         i, j, s = m
  #         tran = get_trans(i, j, gt_trans)
  #         start = time.time()
  #         result, partial_times, original_num, performance = do_single_pair_matching_with_sim(feature_path, set_name, m, voxel_size, tran[1])
  #         times.append(time.time() - start)
  #         ratios.append(ratio)
  #         original_nums.append(original_num)
  #         results.append(result)
  #         precisions1.append(performance[0])
  #         sample_nums1.append(performance[1])
  #         precisions2.append(performance[2])
  #         sample_nums2.append(performance[3])
  #         if gt_idx ==5:
  #             break
  #     print(set_name)
  #     print("mean original num: ", np.mean(original_nums))
  #     print("ratios: ", np.mean(ratios))
  #     print(np.mean(precisions1, axis=0))
  #     print(np.mean(sample_nums1, axis=0))
  #     print(np.mean(precisions2, axis=0))
  #     print(np.mean(sample_nums2, axis=0))
  #     print("time: ", np.mean(times))


def pose_filter(feature_path, pose_path, overlap_path, overlap_ratio):
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    for s in sets:
        set_name = s[0]
        for ratio in overlap_ratio:
            results = []
            traj = read_trajectory(pose_path + set_name + '.log', 4)
            overlap = np.load(overlap_path + name + '_GT_overlap.npy')
            for i, j in zip(overlap, traj):
                if i[2] > ratio and i[2] < ratio + 0.05:
                    results.append([True, j.metadata[0], j.metadata[1], j.metadata[2], np.array(j.pose)])

            traj = gather_results(results)
            write_trajectory(traj, './dataset/threedmatch_test/'+name+"-evaluation/FCGF_{:.2f}.log".format(ratio), 4)

def do_single_pair_evaluation(feature_path,
                              set_name,
                              traj,
                              voxel_size,
                              tau_1=0.1,
                              tau_2=0.05,
                              num_rand_keypoints=-1):
  trans_gth = np.linalg.inv(traj.pose)
  i = traj.metadata[0]
  j = traj.metadata[1]
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)

  # coord and feat form a sparse tensor.
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

  # use the keypoints in 3DMatch
  if num_rand_keypoints > 0:
    # Randomly subsample N points
    Ni, Nj = len(points_i), len(points_j)
    inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
    inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

    sample_i, sample_j = points_i[inds_i], points_j[inds_j]

    key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))
    key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

    key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))
    key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

    inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]
    inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

    coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
    coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

  coord_i = make_open3d_point_cloud(coord_i)
  coord_j = make_open3d_point_cloud(coord_j)

  hit_ratio = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth,
                                       tau_1)

  # logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
  if hit_ratio >= tau_2:
    return True
  else:
    return False


def feature_evaluation(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]

  assert len(
      sets
  ) > 0, "Empty list file. Makesure to run the feature extraction first with --do_extract_feature."

  tau_1 = 0.1  # 10cm
  tau_2 = 0.05  # 5% inlier
  logging.info("%f %f" % (tau_1, tau_2))
  recall = []
  for s in sets:
    set_name = s[0]
    traj = read_trajectory(os.path.join(source_path, set_name + "-evaluation/gt.log"))
    assert len(traj) > 0, "Empty trajectory file"
    results = []
    for i in range(len(traj)):
      results.append(
          do_single_pair_evaluation(feature_path, set_name, traj[i], voxel_size, tau_1,
                                    tau_2, num_rand_keypoints))

    mean_recall = np.array(results).mean()
    std_recall = np.array(results).std()
    recall.append([set_name, mean_recall, std_recall])
    logging.info(f'{set_name}: {mean_recall} +- {std_recall}')
  for r in recall:
    logging.info("%s : %.4f" % (r[0], r[1]))
  scene_r = np.array([r[1] for r in recall])
  logging.info("average : %.4f +- %.4f" % (scene_r.mean(), scene_r.std()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to 3dmatch test dataset')
  parser.add_argument(
      '--source_high_res',
      default=None,
      type=str,
      help='path to high_resolution point cloud')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
      '-m',
      '--model',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', action='store_true')
  parser.add_argument('--evaluate_feature_match_recall', action='store_true')
  parser.add_argument(
      '--evaluate_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument('--with_cuda', action='store_true')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=5000,
      help='Number of random keypoints for each scene')

  args = parser.parse_args()

  device = torch.device('cuda' if args.with_cuda else 'cpu')

  if args.extract_features:
    assert args.model is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    with torch.no_grad():
      extract_features_batch(model, config, args.source, args.target, config.voxel_size,
                             device)

  if args.evaluate_feature_match_recall:
    assert (args.target is not None)
    with torch.no_grad():
      feature_evaluation(args.source, args.target, args.voxel_size,
                         args.num_rand_keypoints)

  if args.evaluate_registration:
    assert (args.target is not None)
    with torch.no_grad():
      registration(args.target, args.voxel_size)
