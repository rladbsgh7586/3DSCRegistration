import argparse
import torch
import os, re
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import math
import time
import pickle
import multiprocessing
import glob
import warnings
from scipy.linalg import expm, norm

from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from lib.eval import find_nn_cpu, find_nn_gpu
from model import load_model
from util.misc import extract_features
from util.file import ensure_dir
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy, get_matching_indices
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results
from util.trajectory import read_trajectory, write_trajectory

from scripts.benchmark_3dmatch import extract_features_batch, registration, do_single_pair_evaluation,\
    feature_evaluation, pose_filter, registration_with_gcn, registration_with_cosine_similarity, extract_features_from_train, registration_with_sim_ete

def execute(number):
    feature_path = "./features_tmp/"
    result_path = "./result/similarity_cosine/parameter_test_LoMatch/"
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    s = sets[number]
    for iter in range(1):
        set_name = s[0]
        pts_num = int(s[1])
        matching_pairs = gen_matching_pair(pts_num)

        for num_rand_keypoints in [2500, 1000, 250, 500, 5000]:
            for num_neighbors in [10, 20, 30, 50]:
                print("parameter_test %s %d %d" % (set_name, num_rand_keypoints, num_neighbors))
                traj = read_trajectory('./dataset/threedmatch_test/' + set_name + '-evaluation/LoMatch_gt.log', 4)
                ratios = []
                for m in matching_pairs:
                    i, j, s = m
                    if not traj:
                        continue
                    if traj[0].metadata[0] != i or traj[0].metadata[1] != j:
                        continue

                    trans_gt = traj[0].pose

                    # load features
                    data_i = np.load("%s%s_%03d.npz" % (feature_path, set_name, i))
                    data_j = np.load("%s%s_%03d.npz" % (feature_path, set_name, j))
                    src = make_open3d_point_cloud(data_i['xyz'])
                    tgt = make_open3d_point_cloud(data_j['xyz'])

                    #     random sampling
                    voxel_size = 0.025
                    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
                    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']
                    # use the keypoints in 3DMatch
                    if num_rand_keypoints > 0:
                        # Randomly subsample N points
                        Ni, Nj = len(coord_i), len(coord_j)
                        inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
                        inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

                        sample_i, sample_j = coord_i[inds_i], coord_j[inds_j]
                        feat_i, feat_j = feat_i[inds_i], feat_j[inds_j]

                    coord_i = make_open3d_point_cloud(sample_i)
                    coord_j = make_open3d_point_cloud(sample_j)

                    # keypoint matching by extracted feature
                    inds = find_nn_cpu(feat_i, feat_j, 1)

                    # get similarity
                    match01_vector = []
                    for i in range(len(inds)):
                        src_point = coord_i.points[i]
                        tgt_point = coord_j.points[inds[i]]
                        vector = [tgt_point[0] - src_point[0], tgt_point[1] - src_point[1],
                                  tgt_point[2] - src_point[2]]
                        distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                        match01_vector.append([vector[0] / distance, vector[1] / distance, vector[2] / distance])

                    similarity = []
                    for i in range(np.shape(coord_i.points)[0]):
                        src_tree = o3d.geometry.KDTreeFlann(coord_i)
                        [k, idx, _] = src_tree.search_knn_vector_3d(coord_i.points[i], num_neighbors)
                        count = 0
                        for j in idx:
                            A = match01_vector[i]
                            B = match01_vector[j]
                            cos_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                            if cos_similarity > 0.95:
                                count += 1
                        similarity.append([inds_i[i], inds_j[inds[i]], count])
                    # get ratios in gt
                    match01_pairs_gt = get_matching_indices(src, tgt, np.linalg.inv(trans_gt), voxel_size, K=1)
                    match01_src_gt = np.array(match01_pairs_gt)[:, 0]
                    sample_match01_gt = []
                    for i in inds_i:
                        if i in match01_src_gt:
                            sample_match01_gt.append(i)
                    ratio = compute_overlap_ratio(src, tgt, np.linalg.inv(trans_gt), voxel_size)


                    ii, jj, s = m
                    for k in range(num_neighbors):
                        total = []
                        count = 0
                        for i in similarity:
                            if i[2] > k:
                                total.append(i[0])
                                if i[0] in match01_src_gt:
                                    count += 1
                        if len(total) == 0:
                            predicted_ratio = 0
                        else:
                            predicted_ratio = count / len(total)
                        ratios.append([ii, jj, count, len(total), ratio, predicted_ratio])
                    del traj[0]

                np.save("%s%s_%d_%d" % (result_path, set_name, num_rand_keypoints, num_neighbors), ratios)

def get_increased_ratio(number):
    feature_path = "./features_tmp/"
    result_path = "./result/similarity_cosine/parameter_test_LoMatch/"
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    for s in sets:
        set_name = s[0]
        pts_num = int(s[1])
        matching_pairs = gen_matching_pair(pts_num)

        increase_ratio = []
        for num_rand_keypoints in [2500]:
            for num_neighbors in [10, 20, 30, 50]:
                data = np.load("%s%s_%d_%d.npy" % (result_path, set_name, num_rand_keypoints, num_neighbors), allow_pickle=True)
                print("%s%s_%d_%d.npy" % (result_path, set_name, num_rand_keypoints, num_neighbors))
                increase_data = []
                for t in range(num_neighbors):
                    gt_increase = []
                    predicted_increase = []
                    sampling_num = []
                    for i in range(int(np.shape(data)[0] / num_neighbors)):
                        gt_increase.append(data[(i)*num_neighbors+t][4])
                        predicted_increase.append(data[(i)*num_neighbors+t][5])
                        sampling_num.append(data[(i)*num_neighbors+t][3])
                    # print("threshold: ", t, "increase ratio: ", np.mean(gt_increase), np.mean(predicted_increase), "sampling_num:" , np.mean(sampling_num))
                    increase_data.append([np.mean(gt_increase), np.mean(predicted_increase), np.mean(sampling_num), t])
                best = [0, 0, 0, 0]
                for data in increase_data:
                    if data[1] > best[1]:
                        best = data
                print(data)
                increase_ratio.append((data[1]-data[0])/data[0])
        print(increase_ratio)
        print(np.mean(increase_ratio))


def make_graph_data(train_data_path, args):
        if args.extract_train_graph:
            target_path = './dataset/threedmatch/features/'
            graph_save_path = "./dataset/threedmatch_graph/train/"
            root = train_data_path
            subset_names = open('./config/train_3dmatch.txt').read().split()
            files = []
            for name in subset_names:
                fname = name + "*.txt"
                fnames_txt = glob.glob(root + "/" + fname)
                assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
                for fname_txt in fnames_txt:
                    with open(fname_txt) as f:
                        content = f.readlines()
                    fnames = [x.strip().split() for x in content]
                    for fname in fnames:
                        files.append([fname[0], fname[1]])

        if args.extract_test_graph:
            target_path = './features_tmp/'
            graph_save_path = "./dataset/threedmatch_graph/test_v_dd/"
            gt_path = "./dataset/threedmatch_test/"
            files = []
            gt_dict = {}
            with open(os.path.join('./features_tmp', "list.txt")) as f:
                sets = f.readlines()
                sets = [x.strip().split() for x in sets]
            for s in sets:
                set_name = s[0]
                with open("./dataset/threedmatch_test/" + set_name + "-evaluation/LoMatch_gt_overlap.log") as f:
                    lines = f.read().splitlines()
                pts_num = int(s[1])
                matching_pairs = gen_matching_pair(pts_num)

                gt_idx = 0
                for m in matching_pairs:
                    ratio_success = 0
                    while ratio_success != 1:
                        overlap_gt = lines[gt_idx].split(',')
                        if m[0] == int(overlap_gt[0]) and m[1] == int(overlap_gt[1]):
                            ratio = float(overlap_gt[2])
                            ratio_success = 1
                        gt_idx += 1
                    if ratio > 0.1:
                        fname0 = set_name + '_%03d.npz' % m[0]
                        fname1 = set_name + '_%03d.npz' % m[1]
                        files.append([fname0, fname1, set_name, m[0], m[1]])

                gts = []
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

                gt_dict[set_name] = gts

        voxel_size = 0.025
        # config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        matching_search_voxel_size = voxel_size * 1.5

        # trans = np.identity(4)

        gt_num = []
        times = []
        nn_true = []
        nn_false = []
        dataset_count = 0
        failed_count = 0

        def get_trans(source_idx, target_idx, gt):
            for i in gt:
                if source_idx == i[0] and target_idx == i[1]:
                    return [True, i[2]]
                if source_idx < i[0]:
                    return [False, np.identity(4)]
                elif source_idx == i[0] and target_idx < i[1]:
                    return [False, np.identity(4)]
            return [False, np.identity(4)]

        # random_sample_idx = np.random.choice(len(files), 10000, replace=False)
        # for inds_idx in range(len(random_sample_idx)):
        for inds_idx in range(len(files)):
            start_time = time.time()

            if args.extract_train_graph:
                idx = random_sample_idx[inds_idx]
                # idx = inds_idx
            if args.extract_test_graph:
                idx = inds_idx
            data_i = np.load(os.path.join(target_path, files[idx][0]))
            data_j = np.load(os.path.join(target_path, files[idx][1]))
            #     random sampling
            coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
            coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']
            inds_i = []
            inds_j = []
            for i in range(len(coord_i)):
                inds_i.append(i)
            for j in range(len(coord_j)):
                inds_j.append(j)
            inds_i = np.array(inds_i)
            inds_j = np.array(inds_j)
            xyz_i = coord_i

            coord_i = make_open3d_point_cloud(coord_i)
            coord_j = make_open3d_point_cloud(coord_j)
            if args.extract_train_graph:
                trans = data_j['T'] @ np.linalg.inv(data_i['T'])
                matches = get_matching_indices(coord_i, coord_j, trans, matching_search_voxel_size)
                if len(matches) < 2:
                    continue
                coord_i_gt = inds_i[np.array(matches)[:, 0]]
            if args.extract_test_graph:
                trans_tmp = get_trans(files[idx][3], files[idx][4], gt_dict[files[idx][2]])
                if trans_tmp[0] == True:
                    trans = trans_tmp[1]
                    matches = get_matching_indices(coord_i, coord_j, np.linalg.inv(trans), matching_search_voxel_size)
                    if len(matches) < 2:
                        coord_i_gt = []
                    else:
                        coord_i_gt = inds_i[np.array(matches)[:, 0]]
                else:
                    coord_i_gt = []
            # keypoint matching by extracted feature
            inds = find_nn_cpu(feat_i, feat_j, 1)
            # get similarity
            match01_vector = []
            feature_matching = []
            label = ['pos', 'neg']
            pos_num = 0

            for i in range(len(inds)):
                src_point = np.floor(coord_i.points[i]/voxel_size)
                tgt_point = np.floor(coord_j.points[inds[i]]/voxel_size)
                # src_point = coord_i.points[i]
                # tgt_point = coord_j.points[inds[i]]
                feature_matching.append([i, inds[i]])
                vector = [tgt_point[0] - src_point[0], tgt_point[1] - src_point[1], tgt_point[2] - src_point[2]]
                distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                if distance == 0:
                    print(files[idx])
                    if inds_i[i] in coord_i_gt:
                        match01_vector.append(
                            [inds_i[i], 0, 0, 0, 0, label[0]])
                    else:
                        match01_vector.append(
                            [inds_i[i], 0, 0, 0, 0, label[1]])
                else:
                    if inds_i[i] in coord_i_gt:
                        pos_num += 1
                        match01_vector.append(
                            [inds_i[i], vector[0] / distance, vector[1] / distance, vector[2] / distance, distance, label[0]])
                    else:
                        match01_vector.append(
                            [inds_i[i], vector[0] / distance, vector[1] / distance, vector[2] / distance, distance, label[1]])
            match01_vector = np.array(match01_vector)
            d_vector = match01_vector[:,4]
            d_vector = np.array(d_vector, dtype = np.float64)
            max = np.max(d_vector, axis = 0)
            min = np.min(d_vector, axis=0)
            for i in range(len(match01_vector)):
                match01_vector[i][4] = (((float(match01_vector[i][4]) - min) / (max - min)) - 0.5) * 2

            pairs = []
            nn = 5
            inds = find_nn_cpu(xyz_i, xyz_i, nn)
            nearest_of_true = []
            nearest_of_false = []
            for i in inds:
                for j in range(nn - 1):
                    pairs.append([i[0], i[j + 1]])

            # for i in result:
            #     pairs.append([i[0], inds[i[0]], i[1]])
            # for i in inds:
            #     nearest_true = 0
            #     for j in range(nn-1):
            #         if i[j+1] in coord_i_gt:
            #             nearest_true +=1
            #         pairs.append([i[0], i[j+1]])
            #     if i[0] in coord_i_gt:
            #         nearest_of_true.append(nearest_true)
            #     else:
            #         nearest_of_false.append(nearest_true)
            # if pos_num > 50:
            #     nn_true.append(np.mean(nearest_of_true))
            #     nn_false.append(np.mean(nearest_of_false))

            if args.extract_train_graph:
                if pos_num > 1000:
                    if dataset_count < 500:
                        np.save(graph_save_path + "train_content_%s" % dataset_count, match01_vector)
                        np.save(graph_save_path + "train_pairs_%s" % dataset_count, pairs)
                    else:
                        np.save("./dataset/threedmatch_graph/valid/" + "valid_content_%s" % dataset_count,
                                match01_vector)
                        np.save("./dataset/threedmatch_graph/valid/" + "valid_pairs_%s" % dataset_count, pairs)
                    dataset_count += 1
                else:
                    failed_count += 1
                times.append(time.time() - start_time)
                print(pos_num, len(inds_i), np.mean(times))
                if dataset_count % 50 == 0:
                    print("success count: ", dataset_count, np.mean(times))
                    print("failed count: ", failed_count)
                if dataset_count == 550:
                    break

            if args.extract_test_graph:
                np.save(graph_save_path + "test_content_%s_%03d_%03d" % (files[idx][2], files[idx][3], files[idx][4]),
                        match01_vector)
                np.save(graph_save_path + "test_pairs_%s_%03d_%03d" % (files[idx][2], files[idx][3], files[idx][4]),
                        pairs)
                np.save(graph_save_path + "test_feature_matching_%s_%03d_%03d" % (
                files[idx][2], files[idx][3], files[idx][4]), feature_matching)
                times.append(time.time() - start_time)
                dataset_count += 1
                if dataset_count % 100 == 0:
                    print("success count: ", dataset_count, np.mean(times))
                # print(np.mean(nn_true), np.mean(nn_false))


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
    parser.add_argument('--extract_traindata_features', action='store_true')
    parser.add_argument('--extract_train_graph', action='store_true')
    parser.add_argument('--extract_test_graph', action='store_true')
    parser.add_argument('--evaluate_feature_match_recall', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--get_overlap_ratio', action='store_true')
    parser.add_argument('--pose_with_overlap_ratio', action='store_true')
    parser.add_argument('--get_correspondence', action='store_true')
    parser.add_argument('--similarity_registration_ete', action='store_true')
    parser.add_argument('--load_3d_low_match', action='store_true')
    parser.add_argument(
        '--graph_registration',
        action='store_true' )
    parser.add_argument(
        '--evaluate_registration_with_similarity',
        action='store_true',)
    parser.add_argument(
        '--evaluate_registration',
        action='store_true',#
        help='The target directory must contain extracted features')
    parser.add_argument('--with_cuda', action='store_true')
    parser.add_argument(
        '--num_rand_keypoints',
        type=int,
        default=5000,
        help='Number of random keypoints for each scene')

    args = parser.parse_args()

    device = torch.device('cuda' if args.with_cuda else 'cpu')

    if args.test:
        feature_path = "./features_tmp/"
        with open(os.path.join(feature_path, "list.txt")) as f:
            sets = f.readlines()
            sets = [x.strip().split() for x in sets]
        # get_increased_ratio(0)
        # for i in range(len(sets)):
        #     my_process = multiprocessing.Process(target=execute, args=(i,))
        #     my_process.start()

        # for s in sets:
        #     set_name = s[0]
        #     pts_num = int(s[1])
        #     traj = read_trajectory("./result/similarity_cosine/registration/"+set_name+"_18.log", 4)
        #     infos = np.load("./result/similarity_cosine/registration/"+set_name+"_info_18.npy")
        #     result = []
        #     for i in range(len(infos)):
        #         if infos[i][1] > 0.1:
        #             result.append(traj[i])
        #     write_trajectory(result, "./dataset/threedmatch_test/"+set_name+"-evaluation/FCGF_18.log")
        for s in sets:
            set_name = s[0]
            pts_num = int(s[1])
            matching_pairs = gen_matching_pair(pts_num)
            results = []
            times = []
            traj = []
            filename = "%s_sim.log" % (os.path.join(feature_path, set_name))
            with open(filename, 'r') as f:
                metastr = f.readline()
                while metastr:
                    mat = np.zeros(shape=(4, 4))
                    for i in range(4):
                        matstr = f.readline()
                        mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
                    traj.append(mat)
                    metastr = f.readline()
            count = 0
            for m in matching_pairs:
                i, j, s = m
                result = [True, i, j, s, traj[count]]
                count+=1
                results.append(result)
            traj = gather_results(results)
            write_trajectory(traj, "%s_sim.log" % (os.path.join(feature_path, set_name)), 4)

    if args.extract_train_graph:
        make_graph_data('./dataset/threedmatch', args)

    if args.extract_test_graph:
        make_graph_data('./dataset/threedmatch', args)

    if args.graph_registration:
        registration_with_gcn('./features_tmp', 0.025)

    if args.load_3d_low_match:
        with open('3DLoMatch.pkl', 'rb') as f:
            pick = pickle.load(f)
        set_name = '7-scenes-redkitchen'
        bin_num = {
            '7-scenes-redkitchen':60,
            'sun3d-home_at-home_at_scan1_2013_jan_1':60,
            'sun3d-home_md-home_md_scan9_2012_sep_30': 60,
            'sun3d-hotel_uc-scan3': 55,
            'sun3d-hotel_umd-maryland_hotel1': 57,
            'sun3d-hotel_umd-maryland_hotel3': 37,
            'sun3d-mit_76_studyroom-76-1studyroom2': 66,
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika': 38
        }
        LoMatch_gt = []
        LoMatch_overlap = []
        # len(pick['src'])
        for i in range(len(pick['src'])):
            # print(pick['src'][i], pick['tgt'][i])
            strings = pick['src'][i].split('/')
            name = strings[1]
            if name != set_name:
                traj = gather_results(LoMatch_gt)
                feature_path_tmp = "./dataset/threedmatch_test/"
                np.save("%s-evaluation/LoMatch_overlap" % (os.path.join(feature_path_tmp, set_name)), LoMatch_overlap)
                write_trajectory(traj, "%s-evaluation/LoMatch_gt.log" % (os.path.join(feature_path_tmp, set_name)))
                set_name = name
                LoMatch_gt = []
                LoMatch_overlap = []
            src_num = int(re.findall('\d+', strings[2])[0])
            strings = pick['tgt'][i].split('/')
            tgt_num = int(re.findall('\d+', strings[2])[0])
            test = np.concatenate([pick['rot'][i], pick['trans'][i]], axis=1).tolist()
            test.append([0, 0, 0, 1])
            LoMatch_gt.append([True, tgt_num, src_num, bin_num[name], np.array(test)])
            LoMatch_overlap.append([tgt_num,src_num,bin_num[name], pick['overlap'][i]])

    if args.get_overlap_ratio:
        data_i = np.load("./features_tmp/7-scenes-redkitchen_000.npz")
        data_j = np.load("./features_tmp/7-scenes-redkitchen_002.npz")
        voxel_size = 0.025
        trans = [
            [9.54999224e-01, 1.08859481e-01, -2.75869135e-01, -3.41060560e-01],
            [-9.89491703e-02, 9.93843326e-01, 4.96360476e-02, -1.78254668e-01],
            [2.79581388e-01, -2.01060700e-02, 9.59896612e-01, 3.54627338e-01],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        pcd0 = make_open3d_point_cloud(data_i['xyz'])
        pcd1 = make_open3d_point_cloud(data_j['xyz'])
        matching01 = get_matching_indices(pcd0, pcd1, np.linalg.inv(trans), voxel_size, 1)
        matching10 = get_matching_indices(pcd1, pcd0, trans, voxel_size, 1)
        result = []
        for i in matching01:
            result.append(i[0])
        np.save("./result/src_overlap.npy",result)

        result = []
        for j in matching10:
            result.append(j[0])
        np.save("./result/tgt_overlap.npy", result)
        # print(matching10)
        # write_overlap_ratio(args.target, args.voxel_size)

    if args.pose_with_overlap_ratio:
        pose_filter('./features_tmp/', './result/', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])


    if args.get_correspondence:
        # feature_path = "./features_tmp/"
        # result_path = "./result/similarity_cosine/"
        # with open(os.path.join(feature_path, "list.txt")) as f:
        #     sets = f.readlines()
        #     sets = [x.strip().split() for x in sets]
        # for s in sets:
        #     set_name = s[0]
        #     ratio = np.load("./result/similarity_cosine/"+set_name+"_18.npy")
        #     print(set_name)
        #     print(np.mean(ratio[:,5]-ratio[:,4]))
        feature_path = "./features_tmp/"
        result_path = "./result/similarity_cosine/"
        with open(os.path.join(feature_path, "list.txt")) as f:
            sets = f.readlines()
            sets = [x.strip().split() for x in sets]

        for s in sets:
            set_name = s[0]
            pts_num = int(s[1])
            matching_pairs = gen_matching_pair(pts_num)
            results = []

            traj = read_trajectory('./dataset/threedmatch_test/'+set_name+'-evaluation/gt.log', 4)

            times = []
            ratios = []
            for m in matching_pairs:
                start = time.time()
                # result = np.load("./result/7-scenes-redkitchen_predict_overlap.npy")
                i, j, s = m

                # print(set_name, "set matching pairs:", i, j)
                filename = "%s49_all/%s/%03d_%03d" % (result_path, set_name, i, j)
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise

                # load features
                data_i = np.load("%s%s_%03d.npz" % (feature_path, set_name, i))
                data_j = np.load("%s%s_%03d.npz" % (feature_path, set_name, j))
                src = make_open3d_point_cloud(data_i['xyz'])
                tgt = make_open3d_point_cloud(data_j['xyz'])
                src_feat = data_i['feature']
                tgt_feat = data_j['feature']
                # print(np.shape(result))
                # print(result)

                #     random sampling
                # num_rand_keypoints = 2500
                voxel_size = 0.025
                coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
                coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']
                # # use the keypoints in 3DMatch
                # if num_rand_keypoints > 0:
                #     # Randomly subsample N points
                #     Ni, Nj = len(coord_i), len(coord_j)
                #     inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
                #     inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)
                #
                #     sample_i, sample_j = coord_i[inds_i], coord_j[inds_j]
                #     feat_i, feat_j = feat_i[inds_i], feat_j[inds_j]

                # coord_i = make_open3d_point_cloud(sample_i)
                # coord_j = make_open3d_point_cloud(sample_j)

                xyz_i = coord_i
                coord_i = make_open3d_point_cloud(coord_i)
                coord_j = make_open3d_point_cloud(coord_j)

                # keypoint matching by extracted feature
                inds = find_nn_cpu(feat_i, feat_j, 1)

                # get similarity
                match01_vector = []
                match01_distance = []
                for i in range(len(inds)):
                    src_point = coord_i.points[i]
                    tgt_point = coord_j.points[inds[i]]
                    vector = [tgt_point[0] - src_point[0], tgt_point[1] - src_point[1], tgt_point[2] - src_point[2]]
                    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
                    match01_vector.append([vector[0] / distance, vector[1] / distance, vector[2] / distance])
                    match01_distance.append(distance)
                similarity = []
                nn = 50
                idxs = find_nn_cpu(xyz_i, xyz_i, nn)
                nearest_of_true = []
                nearest_of_false = []

                def my_func(i):
                    nn = 50
                    A = match01_vector[i[0]]
                    count = 0
                    for j in range(nn - 1):
                        B = match01_vector[i[j + 1]]
                        cos_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                        if cos_similarity > 0.95:
                            count += 1
                    return [i[0], count]


                pool = multiprocessing.Pool()

                result = pool.map(my_func, idxs)
                for i in result:
                    similarity.append([i[0], inds[i[0]], i[1]])
                pool.close()
                # for i in idxs:
                #
                #     similarity.append([i[0], i[j + 1], count])

                # for i in range(np.shape(coord_i.points)[0]):
                #     src_tree = o3d.geometry.KDTreeFlann(coord_i)
                #     [k, idx, _] = src_tree.search_knn_vector_3d(coord_i.points[i], 50)
                #     count = 0
                #     for j in idx:
                #         A = match01_vector[i]
                #         B = match01_vector[j]
                #         cos_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
                #         if cos_similarity > 0.95:
                #             count += 1
                #     similarity.append([inds_i[i], inds_j[inds[i]], count])

                get_sim_time = time.time() - start
                times.append(get_sim_time)
                # print("means of get similarity: ", np.mean(times))

                np.save(filename, similarity)
            print(set_name)
            print("means of get similarity: ", np.mean(times))
            #
            #     # get ratios in gt
            #     i, j, s = m
            #     if not traj:
            #         continue
            #     if traj[0].metadata[0] != i or traj[0].metadata[1] != j:
            #         continue
            #
            #     trans_gt = traj[0].pose
            #
            #     match01_pairs_gt = get_matching_indices(src, tgt, np.linalg.inv(trans_gt), voxel_size, K=1)
            #     match01_src_gt = np.array(match01_pairs_gt)[:, 0]
            #     sample_match01_gt = []
            #     for i in inds_i:
            #         if i in match01_src_gt:
            #             sample_match01_gt.append(i)
            #     ratio = compute_overlap_ratio(src, tgt, np.linalg.inv(trans_gt), voxel_size)
            #
            #     total = []
            #     count = 0
            #     for i in similarity:
            #         if i[2] > 49:
            #             total.append(i[0])
            #             if i[0] in match01_src_gt:
            #                 count += 1
            #     if len(total) == 0:
            #         ratios.append([i, j, count, len(total), ratio, 0])
            #     else:
            #         ratios.append([i, j, count, len(total), ratio, count / len(total)])
            #
            #     del traj[0]
            #
            # np.save("%s49_all/%s_49" % (result_path, set_name), ratios)

    #     get correspondence by feature distance

    #     give similarity point by "some rule"
    #     - translate neighbor(distance) as much as pose between searching point and closest feature point in target pcd
    #     and check distance with feature correspondence point

    if args.evaluate_registration_with_similarity:
        assert (args.target is not None)
        with torch.no_grad():
            registration_with_cosine_similarity(args.target, args.voxel_size, "./result/similarity_cosine/49_all/", 45)

    if args.similarity_registration_ete:
        assert (args.target is not None)
        with torch.no_grad():
            registration_with_sim_ete(args.target, args.voxel_size)

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
    if args.extract_traindata_features:
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
            extract_features_from_train(model, config.voxel_size, device)


    if args.evaluate_feature_match_recall:
        assert (args.target is not None)
        with torch.no_grad():
            feature_evaluation(args.source, args.target, args.voxel_size,
                               args.num_rand_keypoints)

    if args.evaluate_registration:
        assert (args.target is not None)
        with torch.no_grad():
            registration(args.target, args.voxel_size)