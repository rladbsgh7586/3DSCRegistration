import open3d as o3d
import numpy as np
import copy
import torch

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent

def to_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

if __name__ == '__main__':
    data_i = np.load("./features_tmp/7-scenes-redkitchen_000.npz")
    # coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
    data_j = np.load("./features_tmp/7-scenes-redkitchen_002.npz")
    src_raw = o3d.geometry.PointCloud()
    src_raw.points = o3d.utility.Vector3dVector(data_i['xyz'])
    tgt_raw = o3d.geometry.PointCloud()
    tgt_raw.points = o3d.utility.Vector3dVector(data_j['xyz'])
    src_overlap = np.load("./result/src_overlap.npy")
    tgt_overlap = np.load("./result/tgt_overlap.npy")
    src_sim = np.load("./result/similarity.npy")

    trans = [
        [1,0,0,2],
        [0,1,0,-1],
        [0,0,1,2],
        [0,0,0,1]
    ]

    tsfm = [
        [ 9.54999224e-01, 1.08859481e-01, -2.75869135e-01, -3.41060560e-01],
        [-9.89491703e-02, 9.93843326e-01, 4.96360476e-02, -1.78254668e-01],
        [2.79581388e-01, -2.01060700e-02, 9.59896612e-01, 3.54627338e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    ########################################
    # 1. input point cloud
    src_pcd_before = src_raw
    tgt_pcd_before = tgt_raw
    print(src_pcd_before)
    print(src_overlap)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    #
    # ########################################
    # 2. overlap colors
    tmp = np.zeros([np.shape(data_i['xyz'])[0],3])
    for i in src_overlap:
        tmp[i] = [1, 1, 1]
    src_overlap_color = lighter(get_yellow(), 1 - tmp)

    tmp = np.zeros([np.shape(data_j['xyz'])[0], 3])
    for i in tgt_overlap:
        tmp[i] = [1, 1, 1]
    tgt_overlap_color = lighter(get_blue(), 1 - tmp)
    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(np.linalg.inv(tsfm))
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    # ########################################
    # 3. similarity
    tmp = np.zeros([np.shape(data_i['xyz'])[0],3])
    count=0
    for i in src_sim:
        if i[2] > 9:
            count+=1
            src_tree = o3d.geometry.KDTreeFlann(src_raw)
            [k, idx, _] = src_tree.search_knn_vector_3d(src_raw.points[i[0]], 100)
            for x in idx:
                tmp[x] = [1, 1, 1]
    print(count)
    src_sim_color = lighter(get_yellow(), 1 - tmp)

    tmp = np.zeros([np.shape(data_j['xyz'])[0], 3])
    # for i in tgt_overlap:
    #     tmp[i] = [1, 1, 1]
    tgt_sim_color = lighter(get_blue(), 1 - tmp)
    src_pcd_sim = copy.deepcopy(src_pcd_before)
    src_pcd_sim.transform(np.linalg.inv(tsfm))
    tgt_pcd_sim = copy.deepcopy(tgt_pcd_before)
    src_pcd_sim.colors = o3d.utility.Vector3dVector(src_sim_color)
    tgt_pcd_sim.colors = o3d.utility.Vector3dVector(tgt_sim_color)
    #
    # ########################################
    # 4. draw registrations
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(np.linalg.inv(tsfm))

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before.transform(trans))
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap.transform(trans))
    vis2.add_geometry(tgt_pcd_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='similrarity check', width=960, height=540, left=0, top=600)
    vis3.add_geometry(src_pcd_sim.transform(trans))
    vis3.add_geometry(tgt_pcd_sim)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='Our registration', width=960, height=540, left=960, top=0)
    vis4.add_geometry(src_pcd_after)
    vis4.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis3.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        vis4.update_geometry(src_pcd_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()



    #
    # pcd2 = o3d.io.read_point_cloud("./dataset/threedmatch_test/7-scenes-redkitchen/cloud_bin_1.ply")
    # o3d.visualization.draw_geometries([pcd1, pcd2])

        # result = np.load("./result/7-scenes-redkitchen_predict_overlap.npy")
        # print(np.shape(result))
        # print(result)
    #     get correspondence by distance

    #     get correspondence by feature distance

    #     give similarity point by "some rule"
    #     - translate neighbor(distance) as much as pose between searching point and closest feature point in target pcd
    #     and check distance with feature correspondence point
