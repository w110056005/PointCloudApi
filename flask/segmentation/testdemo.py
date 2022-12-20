import logging
import argparse
from statistics import mode
import open3d as o3d
import cv2
import os
from torch import gt
from pipeline import SparseConvSegmentation
from model import SparseEncDec
from dataset import MySemantic3D
import open3d._ml3d as ml3d
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    parser.add_argument('--data_path',
                        help='path to data.npy',
                        required=True)
    parser.add_argument('--ckpt_path',
                        help='path to saved checkpoint')
                        
    parser.add_argument('--test_path',
                        help='path to saved checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args

def test(args,cfg):

    # Initialize the testing by passing parameters
    dataset = MySemantic3D(args.data_path, use_cache=True, **cfg.dataset)

    model = SparseEncDec(dim_input=3,**cfg.model)

    pipeline = SparseConvSegmentation(model=model, dataset=dataset,**cfg.pipeline)

    pipeline.run_test()


if __name__ == '__main__':
    args = parse_args()
    cfg_file = "./config.yml"
    cfg = ml3d.utils.Config.load_from_file(cfg_file)
    test(args,cfg)
    print(args.test_path)
    data = np.load(args.test_path)
    xyz = data[:,:3]
    rgb = data[:,3:6]/255
    gt_labels = data[:,6]

    # visualize colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # o3d.visualization.draw_geometries([pcd])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz)
    pcd1.colors = o3d.utility.Vector3dVector(rgb)

    labels = np.load('predictions/predicted_labels.npy')

    eq = gt_labels == labels
    print("Prediction Accuracy = ", eq.sum()/len(eq))


    # visualize labeled point cloud with fake colors
    vis_colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [255,127,255], [127,255,255], [127,127,255], [127,255,127], [255,127,127], [127,127,127], [127,127,255]])
    labels_color = np.zeros((labels.shape[0],3))
    for i in range(len(vis_colors)):
        labels_color[labels==i] = vis_colors[i]/255

    pcd.colors = o3d.utility.Vector3dVector(labels_color)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("segmentation.ply", pcd)
    o3d.io.write_point_cloud("raw.ply", pcd1)