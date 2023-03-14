import logging
import argparse
from statistics import mode
import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
from pipeline import SparseConvSegmentation
from model import SparseEncDec
from dataset import MySemantic3D
import open3d._ml3d as ml3d
import numpy as np
from pathlib import Path
import open3d as o3d


def plytonpy(file):
    file_dir = file  #文件的路径
    plydata = PlyData.read(file_dir)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    data_np=np.insert(data_np, 6, 0, axis=1)
    np.save('./segmentation-pointcloud/data/test/my.npy', data_np)   
    return data_np  

def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    parser.add_argument('--data_path',
                        help='path to data.npy',
                        required=True)
    parser.add_argument('--ckpt_path',
                        help='path to saved checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args

def test(args,cfg):
    plytonpy(args.data_path)
    data="./segmentation-pointcloud/data"
    #print(data)
    # Initialize the testing by passing parameters
    dataset = MySemantic3D(data, use_cache=True, **cfg.dataset)

    model = SparseEncDec(dim_input=3,**cfg.model)

    pipeline = SparseConvSegmentation(model=model, dataset=dataset,**cfg.pipeline)

    pipeline.run_test()


if __name__ == '__main__':
    args = parse_args()
    cfg_file = "./segmentation-pointcloud/code/config.yml"
    # get id
    id = args.data_path
    # set the real path to arg
    args.data_path = './files/'+id+'/'+id+'.ply'
    cfg = ml3d.utils.Config.load_from_file(cfg_file)
    test(args,cfg)
    viewdata=plytonpy(args.data_path)
    xyz = viewdata[:,:3]
    rgb = viewdata[:,3:6]/255.0
    gt_labels = viewdata[:,6]

    # visualize colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # o3d.visualization.draw_geometries([pcd])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz)
    pcd1.colors = o3d.utility.Vector3dVector(rgb)

    labels = np.load('./segmentation-pointcloud/code/predictions/predicted_labels.npy')

    eq = gt_labels == labels
    

    # visualize labeled point cloud with fake colors
    vis_colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [255,127,255], [127,255,255], [127,127,255], [127,255,127], [255,127,127], [127,127,127], [127,127,255]])
    labels_color = np.zeros((labels.shape[0],3))
    for i in range(len(vis_colors)):
        labels_color[labels==i] = vis_colors[i]/256.0

    pcd.colors = o3d.utility.Vector3dVector(labels_color.astype(np.float64))
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('./files/'+id+'/'+id+"_segmentation.ply", pcd, write_ascii=True)
    # o3d.io.write_point_cloud('./files/'+id+'/'+id+"_raw.ply", pcd1, write_ascii=True)
    file = Path('./segmentation-pointcloud/data/test/my.npy')
    file.unlink()
