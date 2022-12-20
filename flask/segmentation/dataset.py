import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging

from open3d._ml3d.datasets.utils import DataProcessing as DP
from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import make_dir, DATASET

log = logging.getLogger(__name__)


class MySemantic3D(BaseDataset):
    """This class is used to create a dataset based on the Semantic3D dataset,
    and used in visualizer, training, or testing.
    The dataset includes 8 semantic classes and covers a variety of urban
    outdoor scenes.
    """

    def __init__(self,
                 dataset_path,
                 name='MySemantic3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=111620,
                 class_weights=[0.40306796761589736, 0.7111864363583075, 0.6551315310662175, 0, 1.5809526507372207, 1.4324580991247657, 2.3817347700842846, 1.0207030249826257, 1.9866158829601681, 3.3396164317984622, 0.5544461995449984, 5.962606837606837, 0.4802636674210676],
                 ignored_label_inds=[],
                 val_files=[
                     'bildstein_station3_xyz_intensity_rgb',
                     'sg27_station2_intensity_rgb'
                 ],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.
        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (MySemantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            val_files: The files with the data.
            test_result_folder: The folder where the test results should be stored.
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.all_files = glob.glob(str(Path(self.cfg.dataset_path) / 'all/*.npy'))
        self.train_files = glob.glob(str(Path(self.cfg.dataset_path) / 'train/*.npy'))
        self.test_files = glob.glob(str(Path(self.cfg.dataset_path) / 'test/*.npy'))
        self.val_files = glob.glob(str(Path(self.cfg.dataset_path) / 'val/*.npy'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)
        self.val_files = np.sort(self.val_files)

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.
        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10',
            11: '11',
            12: '12'
        }
        return label_to_names

    def get_split(self, split):
        return Semantic3DSplit(self, split=split)
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
	"""

    def get_split_list(self, split):
        """Returns the list of data splits available.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.
        Args:
            attr: The attribute that needs to be checked.
        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.
        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class Semantic3DSplit(BaseDatasetSplit):
    """This class is used to create a split for Semantic3D dataset.
    Initialize the class.
    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.
    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = np.load(pc_path)

        points = pc[:, 0:3]
        rgb = pc[:, 3:6]
        labels = pc[:, 6]

        points = np.array(points, dtype=np.float32)
        rgb = np.array(rgb, dtype=np.float32)/255
        labels = np.array(labels, dtype=np.float32)

        data = {
            'points': points,
            'feat': rgb,
            'label': labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(MySemantic3D)