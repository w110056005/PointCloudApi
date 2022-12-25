from cv2 import moments
import numpy as np
import torch
import torch.nn as nn


from open3d._ml3d.torch.models.base_model import BaseModel
from open3d._ml3d.utils import MODEL
from open3d._ml3d.torch.modules.losses import filter_valid_label
from open3d._ml3d.datasets.augment import SemsegAugmentation
from open3d.ml.torch.layers import SparseConv, SparseConvTranspose
from open3d.ml.torch.ops import voxelize, reduce_subarrays_sum


class SparseEncDec(BaseModel):
    """Semantic Segmentation model.
    Attributes:
        name: Name of model.
          Default to "SparseEncDec".
        device: Which device to use (cpu or cuda).
        voxel_size: Voxel length for subsampling.
        multiplier: min length of feature length in each layer.
        conv_block_reps: repetition of Unet Blocks.
        residual_blocks: Whether to use Residual Blocks.
        in_channels: Number of features(default 3 for color).
        num_classes: Number of classes.
    """

    def __init__(
            self,
            name="SparseEncDec",
            device="cuda",
            multiplier=16,  # Proportional to number of neurons in each layer.
            voxel_size=0.05,
            conv_block_reps=1,  # Conv block repetitions.
            residual_blocks=False,
            in_channels=3,
            num_classes=13,
            grid_size=4096,
            batcher='DefaultBatcher',
            augment=None,
            **kwargs):
        super(SparseEncDec, self).__init__(name=name,
                                             device=device,
                                             multiplier=multiplier,
                                             voxel_size=voxel_size,
                                             conv_block_reps=conv_block_reps,
                                             residual_blocks=residual_blocks,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             grid_size=grid_size,
                                             batcher=batcher,
                                             augment=augment,
                                             **kwargs)
        cfg = self.cfg
        self.device = device
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)
        self.multiplier = cfg.multiplier
        self.input_layer = InputLayer()
        self.sub_sparse_conv1 = SubmanifoldSparseConv(in_channels=in_channels,
                                                     filters=multiplier,
                                                     kernel_size=[3, 3, 3])
        self.sub_sparse_conv2 = SubmanifoldSparseConv(in_channels=multiplier,
                                                     filters=2*multiplier,
                                                     kernel_size=[3, 3, 3])

        self.sub_sparse_conv3 = SubmanifoldSparseConv(in_channels=2*multiplier,
                                                     filters=3*multiplier,
                                                     kernel_size=[5, 5, 5])

        self.sub_sparse_deconv1 = SubmanifoldSparseConvTranspose(in_channels=3*multiplier,
                                                     filters=2*multiplier,
                                                     kernel_size=[5, 5, 5])
        
        self.sub_sparse_deconv2 = SubmanifoldSparseConvTranspose(in_channels=2*multiplier,
                                                     filters=multiplier,
                                                     kernel_size=[3, 3, 3])

        self.sub_sparse_deconv3 = SubmanifoldSparseConvTranspose(in_channels=multiplier,
                                                     filters=in_channels,
                                                     kernel_size=[3, 3, 3])

        self.relu = ReLUBlock()
        self.linear = LinearBlock(in_channels, num_classes)
        self.output_layer = OutputLayer()

    def forward(self, inputs):
        pos_list = []
        feat_list = []
        index_map_list = []

        for i in range(len(inputs['points'])):
            pos = inputs['points'][i]
            feat = inputs['feat'][i]
            # feat, pos, index_map = self.input_layer(feat, pos)
            pos_list.append(pos)
            feat_list.append(feat)
            # index_map_list.append(index_map)

        # Encoder
        voxel_size = 0.02
        feat_list = self.sub_sparse_conv1(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)
        voxel_size *= 2
        feat_list = self.sub_sparse_conv2(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)
        voxel_size *= 2
        feat_list = self.sub_sparse_conv3(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)

        # Decoder
        feat_list = self.sub_sparse_deconv1(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)
        voxel_size /= 2
        feat_list = self.sub_sparse_deconv2(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)
        voxel_size /= 2
        feat_list = self.sub_sparse_deconv3(feat_list, pos_list, voxel_size=voxel_size)
        feat_list = self.relu(feat_list)
        
        output = self.linear(feat_list)[0]

        return output

    
    def preprocess(self, data, attr):
        # If num_workers > 0, use new RNG with unique seed for each thread.
        # Else, use default RNG.

        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng


        points = np.array(data['points'], dtype=np.float32)


        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            raise Exception(
                "SparseConvnet doesn't work without feature values.")

        feat = np.array(data['feat'], dtype=np.float32)

        # Scale to voxel size.
        points *= 1. / self.cfg.voxel_size  # Scale = 1/voxel_size'



        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(points,
                                                          feat,
                                                          labels,
                                                          self.cfg.get(
                                                              'augment', None),
                                                          seed=rng)
        m = points.min(0)
        M = points.max(0)

        # Randomly place pointcloud in 4096 size grid.
        grid_size = self.cfg.grid_size
        offset = -m + np.clip(grid_size - M + m - 0.001, 0, None) * rng.random(
            3) + np.clip(grid_size - M + m + 0.001, None, 0) * rng.random(3)

        points += offset
        idxs = (points.min(1) >= 0) * (points.max(1) < 4096)

        points = points[idxs]
        feat = feat[idxs]
        labels = labels[idxs]

        points = (points.astype(np.int32) + 0.5).astype(
            np.float32)  # Move points to voxel center.

        data = {}
        data['points'] = points
        data['feat'] = feat
        data['label'] = labels

        return data

    def transform(self, data, attr):
        data['points'] = torch.from_numpy(data['points'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])

        return data

    def update_probs(self, inputs, results, test_probs, test_labels):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()
        labels = np.argmax(probs, 1)

        self.trans_point_sampler(patchwise=False)

        return probs, labels

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})
        data['batch_lengths'] = [data['point'].shape[0]]
        data = self.transform(data, {})

        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))

        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.
        Attributes:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).
        
        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = inputs['label']

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler


MODEL._register_module(SparseEncDec, 'torch')


class ReLUBlock(nn.Module):

    def __init__(self):
        super(ReLUBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.relu(torch.cat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l

        return out_list

    def __name__(self):
        return "ReLUBlock"


class LinearBlock(nn.Module):

    def __init__(self, a, b):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(a, b)

    def forward(self, feat_list):
        out_list = []
        for feat in feat_list:
            out_list.append(self.linear(feat))

        return out_list

    def __name__(self):
        return "LinearBlock"


class InputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(InputLayer, self).__init__()
        self.voxel_size = torch.Tensor([voxel_size, voxel_size, voxel_size])

    def forward(self, features, in_positions):
        v = voxelize(
            in_positions,
            torch.LongTensor([0,
                              in_positions.shape[0]]).to(in_positions.device),
            self.voxel_size, torch.Tensor([0, 0, 0]),
            torch.Tensor([4096, 4096, 4096]))

        # Contiguous repeating positions.
        in_positions = in_positions[v.voxel_point_indices]
        features = features[v.voxel_point_indices]

        # Find reverse mapping.
        reverse_map_voxelize = np.zeros((in_positions.shape[0],))
        reverse_map_voxelize[v.voxel_point_indices.cpu().numpy()] = np.arange(
            in_positions.shape[0])
        reverse_map_voxelize = reverse_map_voxelize.astype(np.int32)

        # Unique positions.
        in_positions = in_positions[v.voxel_point_row_splits[:-1]]

        # Mean of features.
        count = v.voxel_point_row_splits[1:] - v.voxel_point_row_splits[:-1]
        reverse_map_sort = np.repeat(np.arange(count.shape[0]),
                                     count.cpu().numpy()).astype(np.int32)

        features_avg = in_positions.clone()
        features_avg[:, 0] = reduce_subarrays_sum(features[:, 0],
                                                  v.voxel_point_row_splits)
        features_avg[:, 1] = reduce_subarrays_sum(features[:, 1],
                                                  v.voxel_point_row_splits)
        features_avg[:, 2] = reduce_subarrays_sum(features[:, 2],
                                                  v.voxel_point_row_splits)

        features_avg = features_avg / count.unsqueeze(1)

        return features_avg, in_positions, reverse_map_sort[
            reverse_map_voxelize]


class OutputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def forward(self, features_list, index_map_list):
        out = []
        for feat, index_map in zip(features_list, index_map_list):
            out.append(feat[index_map])
        return torch.cat(out, 0)


class SubmanifoldSparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(SubmanifoldSparseConv, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = 0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self,
                features_list,
                in_positions_list,
                out_positions_list=None,
                voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = in_positions_list

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "SubmanifoldSparseConv"



class SubmanifoldSparseConvTranspose(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(SubmanifoldSparseConvTranspose, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = 0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConvTranspose(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self,
                features_list,
                in_positions_list,
                out_positions_list=None,
                voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = in_positions_list

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "SubmanifoldSparseConvTranspose"


def calculate_grid(in_positions):
    filter = torch.Tensor([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0],
                           [0, -1, -1], [0, -1, 0], [0, 0, -1],
                           [0, 0, 0]]).to(in_positions.device)

    out_pos = in_positions.long().repeat(1, filter.shape[0]).reshape(-1, 3)
    filter = filter.repeat(in_positions.shape[0], 1)

    out_pos = out_pos + filter
    out_pos = out_pos[out_pos.min(1).values >= 0]
    out_pos = out_pos[(~((out_pos.long() % 2).bool()).any(1))]
    out_pos = torch.unique(out_pos, dim=0)

    return out_pos + 0.5


class Convolution(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(Convolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_positions_list = []
        for in_positions in in_positions_list:
            out_positions_list.append(calculate_grid(in_positions))

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        out_positions_list = [out / 2 for out in out_positions_list]

        return out_feat, out_positions_list

    def __name__(self):
        return "Convolution"


class DeConvolution(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(DeConvolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = SparseConvTranspose(in_channels=in_channels,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       use_bias=use_bias,
                                       offset=offset,
                                       normalize=normalize)

    def forward(self,
                features_list,
                in_positions_list,
                out_positions_list,
                voxel_size=1.0):
        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "DeConvolution"



def load_unet_wts(net, path):
    wts = list(torch.load(path).values())
    state_dict = net.state_dict()
    i = 0
    for key in state_dict:
        if 'offset' in key or 'tracked' in key:
            continue
        if len(wts[i].shape) == 4:
            shp = wts[i].shape
            state_dict[key] = np.transpose(
                wts[i].reshape(int(shp[0]**(1 / 3)), int(shp[0]**(1 / 3)),
                               int(shp[0]**(1 / 3)), shp[-2], shp[-1]),
                (2, 1, 0, 3, 4))
        else:
            state_dict[key] = wts[i]
        i += 1

    net.load_state_dict(state_dict)