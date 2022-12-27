import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import yaml

# torch imports
import torch
import torch.utils.data

# lightconvpoint imports
import lightconvpoint.utils.data_utils as data_utils
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils.network import get_conv, get_search
import lightconvpoint.utils.transformations as lcp_transfo
from lightconvpoint.knn import knn
from lightconvpoint.datasets.s3dis import S3DIS_Pillar as Dataset

from fkaconv.networks.kpconv import KPConvSeg as Network
from fkaconv.networks.fusion import Fusion as NetworkFusion

def batched_index_select(input, dim, index):
        index_shape = index.shape
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(input.size(0), -1, index_shape[1], index_shape[2])

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    pts_src = pts_src.unsqueeze(0).cpu().clone()
    pts_dest = pts_dest.unsqueeze(0).cpu().clone()
    indices = knn(pts_src, pts_dest, K)
    if K==1:
        indices = indices[0, :, 0]
        data_dest = data_src.transpose(0,1)[indices].transpose(0,1)
    else:
        data_dest = batched_index_select(data_src.unsqueeze(0).cpu(), 2, indices)
        data_dest = data_dest.mean(3)[0]
    return data_dest

def main(_config):

    print(_config)

    savedir_root = _config['training']['savedir']
    device = torch.device(_config['misc']['device'])
    rootdir = _config['dataset']['dir']

    N_CLASSES = 8

    # create the network
    print("Creating the network...", end="", flush=True)
    if "Fusion" == _config["network"]["model"]:
        def network_function():
            return NetworkFusion(
                3, N_CLASSES,
                get_conv(_config["network"]["backend_conv"]),
                get_search(_config["network"]["backend_search"]),
                config=_config
        )
    else:
        def network_function():
            return Network(
                3, N_CLASSES,
                get_conv(_config["network"]["backend_conv"]),
                get_search(_config["network"]["backend_search"]),
                config=_config
            )
    net = network_function()
    net.load_state_dict(torch.load(os.path.join(savedir_root, "checkpoint.pth"))["state_dict"])
    net.to(device)
    net.eval()
    print("Done")

    validation_transformations_data = []
    validation_transformations_features = []

    if not _config['training']['rgb']:
        validation_transformations_features.append(lcp_transfo.NoColor())

    ds = Dataset(rootdir, _config, split='test', network_function=network_function,
            transformations_data=validation_transformations_data,
            transformations_features=validation_transformations_features)

    # create the global confusion matrix
    cm_global = np.zeros((N_CLASSES, N_CLASSES))

    for file_id in range(ds.size()):

        ds.compute_sliding_window(file_id, _config["test"]["step"], _config["dataset"]["num_points"])

        test_loader = torch.utils.data.DataLoader(ds, batch_size=_config['test']['batch_size'], shuffle=False,
                                            num_workers=_config['misc']['threads']
                                            )

        filename = ds.filelist[file_id]

        # create a score accumulator
        scores = np.zeros((ds.get_points().shape[0], N_CLASSES))

        # iterate over the dataloader
        t = tqdm(test_loader, ncols=100, desc=filename)
        with torch.no_grad():
            for data in t:

                pts = data['pts'].to(device)
                features = data['features'].to(device)
                pts_ids = data['pts_ids']
                net_ids = data["net_indices"]
                net_pts = data["net_support"]
                for i in range(len(net_ids)):
                    net_ids[i] = net_ids[i].to(device)
                for i in range(len(net_pts)):
                    net_pts[i] = net_pts[i].to(device)
                
                outputs = net(features, pts, indices=net_ids, support_points=net_pts)
                outputs_np = outputs.transpose(1,2).cpu().detach().numpy().reshape((-1, N_CLASSES))
                scores[pts_ids.numpy().ravel()] += outputs_np

        # get the original points
        original_points = ds.get_points()
        original_labels = ds.get_labels()

        # mask = np.logical_and((np.abs(scores).sum(1) > 0), np.argmax(scores, axis=1) == np.argmax(scores_noc, axis=1))

        # compute the mask of points seen at prediction time
        mask = (np.abs(scores).sum(1) > 0)
        seen_scores = scores[mask]
        seen_points = original_points[mask]

        # project the scores on the original points
        scores = nearest_correspondance(
                    torch.from_numpy(seen_points).float().transpose(0,1), 
                    torch.from_numpy(original_points).float().transpose(0,1), 
                    torch.from_numpy(seen_scores).float().transpose(0,1), K=1).transpose(0,1).numpy()
        original_preds = np.argmax(scores, axis=1)

        # confusion matrix
        cm = confusion_matrix(original_labels, original_preds, labels=list(range(N_CLASSES)))
        cm_global += cm

        print("IoU", metrics.stats_iou_per_class(cm)[0])

        # saving results
        savedir_results = os.path.join(savedir_root, f"results_step{_config['test']['step']}" ,filename)

        # saving labels
        if _config['test']['savepreds']:
            os.makedirs(savedir_results, exist_ok=True)
            np.savetxt(os.path.join(savedir_results, "pred.txt"), original_preds,fmt='%d')

        if _config['test']['savepts']:
            os.makedirs(savedir_results, exist_ok=True)
            original_preds = np.expand_dims(original_preds,1).astype(int)
            original_points = ds.get_points()
            original_points = np.concatenate([original_points, original_preds], axis=1)
            np.savetxt(os.path.join(savedir_results, "pts.txt"), original_points, fmt=['%.4f','%.4f','%.4f','%d','%d','%d','%d'])

    print("WARNING: The next scores requires to be check with evaluation script at Convpoint repo")
    print("TODO: check if OK with eval scrip")
    iou = metrics.stats_iou_per_class(cm_global)
    print("Global IoU")
    print(iou[0])
    print("Global IoU per class")
    print(iou[1])

    print(f"{iou[0]} ", end="", flush=True)
    for i in range(iou[1].shape[0]):
        print(f"{iou[1][i]} ", end="")
    print("")


if __name__ == "__main__":

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config file in savedir")
    parser.add_argument("--step", "-s", type=float, default=0.5, help="sliding window parameter")
    parser.add_argument("--savepts", "-p", action="store_true", help="save the points in txt format")
    args = parser.parse_args()

    # update the base directory
    # makes it possible to move the directory
    # without editing the config file
    savedir = os.path.dirname(args.config)

    # load the configuration
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config["training"]["savedir"] = savedir
    config["test"]["step"] = args.step
    config["test"]["savepts"] = args.savepts

    # call the main function
    main(config)