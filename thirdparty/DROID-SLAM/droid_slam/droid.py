import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args, device=None):
        super(Droid, self).__init__()
        # 支持外部传入 device，否则默认 cuda:0
        if device is None:
            device = "cuda:0"
        self.device = device
        self.load_weights(args.weights, device)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo, device=self.device)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh, device=self.device)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            # from visualization import droid_visualization
            from vis_headless import droid_visualization
            print('Using headless ...')
            self.visualizer = Process(target=droid_visualization, args=(self.video, '.'))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video, device=self.device)


    def load_weights(self, weights, device=None):
        """ load trained model weights """
        if device is None:
            device = getattr(self, 'device', 'cuda:0')

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to(device).eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, mask)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None, backend=True):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        if backend:
            if self.device != "cpu":
                torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(7)

            if self.device != "cpu":
                torch.cuda.empty_cache()
            # print("#" * 32)
            self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

    def compute_error(self):
        """ compute slam reprojection error """

        del self.frontend

        if self.device != "cpu":
            torch.cuda.empty_cache()
        self.backend(12)

        return self.backend.errors[-1]


