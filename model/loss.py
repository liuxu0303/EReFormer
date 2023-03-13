import torch
import torch.nn.functional as F
# local modules
from PerceptualSimilarity import models
from utils import loss
from kornia.filters.sobel import spatial_gradient, sobel

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
    # print(y_target.shape)
    mask = (y_target > 0) & (y_target < 1.0)
    y_input  = y_input[mask]
    y_target = y_target[mask]
    # print(y_target.shape)
    log_diff = y_input - y_target
    # is_nan = torch.isnan(log_diff)
    # return weight * (((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))**0.5)
    return weight * (((log_diff**2).mean()-(n_lambda*(log_diff.mean())**2))**0.5)


def scale_invariant_log_loss(y_input, y_target, n_lambda = 1.0):
    log_diff = torch.log(y_input)-torch.log(y_target)
    is_nan = torch.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2)


def mse_loss(y_input, y_target):
    return F.mse_loss(y_input[~torch.isnan(y_target)], y_target[~torch.isnan(y_target)])


class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]
        
        loss_value = 0
        diff = prediction - target
        _,_,H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []
        nbr_ignored_scales = 0

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                new_loss = torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
                # * batch size * 2 (because kornia spatial product has two outputs).
                # replaces the following line to be able to deal with nan's.
                # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

                # ignore losses that are nan due to target = nan
                if new_loss != new_loss:
                    nbr_ignored_scales += 1
                else:
                    loss_value += new_loss

        if preview:
            return record
        else:
            return loss_value / (self.num_scales - nbr_ignored_scales)


multi_scale_grad_loss_fn = MultiScaleGradient()


def multi_scale_grad_loss(prediction, target, preview = False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)

class combined_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred_img, pred_flow, target_img, target_flow):
        """
        image is tensor of N x 2 x H x W, flow of N x 2 x H x W
        These are concatenated, as perceptualLoss expects N x 3 x H x W.
        """
        pred = torch.cat([pred_img, pred_flow], dim=1)
        target = torch.cat([target_img, target_flow], dim=1)
        dist = self.loss(pred, target, normalize=False)
        return dist * self.weight


class warping_flow_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.warping_flow_loss
        self.weight = weight
        self.L0 = L0
        self.default_return = None

    def __call__(self, i, image1, flow):
        """
        flow is from image0 to image1 (reversed when passed to
        warping_flow_loss function)
        """
        loss = self.default_return if i < self.L0 else self.weight * self.loss(
                self.image0, image1, -flow)
        self.image0 = image1
        return loss


class voxel_warp_flow_loss():
    def __init__(self, weight=1.0):
        self.loss = loss.voxel_warping_flow_loss
        self.weight = weight

    def __call__(self, voxel, displacement, output_images=False):
        """
        Warp the voxel grid by the displacement map. Variance 
        of resulting image is loss
        """
        loss = self.loss(voxel, displacement, output_images)
        if output_images:
            loss = (self.weight * loss[0], loss[1])
        else:
            loss *= self.weight
        return loss


class flow_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target):
        """
        pred and target are Tensors with shape N x 2 x H x W
        PerceptualLoss expects N x 3 x H x W.
        """
        dist_x = self.loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], normalize=False)
        dist_y = self.loss(pred[:, 1:2, :, :], target[:, 1:2, :, :], normalize=False)
        return (dist_x + dist_y) / 2 * self.weight


class flow_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


# keep for compatibility
flow_loss = flow_l1_loss


class perceptual_loss():
    def __init__(self, net='vgg', use_gpu=True):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu)
        # self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return dist.mean()

PL_loss = perceptual_loss()

def perceptual_loss_fc(prediction, target, normalize = True):
    return PL_loss(prediction, target, normalize)

class l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class temporal_consistency_loss():
    def __init__(self, L0=2):
        assert L0 > 0
        self.loss = loss.temporal_consistency_loss
        # self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        """
        flow is from image0 to image1 (reversed when passed to
        temporal_consistency_loss function)
        """
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             -flow, output_images=output_images)
            if output_images:
                loss = (loss[0], loss[1])
            else:
                loss = loss
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss

TC_loss = temporal_consistency_loss()

def temporal_consistency_loss_fc(i, image1, processed1, flow, output_images=False):
    return TC_loss(i, image1, processed1, flow, output_images)