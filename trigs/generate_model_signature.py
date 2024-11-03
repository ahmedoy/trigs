# Import Necessary Packages
from typing import List, Optional, Tuple
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torchvision.transforms import functional as ttf
from trigs.utils import load_models_into_program, get_device
import argparse
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import json
import warnings


try:
    from torchmetrics.image import TotalVariation
except ImportError:
    TotalVariation = None

warnings.filterwarnings("ignore")


def tv(img):
    if img.ndim != 4:
        raise RuntimeError(f"Expected input `img` to be an 4D tensor, but got {img.shape}")
    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res1 + res2
    return score


class ClassSpecificImageGeneration:
    """
    Produces an image that maximizes the logit of a certain class with gradient ascent
    """
    def __init__(
            self,
            model: nn.Module,
            mean: List[float],
            std: List[float],
            width: int,
            height: int,
            channels: int,
            lr: float,
            optim_type: str = 'ADAM',
            iterations: Optional[int] = 60,
            blur_freq: Optional[int] = None,
            blur_sigma: Optional[float] = None,
            blur_hks: Optional[int] = 2,
            clipping_val: Optional[float] = 0.0,
            normalize_grad: Optional[bool] = False,
            loss_type: Optional[str] = 'logit',
            wd: Optional[float] = 1e-5,
            standardize_output: Optional[bool] = True,
            clamp_pixels_freq: Optional[int] = None,
            lambda_tv: Optional[float] = 0.,
    ):
        self.mean = mean
        self.std = std

        self.w = width
        self.h = height
        self.c = channels
        self.lr = lr
        self.iterations = iterations
        self.blur_freq = blur_freq
        self.blur_sigma = blur_sigma
        self.blur_ks = 2 * blur_hks + 1
        self.clipping_val = clipping_val
        self.normalize_grad = normalize_grad
        self.loss_type = loss_type
        self.wd = wd
        self.standardize_output = standardize_output
        self.clamp_pixels_freq = clamp_pixels_freq
        self.lambda_tv = lambda_tv
        self.tv = tv if TotalVariation is None else TotalVariation(reduction='none').to(get_device())

        if optim_type == 'ADAM':
            self.optim_class = Adam
        elif optim_type == 'SGD':
            self.optim_class = SGD
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")
        
        self.model = model
        if self.loss_type == "grad_norm":
            self.model = self.swap_model_activation(self.model)
            
        self.model.eval()

        # Generate a random image
        rng = np.random.default_rng(seed=928462)
        self.init_image = np.uint8(
            rng.uniform(0, 255, (width, height, channels))
        )

    def generate(self, target_class: [int, List[int]] = 0, ascent: bool = True) -> Tuple[np.ndarray, int]:
        """Generates class specific image
        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})
        Returns:
            np.ndarray -- Final maximally activated class image
        """
        if isinstance(target_class, int):
            target_class = [target_class]
        created_images = np.tile(self.init_image, [len(target_class), 1, 1, 1])
        processed_images = self.images_to_tensors(created_images)
        processed_images = processed_images.to(get_device())
        processed_images.requires_grad = True
        targets = torch.as_tensor(target_class, device=get_device())
        indices = torch.arange(end=len(target_class), device=get_device())
        optimizer = self.optim_class([processed_images], lr=self.lr, weight_decay=self.wd)
        min_losses = [float('inf')] * len(target_class)
        best_images = processed_images.clone()
        for i in range(self.iterations):
            # Process image and return variable

            # implement gaussian blurring every blur_freq iteration to improve output
            if self.blur_freq and i % self.blur_freq == 0:
                processed_images = ttf.gaussian_blur(processed_images.detach(),
                                                     [self.blur_ks, self.blur_ks],
                                                     self.blur_sigma)
                processed_images.requires_grad = True
                optimizer = self.optim_class([processed_images], lr=self.lr, weight_decay=self.wd)
            # Forward
            output = self.model(processed_images)

            # Target specific class
            if self.loss_type == 'ce':
                class_loss = F.cross_entropy(output, targets)
            elif self.loss_type == 'logit':
                class_loss = -output[indices, targets]
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
            if ascent:
                class_loss = -class_loss

            tv_loss = (self.lambda_tv * self.tv(processed_images)) if self.lambda_tv > 0. else 0.

            total_loss = class_loss + tv_loss

            for li, loss in enumerate(total_loss):
                if loss.item() < min_losses[li]:
                    min_losses[li] = loss.item()
                    best_images[li] = processed_images[li].clone()

            optimizer.zero_grad()

            # Backward
            total_loss.sum().backward()

            if self.normalize_grad:
                for gi in range(len(target_class)):
                    img_grad = processed_images.grad[gi]
                    norm = torch.norm(img_grad, dim=[1, 2])
                    img_grad = img_grad.permute(1, 2, 0)
                    img_grad /= (norm + 1e-8)
                    img_grad = img_grad.permute(2, 0, 1)
                    processed_images.grad[gi] = img_grad

            if self.clipping_val > 0:
                torch.nn.utils.clip_grad_norm(processed_images, self.clipping_val)
            # Update image
            optimizer.step()

            if self.clamp_pixels_freq and i % self.clamp_pixels_freq == 0:
                tmp_images = processed_images.detach()
                for channel in range(tmp_images.shape[1]):
                    tmp_images[:, channel, ...] *= self.std[channel]
                    tmp_images[:, channel, ...] += self.mean[channel]
                tmp_images = torch.clamp(tmp_images, 0., 1.)
                for channel in range(tmp_images.shape[1]):
                    tmp_images[:, channel, ...] -= self.mean[channel]
                    tmp_images[:, channel, ...] /= self.std[channel]
                # For some reason, I have to use contiguous() call in the following line
                processed_images = tmp_images.contiguous().requires_grad_()
                optimizer = self.optim_class([processed_images], lr=self.lr, weight_decay=self.wd)

        predictions = torch.argmax(self.model(best_images), dim=1).detach().cpu().numpy()

        # Recreate image
        created_images = self.recreate_images(best_images.cpu())

        return created_images, predictions

    def images_to_tensors(self, imgs):
        """
            Processes image with optional Gaussian blur for CNNs
        Args:
            imgs (nd.array): numpy array to process
        returns:
            im_as_param (torch.Tensor): a tensor that requires gradient
        """
        ims_as_arrs = np.float32(imgs)
        ims_as_arrs = ims_as_arrs.transpose(0, 3, 1, 2)  # Convert array to B,C,H,W
        ims_as_arrs /= 255
        # Normalize the channels
        for channel in range(ims_as_arrs.shape[1]):
            ims_as_arrs[:, channel, ...] -= self.mean[channel]
            ims_as_arrs[:, channel, ...] /= self.std[channel]
        # Convert to float tensor
        ims_as_tens = torch.from_numpy(ims_as_arrs)
        return ims_as_tens

    def recreate_images(self, ims_as_tensors):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            ims_as_tensors (torch variable): Images to recreate
        returns:
            recreated_ims (numpy arr): Recreated images in array
        """
        recreated_ims = ims_as_tensors.detach().cpu().numpy()
        for channel in range(recreated_ims.shape[1]):
            recreated_ims[:, channel, ...] *= self.std[channel]
            recreated_ims[:, channel, ...] += self.mean[channel]
        if self.standardize_output:
            for img in recreated_ims:
                img -= img.mean()
                img /= img.std() + 1e-5
                img *= 0.25
                img += 0.5
        recreated_ims[recreated_ims > 1] = 1
        recreated_ims[recreated_ims < 0] = 0
        recreated_ims = np.round(recreated_ims * 255)

        recreated_ims = np.uint8(recreated_ims).transpose(0, 2, 3, 1)
        return recreated_ims




    def swap_model_activation(self, model: nn.Module) -> nn.Module:
        # TODO Train the model to work after swapping SiLU
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(model, name, nn.SiLU())
            else:
                self.swap_model_activation(module)  # Recursively check child modules
        return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        help='Name of dataset used to train the probe models. '
                             'Use Cifar10Ext for CIFAR10 and TINExt for Tiny ImageNet')
    parser.add_argument("--model_name", type=str, default="",
                        help='Name of the architecture for the probe model. '
                             'Use vggmod with CIFAR10, resnet18_mod with Tiny ImageNet, and vit16b with ImageNet.')
    parser.add_argument("--weights_path", type=str, default="", help='Path to model weights file.')
    parser.add_argument("--iterations", type=int, default=200,
                        help='Number of optimization iterations. Use default to reproduce paper results.')
    parser.add_argument("--learning_rate", type=float, required=True,
                        help='Learning rate. Use 10 with CIFAR10 and 0.1 with others to reproduce paper results.')
    parser.add_argument("--output_dir", type=str, required=True, help="Path to store the resulting signature images.")
    parser.add_argument("--opt_type", type=str, default='ADAM', choices=['ADAM', 'SGD'],
                        help='Optimizer to use. Use default to reproduce paper results.')
    parser.add_argument("--blur_freq", type=int, default=None,
                        help='Frequency of applying Gaussian bluring. Use default to reproduce paper results.')
    parser.add_argument("--blur_sigma", type=float, default=None,
                        help='Sigma for the Gaussian blur. Use default to reproduce paper results.')
    parser.add_argument("--blur_hks", type=int, default=2,
                        help='Half kernel width for the Gaussian blur. Use default to reproduce paper results.')
    parser.add_argument("--clipping_val", type=float, default=0.0,
                        help='Value beyond which to clip normalized gradient. '
                             '0 means no clipping. Use default to reproduce paper results.')
    parser.add_argument("--normalize_grad", action='store_true', default=False,
                        help='If used, gradients will be normalized to a unit vector before every opt steps. '
                             'Do NOT use to reproduce paper results.')
    parser.add_argument("--loss_type", type=str, default='logit', choices=['logit', 'ce'],
                        help='''Whether to optimize the logit value or the cross entropy loss. Use default to reproduce 
                        paper results.''')
    parser.add_argument("--standardize_output", action='store_true', default=False,
                        help='Convert each signature image to have 0.5 mean and 0.25 std. Use only with CIFAR10.')
    parser.add_argument("--clamp_pixels_freq", type=int, default=None,
                        help='Frequency of clamping pixel values to valid range. Use default to reproduce paper results.')
    parser.add_argument("--lambda_tv", type=float, default=0.0,
                        help='Weight of the TV loss. Use 0.01 with Tiny ImageNet and 0.001 with others.')
    parser.add_argument("--batch_size", type=int, default=None,
                        help='Number of signature images to create at the same time. '
                             'Adjust based on your GPU memory for ImageNet and use default for others.')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='Debug mode. Do NOT use to reproduce paper results.')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    weights_path = args.weights_path
    iterations = args.iterations
    learning_rate = args.learning_rate
    output_dir = args.output_dir
    opt_type = args.opt_type
    blur_freq = args.blur_freq
    blur_sigma = args.blur_sigma
    blur_hks = args.blur_hks
    clipping_val = args.clipping_val
    normalize_grad = args.normalize_grad
    loss_type = args.loss_type
    standardize_output = args.standardize_output
    clamp_pixels_freq = args.clamp_pixels_freq
    lambda_tv = args.lambda_tv
    batch_size = args.batch_size
    debug = args.debug

    classifier = load_models_into_program(
        dataset_name=dataset_name, model_name=model_name, weights=weights_path
    )

    if dataset_name == "TINExt":
        height, width, channels = 64, 64, 3
        n_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == 'Cifar10Ext':
        height, width, channels = 32, 32, 3
        n_classes = 10
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
    elif dataset_name == 'ImageNet':
        height, width, channels = 224, 224, 3
        n_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == 'TrojAI':
        height, width, channels = 224, 224, 3
        config_path = os.path.join(os.path.dirname(weights_path), "config.json")
        with open(config_path, "rt") as fp:
            cfg = json.load(fp)
        n_classes = cfg["NUMBER_CLASSES"]
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    if not batch_size:
        batch_size = n_classes

    image_gen = ClassSpecificImageGeneration(
        model=classifier,
        mean=mean,
        std=std,
        width=width,
        height=height,
        channels=channels,
        iterations=iterations,
        lr=learning_rate,
        optim_type=opt_type,
        blur_freq=blur_freq,
        blur_sigma=blur_sigma,
        blur_hks=blur_hks,
        clipping_val=clipping_val,
        normalize_grad=normalize_grad,
        loss_type=loss_type,
        standardize_output=standardize_output,
        clamp_pixels_freq=clamp_pixels_freq,
        lambda_tv=lambda_tv,
    )

    for start_class_no in tqdm(range(0, n_classes, batch_size)):
        if debug and start_class_no > 5:
            break
        end_class_no = min(start_class_no + batch_size, n_classes) - 1


        # check if using grad_norm mode
        if loss_type == 'grad_norm':
            # create the hessian/grad_norm signature channels
            last_file_name = os.path.join(
                output_dir, f"h_{str(end_class_no).zfill(3)}.png")
            if not os.path.exists(last_file_name):
                imgs, _ = image_gen.generate(
                    target_class=list(range(start_class_no, end_class_no + 1)),
                    ascent=False
                )
                class_no = start_class_no
                for img in imgs:
                    im = Image.fromarray(img)
                    out_file_name = os.path.join(
                        output_dir, f"h_{str(class_no).zfill(3)}.png")
                    im.save(out_file_name)
                    class_no += 1
        else:
            # create the minimization signature channels
            last_file_name = os.path.join(
                output_dir, f"+{str(end_class_no).zfill(3)}.png")
            if not os.path.exists(last_file_name):
                imgs, _ = image_gen.generate(
                    target_class=list(range(start_class_no, end_class_no + 1)),
                    ascent=True
                )
                class_no = start_class_no
                for img in imgs:
                    im = Image.fromarray(img)
                    out_file_name = os.path.join(
                        output_dir, f"+{str(class_no).zfill(3)}.png")
                    im.save(out_file_name)
                    class_no += 1
            # create the maximization signature channels
            last_file_name = os.path.join(
                output_dir, f"-{str(end_class_no).zfill(3)}.png")
            if not os.path.exists(last_file_name):
                imgs, _ = image_gen.generate(
                    target_class=list(range(start_class_no, end_class_no + 1)),
                    ascent=False
                )
                class_no = start_class_no
                for img in imgs:
                    im = Image.fromarray(img)
                    out_file_name = os.path.join(
                        output_dir, f"-{str(class_no).zfill(3)}.png")
                    im.save(out_file_name)
                    class_no += 1
    print("SUCCESS!")


if __name__ == "__main__":
    main()
