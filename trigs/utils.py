import random
from torchvision import models
from trigs.custom_models import VGG_MOD, VGG16TIN, resnet18_mod
import torch
from collections import defaultdict
from typing import Tuple, Union, Optional


def deterministic_subset_label_distribution(labels: list, subset_start: float, subset_end: float) -> Union[list, None]:
    assert 0.0 <= subset_start < subset_end <= 1.0, \
        "subset_start and subset_end must be in [0,1] and subset_start must be smaller than subset_end"
    if subset_start == 0.0 and subset_end == 1.0:
        return None
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)
    subset_indices = list()
    for label, indices in label_indices.items():
        start_idx = int(subset_start * len(indices))
        end_idx = int(subset_end * len(indices))
        subset_indices.extend(sorted(indices)[start_idx:end_idx])
    return subset_indices


def random_split_label_distribution(labels: list, fractions: tuple, random_state: int) -> Tuple[list, list]:
    assert 0.0 < fractions[0] < 1.0 and 0.0 < fractions[1] < 1.0, "fractions must be in (0,1)"
    assert fractions[0] + fractions[1] == 1.0, "fractions must add up to 1"
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)
    rng = random.Random(random_state)
    list1 = list()
    list2 = list()
    for label, indices in label_indices.items():
        split_idx = int(fractions[0] * len(indices))
        rng.shuffle(indices)
        list1.extend(indices[:split_idx])
        list2.extend(indices[split_idx:])
    return list1, list2


def get_model(net, n_classes, device):
    if net == "vgg16tin":
        model = VGG16TIN(num_classes=n_classes)
    elif net == "resnet18_mod":
        model = resnet18_mod(num_classes=n_classes)
    elif net == "vggmod":
        model = VGG_MOD(num_classes=n_classes)
    else:
        raise ValueError(f"Unsupported model name: {net}")

    model = model.to(device)

    return model


def load_models_into_program(
        dataset_name: str, model_name: str, weights: Optional[str] = None
):
    if dataset_name == "ImageNet":
        if model_name == "resnet50":
            nnet = models.resnet50(pretrained=not weights, progress=True)
        elif model_name == "alexnet":
            nnet = models.alexnet(pretrained=not weights, progress=True)
        elif model_name == "vgg16":
            nnet = models.vgg16(pretrained=not weights, progress=True)
        elif model_name == "vitb16":
            nnet = models.vit_b_16(weights=None if weights else models.ViT_B_16_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid Model Name Specified for Imagenet")
    elif dataset_name == "Cifar10":
        if model_name == "resnet56":
            nnet = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=not weights)
        elif model_name == "vgg19_bn":
            nnet = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=not weights)
        elif model_name == "mobilenetv2_x1_4":
            nnet = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_4", pretrained=not weights)
        else:
            raise ValueError(f"Invalid Model Name Specified for Imagenet")
    elif dataset_name == "Cifar10Ext":
        assert model_name in ["vggmod"], "Invalid Model name for Cifar10Ext dataset"
        assert weights, "Weights Path should not be none"
        nnet = get_model(net=model_name, n_classes=10, device=get_device())
    elif dataset_name == "TINExt":
        assert model_name in ["resnet18_mod", "vgg16tin"], "Invalid Model name for TINExt dataset"
        assert weights, "Weights Path should not be none"
        nnet = get_model(net=model_name, n_classes=200, device=get_device())
    elif dataset_name == "TrojAI":
        assert weights, "Weights Path should not be none"
        nnet = torch.load(weights, map_location=lambda storage, loc: storage)
    elif dataset_name == "GTSRB":
        assert model_name in ["resnet18"], "Invalid Model name for TINExt dataset"
        assert weights, "Weights Path should not be none"
        nnet = get_model(net=model_name, n_classes=43, device=get_device())
    else:
        raise ValueError(f"Dataset Not Supported")
    nnet = nnet.to(get_device())
    if dataset_name != "TrojAI":  # TrojAI loads the weights directly without having to use load_state_dict
        if weights:
            nnet.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))
    return nnet


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device
