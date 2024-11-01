import numpy as np
from matplotlib import pyplot as plt
from lightning import LightningModule, Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
import torch.nn.functional as F
from trigs.utils import get_device
import argparse
from trigs.custom_models import baseline_classifier, hybrid_transfomer
from copy import deepcopy
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    auc,
)
import os
from trigs.signature_dm import SignatureDataModule
import pandas as pd
import json


class SignatureClassifierWrapper(LightningModule):
    def __init__(
        self, model: nn.Module, test_prefix: str = "", learning_rate: float = 1e-4
    ) -> None:
        super().__init__()

        self.model = model
        self.test_prefix = test_prefix
        self.learning_rate = learning_rate
        self.test_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        z = self.model(x)
        y = y.type(torch.LongTensor).to(get_device())
        z_softmax = torch.softmax(z, dim=1)
        n_correct = torch.sum(torch.argmax(z_softmax, dim=1).eq(y)).item()
        n_samples = len(y)
        self.test_step_outputs.append([n_correct, n_samples, z_softmax, y])
        return n_correct, n_samples, z_softmax, y

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        tot_correct = tot_samples = 0
        z = outputs[0][2]
        t = outputs[0][3]

        for idx, (nc, ns, z_sig, yt) in enumerate(outputs):
            tot_correct += nc
            tot_samples += ns
            if idx >= 1:
                z = torch.cat((z, z_sig))
                t = torch.cat((t, yt))
        self.log(self.test_metric_prerfix + "Accuracy", tot_correct / tot_samples)

        z_ = []
        for idx, gt in enumerate(t):
            z_.append(deepcopy(z[idx][1].item()))
        z_ = torch.tensor(z_).to(get_device())
        assert z_.shape == t.shape, f"Invalid Tensor Shape {z_.shape}, {t.shape}"
        val = roc_auc_score(t.cpu().numpy(), z_.cpu().numpy())
        self.log(self.test_metric_prerfix + "AUROC", float(val))
        self.test_step_outputs.clear()

    def compute_loss_(self, batch, _):
        x, y = batch
        y = y.type(torch.LongTensor).to(get_device())
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        return loss, z, y

    def training_step(self, train_batch, train_batch_idx):
        loss, z, y = self.compute_loss_(train_batch, train_batch_idx)
        self.log("train_loss", loss)
        z_softmax = torch.softmax(z, dim=1)
        n_correct = torch.sum(torch.argmax(z_softmax, dim=1).eq(y)).item()
        n_samples = len(y)
        self.log("train_acc_step", n_correct / n_samples)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        loss, z, y = self.compute_loss_(val_batch, val_batch_idx)
        z_softmax = torch.softmax(z, dim=1)
        n_correct = torch.sum(torch.argmax(z_softmax, dim=1).eq(y)).item()
        n_samples = len(y)
        self.log("val_acc_step", n_correct / n_samples)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        loss = torch.mean(torch.as_tensor(self.val_step_outputs))
        self.log("val_loss", loss)
        self.val_step_outputs.clear()

    def predict_step(self, pred_batch, batch_idx, *_):
        # I am not sure why Sudharshan made it this way, but this method works correctly only if batch size is 1, which
        # is currently the case for the testing data loader.
        x, y = pred_batch
        assert len(x) == 1, "Test dataset batch size must be 1"  # to make sure batch_size is 1
        z = self.model(x)
        z = torch.softmax(z, dim=1)
        pred = torch.argmax(z, dim=1)

        return pred.cpu().numpy()[0], y.cpu().numpy()[0], z[0][1].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        help='The name of the dataset used to train the probe models. ImageNet or TIN (Tiny ImageNet).')
    parser.add_argument("--train_ds_use_frac", type=float, default=1.0,
                        help='Fraction to use from the training dataset of shadow models.')
    parser.add_argument("--model_name", type=str, default="Baseline", choices=['Baseline', 'hybrid_transformer'],
                        help='Name of the model architecture to be used for the classifier. '
                             '"Baseline" is the one used in the paper.')
    parser.add_argument("--backbone_name", type=str, default="ResNext50_32x4d",
                        help='The convolutional backbone. Use default to reproduce paper results.')
    parser.add_argument("--embed_dim", type=int, default=256,
                        help='Embedding dimension for the Transformer model. Use default to reproduce paper results.')
    parser.add_argument("--num_heads", type=int, default=8,
                        help='''The number of self-attention heads in the Transformer model. Use default to reproduce 
                        paper results.''')
    parser.add_argument("--depth", type=int, default=3,
                        help='''Number of transformer layers in the Transformer model. Use default to reproduce paper 
                        results.''')
    parser.add_argument("--epochs", type=int, required=True,
                        help='''Number of training epochs. 100 was used for results in the paper with the for Baseline 
                        model. 20 was found to be sufficient for the hybrid_transformer models.''')
    parser.add_argument("--ckpt_path", type=str, required=True, help='Path to save checkpoints.')
    parser.add_argument("--data_path", type=str, required=True, help='Path where model signatures are saved.')
    parser.add_argument("--opt_mode", type=str, required=True,
                        help='''Whether to use activation minimization ("min"), maximization ("max"), both ("all"), or 
                        their stats ("stats-qXhYY"), where X is the number of quantiles (3 or 7) and YY is the number of
                         histogram bins (04, 08, 12, or 16)).''')
    parser.add_argument("--split_rand_seed", type=int, required=True,
                        help='Random seed used to split training dataset into val and train.')

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    model_name = args.model_name
    embed_dim = args.embed_dim
    backbone_name = args.backbone_name
    num_heads = args.num_heads
    depth = args.depth
    ckpt_path = args.ckpt_path
    data_path = args.data_path
    epochs = args.epochs
    opt_mode = args.opt_mode
    split_rand_seed = args.split_rand_seed
    train_ds_use_frac = args.train_ds_use_frac

    assert 0. < train_ds_use_frac <= 1., "train_ds_use_frac must be in (0, 1]"
    assert not (opt_mode.startswith("stats") and model_name == "hybrid_transformer")

    with open(os.path.join(ckpt_path, "params.json"), 'wt') as fp:
        json.dump(args.__dict__, fp, indent=4)

    num_trojan_classes = 2
    if opt_mode.startswith("stats"):
        stat_type = opt_mode.split("-")[1]
        num_channels = 3 * (4 + int(stat_type[1]) + int(stat_type[-2:]))
    else:
        in_chans = 3
        if dataset_name.startswith("TIN"):
            num_classes = 200
        elif dataset_name.startswith("ImageNet"):
            num_classes = 1000
        elif dataset_name == "CIFAR10":
            num_classes = 10
        elif dataset_name == "TrojAI-Round01":
            num_classes = 5
        elif dataset_name in ["TrojAI-Round02", "TrojAI-Round03"]:
            num_classes = 25
        elif dataset_name == "TrojAI-Round04":
            num_classes = 45
        else:
            raise ValueError(f"Invalide dataset name: {dataset_name}")
        if opt_mode in ["min", "max"]:
            num_maps = num_classes
        elif opt_mode == "all":
            num_maps = 2 * num_classes
        else:
            raise ValueError(f"Invalid optimization mode: {opt_mode}")
        num_channels = in_chans * num_maps
    print(
        "==============================================================================================="
    )
    print(f"Creating Instance of {model_name} Model\n")
    if model_name == "hybrid_transformer":
        model = hybrid_transfomer(
            num_classes=num_trojan_classes, num_heads=num_heads, num_layers=depth, latent_dim=embed_dim,
        )
        model = model.to(get_device())
    elif model_name == "Baseline":
        model = baseline_classifier(num_classes=num_trojan_classes, num_channels=num_channels)
        model = model.to(get_device())
    else:
        raise ValueError("Invalid Model Name")
    print(
        "==============================================================================================="
    )

    """ For some reason the summary function causes a weird error. This is why it is commented out here.
    """
    # print(f"Summary of {model_name} Model\n")
    # if model_name == "Baseline":
    #     print(summary(model, (sign_batch * in_chans, img_size, img_size)))
    # else:
    #     print(summary(model, (sign_batch, in_chans, img_size, img_size)))
    # print(
    #     "\n==============================================================================================="
    # )

    print(
        "==============================================================================================="
    )
    print(f"Instantiating Pytorch Lightning Model\n")

    classif_module = SignatureClassifierWrapper(model=model, learning_rate=1e-4)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    exp_name = (
        f"{dataset_name}_{model_name}_{embed_dim}_{backbone_name}_{num_heads}_{depth}_{opt_mode}"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
        dirpath=ckpt_path,
    )

    wandb_logger = pl_loggers.WandbLogger(
        project=f"TRIGS",
        name=f"{exp_name}",
        id=f"{exp_name}",
        config=args.__dict__,
        save_dir=os.path.join("wandb", exp_name),
        resume="allow",
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        detect_anomaly=False,
    )

    print(
        "==============================================================================================="
    )
    print(f"Loading Dataset\n")

    bool_val = True if model_name == "Baseline" else False
    data_module = SignatureDataModule(data_path=data_path, dataset_name=dataset_name, baseline=bool_val,
                                      opt_mode=opt_mode, split_rand_seed=split_rand_seed,
                                      train_ds_use_frac=train_ds_use_frac)

    print(
        "==============================================================================================="
    )
    print("Fitting the Classification model ...")
    last_ckpt_path = os.path.join(ckpt_path, "last.ckpt")
    print(" Last Known Checkpoint Location ... ")
    print(last_ckpt_path)

    trainer.fit(
        classif_module,
        data_module,
        ckpt_path=last_ckpt_path if os.path.exists(last_ckpt_path) else None,
    )
    print(
        "==============================================================================================="
    )
    print(f"\nLoading the best model from {checkpoint_callback.best_model_path}")

    best_classif_module = SignatureClassifierWrapper.load_from_checkpoint(
        checkpoint_path=str(os.path.join(ckpt_path, os.path.basename(checkpoint_callback.best_model_path))),
        strict=True,
        model=model,
        learning_rate=1e-4,
    )

    best_classif_module.test_metric_prerfix = "Test_"
    test_results = trainer.test(best_classif_module, data_module)
    results_dict = {k: [v] for k, v in test_results[0].items()}
    pd.DataFrame(results_dict).to_csv(os.path.join(ckpt_path, "test_results.csv"), index=False)

    pred, truth, y_score = zip(*trainer.predict(best_classif_module, data_module))

    pred, truth, y_score = np.asarray(pred), np.asarray(truth), np.asarray(y_score)
    print(pred)
    print("\n")
    print(truth)
    fpr, tpr, thresholds = roc_curve(truth, y_score)
    auc_score = auc(fpr, tpr)
    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=auc_score, estimator_name="Poisoned Model Detector"
    )
    display.plot(
        **{"color": "k", "lw": 2, "path_effects": [pe.Stroke(linewidth=5, foreground="g"), pe.Normal()]}
    )
    disp_model_name = model_name if model_name == 'Baseline' else 'Hybrid-Transformer'
    plt.title(f"{disp_model_name} Model")
    plt.show()
    plt.savefig(f"{dataset_name}_{model_name}_{opt_mode}_ep{epochs:03d}_ROC_CURVE.png")

    cm = confusion_matrix(truth, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0., 1.])
    disp.plot()
    plt.show()
    plt.title(f"{disp_model_name} Model")
    plt.savefig(f"{dataset_name}_{model_name}_{opt_mode}_ep{epochs:03d}_Confusion_matrix.png")


if __name__ == "__main__":
    main()
