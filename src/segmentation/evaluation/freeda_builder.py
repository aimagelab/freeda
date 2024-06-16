# ------------------------------------------------------------------------------
# FreeDA
# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
import mmcv
import torch
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from datasets import get_template

from .freeda_seg import FreeDASegInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_freeda_seg_inference(
    model,
    dataset,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.dataset.CLASSES[1:]
    else:
        classnames = dataset.dataset.CLASSES
    if model.use_mask_proposer and model.mask_proposer == "mask_cut" and model.save_mask_cut_masks:
        text_embedding = torch.zeros((len(classnames), 768)).to(device)
        proto_embedding = torch.zeros((len(classnames), model.k_clustering, 1024)).to(device)
    else:
        text_tokens = model.build_dataset_class_tokens(config.evaluate.template, classnames)
        text_embedding, proto_embedding = model.build_proto_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "FreeDA":
        seg_model = FreeDASegInference(model, text_embedding, proto_embedding, classnames, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.dataset.CLASSES
    seg_model.PALETTE = dataset.dataset.PALETTE

    return seg_model
