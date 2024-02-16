import os
import torch
from torch import nn
from torchvision.datasets.utils import download_file_from_google_drive
from collections import OrderedDict
import resnet3d
from video_swin_transformer import SwinTransformer3D





def mvit(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    # model = torch.hub.load("facebookresearch/pytorchvideo", model='mvit_base_32x3', pretrained=pretrained)
    model = torch.hub.load("facebookresearch/pytorchvideo", model='mvit_base_32x3', pretrained=pretrained)
    feature_dims = model.head.proj.in_features

    # Adding dropout layer before the classification layer
    model.head.proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


def load_pytorchvideo_model(name, pretrained=True):
    return torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=pretrained)


class PackPathway(nn.Module):
    def __init__(self):
        self.alpha = 4
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // self.alpha
            ).long().to(frames.device),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def slowfast_r50(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    model = load_pytorchvideo_model("slowfast_r50", pretrained=pretrained)
    feature_dims = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    model = nn.Sequential(PackPathway(), model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def slowfast_r101(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    model = load_pytorchvideo_model("slowfast_r101", pretrained=pretrained)
    feature_dims = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    model = nn.Sequential(PackPathway(), model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def slow_r50(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    model = load_pytorchvideo_model("slow_r50", pretrained=pretrained)
    feature_dims = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def r2plus1d_r50(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    model = load_pytorchvideo_model("r2plus1d_r50", pretrained=pretrained)
    feature_dims = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    model.blocks[5].activation = Identity()
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def csn_r101(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    model = load_pytorchvideo_model("csn_r101", pretrained=pretrained)
    feature_dims = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def x3d(pretrained=True, num_classes=2, dropout_rate=0.5, **kwargs):
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load("facebookresearch/pytorchvideo", model='x3d_m', pretrained=pretrained)
    feature_dims = model._modules['blocks'][-1].proj.in_features
    model._modules['blocks'][-1].proj = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(feature_dims, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
    model._modules['blocks'][-1].activation = Identity()
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def swin_transformer_3d(pretrained=True):
    model = SwinTransformer3D(  # this is the Swin-T config
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),  # pretty sure
        drop_path_rate=0.1, mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.,
        patch_norm=True,
        drop_rate=0.

    )

    if pretrained:
        ckpt_path = './data/ckpt/swin_tiny_patch244_window877_kinetics400_1k.pth'

        checkpoint = torch.load(ckpt_path)
        # model.load_state_dict(checkpoint, strict=False)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    return model


def resnet_3d_50(pretrained=True, **kwargs):
    model = resnet3d.generate_model(50, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d50_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "1fCKSlakRJ54b3pEWqgBmuJi0nF7HXQc0", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"], strict=False)

        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    return model


def resnet_3d_34(pretrained=True, **kwargs):
    model = resnet3d.generate_model(34, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d34_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "12FxrQY2hX-bINbmSrN9q2Z5zJguJhy6C", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"], strict=False)

        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    return model


def resnet_3d_18(pretrained=True, **kwargs):
    model = resnet3d.generate_model(18, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d18_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "12FxrQY2hX-bINbmSrN9q2Z5zJguJhy6C", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"], strict=False)

        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    return model
