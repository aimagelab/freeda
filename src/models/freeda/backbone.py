import timm
from torchvision.transforms import Compose, Resize

timm_backbones = [
    # MAE
    'vit_base_patch16_224.mae',
    'vit_large_patch16_224.mae',
    # DINO
    'vit_small_patch8_224.dino',
    'vit_base_patch8_224.dino',
    'vit_small_patch16_224.dino',
    'vit_base_patch16_224.dino',
    # DINOv2
    'vit_small_patch14_dinov2.lvd142m'
    'vit_base_patch14_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m']

moco_backbones = [
    'moco-vit-s',
    'moco-vit-b']


class BackboneManager:
    def __init__(self, backbone_model_name):
        self.backbone_model_name = backbone_model_name

    def get_backbone_and_transforms(self):
        if self.backbone_model_name == "clip":
            raise NotImplementedError
        elif self.backbone_model_name in timm_backbones:
            backbone_model, backbone_transforms = self.get_timm_backbone()
        elif self.backbone_model_name in moco_backbones:
            raise NotImplementedError
        else:
            raise NotImplementedError
        backbone_model = backbone_model.eval()
        return backbone_model, backbone_transforms

    def get_timm_backbone(self):
        backbone_model = timm.create_model(
            self.backbone_model_name,
            pretrained=True,
        )
        data_config = timm.data.resolve_model_data_config(backbone_model)
        backbone_transforms = timm.data.create_transform(**data_config, is_training=False)
        if self.backbone_model_name == "vit_large_patch16_224.mae":
            # resize_dim = self.backbone_transforms.transforms[1].size[0]
            backbone_transforms = Compose([
                Resize(backbone_transforms.transforms[1].size),
                lambda x: x / 255,
                backbone_transforms.transforms[-1]
            ])
        elif self.backbone_model_name == "vit_base_patch8_224.dino":
            backbone_transforms = Compose([
                Resize(size=(224, 224)),
                lambda x: x / 255,
                backbone_transforms.transforms[-1]
            ])
        else:
            # resize_dim = backbone_transforms.transforms[0].size
            backbone_transforms = Compose([
                Resize(
                    (backbone_transforms.transforms[0].size, backbone_transforms.transforms[0].size)) if type(
                    backbone_transforms.transforms[0].size) == int else Resize(
                    backbone_transforms.transforms[0].size),
                lambda x: x / 255,
                backbone_transforms.transforms[-1]
            ])
        return backbone_model, backbone_transforms
