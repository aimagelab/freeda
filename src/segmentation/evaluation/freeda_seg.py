import mmcv
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder
from utils import get_logger


class FreeDASegInference(EncoderDecoder):
    def __init__(
            self,
            model,
            text_embedding,
            proto_embedding,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            # kp_w=0.3,
            **kwargs,
    ):
        super(EncoderDecoder, self).__init__()  # init BaseSegmenter (parent of EncoderDecoder)

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy
        # self.kp_w = kp_w

        self.model = model
        self.register_buffer("proto_embedding", proto_embedding)
        self.register_buffer("text_embedding", text_embedding)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(proto_embedding) + 1
        else:
            self.num_classes = len(proto_embedding)

        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building FreeDASegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}"
        )

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        """
        assert img.shape[0] == 1, "batch size must be 1"

        # masks [B, N, H, W]
        # simmap [B, N, H//4, W//4]
        # soft mask (logit-like) is required
        masks, simmap, dino_masks = self.model.generate_masks(
            img,
            img_metas,
            self.text_embedding,
            self.proto_embedding,
            self.classnames,
            apply_pamr=self.pamr,
            # kp_w=self.kp_w,
        )

        B, N, H, W = masks.shape

        if self.with_bg:

            dino_masks = dino_masks.cpu()
            masks = masks.cpu()

            if self.bg_strategy == "only_dino":
                max_sim = dino_masks.max(dim=1, keepdim=True)[0]
                background = torch.zeros([B, 1, H, W])
                background[max_sim < self.bg_thresh] = 1
            elif self.bg_strategy == "dino_after_ensemble":
                max_indices = masks.max(dim=1, keepdim=True)[1] # [B, 1, H, W]
                indices_masks = (torch.ones(B, N, H, W) * torch.arange(N).reshape(1, N, 1, 1).repeat(B, 1, 1, 1)) == max_indices # [B, C, H, W]
                max_sim = (dino_masks * indices_masks).sum(1).unsqueeze(1)
                background = torch.zeros([B, 1, H, W])
                background[max_sim < self.bg_thresh] = 1
            else:
                background = torch.full(
                    [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
                )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        return masks
