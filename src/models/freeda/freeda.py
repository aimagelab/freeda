import itertools
import os
import pickle
from math import sqrt
import re

import faiss
import numpy as np
import open_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision.transforms import Compose, Resize

from models.builder import MODELS
from models.freeda.mask_proposer.superpixel import SuperpixelExtractor
from models.freeda.pamr import PAMR
from models.freeda.masker import FreeDAMasker
import us
from datasets import get_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODELS.register_module()
class FreeDA(nn.Module):
    def __init__(
            self, clip_model, clip_weights, backbone_model_name, collection_index, k_search, k_clustering,
            last_attn_keys, collection_embeddings, use_mask_proposer, mask_proposer, ensemble_dino_clip,
            ensemble_max_mean, masker, save_mask_cut_masks, save_mask_cut_masks_dir, category_pre_filtering,
            save_category_pre_filtering, save_category_pre_filtering_dir, region_pooling, embedding_list_dir,
            save_dino_features_dir, save_dino_features, ensemble_mode="arithmetic", similarity_type="cosine",
            clustering_type="euclidean", ef_search=50000, cutler=None, mask_cut=None, superpixel=None, use_k_nn=False, k_nn=None,
            skip_clustering=False, k_parts=1, retrieval_threshold=None, visual_category_pre_filtering=None,
            visual_category_pre_filtering_threshold=None, save_clip_features=False, save_clip_features_dir=None,
            prototype_mean=None, measure_faiss_time=None, measure_faiss_time_retrieval=None, measure_distances=None,
            into_the_wild=False, coco_annotations_path=None
    ):
        """
        Args:
            clip_model (str): The name of the CLIP model to use (e.g., ViT-L-14).
            clip_weights (str): The path to the CLIP model weights (openai).
            backbone_model_name (str): The name of the backbone model to use (e.g., vit_large_patch14_dinov2.lvd142m).
            collection_index (str): The path to the collection index.
            k_search (int): The number of embeddings to retrieve for each category.
            k_clustering (int): The number of clusters (and, consequently, prototypes) to use for each category.
            last_attn_keys (bool): Whether to use the keys of the last attention layer instead of the values.
            collection_embeddings (str): The path to the collection embeddings extracted with the backbone (assert that
                the backbone is the same as the one used for the current model).
            use_mask_proposer (bool): Whether to use a mask proposer.
            mask_proposer (str): The name of the mask proposer to use (e.g., superpixel).
            cutler (dict): Parameters of the Cutler mask proposer.
            ensemble_dino_clip (float): Parameter to ensemble the backbone and CLIP on proposed regions.
            ensemble_max_mean (float): Parameter to ensemble the max and mean similarities toward a certain category.
            category_pre_filtering (bool): Whether to use category pre-filtering.
            save_category_pre_filtering (bool): Whether to save the list of pre-filtered categories.
            save_category_pre_filtering_dir (str): The path to the directory where to save/load the list of pre-filtered
                categories.
            region_pooling (bool): Whether to use region pooling on the proposed regions or directly provide them to the
                backbone model.
            embedding_list_dir (str): The path to the directory where to save/load the list of embeddings in the
                collection directory. Avoids reading the list of files in the collection directory and sorting it.
            save_dino_features_dir (str): The path to the directory where to save/load the backbone features.
                Use None to avoid saving/loading the backbone features.
            save_dino_features (bool): Whether to save the backbone features.
            ensemble_mode (str): The mode to use to ensemble the max and mean similarities. Can be either 'arithmetic'
                or 'geometric'.
            similarity_type (str): The type of similarity to use. Can be either 'cosine' or 'euclidean'.
            clustering_type (str): The type of clustering to use. Can be either 'euclidean' or 'cosine'.
            ef_search (int): Parameter for the approximated index of faiss.
            superpixel (str): Superpixel algorithm to be used.
            k_nn (int): The number of nearest neighbors to use for the kNN similarity.
        """
        super().__init__()
        self.similarity_type = similarity_type
        self.clustering_type = clustering_type
        self.use_k_nn = use_k_nn
        self.ef_search = ef_search
        self.pamr = None  # lazy init
        self.skip_clustering = skip_clustering
        self.k_parts = k_parts
        self.prototype_mean = prototype_mean
        self.measure_faiss_time = measure_faiss_time
        self.measure_faiss_time_retrieval = measure_faiss_time_retrieval
        self.measure_distances = measure_distances

        self.ensemble_mode = ensemble_mode

        assert self.ensemble_mode in ["arithmetic", "geometric"], "ensemble_mode must be either 'arithmetic' or 'geometric'"
        assert self.similarity_type in ["cosine", "euclidean"], "similarity_type must be either 'cosine' or 'euclidean'"
        if self.use_k_nn:
            assert k_nn is not None, "k_nn must be specified when using knn similarity"
        self.k_nn = k_nn
        print("Ensemble mode:", self.ensemble_mode)
        print("Similarity type:", self.similarity_type)
        print("Clustering type:", self.clustering_type)

        self.save_mask_cut_masks = save_mask_cut_masks
        self.save_mask_cut_masks_dir = save_mask_cut_masks_dir
        self.filenames = {}

        self.dino_filenames = {}
        self.save_dino_features_dir = save_dino_features_dir
        self.save_dino_features = save_dino_features

        self.use_mask_proposer = use_mask_proposer
        self.mask_proposer = mask_proposer

        if not self.use_mask_proposer or not self.mask_proposer == "mask_cut" or not self.save_mask_cut_masks:
            self.clip_model, _, self.clip_image_preprocess = open_clip.create_model_and_transforms(clip_model,
                                                                                                   pretrained=clip_weights)
            self.clip_resize_dim = self.clip_image_preprocess.transforms[0].size
            self.clip_image_preprocess = Compose([
                Resize(
                    (self.clip_image_preprocess.transforms[0].size, self.clip_image_preprocess.transforms[0].size),
                    interpolation=self.clip_image_preprocess.transforms[0].interpolation, antialias=None),
                lambda x: x / 255,
                self.clip_image_preprocess.transforms[4]
            ])

            self.clip_model = self.clip_model.to(device)

            self.clip_model = self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer(clip_model)

        if (self.save_dino_features and self.save_dino_features_dir is not None) or self.save_dino_features_dir is None:
            img_size = 518 if "patch14" in backbone_model_name else 592

            self.backbone_model = timm.create_model(
                backbone_model_name,
                pretrained=True,
                img_size=img_size,
            ).to(device if not self.save_mask_cut_masks else "cpu")
            self.backbone_model = self.backbone_model.eval()

            data_config = timm.data.resolve_model_data_config(self.backbone_model)
            self.backbone_transforms = timm.data.create_transform(**data_config, is_training=False)
            if backbone_model_name == "vit_base_patch14_dinov2.lvd142m":
                self.backbone_transforms = Compose([
                    Resize((518, 518), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2 + 1), "Wrong number of positional embeddings"
            elif backbone_model_name == "vit_base_patch16_224.dino":
                self.backbone_transforms = Compose([
                    Resize((592, 592), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2 + 1), "Wrong number of positional embeddings"
            elif backbone_model_name == "vit_large_patch14_clip_224.openai":
                self.backbone_transforms = Compose([
                    Resize((518, 518), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2 + 1), "Wrong number of positional embeddings"
            elif backbone_model_name == "vit_base_patch16_clip_224.openai":
                self.backbone_transforms = Compose([
                    Resize((592, 592), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2 + 1), "Wrong number of positional embeddings"
            elif backbone_model_name == "vit_large_patch16_224.mae":
                self.backbone_transforms = Compose([
                    Resize((592, 592), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2 + 1), "Wrong number of positional embeddings"
            elif backbone_model_name == "deit3_large_patch16_224.fb_in1k":
                self.backbone_transforms = Compose([
                    Resize((592, 592), antialias=None),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
                assert self.backbone_model.pos_embed.size(1) == (37 ** 2), "Wrong number of positional embeddings"  # <== Doesn't use the cls token
            elif backbone_model_name == "vit_base_patch8_224.dino":
                self.backbone_transforms = Compose([
                    Resize(size=(224, 224)),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
            else:
                # resize_dim = self.backbone_transforms.transforms[0].size
                self.backbone_transforms = Compose([
                    Resize(
                        (self.backbone_transforms.transforms[0].size, self.backbone_transforms.transforms[0].size)) if type(
                        self.backbone_transforms.transforms[0].size) == int else Resize(
                        self.backbone_transforms.transforms[0].size),
                    lambda x: x / 255,
                    self.backbone_transforms.transforms[-1]
                ])
            self.backbone_resize_dim = self.backbone_transforms.transforms[0].size[0]

        self.last_attn_keys = last_attn_keys

        if last_attn_keys:
            self.model_output_feats = {}
            self.backbone_model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                self.last_attn_keys_hook)

        # self.patch_size = self.clip_model.visual.patch_size

        self.collection_index = faiss.read_index(collection_index)
        self.k_search = k_search
        self.k_clustering = k_clustering
        self.collection_embeddings_path = collection_embeddings
        if embedding_list_dir is not None and os.path.exists(embedding_list_dir):
            with open(embedding_list_dir, "rb") as f:
                self.collection_embeddings = pickle.load(f)
        elif embedding_list_dir is not None and not os.path.exists(embedding_list_dir):
            self.collection_embeddings = sorted(os.listdir(self.collection_embeddings_path))
            with open(embedding_list_dir, "wb") as f:
                pickle.dump(self.collection_embeddings, f)
        else:
            self.collection_embeddings = sorted(os.listdir(self.collection_embeddings_path))

        self.masker = FreeDAMasker(ensemble_max_mean, similarity_type)
        self.masker = self.masker.eval()

        self.ensemble_dino_clip = ensemble_dino_clip
        self.ensemble_max_mean = ensemble_max_mean

        if self.use_mask_proposer and self.mask_proposer == "superpixel":
            self.mask_proposer_algorithm = SuperpixelExtractor(superpixel)

        self.save_category_pre_filtering = save_category_pre_filtering
        self.save_category_pre_filtering_dir = save_category_pre_filtering_dir
        self.use_category_pre_filtering = category_pre_filtering
        self.category_pre_filtering_filenames = {}
        self.region_pooling = region_pooling

        self.retrieval_threshold = retrieval_threshold
        self.visual_category_pre_filtering = visual_category_pre_filtering
        self.visual_category_pre_filtering_threshold = visual_category_pre_filtering_threshold

        self.save_clip_features = save_clip_features
        self.save_clip_features_dir = save_clip_features_dir
        if self.save_clip_features_dir is not None:
            os.makedirs(self.save_clip_features_dir, exist_ok=True)
        self.clip_features_filenames = {}

    def last_attn_keys_hook(self, module, input, output):
        B, P, D = output.shape
        output = output.reshape(B, P, 3, -1)
        self.model_output_feats['keys'] = output[:, :, 1, :]

    def forward(self, image, text):
        pass

    @torch.no_grad()
    def build_dataset_class_tokens(self, template_set, classnames):
        tokens = []
        templates = get_template(template_set)
        for classname in classnames:
            tokens.append(
                self.clip_tokenizer([template.format(classname) for template in templates])
            )
        # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
        tokens = torch.stack(tokens)

        return tokens

    @torch.no_grad()
    def build_proto_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            proto_embs
        """
        text = text.to(device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 1024
        N = text.size(0)
        text_embs = torch.cat([
            self.clip_model.encode_text(text[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1)
        text_embs = us.normalize(text_embs, dim=-1)

        # TODO: retrieve from faiss index using text_embs

        print("Faiss class type: ", str(type(self.collection_index)))
        if str(type(self.collection_index)) == "<class 'faiss.swigfaiss.IndexHNSWFlat'>" \
                or str(type(self.collection_index)) == "<class 'faiss.swigfaiss_avx2.IndexHNSWFlat'>":
            print(f"Setting faiss efSearch to {self.ef_search}")
            faiss.ParameterSpace().set_index_parameter(self.collection_index, 'efSearch', self.ef_search)

        similarities, indices = self.collection_index.search(text_embs.cpu().numpy(), self.k_search)
        proto_embeds = []
        printed = False
        min_len = self.k_search
        for i, category_index in enumerate(indices):

            print(f"Processing class {i + 1} / {len(indices)}")

            category_proto_embeds = []
            for j, retrieval_index in enumerate(category_index):
                if self.retrieval_threshold is not None and similarities[i][j].item() < self.retrieval_threshold and j != 0:
                    break
                if retrieval_index == -1:
                    print(f"Warning: Found an index -1.")
                    break
                category_proto_embeds.append(
                    np.load(os.path.join(self.collection_embeddings_path, self.collection_embeddings[retrieval_index])))
            for index, elem in reversed(list(enumerate(category_proto_embeds))):
                if not np.isfinite(elem).all():
                    print(f"Warning: Found an embedding that is not finite.")
                    category_proto_embeds.pop(index)
            if self.k_parts == 1:
                category_proto_embeds = np.stack(category_proto_embeds)
            else:
                category_proto_embeds = np.concatenate(category_proto_embeds, axis=0)
            if self.retrieval_threshold is not None:
                print("Number of retrieved images after thresholding: ", len(category_proto_embeds))
            if self.skip_clustering:
                if len(category_proto_embeds) < min_len:
                    min_len = len(category_proto_embeds)
                proto_embeds.append(category_proto_embeds)
            elif self.prototype_mean is not None and self.prototype_mean:
                proto_embeds.append(np.expand_dims(category_proto_embeds.mean(axis=0), 0))
            else:
                kmeans = faiss.Kmeans(
                    d=category_proto_embeds.shape[1],
                    k=self.k_clustering,
                    niter=20,
                    verbose=False,
                    nredo=10,
                    gpu=False,
                    seed=42,
                    spherical=self.clustering_type == "cosine",
                )
                if not printed:
                    print("Kmean parameters: ")
                    print(f"- k: {self.k_clustering}")
                    print(f"- seed: {kmeans.cp.seed}")
                    print(f"- niter: {kmeans.cp.niter}")
                    print(f"- nredo: {kmeans.cp.nredo}")
                    print(f"- spherical: {kmeans.cp.spherical}")
                    print(f"- min_points_per_centroid: {kmeans.cp.min_points_per_centroid}")
                    print(f"- max_points_per_centroid: {kmeans.cp.max_points_per_centroid}")
                    print(f"- decode_block_size: {kmeans.cp.decode_block_size}")
                    printed = True

                # apply L2 norm to the embeddings
                if self.clustering_type == "cosine":
                    category_proto_embeds = category_proto_embeds / np.linalg.norm(category_proto_embeds, axis=-1,
                                                                                   keepdims=True)
                kmeans.train(category_proto_embeds.astype(np.float32))
                proto_embeds.append(kmeans.centroids)
        if self.skip_clustering:
            for i in range(len(proto_embeds)):
                proto_embeds[i] = proto_embeds[i][:min_len]
        proto_embeds = torch.from_numpy(np.stack(proto_embeds)).to(device)

        return text_embs, proto_embeds

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        if self.pamr is None:
            pamr_iter = 10
            pamr_kernel = [1, 2, 4, 8, 12, 24]
            self.pamr = PAMR(pamr_iter, pamr_kernel)
            self.pamr.eval()
            self.pamr.to(next(self.parameters()).device)

        mask = self.pamr(image, mask)
        return mask

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    @torch.no_grad()
    def masked_crop(self, images, masks, num_masks, resize_dim):

        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)

        sum_y = torch.sum(masks, dim=2)
        cumsum_x = torch.cumsum(sum_y, dim=2).float()
        xmaxs = torch.argmax(cumsum_x, dim=2, keepdim=True)
        cumsum_x[cumsum_x == 0] = np.inf
        xmins = torch.argmin(cumsum_x, dim=2, keepdim=True)
        sum_x = torch.sum(masks, dim=3)
        cumsum_y = torch.cumsum(sum_x, dim=2).float()
        ymaxs = torch.argmax(cumsum_y, dim=2, keepdim=True)
        cumsum_y[cumsum_y == 0] = np.inf
        ymins = torch.argmin(cumsum_y, dim=2, keepdim=True)
        del cumsum_x, cumsum_y, sum_x, sum_y

        ymaxs = ymaxs.reshape(-1, 1) + 1
        ymins = ymins.reshape(-1, 1)
        xmaxs = xmaxs.reshape(-1, 1) + 1
        xmins = xmins.reshape(-1, 1)

        # batch_index = torch.arange(batch_size).unsqueeze(1).repeat(1, num_masks).reshape(-1, 1).to(image_segmentations.device)
        indexes = torch.cat([torch.ones(elem) * i for i, elem in enumerate(num_masks)]).unsqueeze(1).to(device)
        mask_indexes = torch.arange(masks.shape[0]).unsqueeze(1).to(device)
        boxes = torch.cat((indexes, xmins, ymins, xmaxs, ymaxs), 1)
        mask_boxes = torch.cat((mask_indexes, xmins, ymins, xmaxs, ymaxs), 1)
        del xmins, ymins, xmaxs, ymaxs, indexes

        box_masked_images = torchvision.ops.roi_align(images.float(), boxes.float(), resize_dim, aligned=True)
        box_binary_masks = torchvision.ops.roi_align(masks.float(), mask_boxes.float(), resize_dim, aligned=True)
        box_binary_masks = torch.clip(torch.round(box_binary_masks), 0, 1).bool()
        box_masked_images = (box_masked_images * box_binary_masks.float()).int()

        return box_masked_images

    @torch.no_grad()
    def replace_covered_pixel_similarities(self, mask, new_similarities, covered_pixels):
        tmp_mask = mask.permute(0, 2, 3, 1)
        tmp_new_similarities = new_similarities.permute(0, 2, 3, 1)
        tmp_mask[covered_pixels] = tmp_new_similarities[covered_pixels]
        return tmp_mask.permute(0, 3, 1, 2)

    @torch.no_grad()
    def get_clip_similarities(self, image, pred_masks_batch, num_pred_masks, covered_pixels_batch, assigned_masks_batch,
                              text_emb, batch_size, pH, pW, num_classes):
        interpolated_pred_masks_batch = F.interpolate(pred_masks_batch.unsqueeze(1).float(),
                                                      (image.shape[2], image.shape[3]),
                                                      mode='nearest').bool().squeeze(1)
        box_masked_images = self.masked_crop(image, interpolated_pred_masks_batch, num_pred_masks,
                                             (self.clip_resize_dim, self.clip_resize_dim))
        inputs = torch.cat([self.clip_image_preprocess(image), self.clip_image_preprocess(box_masked_images)], dim=0)
        clip_output = self.clip_model.encode_image(inputs)
        clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True)
        encoded_whole_images = clip_output[:batch_size]
        clip_output = clip_output[batch_size:]
        clip_similarities = clip_output @ text_emb.T
        dense_clip_similarities = torch.ones((batch_size, num_classes, pH, pW)).to(device) * -1e5
        for i in range(batch_size):
            dense_clip_similarities[i][:, covered_pixels_batch[i]] = clip_similarities[assigned_masks_batch[i]][
                covered_pixels_batch[i]].permute(1, 0)
        return encoded_whole_images, torch.sigmoid(dense_clip_similarities)

    @torch.no_grad()
    def get_backbone_region_similarities(self, image, dense_features, num_tokens_side, pred_masks_batch, covered_pixels_batch,
                                         assigned_masks_batch, proto_emb, num_pred_masks, batch_size, pH, pW,
                                         num_classes):

        if not self.region_pooling:
            interpolated_pred_masks_batch = F.interpolate(pred_masks_batch.unsqueeze(1).float(), (image.shape[2], image.shape[3]),
                                                          mode='nearest').bool().squeeze(1)
            box_masked_images = self.masked_crop(image, interpolated_pred_masks_batch, num_pred_masks,
                                                 (self.backbone_resize_dim, self.backbone_resize_dim))
            count = 0
            num_iter = int(box_masked_images.shape[0] / 8) if box_masked_images.shape[0] % 8 == 0 else int(box_masked_images.shape[0] / 8) + 1
            mask_features_batch = []
            for i in range(num_iter):
                num_images = 8 if i != num_iter - 1 else box_masked_images.shape[0] % 8
                mask_features_batch.append(self.backbone_model(self.backbone_transforms(box_masked_images[count:count + num_images])))
                count += num_images
            mask_features_batch = torch.cat(mask_features_batch, dim=0)
            mask_features_batch = mask_features_batch / mask_features_batch.norm(dim=-1, keepdim=True)

        output_similarities = torch.ones((batch_size, num_classes, pH, pW)).to(device) * -1e5
        count = 0
        if self.similarity_type == 'cosine':
            norm_proto_emb = proto_emb / proto_emb.norm(dim=-1, keepdim=True)
        for i, num_pred_masks_per_image in enumerate(num_pred_masks):
            if self.region_pooling:
                pred_masks = pred_masks_batch[count:count + num_pred_masks_per_image]
                pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), (num_tokens_side, num_tokens_side),
                                           mode='bilinear', align_corners=True)
                pred_masks = pred_masks.squeeze(1)
                mask_features = pred_masks.unsqueeze(1) * dense_features[i].unsqueeze(0)
                mask_features = mask_features.sum(dim=(2, 3)) / (pred_masks.sum(dim=(1, 2)).unsqueeze(1) +
                                                                 torch.finfo(torch.float32).eps)
                # pred_masks_batch[mask_features.isnan().sum(-1).bool()] = torch.zeros(pred_masks_batch.shape[-2],
                # pred_masks_batch.shape[-1])
            else:
                mask_features = mask_features_batch[count:count + num_pred_masks_per_image]
            if self.similarity_type == 'cosine':
                mask_features = mask_features / mask_features.norm(dim=-1, keepdim=True)
                covered_pixels_batch[i, pred_masks_batch[count:count + num_pred_masks_per_image][mask_features.isnan().sum(-1).bool()].sum(0).bool()] = False
                mask_similarities = torch.einsum("bc,nkc->bnk", mask_features, norm_proto_emb)
            elif self.similarity_type == 'euclidean':
                b, c = mask_features.shape
                n, k, c = proto_emb.shape
                mask_similarities = torch.zeros((b, n, k)).to(device)
                for j in range(mask_features.shape[0]):
                    mask_similarities[j] = 1 / (1 + torch.cdist(proto_emb, mask_features[j].unsqueeze(0), p=2))[:, :, 0]

            if not self.use_k_nn:
                mask_similarities_max, _ = torch.max(mask_similarities, dim=-1)
                mask_similarities_mean = torch.mean(mask_similarities, dim=-1)

                if self.ensemble_mode == "arithmetic":
                    # use arithmetic mean
                    mask_similarities = self.ensemble_max_mean * mask_similarities_max + (1 - self.ensemble_max_mean) * mask_similarities_mean
                elif self.ensemble_mode == "geometric":
                    # use geometric mean
                    mask_similarities = torch.pow(mask_similarities_max, self.ensemble_max_mean) * torch.pow(mask_similarities_mean, (1 - self.ensemble_max_mean))
                else:
                    raise NotImplementedError(f"Ensemble mode '{self.ensemble_mode}' not implemented")
            else:
                b, n, k = mask_similarities.shape
                values, indices = torch.topk(mask_similarities.reshape(b, n * k), k=self.k_nn, dim=-1)
                labels = torch.floor(indices / k).int()
                values_one_hot = torch.zeros(b, n, self.k_nn).to(labels.device).scatter_(1, labels.unsqueeze(1).long(),
                                                                                         values.unsqueeze(1))
                mask_similarities = values_one_hot.sum(dim=-1)

            output_similarities[i][:, covered_pixels_batch[i]] = mask_similarities[assigned_masks_batch[i]][
                covered_pixels_batch[i]].permute(1, 0)
            count += num_pred_masks_per_image
        return torch.sigmoid(output_similarities)

    def multi_classname_prompt(self, combination):
        return " and ".join([f"{classname}" for classname in combination])

    @torch.no_grad()
    def category_pre_filtering(self, image_clip_embedding, text_emb, classnames):
        clip_similarities = image_clip_embedding @ text_emb.T
        # mean_similarities = clip_similarities.mean(dim=-1)
        # return clip_similarities > mean_similarities
        k = 10
        top_k = torch.topk(clip_similarities, k=k, dim=-1).indices.cpu().numpy().tolist()
        output = torch.zeros((clip_similarities.shape[0], clip_similarities.shape[1])).bool().to(device)
        for i in range(clip_similarities.shape[0]):
            combinations = []
            for j in range(1, k + 1):
                combinations.extend(list(itertools.combinations(top_k[i], j)))
            combinations = [list(combination) for combination in combinations]
            classnames_combinations = [[classnames[index] for index in combination] for combination in combinations]
            multi_class_labels = [self.multi_classname_prompt(combination) for combination in classnames_combinations]
            multi_class_labels = self.clip_tokenizer(multi_class_labels).to(device)
            chunk_size = 512
            N = multi_class_labels.size(0)
            multi_class_embs = torch.cat([
                self.clip_model.encode_text(multi_class_labels[n:n + chunk_size])
                for n in range(0, N, chunk_size)
            ])
            multi_class_embs = multi_class_embs / multi_class_embs.norm(dim=-1, keepdim=True)
            multi_class_similarities = image_clip_embedding[i] @ multi_class_embs.T
            max_similarity_index = torch.argmax(multi_class_similarities)
            for index in combinations[max_similarity_index]:
                output[i][index] = True
        return output

    @torch.no_grad()
    def store_category_pre_filtering(self, filenames, category_masks):
        for i, (filename, category_mask) in enumerate(zip(filenames, category_masks)):
            if filename not in self.category_pre_filtering_filenames:
                self.category_pre_filtering_filenames[filename] = 0
            else:
                self.category_pre_filtering_filenames[filename] += 1
            np.save(os.path.join(self.save_category_pre_filtering_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.category_pre_filtering_filenames[filename])), category_mask.cpu().numpy())

    @torch.no_grad()
    def load_category_pre_filtering(self, filenames):
        category_masks = []
        for filename in filenames:
            if filename not in self.category_pre_filtering_filenames:
                self.category_pre_filtering_filenames[filename] = 0
            else:
                self.category_pre_filtering_filenames[filename] += 1
            category_mask = torch.from_numpy(np.load(os.path.join(self.save_category_pre_filtering_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.category_pre_filtering_filenames[filename]))))
            category_masks.append(category_mask)
        category_masks = torch.stack(category_masks).to(device)
        return category_masks

    @torch.no_grad()
    def store_clip_features(self, filenames, clip_features):
        for i, (filename, current_clip_features) in enumerate(zip(filenames, clip_features)):
            if filename not in self.clip_features_filenames:
                self.clip_features_filenames[filename] = 0
            else:
                self.clip_features_filenames[filename] += 1
            np.save(os.path.join(self.save_clip_features_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.clip_features_filenames[filename])), current_clip_features.cpu().numpy())

    @torch.no_grad()
    def load_clip_features(self, filenames):
        clip_features = []
        for filename in filenames:
            if filename not in self.clip_features_filenames:
                self.clip_features_filenames[filename] = 0
            else:
                self.clip_features_filenames[filename] += 1
            current_clip_features = torch.from_numpy(np.load(os.path.join(self.save_clip_features_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.clip_features_filenames[filename]))))
            clip_features.append(current_clip_features)
        clip_features = torch.stack(clip_features).to(device)
        return clip_features

    def save_extracted_dino_features(self, image_feat, filenames):
        for filename in filenames:
            if filename not in self.dino_filenames:
                self.dino_filenames[filename] = 0
            else:
                self.dino_filenames[filename] += 1
            np.save(os.path.join(self.save_dino_features_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.dino_filenames[filename])), image_feat.cpu().numpy())

    def load_extracted_dino_features(self, filenames):
        image_feat_list = []
        for filename in filenames:
            if filename not in self.dino_filenames:
                self.dino_filenames[filename] = 0
            else:
                self.dino_filenames[filename] += 1
            image_feat_list.append(np.load(os.path.join(self.save_dino_features_dir, filename.replace(".jpg", '').replace(".png", '').replace("/", "_") + '_{}.npy'.format(self.dino_filenames[filename]))))
        image_feat_list = torch.from_numpy(np.concatenate(image_feat_list)).to(device)
        return image_feat_list

    @torch.no_grad()
    def generate_masks(
            self, image, img_metas, text_emb, proto_emb, classnames, text_is_token=False, apply_pamr=False,
            # kp_w=0.3,
    ):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]
            proto_emb [N, K, C]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """

        H, W = image.shape[2:]  # original image shape

        # padded image size
        pH, pW = image.shape[2:]
        num_classes = proto_emb.shape[0]
        batch_size = image.shape[0]

        image = image[:, [2, 1, 0], :, :]  # BGR to RGB
        image_clip_embedding = None

        ############### Generate mask ################
        # soft mask
        if self.use_mask_proposer:
            if self.mask_proposer == "superpixel":
                pred_masks_batch, num_pred_masks, covered_pixels_batch, assigned_masks_batch = \
                    self.mask_proposer_algorithm(image)
                pred_masks_batch = pred_masks_batch.to(image.device)
                covered_pixels_batch = covered_pixels_batch.to(image.device)
                assigned_masks_batch = assigned_masks_batch.to(image.device)
            else:
                raise NotImplementedError
            zero_masks = pred_masks_batch.shape[0] == 0

        if (self.use_category_pre_filtering and image_clip_embedding is None) or self.ensemble_dino_clip != 1.0:
            if self.save_clip_features and self.save_clip_features_dir is not None:
                image_clip_embedding = self.clip_model.encode_image(self.clip_image_preprocess(image))
                image_clip_embedding = image_clip_embedding / image_clip_embedding.norm(dim=-1, keepdim=True)
                filenames = [img_metas[i]['ori_filename'] for i in range(len(img_metas))]
                self.store_clip_features(filenames, image_clip_embedding)
            elif not self.save_clip_features and self.save_clip_features_dir is not None:
                filenames = [img_metas[i]['ori_filename'] for i in range(len(img_metas))]
                image_clip_embedding = self.load_clip_features(filenames)
            else:
                image_clip_embedding = self.clip_model.encode_image(self.clip_image_preprocess(image))
                image_clip_embedding = image_clip_embedding / image_clip_embedding.norm(dim=-1, keepdim=True)
            if self.ensemble_dino_clip != 1.0:
                clip_similarities = image_clip_embedding @ text_emb.T

        if image_clip_embedding is not None and self.use_category_pre_filtering:
            if self.save_category_pre_filtering:
                category_mask = self.category_pre_filtering(image_clip_embedding, text_emb, classnames)
                filenames = [img_metas[i]['ori_filename'] for i in range(len(img_metas))]
                self.store_category_pre_filtering(filenames, category_mask)
            else:
                filenames = [img_metas[i]['ori_filename'] for i in range(len(img_metas))]
                category_mask = self.load_category_pre_filtering(filenames)

        ori_image = image.clone()
        filenames = [img_metas[i]['ori_filename'] for i in range(len(img_metas))]
        if (self.save_dino_features and self.save_dino_features_dir is not None) or self.save_dino_features_dir is None:
            image = self.backbone_transforms(image)
            image_feat = self.backbone_model.forward_features(image)
        if self.save_dino_features and self.save_dino_features_dir is not None:
            self.save_extracted_dino_features(image_feat, filenames)
        elif not self.save_dino_features and self.save_dino_features_dir is not None:
            image_feat = self.load_extracted_dino_features(filenames)
        if self.last_attn_keys:
            image_feat = self.model_output_feats["keys"]
        num_tokens_side = int(sqrt(image_feat.shape[1] - 1))
        image_feat = image_feat[:, 1::, :].reshape(image_feat.shape[0], int(num_tokens_side), int(num_tokens_side),
                                                   image_feat.shape[-1])
        image_feat = image_feat.permute(0, 3, 1, 2)
        mask, simmap = self.masker.forward_seg(image_feat, proto_emb, hard=False, use_k_nn=self.use_k_nn,
                                               k_nn=self.k_nn)  # [B, N, H', W']

        # resize
        mask = F.interpolate(mask, (pH, pW), mode='bilinear', align_corners=True)  # [B, N, H, W]
        if self.use_mask_proposer and not zero_masks:
            ensembled_similarities = self.get_backbone_region_similarities(ori_image, image_feat, num_tokens_side,
                                                                           pred_masks_batch,
                                                                           covered_pixels_batch,
                                                                           assigned_masks_batch, proto_emb,
                                                                           num_pred_masks, batch_size, pH, pW,
                                                                           num_classes)

            mask = self.replace_covered_pixel_similarities(mask, ensembled_similarities, covered_pixels_batch)

        dino_masks = mask.clone()

        if self.ensemble_dino_clip != 1.0:
            mask = self.ensemble_dino_clip * mask + (1 - self.ensemble_dino_clip) * clip_similarities \
                .reshape(1, num_classes, 1, 1).repeat(1, 1, pH, pW)

        if self.visual_category_pre_filtering is not None:
            visual_category_mask = self.get_visual_category_pre_filtering(image_feat, proto_emb)
            mask[~visual_category_mask] = -1e5

        if self.use_category_pre_filtering:
            mask[~category_mask] = -1e5

        if apply_pamr:
            for c in range(0, mask.shape[1], 30):
                mask[:, c:c + 30] = self.apply_pamr(ori_image, mask[:, c:c + 30])

        assert mask.shape[2] == H and mask.shape[3] == W, f"shape mismatch: ({H}, {W}) / {mask.shape}"

        return mask, simmap, dino_masks
