import cv2
import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

if __name__ == '__main__':
    print("Running superpixel.py as main file")
    from superpixel_configs import felzenszwalb_parameters_dict, slic_parameters_dict, quickshift_parameters_dict, watershed_parameters_dict, seeds_parameters_dict
else:
    from .superpixel_configs import felzenszwalb_parameters_dict, slic_parameters_dict, quickshift_parameters_dict, watershed_parameters_dict, seeds_parameters_dict


class SuperpixelExtractor:
    """Proposes class-agnostic masks for the given images using superpixels.

    Args:
        parameters_dict (dict): The parameters for the superpixel algorithm. Contains also which algorithm to use.
    """

    def __init__(self, algorithm):
        print("Parameters dict at beginning of init", algorithm)

        if isinstance(algorithm, str):

            self.algorithm = algorithm
            if algorithm == "felzenszwalb":
                parameters_dict = felzenszwalb_parameters_dict
            elif algorithm == "slic":
                parameters_dict = slic_parameters_dict
            elif algorithm == "quickshift":
                parameters_dict = quickshift_parameters_dict
            elif algorithm == "watershed":
                parameters_dict = watershed_parameters_dict
            elif algorithm == "seeds":
                parameters_dict = seeds_parameters_dict
                self.num_iterations = parameters_dict.pop("num_iterations")
            else:
                raise NotImplementedError(f"Superpixel algorithm {algorithm} not implemented")
        elif isinstance(algorithm, dict):

            self.algorithm = algorithm.pop("algorithm")
            if self.algorithm == "seeds":
                self.num_iterations = algorithm.pop("num_iterations")
            parameters_dict = algorithm
        else:
            raise TypeError(f"Algorithm must be either a string or a dictionary, but is {type(algorithm)}")

        self.parameters_dict = parameters_dict

    def __call__(self, images):
        """
        Args:
            images (torch.Tensor): [B, C, H, W]
        Output:
            pred_masks_batch: torch.Tensor [NUM_MASKS, IMAGE_HEIGHT, IMAGE_WIDTH]
                Contains the list of binary masks for each image in the batch. NUM_MASKS is the number of total proposed
                masks for all images.
            n_pred_masks: List [BATCH_SIZE]
                Number of proposed masks for each image in the batch.
            covered_pixels_batch: torch.Tensor [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH]
                Indicates for each pixel whether it is covered by a mask.
            assigned_masks_batch: torch.Tensor [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH]
                Indicates for each pixel which mask it is assigned to.
        """
        pred_masks_batch = []
        n_pred_masks = []
        assigned_masks_batch = []
        covered_pixels_batch = torch.ones(images.shape[0], images.shape[2], images.shape[3]).type(torch.bool)

        for img in images:
            if self.algorithm == "seeds":
                img = img.permute(1, 2, 0).cpu().numpy()
                img = np.ascontiguousarray(img.astype(np.uint8))
            else:
                img = img.permute(1, 2, 0).cpu().numpy() / 255

            if self.algorithm == "felzenszwalb":
                superpixel_mask = felzenszwalb(img, **self.parameters_dict)

            elif self.algorithm == "slic":
                superpixel_mask = slic(img, **self.parameters_dict)

            elif self.algorithm == "quickshift":
                superpixel_mask = quickshift(img, **self.parameters_dict)

            elif self.algorithm == "watershed":
                gradient = sobel(rgb2gray(img))
                superpixel_mask = watershed(gradient, **self.parameters_dict)

            elif self.algorithm == "seeds":
                superpix_seeds = cv2.ximgproc.createSuperpixelSEEDS(**self.parameters_dict)
                superpix_seeds.iterate(img, self.num_iterations)
                superpixel_mask = superpix_seeds.getLabels()
                num_superpixels = superpix_seeds.getNumberOfSuperpixels()
            else:
                raise NotImplementedError(f"Superpixel algorithm {self.algorithm} not implemented.")

            # create a binary mask for each superpixel
            if self.algorithm == "seeds":
                superpixel_mask_binary = np.array([superpixel_mask == i for i in np.arange(num_superpixels)])
            else:
                superpixel_mask_binary = np.array([superpixel_mask == i for i in np.unique(superpixel_mask)])
            num_superpixel = superpixel_mask_binary.shape[0]

            pred_masks_batch.append(superpixel_mask_binary)
            n_pred_masks.append(num_superpixel)
            assigned_masks_batch.append(superpixel_mask[None, :, :])

        pred_masks_batch = torch.Tensor(np.concatenate(pred_masks_batch, axis=0)).type(torch.bool)
        # n_pred_masks = torch.Tensor(n_pred_masks).type(torch.long)
        assigned_masks_batch = torch.Tensor(np.concatenate(assigned_masks_batch, axis=0)).type(torch.long)

        if self.algorithm == "watershed":
            assigned_masks_batch = assigned_masks_batch - 1

        return pred_masks_batch, n_pred_masks, covered_pixels_batch, assigned_masks_batch


class SEEDSSuperpixelExtractor:

    def __init__(self, num_superpixels, compactness_superpixels):
        self.num_superpixels = num_superpixels
        self.compactness_superpixels = compactness_superpixels

    def calculate_pe_opencv_seeds(self, img):
        img = np.ascontiguousarray(img.astype(np.uint8))
        image_height, image_width = img.shape[:2]

        num_levels = 4  # SEEDS Number of Levels
        prior = int(self.compactness_superpixels)  # SEEDS Smoothing Prior | range: [0, 5] | default: 1
        num_histogram_bins = 5  # SEEDS histogram bins
        double_step = False  # SEEDS two steps
        num_iterations = 10  # Iterations

        superpix_seeds = cv2.ximgproc.createSuperpixelSEEDS(
            image_width,
            image_height,
            3,
            self.num_superpixels,
            num_levels,
            prior,
            histogram_bins=num_histogram_bins,
            double_step=double_step,
        )
        superpix_seeds.iterate(img, num_iterations)

        segments = superpix_seeds.getLabels()

        if segments.min() == 0:
            segments = segments + 1

        regions_seeds = regionprops(segments + 1)
        centroids = np.array([np.rint(c.centroid).astype(np.int32) for c in regions_seeds])

        return segments, centroids
