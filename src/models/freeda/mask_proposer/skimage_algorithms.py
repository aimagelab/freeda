"""
Utils:
- https://github.com/davidstutz/superpixel-benchmark/tree/master
"""

import time

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, mark_boundaries
from skimage.transform import resize
from skimage.util import img_as_float

import matplotlib.pyplot as plt

# try:
#     from fast_slic.avx2 import SlicAvx2 as Slic
#
#     print("Using fast_slic.avx2")
# except Exception:
#     from fast_slic import Slic as Slic

# print("Using fast_slic")

img = img_as_float(astronaut())  # [::12, ::12]

# image_size = (512, 512)
image_size = (1024, 1024)
img = resize(img, image_size, anti_aliasing=True)

num_levels = 4  # SEEDS Number of Levels
prior = 1  # SEEDS Smoothing Prior
num_histogram_bins = 5  # SEEDS histogram bins
double_step = False  # SEEDS two steps
num_iterations = 10  # Iterations

print(f"Input Image Size: {img.shape}")
print("-" * 10)

# num_superpixels = [50, 200, 1024, 2028, 4096, 8192, 16384, 28000, 32768]
# num_superpixels = [8, 16, 32, 64, 128, 256, 512, 1024, 2028, 4096, 8192, 16384, 28000, 32768]
# num_superpixels = [1024, 2028, 4096, 8192, 16384, 28000, 32768]
num_superpixels = [100]

fastslic_vs_watershed = []
fastslic_vs_seeds = []

print("Image Size: ", img.shape)
for n_segments in num_superpixels:
    print(f">>> #Segments={n_segments} <<<")
    t1 = time.time()
    segments_fz = felzenszwalb(img, scale=800, sigma=0.5, min_size=1000)
    t2 = time.time()
    regions_fz = regionprops(segments_fz + 1)
    print(f"{'Felzenszwalb number of segments:':33} {len(np.unique(segments_fz))}/({n_segments})"
          f" -- Time: {(t2 - t1) * 1000:6.2f} msec")

    plt.imshow(mark_boundaries(img, segments_fz, color=(1, 1, 0)))
    # plt.scatter([x.centroid[1] for x in regions_fz], [y.centroid[0] for y in regions_fz], c='red')
    plt.title(f"Felzenszwalb #{n_segments}")
    plt.tight_layout()
    plt.show()
    plt.close()

    # ------------------------------------------------------------

    t2 = time.time()
    segments_slic = slic(img, n_segments=n_segments, compactness=10, sigma=1, start_label=0, min_size_factor=0.05)
    t3 = time.time()
    regions_slic = regionprops(segments_slic + 1)
    print(f"{'SLIC number of segments:':33} {len(np.unique(segments_slic))}/({n_segments})"
          f" -- Time: {(t3 - t2) * 1000:6.2f} msec")

    plt.imshow(mark_boundaries(img, segments_slic, color=(1, 1, 0)))
    # plt.scatter([x.centroid[1] for x in regions_slic], [y.centroid[0] for y in regions_slic], c='red')
    plt.title(f"slic #{n_segments}")
    plt.tight_layout()
    plt.show()
    plt.close()

    # ------------------------------------------------------------

    # segments_slic_fast = Slic(num_components=n_segments, compactness=20, min_size_factor=0.05)
    # t5 = time.time()
    # assignment = segments_slic_fast.iterate(np.ascontiguousarray(img.astype(np.uint8)))  # Cluster Map
    # t6 = time.time()
    #
    # time_fast_slic = t6 - t5
    #
    # # resize assignment to (32, 32) with nearest neighbor interpolation
    # assignment = resize(assignment, (32, 32), order=0, anti_aliasing=False, preserve_range=True)
    #
    # centroids = np.array([d['yx'] for d in segments_slic_fast.slic_model.clusters], dtype=np.int32)
    # regions_slic_fast = regionprops(assignment + 1)
    # centroids_props = [np.rint(c.centroid).astype(np.int32) for c in regions_slic_fast]
    # print(f"{'Fast-SLIC number of segments:':33} {len(np.unique(assignment))}/({n_segments})"
    #       f" -- Time: {(t6 - t5) * 1000:6.2f} msec")

    # plt.imshow(mark_boundaries(resize(astronaut(), (32, 32), anti_aliasing=True), assignment, color=(1, 1, 0)))
    # # plt.scatter([x.centroid[1] for x in regions_slic_fast], [y.centroid[0] for y in regions_slic_fast], c='red')
    # plt.title(f"fast slic #{n_segments}")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # ------------------------------------------------------------

    t3 = time.time()
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    t4 = time.time()
    regions_quick = regionprops(segments_quick + 1)
    print(f"{'Quickshift number of segments:':33} {len(np.unique(segments_quick))}/({n_segments})"
          f" -- Time: {(t4 - t3) * 1000:6.2f} msec")

    plt.imshow(mark_boundaries(img, segments_quick, color=(1, 1, 0)))
    # plt.scatter([x.centroid[1] for x in regions_quick], [y.centroid[0] for y in regions_quick], c='red')
    plt.title(f"Quickshift #{n_segments}")
    plt.tight_layout()
    plt.show()
    plt.close()

    # ------------------------------------------------------------

    t4 = time.time()
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=n_segments, compactness=0.0001)
    t5 = time.time()
    regions_watershed = regionprops(segments_watershed + 1)
    print(f"{'Watershed number of segments:':33} {len(np.unique(segments_watershed))}/({n_segments})"
          f" -- Time: {(t5 - t4) * 1000:6.2f} msec")

    watershed_time = t5 - t4

    plt.imshow(mark_boundaries(img, segments_watershed, color=(1, 1, 0)))
    # plt.scatter([x.centroid[1] for x in regions_watershed], [y.centroid[0] for y in regions_watershed], c='red')
    plt.title(f"Watershed #{n_segments}")
    plt.tight_layout()
    plt.show()
    plt.close()
    # ------------------------------------------------------------
    image_height, image_width = img.shape[:2]
    image_channels = 1 if len(img.shape) == 2 else img.shape[2]
    t1 = time.time()
    superpix_seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image_width,
        image_height,
        image_channels,
        n_segments,
        num_levels,
        prior,
        histogram_bins=num_histogram_bins,
        double_step=double_step,
    )
    superpix_seeds.iterate((img * 255).astype('uint8'), num_iterations)
    t2 = time.time()

    npix = superpix_seeds.getNumberOfSuperpixels()
    segments_seeds = superpix_seeds.getLabels()

    seeds_time = t2 - t1

    print(f"{'SEEDS number of segments:':33} {npix}/({n_segments})"
          f" -- Time: {(t2 - t1) * 1000:6.2f} msec")

    plt.imshow(mark_boundaries(img, segments_seeds, color=(1, 1, 0)))
    # plt.scatter([x.centroid[1] for x in regions_watershed], [y.centroid[0] for y in regions_watershed], c='red')
    plt.title(f"SEEDS #{n_segments}")
    plt.tight_layout()
    plt.show()
    plt.close()

    # ------------------------------------------------------------
    # print how many times watershed time is slower than fast slic
    # fastslic_vs_watershed.append(watershed_time / time_fast_slic)
    # print(f"Watershed is {watershed_time / time_fast_slic:6.2f} times slower than Fast-SLIC\n")
    #
    # fastslic_vs_seeds.append(seeds_time / time_fast_slic)
    # print(f"SEEDS is {seeds_time / time_fast_slic:6.2f} times slower than Fast-SLIC\n")

    # ------------------------------------------------------------

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # Labels with value 0 are ignored!
    # ax[0, 0].imshow(mark_boundaries(img, segments_fz, color=(1, 1, 0)))
    # ax[0, 0].scatter([x.centroid[1] for x in regions_fz], [y.centroid[0] for y in regions_fz], c='red')
    # ax[0, 0].set_title("Felzenszwalbs's method")
    #
    # ax[0, 1].imshow(mark_boundaries(img, segments_slic, color=(1, 1, 0)))
    # ax[0, 1].scatter([x.centroid[1] for x in regions_slic], [y.centroid[0] for y in regions_slic], c='red')
    # ax[0, 1].set_title('SLIC')
    #
    # ax[1, 0].imshow(mark_boundaries(img, segments_quick, color=(1, 1, 0)))
    # ax[1, 0].scatter([x.centroid[1] for x in regions_quick], [y.centroid[0] for y in regions_quick], c='red')
    # ax[1, 0].set_title('Quickshift')

    # ax[1, 1].imshow(mark_boundaries(img, segments_watershed, color=(1, 1, 0)))
    # ax[1, 1].scatter([x.centroid[1] for x in regions_watershed], [y.centroid[0] for y in regions_watershed], c='red')
    # ax[1, 1].set_title('Compact watershed')

    # for a in ax.ravel():
    #     a.set_axis_off()

    # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    #
    # color = np.random.choice(range(256), size=3 * (np.max(segments_fz) + 1)).reshape(np.max(segments_fz) + 1, 3)
    # ax[0, 0].imshow(color[segments_fz])
    # ax[0, 0].set_title("Felzenszwalbs's method")
    #
    # color = np.random.choice(range(256), size=3 * (np.max(segments_slic) + 1)).reshape(np.max(segments_slic) + 1, 3)
    # ax[0, 1].imshow(color[segments_slic])
    # ax[0, 1].set_title('SLIC')
    #
    # color = np.random.choice(range(256), size=3 * (np.max(segments_quick) + 1)).reshape(np.max(segments_quick) + 1, 3)
    # ax[1, 0].imshow(color[segments_quick])
    # ax[1, 0].set_title('Quickshift')
    #
    # color = np.random.choice(range(256), size=3 * (np.max(segments_watershed) + 1)).reshape(
    #     np.max(segments_watershed) + 1, 3)
    # ax[1, 1].imshow(color[segments_watershed])
    # ax[1, 1].set_title('Compact watershed')
    #
    # for a in ax.ravel():
    #     a.set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()
    #
    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    #
    # ax[0, 0].imshow(label2rgb(segments_fz, img, kind='avg', bg_label=-1))
    # ax[0, 0].set_title("Felzenszwalbs's method")
    #
    # ax[0, 1].imshow(label2rgb(segments_slic, img, kind='avg', bg_label=-1))
    # ax[0, 1].set_title('SLIC')
    #
    # ax[1, 0].imshow(label2rgb(segments_quick, img, kind='avg', bg_label=-1))
    # ax[1, 0].set_title('Quickshift')
    #
    # ax[1, 1].imshow(label2rgb(segments_watershed, img, kind='avg', bg_label=-1))
    # ax[1, 1].set_title('Compact watershed')
    #
    # for a in ax.ravel():
    #     a.set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()

# print(f"{'Fast-SLIC vs Watershed:':33} {np.mean(fastslic_vs_watershed):6.2f} times slower")
# print(f"{'Fast-SLIC vs SEEDS:':33} {np.mean(fastslic_vs_seeds):6.2f} times slower")
