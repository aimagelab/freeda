felzenszwalb_parameters_dict = {
    "scale": 600,  # Higher scale means less and larger segments
    "sigma": 0.8,  # is the diameter of a Gaussian kernel, used for smoothing the image prior to segmentation.
    "min_size": 400,  # Minimum component size. Enforced using postprocessing.
}

slic_parameters_dict = {
    "n_segments": 200,  # 100  # The (approximate) number of labels in the segmented output image.
    "compactness": 10,
    # Balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. We recommend exploring possible values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
    "sigma": 1,  # 0,  # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
    "start_label": 0,
    "min_size_factor": 0.5,  # Proportion of the minimum segment size to be removed with respect to the supposed segment size `depth*width*height/n_segments`
    "max_num_iter": 10,  # Maximum number of iterations of k-means
    "enforce_connectivity": True,  # Whether the generated segments are connected or not
}

quickshift_parameters_dict = {
    "ratio": 5.0,  # 1.0,  # Balances color-space proximity and image-space proximity. Higher values give more weight to color-space.
    "kernel_size": 10,  # 5,  # Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
    "max_dist": 10,  # Cut-off point for data distances. Higher means fewer clusters.
    "sigma": 1,  # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
}

watershed_parameters_dict = {
    "markers": 200,  # The number of markers, i.e. the number of segments in the output segmentation.
    "compactness": 1e-5,  # Use compact watershed with given compactness parameter. Higher values result in more regularly-shaped watershed basins.
}

seeds_parameters_dict = {
    "image_width": 448,
    "image_height": 448,
    "image_channels": 3,
    "num_superpixels": 200,  # Desired number of superpixels. Note that the actual number may be smaller due to restrictions (depending on the image size and num_levels). Use getNumberOfSuperpixels() to get the actual number.
    "num_levels": 4,  # Number of block levels. The more levels, the more accurate is the segmentation, but needs more memory and CPU time.
    "prior": 1,  # enable 3x3 shape smoothing term if >0. A larger value leads to smoother shapes. prior must be in the range [0, 5].
    "histogram_bins": 5,  # Number of histogram bins.
    "double_step": False,  # If true, iterate each block level twice for higher accuracy.
    "num_iterations": 10,  # Number of iterations. Higher number improves the result.
}
