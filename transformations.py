from typing import List, Callable, Tuple

import numpy as np
import SimpleITK as sitk
import albumentations as A
from sklearn.externals._pilutil import bytescale
from skimage.util import crop


def normalize_01(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    """Normalize based on mean and standard deviation."""
    inp_out = (inp - mean) / std
    return inp_out


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def center_crop_to_size(x: np.ndarray,
                        size: Tuple,
                        copy: bool = False,
                        ) -> np.ndarray:
    """
    Center crops a given array x to the size passed in the function.
    Expects even spatial dimensions! But can handle one extra dimension.
    """
    x_shape = np.array(x.shape)
    size = np.array(size)
    if x_shape.shape[0] - size.shape[0] == 1: np.append(size, np.array([0]))
    params_list = ((x_shape - size) / 2).tolist()
    params_tuple = tuple([(int(np.floor(i)), int(np.ceil(i))) for i in params_list])
    cropped_image = crop(x, crop_width=params_tuple, copy=copy)
    return cropped_image


def resample_to_size_itk(x: sitk.Image, size: tuple, interpolator=sitk.sitkNearestNeighbor):
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputPixelType(x.GetPixelID())
    resample_filter.SetSize(size)
    resample_filter.SetOutputDirection(x.GetDirection())
    resample_filter.SetOutputOrigin(x.GetOrigin())
    resample_filter.SetOutputSpacing(tuple((size1 / size2) * space for size1, size2, space in zip(x.GetSize(), size,
                                                                                                  x.GetSpacing())))
    x = resample_filter.Execute(x)

    return x


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


def random_flip(inp: np.ndarray, tar: np.ndarray, ndim_spatial: int):
    flip_dims = [np.random.randint(low=0, high=2) for dim in range(ndim_spatial)]

    flip_dims_inp = tuple([i + 1 for i, element in enumerate(flip_dims) if element == 1])
    flip_dims_tar = tuple([i for i, element in enumerate(flip_dims) if element == 1])

    inp_flipped = np.flip(inp, axis=flip_dims_inp)
    tar_flipped = np.flip(tar, axis=flip_dims_tar)

    return inp_flipped, tar_flipped


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar


class FunctionWrapperCustom(Repr):
    """ A custom function wrapper that returns a partial for a post-operative scan
    with corresponding masks of the resection cavities, brain, gray-matter and
    resectable areas and the defomation vector field."""

    def __init__(self, function: Callable, image: bool = True, masks: bool = True, dvf: bool = True, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.image = image
        self.masks = masks
        self.dvf = dvf

    def __call__(self, image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field):
        if self.image:
            image = self.function(image)
            noise_image = self.function(noise_image)
        if self.masks:
            brain_mask = self.function(brain_mask)
            resection_mask = self.function(resection_mask)
            gray_matter_mask = self.function(gray_matter_mask)
        if self.dvf: vector_field = self.function(vector_field)
        return image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class ComposeCustom(Compose):
    """Composes transforms for a post-operative scan with corresponding masks of
    the resection cavities, brain, gray-matter and resectable areas and the
    defomation vector field."""

    def __call__(self, image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field):
        for t in self.transforms:
            image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = \
                t(image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field)
        return image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field


class AlbuReg2d(Repr):
    """
    Wrapper for albumentations' registration-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (C, spatial_dims)
    Expected target: (spatial_dims) -> No (C)hannel dimension
    """

    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        out_dict = self.albumentation(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out


class AlbuReg3d(Repr):
    """
    Wrapper for albumentations' registration-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (spatial_dims)  -> No (C)hannel dimension
    Expected target: (spatial_dims) -> No (C)hannel dimension
    Iterates over the slices of a input-target pair stack and performs the same albumentation function.
    """

    def __init__(self, albumentation: Callable):
        self.albumentation = A.ReplayCompose([albumentation])

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        tar = tar.astype(np.uint8)  # target has to be in uint8

        input_copy = np.copy(inp)
        target_copy = np.copy(tar)

        replay_dict = self.albumentation(image=inp[0])[
            'replay']  # perform an albu on one slice and access the replay dict

        # TODO: consider cases with RGB 3D or multimodal 3D input

        # only if input_shape == target_shape
        for index, (input_slice, target_slice) in enumerate(zip(inp, tar)):
            result = A.ReplayCompose.replay(replay_dict, image=input_slice, mask=target_slice)
            input_copy[index] = result['image']
            target_copy[index] = result['mask']

        return input_copy, target_copy