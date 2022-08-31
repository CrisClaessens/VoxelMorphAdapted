import torch
import SimpleITK as sitk
from copy import deepcopy
from torch.utils import data
from math import tau
import numpy as np
import resector


class RegistrationDataSet(data.Dataset):
    """Image registration dataset with caching, pretransforms and multiprocessing."""

    def __init__(self,
                 brain_image: list,
                 brain_mask: list,
                 resection_mask: list,
                 gray_matter_mask: list,
                 noise_image: list,
                 vector_field: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 use_init_dvf=True,
                 sigmas_range=(0.5, 1),
                 radii_ratio_range=(1, 2),
                 volumes_range=(2000, 5000)  # (1.5, 57.2)
                 ):
        self.brain_image = brain_image
        self.brain_mask = brain_mask
        self.resection_mask = resection_mask
        self.gray_matter_mask = gray_matter_mask
        self.noise_image = noise_image
        self.vector_field = vector_field
        self.transform = transform
        self.image_dtype = torch.float32
        self.mask_dtype = torch.float32
        self.vector_dtype = torch.float32
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.use_init_dvf = use_init_dvf
        self.sigmas_range = sigmas_range
        self.radii_range = radii_ratio_range
        self.volumes_range = volumes_range

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(brain_image, brain_mask, resection_mask,
                                                                      gray_matter_mask, noise_image, vector_field,
                                                                      repeat(self.pre_transform)))
                print('data caching ready')

    def __len__(self):
        return len(self.brain_image)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = self.cached_data[
                index]
        else:
            # Select the sample
            brain_image_ID = self.brain_image[index]
            brain_mask_ID = self.brain_mask[index]
            resection_mask_ID = self.resection_mask[index]
            gray_matter_mask_ID = self.gray_matter_mask[index]
            noise_image_ID = self.noise_image[index]
            vector_field_ID = self.vector_field[index]

            # Load images and masks
            brain_image = sitk.ReadImage(str(brain_image_ID))

            brain_mask = sitk.ReadImage(str(brain_mask_ID), sitk.sitkFloat32)
            brain_mask.SetOrigin(brain_image.GetOrigin())
            brain_mask.SetSpacing(brain_image.GetSpacing())
            brain_mask.SetDirection(brain_image.GetDirection())

            resection_mask = sitk.ReadImage(str(resection_mask_ID))
            resection_mask.SetOrigin(brain_image.GetOrigin())
            resection_mask.SetSpacing(brain_image.GetSpacing())
            resection_mask.SetDirection(brain_image.GetDirection())

            gray_matter_mask = sitk.ReadImage(str(gray_matter_mask_ID))
            gray_matter_mask.SetOrigin(brain_image.GetOrigin())
            gray_matter_mask.SetSpacing(brain_image.GetSpacing())
            gray_matter_mask.SetDirection(brain_image.GetDirection())

            noise_image = sitk.ReadImage(str(noise_image_ID))
            noise_image.SetOrigin(brain_image.GetOrigin())
            noise_image.SetSpacing(brain_image.GetSpacing())
            noise_image.SetDirection(brain_image.GetDirection())

            vector_field = sitk.ReadImage(str(vector_field_ID), sitk.sitkVectorFloat32)
            vector_field.SetOrigin(brain_image.GetOrigin())
            vector_field.SetSpacing(brain_image.GetSpacing())
            vector_field.SetDirection(brain_image.GetDirection())

        # Preprocessing
        if self.transform is not None:
            brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = \
                self.transform(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field)

        # Create artificial post-operative scan
        volume = torch.FloatTensor(1).uniform_(*self.volumes_range).item()
        sigmas = torch.FloatTensor(3).uniform_(*self.sigmas_range).tolist()
        radii_ratio = torch.FloatTensor(1).uniform_(*self.radii_range).item()
        radius = (3 / 2 * volume / tau) ** (1 / 3)
        a = radius
        b = a * radii_ratio
        c = radius ** 3 / (a * b)
        radii = [a, b, c]
        brain_image_intra, resection_intra, _, _ = resector.resect(brain_image, gray_matter_mask, resection_mask,
                                                                   sigmas, radii,
                                                                   noise_image, noise_offset=1000, angles=(0, 0, 0))

        # Transform image and mask inside skull and reverse the applied vector field
        if not self.use_init_dvf:
            vector_field = None

        brain_image_post, resection_post, dvf = produceRandomlyDeformedImage(brain_image_intra, resection_intra,
                                                                             vector_field, brain_mask)

        # Typecasting
        brain_image_post, dvf, resection_post, resection_intra = sitk.GetArrayFromImage(brain_image_post), \
                                                                 np.moveaxis(sitk.GetArrayFromImage(dvf), -1, 0), \
                                                                 sitk.GetArrayFromImage(resection_post), \
                                                                 sitk.GetArrayFromImage(resection_intra)
        brain_image_post, dvf, resection_post, resection_intra = torch.from_numpy(brain_image_post).type(
            self.image_dtype), torch.from_numpy(dvf).type(self.vector_dtype), torch.from_numpy(resection_post).type(
            self.mask_dtype), torch.from_numpy(resection_intra).type(self.mask_dtype)

        # Add channel axis
        brain_image_post, resection_post, resection_intra = torch.unsqueeze(brain_image_post, 0), \
                                                            torch.unsqueeze(resection_post, 0), \
                                                            torch.unsqueeze(resection_intra, 0)

        # Transform to input/output
        input = [brain_image_post, resection_post, resection_intra]
        output = [dvf]

        return input, output

    @staticmethod
    def read_images(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field,
                    pre_transform):

        brain_image = sitk.ReadImage(str(brain_image))

        brain_mask = sitk.ReadImage(str(brain_mask), sitk.sitkFloat32)
        brain_mask.SetOrigin(brain_image.GetOrigin())
        brain_mask.SetSpacing(brain_image.GetSpacing())
        brain_mask.SetDirection(brain_image.GetDirection())

        resection_mask = sitk.ReadImage(str(resection_mask))
        resection_mask.SetOrigin(brain_image.GetOrigin())
        resection_mask.SetSpacing(brain_image.GetSpacing())
        resection_mask.SetDirection(brain_image.GetDirection())

        gray_matter_mask = sitk.ReadImage(str(gray_matter_mask))
        gray_matter_mask.SetOrigin(brain_image.GetOrigin())
        gray_matter_mask.SetSpacing(brain_image.GetSpacing())
        gray_matter_mask.SetDirection(brain_image.GetDirection())

        noise_image = sitk.ReadImage(str(noise_image))
        noise_image.SetOrigin(brain_image.GetOrigin())
        noise_image.SetSpacing(brain_image.GetSpacing())
        noise_image.SetDirection(brain_image.GetDirection())

        vector_field = sitk.ReadImage(str(vector_field), sitk.sitkVectorFloat32)
        vector_field.SetOrigin(brain_image.GetOrigin())
        vector_field.SetSpacing(brain_image.GetSpacing())
        vector_field.SetDirection(brain_image.GetDirection())

        if pre_transform:
            brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = \
                pre_transform(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field)
        return brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field


class RegistrationDataSet2(object):
    """Image registration dataset with caching, pretransforms and multiprocessing."""

    def __init__(self,
                 brain_image: list,
                 brain_mask: list,
                 resection_mask: list,
                 gray_matter_mask: list,
                 noise_image: list,
                 vector_field: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 use_init_dvf=True,
                 sigmas_range=(0.5, 1),
                 radii_ratio_range=(1, 2),
                 volumes_range=(2000, 5000)  # (1.5, 57.2)
                 ):
        self.brain_image = brain_image
        self.brain_mask = brain_mask
        self.resection_mask = resection_mask
        self.gray_matter_mask = gray_matter_mask
        self.noise_image = noise_image
        self.vector_field = vector_field
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.use_init_dvf = use_init_dvf
        self.sigmas_range = sigmas_range
        self.radii_range = radii_ratio_range
        self.volumes_range = volumes_range
        self.index = 0

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(brain_image, brain_mask, resection_mask,
                                                                      gray_matter_mask, noise_image, vector_field,
                                                                      repeat(self.pre_transform)))
                print('data caching ready')

    def __len__(self):
        return len(self.brain_image)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        try:
            if self.use_cache:
                brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field, subject = \
                    self.cached_data[self.index - 1]
            else:
                # Select the sample
                brain_image_ID = self.brain_image[self.index - 1]
                brain_mask_ID = self.brain_mask[self.index - 1]
                resection_mask_ID = self.resection_mask[self.index - 1]
                gray_matter_mask_ID = self.gray_matter_mask[self.index - 1]
                noise_image_ID = self.noise_image[self.index - 1]
                vector_field_ID = self.vector_field[self.index - 1]

                subject = str(brain_image_ID).split('/')[-1][:-4]

                # Load images and masks
                brain_image = sitk.ReadImage(str(brain_image_ID))

                brain_mask = sitk.ReadImage(str(brain_mask_ID), sitk.sitkFloat32)
                brain_mask.SetOrigin(brain_image.GetOrigin())
                brain_mask.SetSpacing(brain_image.GetSpacing())
                brain_mask.SetDirection(brain_image.GetDirection())

                resection_mask = sitk.ReadImage(str(resection_mask_ID))
                resection_mask.SetOrigin(brain_image.GetOrigin())
                resection_mask.SetSpacing(brain_image.GetSpacing())
                resection_mask.SetDirection(brain_image.GetDirection())

                gray_matter_mask = sitk.ReadImage(str(gray_matter_mask_ID))
                gray_matter_mask.SetOrigin(brain_image.GetOrigin())
                gray_matter_mask.SetSpacing(brain_image.GetSpacing())
                gray_matter_mask.SetDirection(brain_image.GetDirection())

                noise_image = sitk.ReadImage(str(noise_image_ID))
                noise_image.SetOrigin(brain_image.GetOrigin())
                noise_image.SetSpacing(brain_image.GetSpacing())
                noise_image.SetDirection(brain_image.GetDirection())

                vector_field = sitk.ReadImage(str(vector_field_ID), sitk.sitkVectorFloat32)
                vector_field.SetOrigin(brain_image.GetOrigin())
                vector_field.SetSpacing(brain_image.GetSpacing())
                vector_field.SetDirection(brain_image.GetDirection())

            # Preprocessing
            if self.transform is not None:
                brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = \
                    self.transform(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field)

            # Create artificial post-operative scan
            volume = torch.FloatTensor(1).uniform_(*self.volumes_range).item()
            sigmas = torch.FloatTensor(3).uniform_(*self.sigmas_range).tolist()
            radii_ratio = torch.FloatTensor(1).uniform_(*self.radii_range).item()
            radius = (3 / 2 * volume / tau) ** (1 / 3)
            a = radius
            b = a * radii_ratio
            c = radius ** 3 / (a * b)
            radii = [a, b, c]
            brain_image_intra, resection_intra, _, _ = resector.resect(brain_image, gray_matter_mask, resection_mask,
                                                                       sigmas, radii,
                                                                       noise_image, noise_offset=1000, angles=(0, 0, 0))

            # Transform image and mask inside skull and reverse the applied vector field
            if not self.use_init_dvf:
                vector_field = None

            brain_image_post, resection_post, dvf = produceRandomlyDeformedImage(brain_image_intra, resection_intra,
                                                                                 vector_field, brain_mask)

            brain_image_post.SetOrigin(brain_image.GetOrigin())
            brain_image_post.SetSpacing(brain_image.GetSpacing())
            brain_image_post.SetDirection(brain_image.GetDirection())

            brain_image_intra.SetOrigin(brain_image.GetOrigin())
            brain_image_intra.SetSpacing(brain_image.GetSpacing())
            brain_image_intra.SetDirection(brain_image.GetDirection())

            resection_post.SetOrigin(brain_image.GetOrigin())
            resection_post.SetSpacing(brain_image.GetSpacing())
            resection_post.SetDirection(brain_image.GetDirection())

            resection_intra.SetOrigin(brain_image.GetOrigin())
            resection_intra.SetSpacing(brain_image.GetSpacing())
            resection_intra.SetDirection(brain_image.GetDirection())

            dvf.SetOrigin(brain_image.GetOrigin())
            dvf.SetSpacing(brain_image.GetSpacing())
            dvf.SetDirection(brain_image.GetDirection())

            return subject, brain_image_post, brain_image_intra, resection_post, resection_intra, dvf
        except IndexError:
            self.index = 0
            return self.__next__()

    @staticmethod
    def read_images(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field,
                    pre_transform):

        subject = str(brain_image).split('/')[-1][:-4]
        brain_image = sitk.ReadImage(str(brain_image))

        brain_mask = sitk.ReadImage(str(brain_mask), sitk.sitkFloat32)
        brain_mask.SetOrigin(brain_image.GetOrigin())
        brain_mask.SetSpacing(brain_image.GetSpacing())
        brain_mask.SetDirection(brain_image.GetDirection())

        resection_mask = sitk.ReadImage(str(resection_mask))
        resection_mask.SetOrigin(brain_image.GetOrigin())
        resection_mask.SetSpacing(brain_image.GetSpacing())
        resection_mask.SetDirection(brain_image.GetDirection())

        gray_matter_mask = sitk.ReadImage(str(gray_matter_mask))
        gray_matter_mask.SetOrigin(brain_image.GetOrigin())
        gray_matter_mask.SetSpacing(brain_image.GetSpacing())
        gray_matter_mask.SetDirection(brain_image.GetDirection())

        noise_image = sitk.ReadImage(str(noise_image))
        noise_image.SetOrigin(brain_image.GetOrigin())
        noise_image.SetSpacing(brain_image.GetSpacing())
        noise_image.SetDirection(brain_image.GetDirection())

        vector_field = sitk.ReadImage(str(vector_field), sitk.sitkVectorFloat32)
        vector_field.SetOrigin(brain_image.GetOrigin())
        vector_field.SetSpacing(brain_image.GetSpacing())
        vector_field.SetDirection(brain_image.GetDirection())

        if pre_transform:
            brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field = \
                pre_transform(brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field)
        return brain_image, brain_mask, resection_mask, gray_matter_mask, noise_image, vector_field, subject


def produceRandomlyDeformedImage(image, label, init_dvf=None, brain_mask=None, numcontrolpoints=2, stdDef=2, seed=None,
                                 exclude_z=False):
    '''
    Part of this function comes from V-net，deform an image by B-spine interpolation
    :param image: images ，numpy array
    :param label: labels，numpy array
    :param init_dvf: initial deformation vector field to initialize random field, take None for default
    :param numcontrolpoints: control point，B-spine interpolation parameters，take 2 for default
    :param stdDef: Deviation，B-spine interpolation parameters，take 5 for default
    :return: Deformed images and GT in numpy array
    '''

    transfromDomainMeshSize = [numcontrolpoints] * image.GetDimension()

    tx = sitk.BSplineTransformInitializer(
        image, transfromDomainMeshSize)

    params = tx.GetParameters()

    paramsNp = np.asarray(params, dtype=float)

    if seed:
        np.random.seed(seed)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * stdDef

    # remove z deformations! The resolution in z is too bad
    if exclude_z:
        paramsNp[0:int(len(params) / 3)] = 0

    params = tuple(paramsNp)
    tx.SetParameters(params)

    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(image)
    displacement_filter.SetOutputPixelType(sitk.sitkVectorFloat32)
    displacement_field = displacement_filter.Execute(tx)  # pixelID=21

    if init_dvf is not None:
        displacement_field.SetOrigin(init_dvf.GetOrigin())
        addition_filter = sitk.AddImageFilter()
        displacement_field = addition_filter.Execute(init_dvf, displacement_field)

    if brain_mask is not None:
        multiply_filter = sitk.MultiplyImageFilter()
        component_filter = sitk.VectorIndexSelectionCastImageFilter()
        component_filter.SetOutputPixelType(sitk.sitkFloat32)
        component_filter.SetIndex(0)
        displacement_field_0 = multiply_filter.Execute(brain_mask, component_filter.Execute(displacement_field))
        displacement_field_0 = sitk.GetArrayFromImage(displacement_field_0)

        component_filter.SetIndex(1)
        displacement_field_1 = multiply_filter.Execute(brain_mask, component_filter.Execute(displacement_field))
        displacement_field_1 = sitk.GetArrayFromImage(displacement_field_1)

        component_filter.SetIndex(2)
        displacement_field_2 = multiply_filter.Execute(brain_mask, component_filter.Execute(displacement_field))
        displacement_field_2 = sitk.GetArrayFromImage(displacement_field_2)

        displacement_field_new = sitk.GetImageFromArray(np.stack([displacement_field_0,
                                                                  displacement_field_1,
                                                                  displacement_field_2], axis=-1),
                                                        isVector=True)
        displacement_field_new.CopyInformation(displacement_field)
        displacement_field_new = sitk.Cast(displacement_field_new, sitk.sitkVectorFloat64)
    else:
        displacement_field_new = displacement_field

    displacement_field2 = deepcopy(displacement_field_new)
    tx = sitk.DisplacementFieldTransform(3)
    tx.SetDisplacementField(displacement_field2)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimg = resampler.Execute(image)
    outlbl = resampler.Execute(label)
    outdvf = sitk.InvertDisplacementField(displacement_field_new)

    return outimg, outlbl, outdvf


def save_example(batch):
    dvf = np.moveaxis(np.array(batch[1][0][0, :, :, :, :]), 0, -1)
    img = np.array(batch[0][0][0, 0, :, :, :])
    post = np.array(batch[0][1][0, 0, :, :, :])
    intra = np.array(batch[0][2][0, 0, :, :, :])

    sitk.WriteImage(sitk.GetImageFromArray(dvf, isVector=True), 'dvf.nii')
    sitk.WriteImage(sitk.GetImageFromArray(img), 'img.nii')
    sitk.WriteImage(sitk.GetImageFromArray(post), 'post.nii')
    sitk.WriteImage(sitk.GetImageFromArray(intra), 'intra.nii')

    return