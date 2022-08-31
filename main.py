import torch
import SimpleITK as sitk

from transformations import ComposeCustom, FunctionWrapperCustom, resample_to_size_itk
from model import RegistrationModel
from trainer import Trainer
from losses import RegistrationLoss
from customdatasets import RegistrationDataSet
from torch.utils.data import DataLoader
#from lr_finder import LearningRateFinder
from visual import plot_training

import pathlib

# root directory
root = pathlib.Path.cwd() / 'data'


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in sorted(path.glob(ext)) if file.is_file()]
    return filenames


# input and target files
brain_images_train = get_filenames_of_path(root / 'pre_op_t1_rigid')[:40]
brain_masks_train = get_filenames_of_path(root / 'brain_mask_pre')[:40]
resection_masks_train = get_filenames_of_path(root / 'resection_mask')[:40]
gray_matter_masks_train = get_filenames_of_path(root / 'gray_matter_mask')[:40]
noise_images_train = get_filenames_of_path(root / 'noise_image')[:40]
deformation_vector_fields_train = get_filenames_of_path(root / 'dvf_pre_post')[:40]

brain_images_val = get_filenames_of_path(root / 'pre_op_t1_rigid')[40:50]
brain_masks_val = get_filenames_of_path(root / 'brain_mask_pre')[40:50]
resection_masks_val = get_filenames_of_path(root / 'resection_mask')[40:50]
gray_matter_masks_val = get_filenames_of_path(root / 'gray_matter_mask')[40:50]
noise_images_val = get_filenames_of_path(root / 'noise_image')[40:50]
deformation_vector_fields_val = get_filenames_of_path(root / 'dvf_pre_post')[40:50]


# training transformations and augmentations
transforms = ComposeCustom([
    FunctionWrapperCustom(resample_to_size_itk, image=True, masks=False, dvf=True, size=(160, 240, 240),
                          interpolator=sitk.sitkLinear),
    FunctionWrapperCustom(resample_to_size_itk, image=False, masks=True, dvf=False, size=(160, 240, 240),
                          interpolator=sitk.sitkNearestNeighbor)
])

# dataset training
train_dataset = RegistrationDataSet(brain_image=brain_images_train,
                                    brain_mask=brain_masks_train,
                                    resection_mask=resection_masks_train,
                                    gray_matter_mask=gray_matter_masks_train,
                                    noise_image=noise_images_train,
                                    vector_field=deformation_vector_fields_train,
                                    use_init_dvf=True,
                                    use_cache=True,
                                    pre_transform=transforms)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=2,
                              shuffle=True)

val_dataset = RegistrationDataSet(brain_image=brain_images_val,
                                  brain_mask=brain_masks_val,
                                  resection_mask=resection_masks_val,
                                  gray_matter_mask=gray_matter_masks_val,
                                  noise_image=noise_images_val,
                                  vector_field=deformation_vector_fields_val,
                                  use_init_dvf=True,
                                  use_cache=True,
                                  pre_transform=transforms)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=2,
                            shuffle=False)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model = RegistrationModel(inshape=(240, 240, 160), nb_unet_features=[[32, 64, 64], [64, 64, 32, 32, 16, 16]]).to(device)

lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

criterion = RegistrationLoss()

"""lr_finder = LearningRateFinder(model=model,
                               criterion=criterion,
                               optimizer=optimizer,
                               device=device)
lr_finder.fit(data_loader=dataloader, max_lr=1e3, steps=200)
lr_finder.plot()"""

trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=train_dataloader,
                  validation_DataLoader=val_dataloader,
                  epochs=100,
                  epoch=0)

training_losses, validation_losses, lr_rates = trainer.run_trainer()

fig = plot_training(training_losses,
                    validation_losses,
                    lr_rates,
                    gaussian=True,
                    sigma=1,
                    figsize=(10, 4))
