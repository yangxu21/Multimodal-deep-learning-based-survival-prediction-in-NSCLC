import cv2
import numpy as np
import os
import openslide
from PIL import Image
from tqdm import tqdm

def resize_wsi(image_path, output_folder, target_um = 0.55):
    # read the image
    image = openslide.OpenSlide(image_path)
    # get the image size
    width, height = image.dimensions
    # calculate the scale
    scale = float(image.properties['openslide.mpp-x']) / target_um
    # calculate the new size
    new_width = int(width * scale)
    new_height = int(height * scale)
    # get the image name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_name_1 = image_name + '.jpg'
    image_name_2 = image_name + '.png'
    # create the output folder
    os.makedirs(output_folder, exist_ok=True)
    # create the output path
    output_path_1 = os.path.join(output_folder, image_name_1)
    output_path_2 = os.path.join(output_folder, image_name_2)
    # check if the jpg or png image is already resized
    if os.path.exists(output_path_1) or os.path.exists(output_path_2):
        return
    # create the image
    new_image = np.zeros((new_height + 4, new_width + 4, 3), dtype=np.uint8)
    # create the patches
    patches = []
    for i in range(4):
        for j in range(4):
            # get the patch
            patch = image.read_region((i * width // 4, j * height // 4), 0, (width // 4, height // 4))
            # convert the patch to numpy array
            patch = np.array(patch)[:, :, :3]
            # resize the patch
            patch = cv2.resize(patch, (new_width // 4, new_height // 4), interpolation=cv2.INTER_CUBIC)
            # add the patch to the list
            patches.append(patch)
    # paste the patches back together
    for i in range(4):
        for j in range(4):
            new_image[j * patches[0].shape[0]: (j * patches[0].shape[0] + patches[0].shape[0]), i * patches[0].shape[1]: (i * patches[0].shape[1] + patches[0].shape[1])] = patches[i * 4 + j]
    # close the image
    image.close()

    # save the image
    new_image = np.array(new_image)
    new_image = Image.fromarray(new_image)
    try:
        new_image.save(output_path_1)
    except:
        new_image.save(output_path_2)


image_paths = os.listdir('original_WSI_files')
image_paths = [os.path.join('original_WSI_files', image_path) for image_path in image_paths]
output_folder = 'resized_image_file'

for image_path in tqdm(image_paths):
    resize_wsi(image_path, output_folder)

