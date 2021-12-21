import os
import glob
import fnmatch
import numpy as np
import cv2
import random
from skimage import io, color
from skimage.measure import label
from skimage.transform import rotate
from skimage.util import view_as_windows
from PIL import Image
import sys
from skimage.morphology import skeletonize



def more_patches(inputPath, path, rotation_angles, gamma_correction_p, down_scale, extension, recursive):

    patch_size = (512, 512)
    #out_size = patch_size + (1,)
    img_paths, mask_img_paths = get_images(inputPath, extension, recursive)

    sum_patch = np.zeros(patch_size) # to get mean and std

    train_images = []
    train_masks = []

    count = 0
    for ind in range(len(mask_img_paths)):
        img = io.imread(img_paths[ind]).astype('float32')
        mask = io.imread(mask_img_paths[ind]).astype('float32')

        img /= np.max(img)
        img = color.rgb2gray(img)
        mask = mask / 255.
        mask = color.rgb2gray(mask)
        org_img = img
        org_mask = mask
    ####################
        for scale in np.nditer(down_scale):
            print(np.nditer(down_scale))
            org_img = cv2.resize(org_img, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_LANCZOS4)
            #
            org_mask = cv2.resize(org_mask, None, fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_AREA)

            print (ind+1,'/',len(img_paths))
            try:
                img_patches = view_as_windows(org_img, patch_size, step=int(patch_size[0]/4))
            except:
                import pdb; pdb.set_trace()
            mask_patches = view_as_windows(org_mask, patch_size, step=int(patch_size[0]/4))
            # import pdb; pdb.set_trace()



            for patch_ind_0 in range(img_patches.shape[0]):
                for patch_ind_1 in range(img_patches.shape[1]):

                    mask_patch = mask_patches[patch_ind_0,patch_ind_1]

                    if (mask_patch > 0.5).sum() > 200:# prev 70
                        img_patch = img_patches[patch_ind_0,patch_ind_1]
                        brt = random.uniform(0.4,1.4)
                        brt = np.min((brt, 1))
                        img_patch *= brt
                        for angle in np.nditer(rotation_angles):
                            for gamma in np.nditer(gamma_correction_p):
                                # gamma = random.uniform(0.6,1.2)
                                gamma = 1
                                count += 1
                                one_image_patch = gamma_correction(rotate(img_patch, angle), gamma).astype('float32')
                                one_mask_patch = rotate(mask_patch, angle).astype('float32')
                                train_images.append(one_image_patch.astype('float32'))
                                cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                                cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        for gamma in np.nditer(gamma_correction_p):
                            # gamma = random.uniform(0.6,1.4)
                            gamma = 1
                            count += 1
                            one_image_patch = gamma_correction(np.fliplr(img_patch), gamma).astype('float32')
                            one_mask_patch = np.fliplr(mask_patch).astype('float32')

                            train_images.append(one_image_patch.astype('float32'))
                            cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                            cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        for gamma in np.nditer(gamma_correction_p):
                            # gamma = random.uniform(0.6,1.4)
                            gamma = 1
                            count += 1

                            one_image_patch = gamma_correction(np.flipud(img_patch), gamma).astype('float32')
                            one_mask_patch = np.flipud(mask_patch).astype('float32')

                            train_images.append(one_image_patch.astype('float32'))
                            cv2.imwrite(img_dir + '/img_patch_' + str(count) + '.png', one_image_patch * 255.)
                            cv2.imwrite(mask_dir + '/mask_patch_' + str(count) + '.png', one_mask_patch * 255.)
                        print ("num_of_patches is %d              \r" %count) ,

    train_images = np.array(train_images)
    meanVal = np.mean(train_images)
    stdVal = np.std(train_images)
    f= open(fileName + "/mean_std.txt","w+")
    f.write("#mean:\n %.20f \n#std:\n %.20f\n" % (meanVal, stdVal))
    print (count)

    return

def gamma_correction(img, gamma):
    img = np.power(img, gamma)
    return img

def get_images(path, extension, recursive):
    image_path = path + '/img_list'
    mask_path = path + '/mask_list'
    img_paths = []
    mask_img_paths = []


    for root, directories, filenames in os.walk(image_path):
      for filename in fnmatch.filter(filenames, extension):
        img_paths.append(os.path.join(root,filename))

    for root, directories, filenames in os.walk(mask_path):
      for filename in fnmatch.filter(filenames, extension):
        mask_img_paths.append(os.path.join(root,filename))

    img_paths.sort()
    mask_img_paths.sort()

    return img_paths, mask_img_paths

# fileName = sys.argv[1]
fileName = 'training'


rotation_angles = np.array([0, 90, 180, 270])
gamma_correction_p = np.array([1])
# gamma_correction_p = np.array([1])
down_scale = np.array([1])
inputPath = 'training_dataset'
outputPath = 'patch' + fileName

img_dir =  outputPath + '/img_list'
mask_dir = outputPath + '/mask_list'

if not os.path.exists(fileName):
    os.makedirs(fileName)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    os.makedirs(mask_dir)

more_patches(inputPath, outputPath, rotation_angles, gamma_correction_p, down_scale, extension='*.tif', recursive = True)
