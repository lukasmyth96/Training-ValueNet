"""
Script for resizing and renaming an image classification dataset.
Makes a copy of the dataset.
Assumes images are in the format

>dataset_dir
  > subset_1
      > class_1
          image_1
            .
          image_n
          .
          .
      > class_n
  .
  .
  >subset_n
"""
import argparse
import logging
import os
import skimage.io
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_dir', default='/home/luka/Documents/training_value_net/aircraft_7_dataset',
                        help='Directory of dataset to resize')
    parser.add_argument('--output_dataset_dir', default='/home/luka/Documents/training_value_net/aircraft_7_dataset_resized')
    parser.add_argument('--resized_dims', default=(128, 128),
                        help='Dimensions of resize images')
    parser.add_argument('--rename_images', default=True)
    pargs = parser.parse_args()

    logging.basicConfig(filename=None, level=logging.INFO)
    logging.captureWarnings(False)

    # Create output dir for resized dataset
    if not os.path.isdir(pargs.output_dataset_dir):
        os.makedirs(pargs.output_dataset_dir)
        logging.info('Created directory for new dataset: {}'.format(pargs.output_dataset_dir))

    subset_names = [directory for directory in os.listdir(pargs.input_dataset_dir)]
    logging.info('Found dataset with following subsets: {}'.format(subset_names))

    for subset_name in subset_names:
        # Create output dir for subset
        output_subset_dir = os.path.join(pargs.output_dataset_dir, subset_name)
        if not os.path.isdir(output_subset_dir):
            os.mkdir(output_subset_dir)

        input_subset_dir = os.path.join(pargs.input_dataset_dir, subset_name)
        class_names = [directory for directory in os.listdir(input_subset_dir)]
        logging.info('Subset {} contains following class names: {}'.format(subset_name, class_names))

        for class_name in class_names:
            # Create output dir for class
            output_class_dir = os.path.join(output_subset_dir, class_name)
            if not os.path.isdir(output_class_dir):
                os.mkdir(output_class_dir)

            logging.info('Starting class {} from subset {}'.format(class_name, subset_name))
            input_class_dir = os.path.join(input_subset_dir, class_name)

            filename_list = [f for f in os.listdir(input_class_dir)]
            for idx, filename in enumerate(tqdm(filename_list)):
                input_filepath = os.path.join(input_class_dir, filename)

                # Load and resize image
                image = skimage.io.imread(input_filepath)
                resized_image = resize(image, output_shape=pargs.resized_dims)
                resized_image = img_as_ubyte(resized_image)

                # Rename and save resized image
                if pargs.rename_images:
                    filename = '{}_{}_{:07d}.png'.format(subset_name, class_name, idx + 1)

                output_filepath = os.path.join(output_class_dir, filename)
                skimage.io.imsave(output_filepath, resized_image)