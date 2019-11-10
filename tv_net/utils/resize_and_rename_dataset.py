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
import os
from keras.preprocessing.image import load_img, img_to_array, save_img
from tqdm import tqdm
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_dir', default='/home/ubuntu/data_store/training_value_net/aircraft_7_dataset_original',
                        help='Directory of dataset to resize')
    parser.add_argument('--output_dataset_dir', default='/home/ubuntu/data_store/training_value_net/aircraft_7_dataset_224')
    parser.add_argument('--resized_dims', default=(224, 224),
                        help='Dimensions of resize images')
    parser.add_argument('--rename_images', default=True)
    pargs = parser.parse_args()

    # Create output dir for resized dataset
    if not os.path.isdir(pargs.output_dataset_dir):
        os.makedirs(pargs.output_dataset_dir)
        print('Created directory for new dataset: {}'.format(pargs.output_dataset_dir))
        time.sleep(3)
    else:
        raise ValueError('Output directory already exists - are you sure you want to overwrite?')

    subset_names = [directory for directory in os.listdir(pargs.input_dataset_dir)]
    print('Found dataset with following subsets: {}'.format(subset_names))
    time.sleep(3)

    for subset_name in subset_names:
        # Create output dir for subset
        output_subset_dir = os.path.join(pargs.output_dataset_dir, subset_name)
        if not os.path.isdir(output_subset_dir):
            os.mkdir(output_subset_dir)

        input_subset_dir = os.path.join(pargs.input_dataset_dir, subset_name)
        class_names = [directory for directory in os.listdir(input_subset_dir)]
        print('Subset {} contains following class names: {}'.format(subset_name, class_names))
        time.sleep(3)

        for class_name in class_names:
            # Create output dir for class
            output_class_dir = os.path.join(output_subset_dir, class_name)
            if not os.path.isdir(output_class_dir):
                os.mkdir(output_class_dir)

            print('Starting class {} from subset {}'.format(class_name, subset_name))
            time.sleep(3)
            input_class_dir = os.path.join(input_subset_dir, class_name)

            filename_list = [f for f in os.listdir(input_class_dir)]
            for idx, filename in enumerate(tqdm(filename_list)):
                input_filepath = os.path.join(input_class_dir, filename)

                # Load and resize image
                new_img_array = img_to_array(load_img(input_filepath, target_size=pargs.resized_dims))

                # Rename and save resized image
                if pargs.rename_images:
                    filename = '{}_{}_{:07d}.png'.format(subset_name, class_name, idx + 1)

                output_filepath = os.path.join(output_class_dir, filename)
                save_img(output_filepath, new_img_array)