# Training-ValueNet

**A simple tool for automatically cleaning your weakly-supervised data for image classification.**

- Labelling large image classification datasets can be time-consuming, expensive and downright boring!

- It is often possible, however, to automatically infer weak labels for your dataset at a fraction of the cost 

- Unfortunately the presence of mislabeled training examples in weakly-supervised data can severely detriment 
classification performance.

-  Training-ValueNet solves this problem by automatically identifying and removing mislabeled examples from your dataset.

![Alt text](img/training_value_examples.jpg?raw=true "Optional Title")

Table of Contents
=================


  * [How it works](#how-it-works)
  * [Installation](#installation)
  * [Expected Directory Structure](##expected-directory-structure)
  * [Basic Configuration](#basic-configuration)
  * [Running the algorithm](#running-the-algorithm)
  * [Expected Output](#expected-output)
  * [FAQ's](#faqs)
  * [Support](#support)




## How it works

Training-ValueNet works by estimating the  _training-value_ that each image offers to the learning process. Specifically,
we estimate the **expected immediate improvement in validation performance** (loss) that we would observe as a result of training
on that individual training example at some randomly chosen time-step. An image with a negative training-value
 would causes a net detriment to performance which is most probably due to it being mislabeled. All such examples are
therefore discarded leaving only the clean training data that you desire.


**_You must provide..._**  

For the algorithm to work you must provide weakly-labeled training set and a **small** validation set of correctly labeled images. Somewhere between 50 - 100 images per class
is fine for this validation set but the **must** be correctly labeled.   

&nbsp;  


 **_The algorithm includes the following steps..._**  

Step 1) A CNN image classifier is first trained on the entire training set so that we can extract a reasonable feature
vector for each image (usually the output of the final conv layer).

Step 2) Using these extracted features, a small MLP classifier is trained on a randomly chosen **subset** of training
examples from each class. The MLP is trained from scratch over M independent episodes using a batch-size of one with a 
fixed learning rate and no momentum (i.e. vanilla SGD). At every time-step the we record the immediate improvement in validation
loss.

Step 3) Once all M episodes are complete we estimate the training-value of each image as simply the mean improvement in 
validation loss that was observed when that image was trained on. 

Step 4) Using these direct estimates as targets, we train a training-value approximation network (Training-ValueNet) for each
class to predict the training-value of any image labeled as being in that class directly from it's extracted feature vector. 
Each Training-ValueNet is simply a MLP regressor. 

Step 5) The Training-ValueNet for each class is used to predict the training-value of all remaining training examples in the training set. 
Finally, all images with a negative predicted training-value are discarded from the training set. 


**[If you want a more detailed explanation you can read our paper here.](http://empslocal.ex.ac.uk/people/staff/np331/publications/SmythEtAl2019.pdf)**


## Installation

You can clone this repo with the following command:

`git clone https://github.com/lukasmyth96/Training-ValueNet.git`

The requirements are minimal but can be installed using:

`pip install -r requirements.txt`

## Expected Directory Structure

The default implementation assumes that your images are stored in a directory with the following structure:

\|--- dataset  
\| &nbsp;&nbsp;&nbsp; \|-- train  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\|-- class_1  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\| &nbsp; .  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\| &nbsp; .  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\|-- class_K  
\| &nbsp;&nbsp;&nbsp; \|-- val  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\|-- class_1  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\| &nbsp; .  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\| &nbsp; .  
\| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\|-- class_K 

If your data is not structured like that and you really don't want to change it you'll need to edit the `load_dataset` method of  `/tv_net/dataset/Dataset` class.



## Basic Configuration

If you just want to get started with label cleaning asap and your data directory is in the correct format, all you'll need to do is alter a few details in the config class which 
can be found in `/tv_net/config.py`.

You will need to change:

- `TRAIN_DATASET_DIR` - the path to your training set directory
- `VAL_DATASET_DIR` -  the path to your validation set directory
- `OUTPUT_DIR` - to be the path in which to store all output (cleaned dataset etc.)
- `NUM_CLASSES` - number of classes in your dataset
- `IMG_DIMS` - dimensions of your images - a tuple (width, height)

**All other parameters should be left as the default unless you have read the [paper](http://empslocal.ex.ac.uk/people/staff/np331/publications/SmythEtAl2019.pdf) and understand what they do.**

## Running the algorithm

Once you have set your configuration correctly you can perform the label cleaning by simply running the following:

`python ./tv_net/clean_your_dataset.py`


## Expected Output

If the algorithm runs successfully you should see the following in the `OUTPUT_DIR` you specified:

- `clean_training_examples/` - directory containing the cleaned training set (i.e. all examples with positive predicted training-value)
- `dirty_training_examples` - directory containing all discarded training examples (i.e. those with a negative predicted training-value)
- `baseline_classifier_weights/` - directory containing the weights for the baseline classifier from the checkpoint with the highest val acc. 
- `training_value_networks/` - directory containing the weights for the trained Training-ValueNet for each class.
- `visualizations/` - directory containing some useful visualizations including a T-SNE of the extracted image features and histograms of the predicted training values for each class.
 

## FAQ's

**How long does the algorithm take to run?**

The run-time of the algorithm scales linearly with the number of examples in your training set but is proportional to the
square of the number of classes. This means that the algorithm is most suitable for datasets with large numbers of examples but
relatively few classes (i.e. < 10). 

 **We therefore do NOT recommend using this algorithm for more than ~20 classes unless you have a lot of compute power available.**

The run-time is also proportional to a number of parameters which can be adjusted in the config including:

- `TRAIN_SUBSET_NUM_PER_CLASS`
- `VAL_SUBSET_NUM_PER_CLASS`
- `MC_EPISODES`
- `MC_EPOCHS`

Be aware however that any significant decrease in these values may impair label cleaning performance.

**Will I need a GPU to use the algorithm?**

No. The most time-consuming phase of the algorithm in the Monte Carlo estimation phase in which a small MLP is repeatedly trained but with
a batch size of one. Using a GPU will not lead to a significant speed up here.

If your dataset is very large, however, it may take a while to train the initial baseline classifier. In this case it
may be better to train the baseline classifier separately on a GPU. You can then run the remainder of the algorithm on a CPU, skipping the first
stage by specifying the following in the config:

- `TRAIN_BASELINE_CLF = False`
- `LOAD_BASELINE_CLF = True`
- `BASELINE_CLF_WEIGHTS = <path to trained classifier weights>`

**What image classifier is used? How can I change it?**

By default an ImageNet pre-trained ResNet50 CNN is used as the baseline classifier. 

You can easily change this in the `_build_feature_extractor` method of the `tv_net.classifier.Classifier` class but be aware
that other CNNs may require certain image dimensions and preprocessing. 


## Support

If you have any questions about the algorithm or how to use this implementation I would be happy to hear from you - you can email me at [lukasmyth@msn.com]()

If you have find a bug or have a specific issue with the code (not unlikely at this stage) feel free to open an issue on the repo.

