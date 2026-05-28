#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#CNN based models require Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
#FFNN requires Flatten, Dense
from keras.layers import Flatten, Dense
from keras import backend as K
#import collections
import os
import pandas as pd

#import utils
#Early stopping is required when systme realizes that there is no improvement after ceratin epochs
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import PIL.Image
os.getcwd()

from google.colab import drive
#drive.mount('/content/drive')
os.chdir('/content')

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                            prepare_full_dataset_for_flow(
                            train_dir_original='train',
                            test_dir_original='test',
                            target_base_dir='target')


#Convert all images to standard width and height
img_width, img_height = 150, 150
epochs = 2
batch_size = 20
