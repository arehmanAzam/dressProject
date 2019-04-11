import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy  import array,random
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from dataset import load_cached
import cv2
from autoencoders import BodyAutoEncoder,HeadAutoEncoder
import os
import argparse
from prepare_ac_data import PrepareData
import glob
from time import time
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



class TrainingAutoEncoders:
    """
    class containing training tools for auto_encoders
    """
    def __init__(self,total_batch_size,dataset_directory="",pkl_file_name=""):

        self.image_paths_train=[]
        self.cls_train=[]
        self.labels_train=[]
        self.train_batch_size = total_batch_size
        print("Batch size \n")
        print (self.train_batch_size)

        self.dataset_directory=dataset_directory
        self.pkl_file_name=pkl_file_name

        self.images_train_ksplits=[]
        self.labels_train_ksplits=[]

    def load_dataset(self):
        dataset=load_cached(cache_path=self.pkl_file_name,in_dir=self.dataset_directory)
        self.image_paths_train, self.cls_train, self.labels_train = dataset.get_training_set()

    def random_batch(self):
        # Number of images (transfer-values) in the training-set.
        num_images = len(self.image_paths_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=self.train_batch_size,
                               replace=False)

        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = array(self.image_paths_train)[idx]
        y_batch = array(self.labels_train)[idx]

        return x_batch, y_batch
    def make_ksplits(self,ksplits=7):

        x,y=self.random_batch()
        print("Batch size =")
        print (len(x))
        self.images_train_ksplits = np.split(x, ksplits)


    def get_split_images(self,k_split_no=1,total_splits=7,im_size=(224,224,3) ):


        image_paths = self.images_train_ksplits[k_split_no] #start from 1 index
        train_image = np.empty((len(image_paths), im_size[0], im_size[1],im_size[2]))

        for i in range(len(image_paths)):
            image = cv2.imread(image_paths[i])
            if (image is not None):
                resized_image = cv2.resize(image, dsize=(im_size[0], im_size[1]))
                # resized_image_float=im2double(resized_image)
                np_image = np.reshape(resized_image, im_size)
                np_image = np_image.astype('float32')
                train_image[i] = np_image
            else:
                np.delete(image_paths, (i), axis=0)
                np.delete(train_image, (i), axis=0)
            # train_image=np.append(train_image,np_image)
        # del image, resized_image, np_image, image_paths

        if k_split_no == total_splits-1:
            image_paths_val = self.images_train_ksplits[0]
            val_image_label = self.images_train_ksplits[0]
            val_image = np.empty((len(image_paths_val), im_size[0], im_size[1],im_size[2]))
        else:
            image_paths_val = self.images_train_ksplits[k_split_no + 1]
            y_paths_val = self.images_train_ksplits[k_split_no + 1]
            val_image = np.empty((len(image_paths_val), im_size[0], im_size[1],im_size[2]))

        for j in range(len(image_paths_val)):
            image = cv2.imread(image_paths_val[j])
            if (image is not None):
                resized_image = cv2.resize(image, dsize=(im_size[0], im_size[1]))
                # resized_image_float=im2double(resized_image)
                np_image = np.reshape(resized_image, im_size)
                np_image = np_image.astype('float32')
                val_image[j] = np_image
            else:
                np.delete(image_paths_val, (j), axis=0)
                np.delete(val_image, (j), axis=0)
        # np.save('images.npy',train_image_np)
        # del image, resized_image, np_image, image_paths_val

        return train_image,val_image

    def train_autoencoders(self,validation_splits=3,model_train=None,epochs=15,model_save_name='',im_size=(88,88,3)):
        csv_logger = CSVLogger(model_save_name+'.csv',
                                   append=True, separator=',')
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1,
        #                          write_graph=True, write_images=False)
        tensorboard =TrainValTensorBoard(write_graph=False)

        callbacks_list2 = [csv_logger,tensorboard]

        for i in range(epochs):
            self.make_ksplits(ksplits=validation_splits)

            for count in range(validation_splits-1):
                train_images,val_images=self.get_split_images(k_split_no=count+1,total_splits=validation_splits,im_size=im_size)
                model_train.fit(train_images, train_images, epochs=1, batch_size=1,
                                validation_data=(val_images, val_images),callbacks=callbacks_list2)
        model_train.save(model_save_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--xmls", help="path where xmls are existing", type=str,
                        required=True)

    parser.add_argument("--folder_images", help="path where images reside", type=str,
                        required=True)

    parser.add_argument("--target_folder", help="path of folder where the new data will be generated", type=str,
                        required=True)

    parser.add_argument("--epochs", help="no of epochs for each autoencoder  ", type=int, required=True)

    # parser.add_argument("--kfold", help="k for cross k validation ", type=int, required=True)

    args = parser.parse_args()


    #Preparation of the data for the autoencoders

    data_p=PrepareData(folder_xml=args.xmls,folder_images=args.folder_images,folder_target=args.target_folder)
    head_training_Path,ubody_training_Path,lbody_training_Path=data_p.prepare_Images()

    print("Train autoencoder for head \n" )
    total_files=glob.glob(head_training_Path +"training/" + "*.jpg")
    model_instance=HeadAutoEncoder()
    head_encoder, head_decoder,head_autoencoder=model_instance.model_func()
    print (int(len(total_files)))
    training_instance=TrainingAutoEncoders(total_batch_size=int(len(total_files)),dataset_directory=head_training_Path
                                             ,pkl_file_name="head_data.pkl")
    training_instance.load_dataset()
    training_instance.train_autoencoders(validation_splits=3,model_train=head_autoencoder
                                         ,epochs=args.epochs,model_save_name='head_autoEncoder.h5',im_size=(88,88,3))

    print("Train autoencoder for upper body \n")
    total_files = glob.glob(ubody_training_Path+"training/" + "*.jpg")
    model_instance = BodyAutoEncoder()
    ubody_encoder, ubody_decoder, ubody_autoencoder = model_instance.model_func()
    training_instance = TrainingAutoEncoders(total_batch_size=int(len(total_files)),dataset_directory=ubody_training_Path
                                             ,pkl_file_name="ubody_data.pkl")
    training_instance.load_dataset()
    training_instance.train_autoencoders(validation_splits=4, model_train=ubody_autoencoder
                                         , epochs=args.epochs, model_save_name='ubody_autoEncoder.h5',im_size=(128,128,3))

    print("Train autoencoder for lower body \n")
    total_files = glob.glob(lbody_training_Path + "training/" + "*.jpg")
    model_instance = BodyAutoEncoder()
    lbody_encoder, lbody_decoder, lbody_autoencoder = model_instance.model_func()
    training_instance = TrainingAutoEncoders(total_batch_size=int(len(total_files)),
                                             dataset_directory=lbody_training_Path
                                             , pkl_file_name="lbody_data.pkl")
    training_instance.load_dataset()
    training_instance.train_autoencoders(validation_splits=3, model_train=ubody_autoencoder
                                         , epochs=args.epochs, model_save_name='lbody_autoEncoder.h5',im_size=(128,128,3))



