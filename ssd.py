"""SSD class to build, train, eval an SSD network

python ssd.py -e --restore-weights=weights/ResNetv2.h5 --image-file=assets/trial.jpg
"""

import tensorflow as tf
assert tf.version.VERSION.startswith('2.')

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber

import utils
import labels

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from dataset import DataGenerator
from labels import build_label_dictionary
from boxes import show_boxes
from resnet import resnet
from common_utils import print_log

import numpy as np

def conv2d(inputs, filters=32, kernel_size=3, strides=1, name=None):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
        kernel_initializer='he_normal',name=name, padding='same')
    return conv(inputs)


def conv_layer(inputs, filters=32, kernel_size=3, strides=1, use_maxpool=True, 
    postfix=None, activation=None):
    x = conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, name='conv'+postfix)
    x = BatchNormalization(name="bn"+postfix)(x)
    x = ELU(name='elu'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool'+postfix)(x)
    return x

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
     conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', 
        kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet(input_shape, depth, n_layers=4):
    """Adapted from Rowen's implementation"""
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides,
                    activation=activation, batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)

            if res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides,
                    activation=None, batch_normalization=False)
            x = Add()([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    conv = AveragePooling2D(pool_size=4, name='pool1')(x)
    outputs = [conv]
    prev_conv = conv
    n_filters = 64

    for i in range(n_layers - 1):
        postfix = "_layer" + str(i+2)
        conv = conv_layer(prev_conv, n_filters, kernel_size=3, strides=2, use_maxpool=False, postfix=postfix)
        outputs.append(conv)
        prev_conv = conv
        n_filters *= 2
    
    return Model(inputs=inputs, outputs=outputs)


def build_resnet(input_shape, n_layers=4, n=6):
    depth = n * 9 + 2
    model = resnet(input_shape=input_shape, depth=depth, n_layers=n_layers)
    return model


def mask_offset(y_true, y_pred): 
    """Pre-process ground truth and prediction data"""
    offset = y_true[..., 0:4]
    mask = y_true[..., 4:8]
    pred = y_pred[..., 0:4]
    offset *= mask
    pred *= mask
    return offset, pred


def smooth_l1_loss(y_true, y_pred):
    offset, pred = mask_offset(y_true, y_pred)
    return Huber()(offset, pred)


def build_ssd(input_shape, backbone, n_layers=4, n_classes=4, aspect_ratios=(1, 2, 0.5)):
    """Build the SSD model from the backbone"""
    n_anchors = len(aspect_ratios) + 1

    inputs = Input(shape=input_shape)
    base_outputs = backbone(inputs)

    outputs = []
    feature_shapes = []
    out_cls = []
    out_off = []

    for i in range(n_layers):

        conv = base_outputs if n_layers==1 else base_outputs[i]
        name = "cls" + str(i+1)
        classes  = conv2d(conv, n_anchors*n_classes, kernel_size=3, name=name)

        name = "off" + str(i+1)
        offsets  = conv2d(conv, n_anchors*4, kernel_size=3, name=name)
        shape = np.array(K.int_shape(offsets))[1:]
        feature_shapes.append(shape)
        name = "cls_res" + str(i+1)
        classes = Reshape((-1, n_classes), name=name)(classes)
        name = "off_res" + str(i+1)
        offsets = Reshape((-1, 4), name=name)(offsets)
        offsets = [offsets, offsets]
        name = "off_cat" + str(i+1)
        offsets = Concatenate(axis=-1, name=name)(offsets)
        out_off.append(offsets)

        name = "cls_out" + str(i+1)
        classes = Activation('softmax', name=name)(classes)
        out_cls.append(classes)

    if n_layers > 1:
        name = "offsets"
        offsets = Concatenate(axis=1, name=name)(out_off)
        name = "classes"
        classes = Concatenate(axis=1, name=name)(out_cls)
    else:
        offsets = out_off[0]
        classes = out_cls[0]

    outputs = [classes, offsets]
    model = Model(inputs=inputs, outputs=outputs, name='ssd_head')

    return n_anchors, feature_shapes, 


class SSD:
    def __init__(self, args):
        self.args = args
        self.ssd = None
        self.train_generator = None
        self.build_model()


    def build_model(self):
        self.build_dictionary()
        self.input_shape = (self.args.height, self.args.width, self.args.channels)
        self.backbone = self.args.backbone(self.input_shape, n_layers=self.args.layers)

        anchors, features, ssd = build_ssd(self.input_shape, self.backbone, n_layers=self.args.layers, 
            n_classes=self.n_classes)

        self.n_anchors = anchors
        self.feature_shapes = features
        self.ssd = ssd


    def build_dictionary(self):
        path = os.path.join(self.args.data_path, self.args.train_labels)

        self.dictionary, self.classes = build_label_dictionary(path)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))


    def build_generator(self):
        self.train_generator = DataGenerator(args=self.args, 
            dictionary=self.dictionary, n_classes=self.n_classes, 
            feature_shapes=self.feature_shapes, n_anchors=self.n_anchors, shuffle=True)


    def train(self):
        """Train an ssd network."""
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)

        loss = ['categorical_crossentropy', smooth_l1_loss]
        self.ssd.compile(optimizer=optimizer, loss=loss)

        save_dir = os.path.join(os.getcwd(), self.args.save_dir)
        model_name = self.backbone.name
        model_name += '-' + str(self.args.layers) + "layer"
        model_name += "-smooth_l1"

        if self.args.threshold < 1.0:
            model_name += "-extra_anchors" 

        model_name += "-" 
        model_name += self.args.dataset
        model_name += '-{epoch:03d}.h5'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)
        callbacks = [checkpoint]
        self.ssd.fit(self.train_generator, use_multiprocessing=False, callbacks=callbacks, epochs=self.args.epochs)


    def detect_objects(self, image):
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        return image, classes, offsets


    def evaluate(self, image_file=None, image=None):
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True

        image, classes, offsets = self.detect_objects(image)
        class_names, rects, _, _ = show_boxes(args, image, classes, offsets, self.feature_shapes, show=show)
        return class_names, rects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--layers", default=4)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--backbone", default=build_resnet)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--image-file")
    parser.add_argument("--restore-weights")
    parser.add_argument("--evaluate", default=False, action='store_true', help=help_)
    
    parser = ssd_parser()
    args = parser.parse_args()
    ssd = SSD(args)

    if args.evaluate:
        ssd.evaluate(image_file=args.image_file)
            
    if args.train:
        ssd.train()
