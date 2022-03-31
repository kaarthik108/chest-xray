import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model, Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.densenet import DenseNet121
import warnings
warnings.filterwarnings("ignore")
from model_logging import App_Logger

log_writer = App_Logger()
DIR = os.getcwd() + '/data/chest_xray/'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224, 3)
LOG_FILE = open('logs/train_log.txt', 'a+')

class dense_net:
    def __init__(self):
        self.train_path = DIR + 'train'
        self.test_path = DIR + 'test'
        self.val_path = DIR + 'val'
        self.model_path = os.path.join(os.getcwd(),'models/')
        self.epochs = 1
        self.lr_reduction = None
        self.checkpoint = None
        self.cw = None        
        
    def transform(self):
        log_writer.log(LOG_FILE, 'Data Augmentation..')
        train_image_gen = ImageDataGenerator(
                               rotation_range=20,
                               samplewise_center=True,
                               shear_range =0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               rescale=1./255
                              )

        test_image_gen = ImageDataGenerator(rescale=1./255)
        
        self.train = train_image_gen.flow_from_directory(self.train_path,
                                                         target_size=IMAGE_SIZE[:2],
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary')
        log_writer.log(LOG_FILE, 'Train data samples: {}'.format(self.train.samples))
        
        self.val = train_image_gen.flow_from_directory(self.val_path,
                                                       target_size=IMAGE_SIZE[:2],
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='binary')
        log_writer.log(LOG_FILE, 'Validation data samples: {}'.format(self.val.samples))
        
        self.test = test_image_gen.flow_from_directory(self.test_path,
                                                       target_size=IMAGE_SIZE[:2],
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='binary',
                                                       shuffle=False)
        log_writer.log(LOG_FILE, 'Test data samples: {}'.format(self.test.samples))
        
    def callbacks(self):
        self.lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, min_lr=1e-6, mode='max')
        self.checkpoint = ModelCheckpoint(self.model_path + 'checkpoint/densenet.h5', verbose=1, save_best_only=True)
        self.weights = compute_class_weight('balanced', classes=np.unique(self.train.classes), y=self.train.classes)
        self.cw = dict(zip( np.unique(self.train.classes), self.weights))
        
    def model(self):
        log_writer.log(LOG_FILE, 'Creating model..')
        transfer = DenseNet121(include_top=False, weights='imagenet', input_shape=IMAGE_SIZE)
        
        for layer in transfer.layers:
            layer.trainable = False
            
        x = Flatten()(transfer.output)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=transfer.input, outputs=out)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        log_writer.log(LOG_FILE, 'Model created..')
        return model
    
    def train_model(self):
        self.transform()
        self.callbacks()
        self.model = self.model()
        log_writer.log(LOG_FILE, 'Training model..')
        self.model.fit_generator(self.train,
                                 validation_data=self.val,
                                 epochs=self.epochs,
                                 class_weight=self.cw,
                                 callbacks=[self.lr_reduction, self.checkpoint],
                                 steps_per_epoch=len(self.train),
                                 validation_steps=len(self.val)
        )
        
    def save_model(self):
        self.train_model()
        self.model.save(self.model_path + 'densenet.h5')
        log_writer.log(LOG_FILE, 'Model saved..')
        loss, accuracy_score = self.model.evaluate_generator(self.test, steps=self.test.samples)
        self.model.save_weights(self.model_path + 'weights.h5')
        log_writer.log(LOG_FILE, 'loss: {}, Accuracy: {}'.format(loss, accuracy_score))
        
        
        
if __name__ == '__main__':
    dense_net = dense_net()
    dense_net.save_model()

                                                         
                        