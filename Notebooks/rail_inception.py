"""
First Implementation of Incpetion Estimator
"""

import numpy as np
import copy

from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer

from rail.evaluation.metrics.cdeloss import CDELoss
from rail.core.common_params import SHARED_PARAMS

import pandas as pd
import qp

from scipy.stats import median_abs_deviation
from model_inception import *
from tools import *



def processing(img):
    '''
    Preprocesses the images by normalising them sacling 
    '''
    scaling = []
    for i in range(img.shape[-1]):
        sigma = 1.4826*median_abs_deviation(img[...,i].flatten())
        scaling.append(sigma)

    img = np.arcsinh(img / scaling / 3.)
    
    return img, scaling

def train(model, epoch, img_train, img_val, z_train, z_val, lr=0.001, decay=True):
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    LEARNING_RATE=lr
    
    LEARNING_RATE_EXP_DECAY=0.9
    if not decay:
        LEARNING_RATE_EXP_DECAY = 1
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch)
    
    history = model.fit(x = img_train, 
              y = z_train,
              batch_size = 64,
              validation_data=(img_val, z_val),
              steps_per_epoch=len(z_train)//64,
              epochs=epoch,
              callbacks=[lr_decay])
    
    return history, model

def learning_curves(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show();
    
def metrics(z, pred):
    dz = (pred - z) / (1 + z)
    pred_bias = np.mean(dz)
    MAD = np.median(np.abs(dz - np.median(dz)))
    smad = 1.4826 * MAD
    out_frac_1 = np.sum(np.abs(dz) > 0.05) / float(len(z))
    #out_frac_2 = np.sum(dz > 5*smad) / float(len(z))
    
    return dz, pred_bias, smad, out_frac_1


def print_metrics(pred_bias, smad, out_frac):
    print(f'Prediction bias: {pred_bias:.4f}')
    display(Latex(f'$\sigma MAD$: {smad:.4f}'))
    print(f'Outlier fraction: {out_frac*100:.2f}%')

    
def plot_result(z, preds):
    '''
    Plots the predictions compared to the true redshift
    '''
    
    plt.hist2d(z, preds, 64, range=[[0,0.7],[0,0.7]], cmap='gist_stern'); 
    plt.gca().set_aspect('equal');
    plt.plot([0,0.7],[0,0.7],color='r')
    plt.xlabel('Spectroscopic Redshift')
    plt.ylabel('Predicted Redshift');
    

class Inform_Inception(CatInformer):
    """
    Subclass to train a simple point estimate Neural Net photoz
    with the inception model from Pasquet et al. article 
    """
    name = 'Inform_Inception'
    config_options = CatInformer.config_options.copy()
    config_options.update(trainfrac=Param(float, 0.75, msg="fraction of training and validation data"),
                          epoch=Param(int, 1),
                          hdf5_groupname=SHARED_PARAMS)
             
                
    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        # Not sure what .get_data('input') is ?
        training_data = self.get_data('input')
        
        img, scaling = processing(training_data[0])
        print(scaling)
        self.scaling = scaling
        #img = training_data[0]
        ntrain = round(training_data[0].shape[0] * self.config.trainfrac)
        img_train = img[:ntrain]
        img_val = img[ntrain:]
        z_train = training_data[1][:ntrain]
        z_val= training_data[1][ntrain:]
        print(f"Split into {len(z_train)} training and {len(z_val)} validation samples")
        
        model = model_tf2()
        epoch=self.config.epoch
        print("Model training:")
        history, model = train(model, epoch, img_train, img_val, z_train, z_val, lr=0.001, decay=True) 
        learning_curves(history)
        self.model = dict(nnmodel=model, scale=scaling)
        self.add_data('model', self.model)
        
        
class Inception(CatEstimator):
    """Inception estimator
    """
    name = 'KNearNeighPDF'
    config_options = CatEstimator.config_options.copy()
    config_options.update()
    
    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization """
        
        self.nnmodel = None
        self.scale = None
        CatEstimator.__init__(self, args, comm=comm)
        
    
    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is not None:
            self.nnmodel = self.model['nnmodel']
            self.scaling = self.model['scale']

    def run(self):
        test_data = self.get_data('input')
        # Process test images same way as training set
        img_test = np.arcsinh(test_data[0] / self.scaling / 3.)
        preds = self.nnmodel.predict(img_test)
        #preds_non_scale = self.nnmodel.predict(test_data[0])
        self.pred = preds.squeeze()
        #self.pred_non_scale = preds_non_scale.squeeze()
        
    def finalize(self):
        test_data = self.get_data('input')
        dz, pred_bias, smad, out_frac = metrics(test_data[1], self.pred)
        print_metrics(pred_bias, smad, out_frac)
        plot_result(test_data[1], self.pred)
        