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
    Preprocess the images by scaling them with ther MAD
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
    out_frac = np.sum(np.abs(dz) > 0.05) / float(len(z))
    
    return dz, pred_bias, smad, out_frac

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


def plot_results(sz, pz, pred_bias, out_frac, smad, save=False, path=''):
    plt.hist2d(sz, pz, 150, range=[[0,0.6],[0,0.6]], cmap='gist_stern', cmin=1e-3); 
    plt.gca().set_aspect('equal');
    plt.plot([0,0.7],[0,0.7],color='black')
    plt.xlabel('Spectroscopic Redshift' , fontsize=14)
    plt.ylabel('Predicted Redshift', fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label('Samples')
    number = 0.1
    plt.text(0.1, 0.45, '$ \Delta_z =$' + str(round(pred_bias, 4)) + '\n' 
             + '$\eta =$' + str(round(out_frac*100, 2)) + '%' + '\n'
             + '$\sigma_{MAD}=$'+ str(round(smad, 4)),
             bbox=dict(facecolor='w', alpha=0.8, pad=8), fontsize=14);
    if save:
        plt.savefig(path, dpi=150)

        
class Inform_Inception(CatInformer):
    """
    Subclass to train a Neural Net photoz estimator
    with the inception model from Pasquet et al. article 
    """
    name = 'Inform_Inception'
    config_options = CatInformer.config_options.copy()
    config_options.update(trainfrac=Param(float, 0.75, msg="fraction of training and validation data"),
                          epoch=Param(int, 5),
                          hdf5_groupname=SHARED_PARAMS)
             
                
    def __init__(self, args, comm=None):
        CatInformer.__init__(self, args, comm=comm)
        
    def run(self):
        """
        Train a inception NN on a fraction of the training data
        """
        # Not sure what .get_data('input') is ?
        train_data = self.get_data('input')
        
        # Recover img and z from the array
        z = train_data[:, 0]
        img = train_data[:, 1:].reshape((-1, 64, 64, 5))
        
        img, scaling = processing(img)
        self.scaling = scaling
        ntrain = round(img.shape[0] * self.config.trainfrac)
        img_train = img[:ntrain]
        img_val = img[ntrain:]
        z_train = z[:ntrain]
        z_val= z[ntrain:]
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
    name = 'Inception_Estimator'
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
        testing_data = self.get_data('input')
        
        z = testing_data[:, 0]
        img = testing_data[:, 1:].reshape((-1, 64, 64, 5))
        
        # Process test images same way as training set
        img_test = np.arcsinh(img / self.scaling / 3.)
        preds = self.nnmodel.predict(img_test)
        self.pred = preds.squeeze()
        
    def finalize(self):
        testing_data = self.get_data('input')
        z = testing_data[:, 0]
        dz, pred_bias, smad, out_frac = metrics(z, self.pred)
        print_metrics(pred_bias, smad, out_frac)
        plot_results(z, self.pred, pred_bias, out_frac, smad)
