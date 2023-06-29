import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex


def metrics(z, pred):
    """
        Returns the desired performance metrics.
        
        Arguments:
            z (numpy array): true redschift
            pred (numpy array): the redschift prediction
             
        Returns:
            dz (ndarray): Residuals for every test image.
            pred_bias (float): The prediction bias, mean of dz.
            smad (float):The MAD deviation.
            out_frac (float): The fraction of outliers.
    """
    
    dz = (pred - z) / (1 + z)
    pred_bias = np.mean(dz)
    MAD = np.median(np.abs(dz - np.median(dz)))
    smad = 1.4826 * MAD
    out_frac_1 = np.sum(np.abs(dz) > 0.05) / float(len(z))
    
    return dz, pred_bias, smad, out_frac_1


def print_metrics(pred_bias, smad, out_frac):
    """
    Prints the prediction bias, outlier fraction and sigma MAD
    """
    print(f'Prediction bias: {pred_bias:.4f}')
    display(Latex(f'$\sigma MAD$: {smad:.4f}'))
    print(f'Outlier fraction: {out_frac*100:.2f}%')

    
def history_plot(history, title, save=False, path=''):
    plt.figure(figsize=(8,4))
    epoch = np.arange(1, len(history.history['loss'][1:])+1)
    plt.plot(epoch, history.history['loss'][1:])
    plt.plot(epoch, history.history['val_loss'][1:])
    plt.title(title, fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['train', 'validation'], loc='upper right', prop = { "size": 16 })
    if save:
        plt.savefig(path, dpi=150);
        

def plot_results(sz, pz, pred_bias, out_frac, smad, title, save=False, path=''):
    plt.hist2d(sz, pz, 150, range=[[0,0.6],[0,0.6]], cmap='gist_stern', cmin=1e-3); 
    plt.gca().set_aspect('equal');
    plt.plot([0,0.7],[0,0.7],color='black')
    plt.xlabel('Spectroscopic Redshift' , fontsize=14)
    plt.ylabel('Predicted Redshift', fontsize=14)
    plt.title(title, fontsize=18)
    cbar = plt.colorbar()
    cbar.set_label('Samples')
    number = 0.1
    plt.text(0.1, 0.45, '$ \Delta_z =$' + str(round(pred_bias, 4)) + '\n' 
             + '$\eta =$' + str(round(out_frac*100, 2)) + '%' + '\n'
             + '$\sigma_{MAD}=$'+ str(round(smad, 4)),
             bbox=dict(facecolor='w', alpha=0.8, pad=8), fontsize=14);
    if save:
        plt.savefig(path, dpi=150)