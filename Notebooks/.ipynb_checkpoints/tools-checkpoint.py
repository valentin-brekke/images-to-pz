import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex


def metrics(z, pred):
    """
        Returns the desired performance metrics.
        
        Arguments:
            z (numpy model): true redschift
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
    #out_frac_2 = np.sum(dz > 5*smad) / float(len(z))
    
    return dz, pred_bias, smad, out_frac_1


def print_metrics(pred_bias, smad, out_frac):
    print(f'Prediction bias: {pred_bias:.4f}')
    display(Latex(f'$\sigma MAD$: {smad:.4f}'))
    print(f'Outlier fraction: {out_frac*100:.2f}%')