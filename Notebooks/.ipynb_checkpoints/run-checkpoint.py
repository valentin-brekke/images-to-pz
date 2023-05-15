from scipy.stats import median_abs_deviation
from model_inception import *
from tools import *

def load(img_path, z_path):
    '''
    Loads numpy files
    '''
    img = np.load(img_path)
    z = np.load(z_path)
    return img, z

def processing(img):
    '''
    Preprocesses the images by normalising them sacling 
    '''
    scaling = []
    for i in range(img.shape[-1]):
        sigma = 1.4826*median_abs_deviation(img[...,i].flatten())
        scaling.append(sigma)

    img = np.arcsinh(img / scaling / 3.)
    
def split(img, z):
    '''
    Splits into train, validation and test set with respectively 50%, 25% and 25% and the data
    Returns: dictionnary whose values are a list of two elements as follows: [img, z]
    '''
    n = img.shape[0]
    data = {}
    data['train'] = [img[:int(n*.5),...], z[:int(n*.5)]]
    data['val'] = [img[int(n*.5):int(n*.75),...], z[int(n*.5):int(n*.75)]]
    data['test'] = [img[int(n*.75):,...], z[int(n*.75):]]
    return data

def train(model, epoch, data, lr=0.001, decay=True):
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    LEARNING_RATE=lr
    
    LEARNING_RATE_EXP_DECAY=0.9
    if not decay:
        LEARNING_RATE_EXP_DECAY = 1
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch)
    
    history = model.fit(x = data['train'][0], 
              y = data['train'][1],
              batch_size = 64,
              validation_data=(data['val'][0], data['val'][1]),
              steps_per_epoch=len(data['train'][0])//64,
              epochs=epoch,
              callbacks=[lr_decay])
    
    return history, model


def evaluate(model, data):
    '''
    Evalutes the model on the test set 
    Plots the predictions compared to the true redshift
    Prints the test metrics
    '''
    preds = model.predict(data['test'][0]).squeeze()

    plt.hist2d(data['test'][1], preds, 64, range=[[0,0.7],[0,0.7]], cmap='gist_stern'); 
    plt.gca().set_aspect('equal');
    plt.plot([0,0.7],[0,0.7],color='r')
    plt.xlabel('Spectroscopic Redshift')
    plt.ylabel('Predicted Redshift');
    
    dz, pred_bias, smad, out_frac = metrics(data['test'][1], preds)
    print_metrics(pred_bias, smad, out_frac)
    
    return preds