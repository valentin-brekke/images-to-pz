'''
class Inception():
    """
    Subclass to train a simple point estimate Neural Net photoz
    with the inception model from Pasquet et al. article 
    """
    
    def __init__(self, args, comm=None):
    
    def prepare(self):
        
        img_path = '/global/cfs/cdirs/lsst/groups/PZ/valentin_image_data_temp/img_30k.npy'
        z_path = '/global/cfs/cdirs/lsst/groups/PZ/valentin_image_data_temp/z_30k.npy'
        
        img, z = load(img_path, z_path)
        processing(img, z)
        data = split(img, z)
        
    def train(self):
        """Train the NN model
        """
        model = model_tf2()
        history = train(
        
        import sklearn.neural_network as sknn
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma: no cover
            training_data = self.get_data('input')
        speczs = training_data['redshift']
        print("stacking some data...")
        color_data = make_color_data(training_data, self.config.bands,
                                     self.config.ref_band, self.config.nondetect_val)
        input_data = regularize_data(color_data)
        simplenn = sknn.MLPRegressor(hidden_layer_sizes=(12, 12),
                                     activation='tanh', solver='lbfgs',
                                     max_iter=self.config.max_iter)
        simplenn.fit(input_data, speczs)
        self.model = simplenn
        self.add_data('model', self.model)

'''
