# Notebooks overview

## Model Comparison

1. Inception_Pasquet

    Main notebook with the inception model, organised in different parts:
    - Regular Training
    - Training with an additional feature: Galactic Reddening
    - Using the model to output probability distribution and not only point estimates
    
    For the moment, the most important model is the vanila training from part one.
    

2. Convolutional_Models
    
    Notebook that summs up the results from training with CNN, ResNet and Densenet. 
    None of these models would be deployed but they represent a good point of comparaison.

3. Torch-ViT
    
    Same as 2. Just a comparaison to see how Vision Transformers could perfom. The analysis for this kind of model is rather quick here with a single test with the model from [this open vit-pytorch model.](https://github.com/lucidrains/vit-pytorch) 
    

## Towards an insertion of the Inception model into RAIL Development stage 1

1. rail_inception.py
   Here is the inform and estimator class that follow the RAIL pipeline architecture

2. rail_incep.ipynb
   This notebook shows how to call and use those two classes.


## Additional files

1. Tools
   Contains different useful functions used in all notebooks
3. Inception_Pasquet
   Contains the code for the main Inception model
