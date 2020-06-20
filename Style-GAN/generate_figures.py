import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from metrics import metric_base
import config

def randomLatents(seed1,seed2,morphingFactor,shape):
    # 0 < morphingFactor < 1, 0->latents0, 1-> latents1
    latents0 = np.random.RandomState(seed1).randn(1, shape)
    latents1 = np.random.RandomState(seed2).randn(1, shape)
    latents = latents0+(latents1-latents0)*morphingFactor
    return latents

if __name__ == "__main__":
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    _G, _D, Gs = pickle.load(open("network-snapshot-003285.pkl", "rb"))

    # Print network details.
    Gs.print_layers()
    
    '''
    TRUNCATION = 0.7
    SEED1_1 = [4722,2111,3333,1357,2589]
    SEED1_2 = [5944,3111,3766,3421,4171]
    SEED2_1 = [3611,4111,1434,2267,4102]
    SEED2_2 = [4333,5111,3456,4588,4990]
    MORPH_FACTOR = 0.84615

    for i in range(5):
        # Pick latent vector.
        latents_1 = randomLatents(SEED1_1[i],SEED1_2[i],MORPH_FACTOR,Gs.input_shape[1])
        
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_1, None, truncation_psi=TRUNCATION, randomize_noise=True, output_transform=fmt)
    
        # Save image.
        result_dir = 'results/'
        os.makedirs(result_dir, exist_ok=True)
        png_filename = os.path.join(result_dir, 'photo-'+str(i)+'-1.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

        latents_2 = randomLatents(SEED2_1[i],SEED2_2[i],MORPH_FACTOR,Gs.input_shape[1])
        
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_2, None, truncation_psi=TRUNCATION, randomize_noise=True, output_transform=fmt)
    
        # Save image.
        result_dir = 'results/'
        os.makedirs(result_dir, exist_ok=True)
        png_filename = os.path.join(result_dir, 'photo-'+str(i)+'-2.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        
        latents_3 = latents_1 + latents_2
        
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_3, None, truncation_psi=TRUNCATION, randomize_noise=True, output_transform=fmt)
    
        # Save image.
        result_dir = 'results/'
        os.makedirs(result_dir, exist_ok=True)
        png_filename = os.path.join(result_dir, 'photo-'+str(i)+'-3.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    '''
    
    for i in range(100):
        # Pick latent vector.
        TRUNCATION = 0.7
        SEED1 = 1200 + 40 * i;
        SEED2 = 1000 + 50 * i;
        MORPH_FACTOR = 0.84615
        latents_1 = randomLatents(SEED1,SEED2,MORPH_FACTOR,Gs.input_shape[1])
        
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents_1, None, truncation_psi=TRUNCATION, randomize_noise=True, output_transform=fmt)
    
        # Save image.
        result_dir = 'results/'
        os.makedirs(result_dir, exist_ok=True)
        png_filename = os.path.join(result_dir, 'photo-'+str(i)+'-1.jpg')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)