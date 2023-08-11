import numpy as np
from skimage.feature import texture

def quant(img, level=6):
    img = np.asarray(img) // (256 / level)
    return img.astype(int)

def calc_comat(patch, level=6, angles=[0.5, 0.75, 1, 1.25], mean=True):
    angles = [np.pi*x for x in angles]
    glcm = texture.graycomatrix(patch, distances=[1], angles=angles, levels=level)
    if mean:
        glcm = np.expand_dims(glcm.mean(axis=3), 3)
    co = texture.graycoprops(glcm, 'contrast')
    en = texture.graycoprops(glcm, 'energy')
    return np.array([co[0], en[0]])

def calc_scoot(real, fake, level=6, N_blocks=4):
    assert real.size == fake.size
    x_dim = np.floor(real.size[0]/4).astype(int) 
    y_dim = np.floor(real.size[1]/4).astype(int) 

    real_patches = []
    fake_patches = []
    for i in range(N_blocks):
        for j in range(N_blocks):
            real_patches.append(quant(real.crop((x_dim*i, y_dim*j, x_dim*(i+1), y_dim*(j+1))), level=level))
            fake_patches.append(quant(fake.crop((x_dim*i, y_dim*j, x_dim*(i+1), y_dim*(j+1))), level=level))

    scores = []
    for i in range(len(real_patches)):
        real_stat = calc_comat(real_patches[i], level=level)
        fake_stat = calc_comat(fake_patches[i], level=level)
        score = 1 / (1 + np.linalg.norm(real_stat - fake_stat, ord=2))
        scores.append(score)

    return scores, np.mean(scores)