import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

def plotM(*args,**kwargs):
    t = kwargs.get("t", True)
    f, axs = plt.subplots(1,len(args), figsize=(6*len(args), 6), dpi=100)
    if len(args)>1:
        for i in range(len(args)):
            if t:   c= axs[i].imshow(tf.transpose(args[i]))
            else:   c= axs[i].imshow(args[i])
            plt.colorbar(c, ax=axs[i])
    else:
        if t:  c= axs.imshow(tf.transpose(args[0]), **kwargs)
        else:  c= axs.imshow(args[0], **kwargs)
        plt.colorbar(c, ax=axs)

def plot3D(coord, mask):
    fig = plt.figure(figsize=(25, 25),dpi=150)
    gs = fig.add_gridspec(1,coord.shape[0] )
    for i in range(coord.shape[0]):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ncl =tf.cast(tf.reduce_sum(mask[i]),tf.int32).numpy()
        c = np.arange(ncl)
        im = ax.scatter(coord[i,0:ncl,0], coord[i,0:ncl,1], coord[i,0:ncl,2],c=c, s=80)


def dynamic_window(eta):
    aeta = abs(eta)

    if aeta >= 0 and aeta < 0.1:
        deta_up = 0.075
    if aeta >= 0.1 and aeta < 1.3:
        deta_up = 0.0758929 -0.0178571* aeta + 0.0892857*(aeta**2) 
    elif aeta >= 1.3 and aeta < 1.7:
        deta_up = 0.2
    elif aeta >=1.7 and aeta < 1.9:
        deta_up = 0.625 -0.25*aeta
    elif aeta >= 1.9:
        deta_up = 0.15

    if aeta < 2.1: 
        deta_down = -0.075
    elif aeta >= 2.1 and aeta < 2.5:
        deta_down = -0.1875 *aeta + 0.31875
    elif aeta >=2.5:
        deta_down = -0.15
        
    if aeta < 1.9:
        dphi = 0.6
    elif aeta >= 1.9 and aeta < 2.7:
        dphi = 1.075 -0.25 * aeta
    elif aeta >= 2.7:
        dphi = 0.4
            
    return deta_up, deta_down, dphi
