import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd


def view_images(images, colorbar = True, axis_off = False, save=''):
    ''' Function for displaying a list of images (length should be more than >1) '''

    import numpy as np
    plt.rcParams['figure.figsize'] = [15,12]
    
    fig, axes = plt.subplots(ncols=len(images), sharex=True, sharey=True, constrained_layout = True)
    
    for idx,image in enumerate(images):
        im = axes[idx].imshow(np.transpose(image.numpy(), (1, 2, 0)), cmap='gray')

        if colorbar:
            plt.colorbar(im, ax=axes[idx], orientation='horizontal', pad=0.01) 

    if axis_off:
        for ax in axes:
           ax.axis('off')
    
    plt.show()


# def view_recon(img, recon):
#     ''' Function for displaying an image (as a PyTorch Tensor) and its 
#         reconstruction also a PyTorch Tensor 
#     '''
#     fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
#     axes[0].imshow(img.numpy().squeeze())
#     axes[1].imshow(recon.data.numpy().squeeze())
#     for ax in axes:
#         ax.axis('off')
#         ax.set_adjustable('box-forced')


def view_pair(image1, image2, colorbar = True, axis_off = False, save=''):
    ''' Function for displaying an image (as a PyTorch Tensor) and its 
        reconstruction also a PyTorch Tensor 
    '''
    plt.rcParams['figure.figsize'] = [15,12]
    
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, constrained_layout = True)
    
    im1 = axes[0].imshow(np.transpose(image1.numpy(), (1, 2, 0)), cmap='gray')
    im2 = axes[1].imshow(np.transpose(image2.numpy(), (1, 2, 0)), cmap='gray')
   
    if colorbar:
        plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.01) 
        plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.01)

    if axis_off:
        for ax in axes:
           ax.axis('off')
    if save:
        plt.savefig(f'./saved_images/{save}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

        
def plot_train_val_loss(stats):
    stats_df       = pd.DataFrame(stats)
    stats_df.epoch = stats_df['epoch'] - 1

    # Plot and label the training and validation loss values
    plt.figure(figsize=(8,5))
    plt.plot(stats_df.epoch.values, stats_df.train_loss.values, label='Training Loss')
    plt.plot(stats_df.epoch.values, stats_df.val_loss.values, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.show()


def plot_train_loss(stats):
    ''' Plots the train loss across batches across all epochs given the stats as collected during training
        assuming a certain structure in the stats dictionary (as defined by the training loop)
    '''
    stats_df       = pd.DataFrame(stats)
    stats_df.epoch = stats_df['epoch'] - 1
    values = []
    for i in range(len(stats_df.epoch)):
        values += stats_df.per_batch_train_loss[i]

    # Plot and label the training and validation loss values
    plt.figure(figsize=(8,5))
    plt.plot(list(range(len(values))), values, label='Training Loss')

    # Add in a title and axes labels
    plt.title('Training Loss Across Batches')
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.show()


def plot_optimization_loss(loses, title = 'Optimization Loss', ylabel = 'MSE Loss'):
    ''' Plots the optimization loss across batches across all epochs '''
    plt.figure(figsize=(8,5))
    plt.plot(list(range(len(loses))), loses)

    # Add in a title and axes labels
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')

    plt.show()
