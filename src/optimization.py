import torch
import torchvision

from tqdm import tqdm
from functools import partial
from torchvision.models import resnet50, ResNet50_Weights
from utils.persistance import save_tensor, save_hidden, gen_unique_id
from utils.plotting import view_images
from training import to_predictions, TRAIN_CONFIGS

MEAN_LATENT_SPACE = 'mean_latent_space.pt'
STD_LATENT_SPACE  = 'std_latent_space.pt'


def calcLatentSpaceNormAndMean(encoder, train_loader):
    hiddens = []
    
    for batch_idx, (gray_img, mooney_img) in enumerate(train_loader):
        hidden = encoder(mooney_img) # size of batch-latent space
        hiddens.append(hidden)

    # Calculates the mean/std per pixel - across hidden representation of all images in train_loader
    latent_space = torch.cat(hiddens, dim = 0)
    mean_latent  = torch.mean(latent_space, dim=0)
    std_latent   = torch.std(latent_space, dim=0)

    return mean_latent, std_latent 


def get_latent_space_sampler(mean_latent = None, std_latent = None):
    ''' Returns a tensor randomly sampled from a Normal Distribution defined by the mean and std received. 
        Adds noise to standard deviation in points where it is zero.
        Note: mean and std are tensors of size (C, W, H) which are the sizes of the embedding representation
        of the autoencoder.
    '''
    assert mean_latent is not None
    assert std_latent is not None
    mean_latent = mean_latent
    std_latent  = std_latent
    std_latent[std_latent==.0] = 1e-10
    assert (std_latent!=.0).all()

    sampler      = torch.distributions.Normal(mean_latent, std_latent)
    return sampler


def mse_loss(generator_fn, gray_img, embeddings, h): 
    ''' Custom loss function, generates a mooney from h (hidden representation) using the decoder
        of the trained autoencoder and returns the loss between the generated mooney and the given grayscale.
        If embeddings are provided the loss is calculated on the embeddings of the generated mooney and the given
        grayscale.
    '''

    gen_mooney      = generator_fn(h.unsqueeze(dim=0)).squeeze(dim=0) 
    gen_mooney_emb  = embeddings(gen_mooney) if embeddings else gen_mooney
    gray_image_emb  = embeddings(gray_img) if embeddings else gray_img

    assert gen_mooney_emb.requires_grad
    assert not gray_image_emb.requires_grad 
    
    return torch.mean(torch.square(gray_image_emb-gen_mooney_emb))  


def embeddings_resnet50(img):
    ''' Expects the image to be a torch.tensor of shape (C, W, H) and returns
        the embeddings corresponding to the last representation layer of a pre-trained Resnet 50.
        If the image only has one channel (ie C=1), it will expand it to three for compatibility.
    '''

    def to_n_channels(img, n = 3):
        ''' Repeats image across n channels'''
        img_extended = img.expand(n,*img.shape[1:])
        return img_extended

    if (img.shape[0] == 1):
        img = to_n_channels(img, n=3)

    resnet            = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet_hidden_rep = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()

    for p in resnet_hidden_rep.parameters():
        p.requires_grad = False

    resnet_hidden_rep.eval()
    batch             = resnet_transforms(img).unsqueeze(0)
    prediction        = resnet_hidden_rep(batch).squeeze()
    assert prediction.shape[0] == 2048
    return prediction


def get_optimal_mooney_given_grayscale(gray_img, eval_img, hidden, generator_fn, pred_threshold, loss_fn, loss_embeddings, optimizer, device = None, lr = .1, iterations = 10000, max_patience = 100, show_loss_n = 100, show_images_n = 100, run_id = '', save_best_hidden=False):
    """ Searches for the hidden representation that generates the 'optimal mooney' (minimum loss) according to the loss function received as parameter.
        Returns the generated mooney, the best hidden representation found, the optimizer and the losses. 
    """
    run_id       = run_id or gen_unique_id()
    filename     = ''
    counter      = 0
    best_loss    = 100  
    losses       = []
    accuracies   = []
    fn           = partial(loss_fn, generator_fn = generator_fn, gray_img = gray_img, embeddings = loss_embeddings)

    for i in tqdm(range(iterations)):
        
        optimizer.zero_grad()
        loss = fn(h=hidden)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Early stopping + save + eval and print results
        with torch.no_grad():
            output    = generator_fn(hidden.unsqueeze(0))[0] # pixel intensities or logits depending on the model being used
            mooney    = to_predictions(output, pred_threshold = pred_threshold)
            accuracy  = (mooney == eval_img).float().mean().item()
            accuracies.append(accuracy)
            
            if losses[-1] < best_loss:
                best_loss   = losses[-1]
                best_hidden = hidden 
                best_mooney = mooney 
                if save_best_hidden:
                    save_hidden(run_id = run_id, iteration=i, optimizer = optimizer, losses = losses, accuracies = accuracies, best_hidden = best_hidden, best_mooney = best_mooney, hidden_dir = 'hidden/optim') #, gray_img = gray_img, eval_img = eval_img) 
                counter     = 0
            else:
                counter+=1 
            if (counter > max_patience and i > 100):
                break

            if ((i+1)%show_loss_n==0):
                print(f"Run Id {run_id} - Loss: {loss.item()}\tAccuracy: {accuracy}")
                if ((i+1)%show_images_n == 0):
                    view_images([gray_img, output, mooney]) 
                
    return best_hidden.detach(), best_mooney.detach(), losses, accuracies