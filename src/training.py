import torch
import numpy as np
import random

from utils.persistance import save_model, gen_unique_id
from utils.plotting import view_pair
from tqdm import tqdm


# MSE - Prediction threshold is .5 because im predicting between 0 and 1 
# BCE - Prediction threshold is 0 since it predicts logits (where below zero belongs to class 0, otherwise class 1)
TRAIN_CONFIGS =  {'BCE': {'cls': torch.nn.BCEWithLogitsLoss, 'pred_threshold': .0},
                  'MSE': {'cls': torch.nn.MSELoss,           'pred_threshold': .5}} 




def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_predictions(outputs, pred_threshold = .5):
    ''' This method returns predictions over the classes of interest (0-1) of the outputs received.
        The threshold used should depend on the model output type.mro
        E.g. if model was trained to return logits, then .0 is an appropiate threshold. 
    '''
    predictions = outputs.clone()
    predictions[predictions > pred_threshold]  = 1.
    predictions[predictions <= pred_threshold] = 0.
    return predictions



def set_seeds(seed = 0): #42?
  """ Set seeds for random operations. Defaults to 0 """
  torch.manual_seed(seed) # Pytorch random number generator seed
  torch.cuda.manual_seed(seed)
  random.seed(seed)       # Python seed
  np.random.seed(seed)    # Numpy


def train_val(model, train_loader, val_loader, criterion, optimizer, epochs, start_epoch = 0, stats = None, pred_threshold = .5, seed = 0, device = None, show_batch = 10, run_id = None):

  run_id = run_id or gen_unique_id()
  stats  = stats.copy() if stats else [] 
  print(f"Training run id: {run_id}\t Epochs:[{start_epoch+1}-{epochs}]")
  
  set_seeds(seed = seed)
  
  for e in tqdm(range(start_epoch, epochs)):
    epoch = e+1
    
    ############ TRAINING #############
    model.train()
    batch_losses = []
    for batch_idx, (_, mooney) in enumerate(tqdm(train_loader)):

      inputs, targets = mooney.to(device), mooney.to(device)    
      optimizer.zero_grad()
      output = model.forward(inputs)
      loss   = criterion(output, targets) # This loss is avg per sample
      loss.backward()
      optimizer.step()
    
      batch_losses.append(loss.item())
      if ((batch_idx+1) % show_batch == 0):
        print('Train Batch: [{}/{}]\tAvg Batch Loss: {:.6f}'.format(batch_idx+1, len(train_loader), np.mean(batch_losses)))

    ############ EVALUATION #############
    with torch.no_grad():
      model.eval()
      val_losses     = []
      val_accuracies = []
      for _, (_, mooney) in enumerate(tqdm(val_loader)):
          inputs, targets = mooney.to(device), mooney.to(device)
          output          = model.forward(inputs)
          # VAL LOSS
          val_loss        = criterion(output, targets).item()
          val_losses.append(val_loss)
          # VAL ACCURACY
          predictions = to_predictions(output, pred_threshold = pred_threshold)
          accuracy    = (predictions == targets).float().mean()
          val_accuracies.append(accuracy)

      
      ############ VIEW RECONSTRUCTION #############
      img, recon = inputs[0], predictions[0]
      view_pair(img.cpu(), recon.cpu())
          
      ############ PLOT+SAVE MODEL #############
      epoch_train_loss = np.mean(batch_losses)
      epoch_val_loss   = np.mean(val_losses)
      epoch_val_acc    = np.mean(val_accuracies)
      stats.append({'epoch': epoch, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'val_acc': epoch_val_acc, 'per_batch_train_loss': batch_losses})
      filename = save_model(run_id, epoch, optimizer, model, seed, stats = stats)
      print(f"Epoch: [{epoch}/{epochs}]\tTrain Loss: {epoch_train_loss:.4f}\tTest loss: {epoch_val_loss:.4f}\tVal accuracy: {epoch_val_acc:.4f}")
      print(f"Saved model: {filename}")
      
  return model, stats

