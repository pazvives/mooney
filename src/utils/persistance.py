import time
import torch
import torchvision

from torchvision.utils import save_image

def gen_unique_id():
    return str(int(time.time()))

def save_img(img, name=''):
    file_dir  = './saved_images/'
    name      = name or gen_unique_id()
    full_path = f"{file_dir}{name}.png"
    save_image(img, full_path)
    print(f"Saved file: {full_path}")
    return full_path


def save_tensor(tensor, tensor_name, tensors_dir = 'tensors'):
    tensor_path = f'{tensors_dir}/{tensor_name}'
    torch.save(tensor, tensor_path)
    return tensor_path

def load_tensor(tensor_name, device, tensors_dir = 'tensors'):
    tensor_path = f'{tensors_dir}/{tensor_name}'
    tensor      = torch.load(tensor_path, map_location=device)
    return tensor

def save_hidden(run_id, iteration, optimizer, losses, accuracies, best_hidden, best_mooney, hidden_dir = 'hidden'):
    hidden_path = f'{hidden_dir}/{run_id}_{iteration}.pt'
    hidden_dict = {
                'run_id'     : run_id,
                'iteration'  : iteration, 
                'optimizer'  : optimizer, 
                'losses'     : losses,
                'accuracies' : accuracies,
                'best_hidden': best_hidden,
                'best_mooney': best_mooney
}

    torch.save(hidden_dict, hidden_path)
    return hidden_path

def load_hidden(hidden_filename, device):
    hidden_dict = torch.load(hidden_filename, map_location = device)
    return hidden_dict

def save_model(run_id, epoch, optimizer, model, seed, stats = None, model_dir = "models"):
    model_filename = f'model_{model.model_id}_{run_id}_{epoch}.pth' 
    model_path     = f'{model_dir}/{model_filename}'

    model_dict =  { 
          'run_id'        : run_id, 
          'epoch'         : epoch,
          'optimizer'     : optimizer, 
          'seed'          : seed,
          'model_id'      : model.model_id,
          'model_version' : model.version, 
          'model_drop_p'  : model.drop_prob,
          'model_path'    : model_path,
          'model_state'   : model.state_dict(), 
          'model_stats'   : stats,
    }
    
    torch.save(model_dict, model_path)
    return model_path

def load_model(model_cls, model_filename, device, freeze = False):
    
    model_path = model_filename 
    model_dict = torch.load(model_path, map_location = device)

    model      = model_cls( model_id  = model_dict['model_id'], 
                            drop_prob = model_dict['model_drop_p'], 
                            version   = model_dict['model_version'])
    

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model.eval()

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
   
    return model, model_dict


