import torch
import torch.nn.functional as F


class Network(torch.nn.Module):

    def __init__(self, model_id="", drop_prob=0.5, version = 1):
        super().__init__()

        self.model_id  = model_id
        self.version   = version 
        self.drop_prob = drop_prob

        # Encoder
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        
        # Decoder
        self.dec_upconv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1)
        self.dec_upconv2 = torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)
        self.dec_upconv3 = torch.nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1)
        self.dec_upconv4 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.dec_out     = torch.nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1)  
        if self.version == 2:
            self.dec_conv1 = torch.nn.Conv2d(8, 8, 3,   stride=1, padding=1)
            self.dec_conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
            self.dec_conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.dec_conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.drop    = torch.nn.Dropout2d(p=drop_prob)
    
    def forward(self, x):
        x      = self.encode(x)
        logits = self.decode(x)
        return logits


    def encode(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x
        
    def decode(self, x):

        if (self.version == 1):
            x = self.drop(F.relu(self.dec_upconv1(x, output_size=(64,64)))) 
            x = self.drop(F.relu(self.dec_upconv2(x, output_size=(128,128))))
            x = self.drop(F.relu(self.dec_upconv3(x, output_size=(128,128))))
            x = self.drop(F.relu(self.dec_upconv4(x, output_size=(256,256))))
            x = self.dec_out(x)

        elif (self.version == 2):
            x = self.dec_upconv1(x, output_size=(64,64))
            x = self.dec_conv1(x)
            x = self.drop(F.relu(x))

            x = self.dec_upconv2(x, output_size=(128,128))
            x = self.dec_conv2(x)
            x = self.drop(F.relu(x))

            x = self.dec_upconv3(x, output_size=(128,128))
            x = self.dec_conv3(x)
            x = self.drop(F.relu(x))

            x = self.dec_upconv4(x, output_size=(256,256))
            x = self.dec_conv4(x)
            x = self.drop(F.relu(x))

            x = self.dec_out(x)
        
        
        return x