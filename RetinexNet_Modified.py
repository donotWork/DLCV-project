#%matplotlib inline
import argparse
import os
import random
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from IPython.display import HTML
from timeit import default_timer as timer
from dataclasses import dataclass, field
from pathlib import Path
from pytorch_msssim import ssim as ssim_loss
from torchvision.models import vgg16 # importing a pretrained vgg16 model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        # Activated layers!
        self.convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU())
        # Final recon layer
        self.recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, in_img):
        max     = torch.max(in_img, dim=1, keepdim=True)[0]     # Illumination map (grayscale)
        img     = torch.cat((max, in_img), dim=1)               # (B,3,H,W) + (B,1,H,W) -> (B,4,H,W)
        f0      = self.conv0(img)
        fs      = self.convs(f0)
        out_img = self.recon(fs)
        R       = torch.sigmoid(out_img[:, 0:3, :, :])          # Reflectance map (RGB)
        L       = torch.sigmoid(out_img[:, 3:4, :, :])          # Illumination map (grayscale)
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu         = nn.ReLU()
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        # conv layers for skip-connection fusion
        self.deconv1= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.deconv2= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.deconv3= nn.Conv2d(channel*2, channel, kernel_size, padding=1, padding_mode='replicate')
        # fuse skip connection and downsampling and get output
        self.fusion = nn.Conv2d(channel*3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.out_L = nn.Conv2d(channel, 1, kernel_size, padding=0)

    def forward(self, L, R):     
        input_img = torch.cat((R, L), dim=1)        # joining R+L
        out0      = self.conv0(input_img)           # feature extraction
        # downsampling
        out1      = self.relu(self.conv1(out0))     #(B, 64, img/2)
        out2      = self.relu(self.conv2(out1))     #(B, 64, img/4)
        out3      = self.relu(self.conv3(out2))     #(B, 64, img/8)
        # upsampling
        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))      #(B, 128,img/8) ->(B, 64, img/4)
        deconv1   = self.relu(self.deconv1(torch.cat((out3_up, out2), dim=1)))      #(B, 128,img/4) ->(B, 128,img/4) ->pxl(0,1)
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))   #(B, 128,img/4) ->(B, 64,img/2)
        deconv2   = self.relu(self.deconv2(torch.cat((deconv1_up, out1), dim=1)))   #(B, 128,img/2) ->(B, 128,img/2) ->pxl(0,1)
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))   #(B, 128,img/2) ->(B, 64,img)
        deconv3   = self.relu(self.deconv3(torch.cat((deconv2_up, out0), dim=1)))   #(B, 128,img)   ->(B, 128,img) ->pxl(0,1)
        # reshape and add
        deconv1_rs= F.interpolate(deconv1, size=(R.size()[2], R.size()[3]))         #(B, 128, img/4) ->(B, 128, img)
        deconv2_rs= F.interpolate(deconv2, size=(R.size()[2], R.size()[3]))         #(B, 128, img/2) ->(B, 128, img)
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)             #(B, 384, img)
        feats_fus = self.fusion(feats_all)  #(B, 128, img)
        output    = self.out_L(feats_fus)   #(B, 1, img)
        return output
    

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        # instanciate the DecomNet and RelightNet
        self.DecomNet  = DecomNet()
        self.RelightNet= RelightNet()
        self.vgg = vgg16(pretrained=True).features[:8].to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    # function to calculate the Charbonnier loss
    @staticmethod
    def charbonnier(x, y, eps=1e-3):
        diff = x - y
        return torch.mean(torch.sqrt(diff*diff + eps*eps))
    
    # function to calculate the feature level loss
    def perceptual(self, a, b):
        fa = self.vgg(a)
        fb = self.vgg(b)
        return F.l1_loss(fa, fb)
    
    def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = torch.from_numpy(input_low).float().to(device)
        input_high= torch.from_numpy(input_high).float().to(device)
        # Predict R and L for both Low light and Well lit image
        R_low, I_low   = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)
    
        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low) # Predict the change in L to relight the img

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1) # (B, 3, Img)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1) #(B, 3, Img)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1) #(B, 3,Img)

        # Predictions of the model
        pred_low   = R_low  * I_low_3
        pred_high  = R_high * I_high_3
        pred_relit = R_low  * I_delta_3

        # Compute Charbonnier loss losses
        self.recon_loss_low  = self.charbonnier(pred_low,  input_low)
        self.recon_loss_high = self.charbonnier(pred_high, input_high)
        self.recon_loss_mutal_low  = self.charbonnier(R_high * I_low_3, input_low)
        self.recon_loss_mutal_high = self.charbonnier(R_low * I_high_3, input_high)
        self.equal_R_loss = self.charbonnier(R_low,  R_high.detach())
        self.relight_loss = self.charbonnier(pred_relit, input_high)
        
        # Compute SSIM loss
        self.ssim_loss_low      = 1.0 - ssim_loss(
            pred_low,  input_low,  data_range=1.0, size_average=True)
        self.ssim_loss_high     = 1.0 - ssim_loss(
            pred_high, input_high, data_range=1.0, size_average=True)
        self.ssim_loss_mutal_low  = 1.0 - ssim_loss(
            R_high * I_low_3,  input_low,  data_range=1.0, size_average=True)
        self.ssim_loss_mutal_high = 1.0 - ssim_loss(
            R_low  * I_high_3, input_high, data_range=1.0, size_average=True)
        self.ssim_loss_relight = 1.0 - ssim_loss(
            pred_relit, input_high, data_range=1.0, size_average=True)

        # Compute the perceptual loss / feature loss
        self.vgg_loss_low     = self.perceptual(pred_low,  input_low)
        self.vgg_loss_high    = self.perceptual(pred_high, input_high)
        self.vgg_loss_relight = self.perceptual(pred_relit, input_high)

        # Smoothness penalties on illumination maps
        self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = (
              self.recon_loss_low
            + self.recon_loss_high
            + self.ssim_loss_low
            + self.ssim_loss_high
            + 0.001 * (self.recon_loss_mutal_low
                     + self.recon_loss_mutal_high
                     + self.ssim_loss_mutal_low
                     + self.ssim_loss_mutal_high)
            + 0.1   * (self.Ismooth_loss_low
                     + self.Ismooth_loss_high)
            + 0.01  * (self.equal_R_loss 
                       + self.vgg_loss_low 
                       + self.vgg_loss_high)
        )

        self.loss_Relight = (
              self.relight_loss
            + self.ssim_loss_relight
            + 3 * self.Ismooth_loss_delta
            + 0.01 * self.vgg_loss_relight
        )
        
        # Store CPU copies of key outputs for later visualization
        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def grad(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(device) # (2,2) = (H,W) -> (1,1,2,2) = (B,C,H,W)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3) # (1,1,W,H)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1)) # absolut value of grad
        return grad_out
    # get local avg of grad
    def avg_grad(self, input_tensor, direction):
        return F.avg_pool2d(self.grad(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        # convert the rgb Reflectance to grayscale Luminance
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :] #(B,H,W)
        input_R = torch.unsqueeze(input_R, dim=1) #(B, 1, H, W)
        return torch.mean(self.grad(input_I, "x") * torch.exp(-10 * self.avg_grad(input_R, "x")) +
                          self.grad(input_I, "y") * torch.exp(-10 * self.avg_grad(input_R, "y")))
    
    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                       (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.pth'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir   = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self, train_low_data_names, train_high_data_names, eval_low_data_names, batch_size, patch_size, epoch, lr, vis_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print(f"Start training for phase {self.train_phase}, " f"with start epoch {start_epoch} start iter {iter_num} : ")

        start_time = timer()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high= np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img= Image.open(train_high_data_names[image_id])
                    train_high_img= np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _        = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img= np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img= np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img= np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img= np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :]= train_high_img
                    self.input_low = batch_input_low
                    self.input_high= batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()
                
                print(f"{train_phase} Epoch: [{epoch+1:2d}] " f"[{batch_id+1:4d}/{numBatch:4d}], " f"Time: {timer() - start_time:.4f}, " f"loss: {loss:.6f}")

                iter_num += 1
            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names, vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print(f"Finished training for phase {train_phase}")


    def predict(self, test_low_data_names, res_dir, ckpt_dir):
        # Load the network with a pre-trained checkpoint
        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path  = test_low_data_names[idx]
            test_img_name = test_img_path.name
            print('Processing ', test_img_name)
            test_low_img   = Image.open(test_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32")/255.0
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            if save_R_L:
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image= np.concatenate([input, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir + '/' + test_img_name
            im.save(filepath[:-4] + '.jpg')

    def predict_full(self, test_low_data_names, res_dir, ckpt_dir, patch_size: int = 256, stride:    int = None):
        # load your checkpoints exactly as in predict()
        self.train_phase = 'Decom'
        self.load(ckpt_dir)
        self.train_phase = 'Relight'
        self.load(ckpt_dir)

        # ensure eval mode & no grad
        torch.set_grad_enabled(False)
        # manually set all modules to eval mode 
        for m in self.modules():
            m.training = False        
            

        stride = patch_size if stride is None else stride

        for img_path in test_low_data_names:
            img = Image.open(img_path)
            W, H = img.size

            # pad so that W,H are multiples of stride
            pad_w = (stride - W % stride) % stride
            pad_h = (stride - H % stride) % stride
            canvas = Image.new("RGB", (W+pad_w, H+pad_h))
            canvas.paste(img, (0,0))

            # accumulators
            acc_S = torch.zeros(1, 3, H+pad_h, W+pad_w, device=device)
            acc_c = torch.zeros_like(acc_S)

            # slide window
            for y in range(0, H+pad_h, stride):
                for x in range(0, W+pad_w, stride):
                    crop = canvas.crop((x, y, x+patch_size, y+patch_size))
                    arr  = (np.array(crop, dtype="float32")/255.0).transpose(2,0,1)[None]
                    t    = torch.from_numpy(arr).float().to(device)

                    # forward through the full pipeline
                    R_low, I_low = self.DecomNet(t)
                    I_delta      = self.RelightNet(I_low, R_low)
                    S            = R_low * torch.cat([I_delta]*3, dim=1)

                    acc_S[:,:, y:y+patch_size, x:x+patch_size] += S
                    acc_c[:,:, y:y+patch_size, x:x+patch_size] += 1

            # average overlaps and crop back to W×H
            out = (acc_S / acc_c)[:, :, :H, :W]
            out = out.squeeze(0).cpu().numpy().transpose(1,2,0)
            out_im = Image.fromarray((np.clip(out,0,1)*255).astype("uint8"))

            # save
            out_im.save(os.path.join(res_dir, img_path.name))
