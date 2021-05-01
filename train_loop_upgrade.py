#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import cv2
from google.cloud import storage
import os
import time
from uuid import uuid4
import pickle
import datetime
import gc
import importlib
import signal
import sys
import gaussians
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrainLoopUpgrade():
    
    def __init__(self):
        self.restart_needed = False
        
    def signal_handler(self, sig, frame):
        self.restart_needed = True
        
    def train(self, epoch, model, criterion, optimizer, train_torch_image, train_expected_maps, batch_size):
        device = self.device
        epoch_loss = []
        # For each batch
        for start_i in range(0, len(train_torch_image), batch_size):

            if self.restart_needed:
                return None, False

            start_time = time.time()

    #         torch_image_batch = torch_image[ start_i: start_i + batch_size ,:,:,:].to(device)
    #         map_batch         = expected_maps[ start_i: start_i + batch_size ,:,:]

            # Loading in the training images from train/test splits
            torch_image_batch = train_torch_image[ start_i: start_i + batch_size ,:,:,:].to(device).to(torch.float32)
            map_batch         = train_expected_maps[ start_i: start_i + batch_size ,:,:].to(device).to(torch.float32)

            tensor_transfer_time = time.time()

            # Train on batch

            optimizer.zero_grad()

            out = model(torch_image_batch)

            forward_pass_time = time.time()

            batch_loss = criterion(out, map_batch)

            loss_function_time = time.time()

            epoch_loss.append(batch_loss.item())

            batch_loss.backward()

            backprop_time = time.time()

            optimizer.step()

            optimizer_time = time.time()

    #         print(f'Epoch: {epoch}, Batch: {start_i // batch_size}, Batch Distribution Difference Loss: {batch_loss}, JointMaxMSELoss (to see if model is working): {alt_criterion(out, kpt_batch.to(device))}')
            print(f'Epoch: {epoch}, Batch: {start_i // batch_size}, Training Batch Distribution Difference Loss: {batch_loss}')
            print(f'Tensor Transfer:{tensor_transfer_time - start_time}, Forward:{forward_pass_time - tensor_transfer_time}, Criterion:{loss_function_time - forward_pass_time}')
            print(f'Backward:{backprop_time - loss_function_time}, Optimizer_Step:{optimizer_time - backprop_time}, Total:{optimizer_time - start_time}')

            gc.collect()
            torch.cuda.empty_cache()
        return epoch_loss, True


        # In[ ]:


    def validation(self, epoch, model, criterion, val_torch_image, val_expected_maps, batch_size):
        device = self.device
        epoch_loss = []
        # For each batch
        for start_i in range(0, len(val_torch_image), batch_size):

            if self.restart_needed:
                return None, False

            start_time = time.time()

    #         torch_image_batch = torch_image[ start_i: start_i + batch_size ,:,:,:].to(device)
    #         map_batch         = expected_maps[ start_i: start_i + batch_size ,:,:]

            # Loading in the training images from train/test splits
            torch_image_batch = val_torch_image[ start_i: start_i + batch_size ,:,:,:].to(device).to(torch.float32)
            map_batch         = val_expected_maps[ start_i: start_i + batch_size ,:,:].to(device).to(torch.float32)

            tensor_transfer_time = time.time()

            # Train on batch
            with torch.no_grad():
                out = model(torch_image_batch)

                forward_pass_time = time.time()

                batch_loss = criterion(out, map_batch)

                loss_function_time = time.time()

            epoch_loss.append(batch_loss.item())


    #         print(f'Epoch: {epoch}, Batch: {start_i // batch_size}, Batch Distribution Difference Loss: {batch_loss}, JointMaxMSELoss (to see if model is working): {alt_criterion(out, kpt_batch.to(device))}')
            print(f'Epoch: {epoch}, Batch: {start_i // batch_size}, Validation Batch Distribution Difference Loss: {batch_loss}')
            print(f'Tensor Transfer:{tensor_transfer_time - start_time}, Forward:{forward_pass_time - tensor_transfer_time}, Criterion:{loss_function_time - forward_pass_time}')

            gc.collect()
            torch.cuda.empty_cache()
        return epoch_loss, True

    def main(self):
        # Check device availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        print("device: %s" % device)
        # device = 'cpu'

        


        # In[2]:


        with open('annotations/valid.json') as f:
            test_data = json.load(f)
        with open('annotations/train.json') as f:
            train_data = json.load(f)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pose-estimation-2-dc39bc540ba3.json"

        storage_client = storage.Client("pose_estimation_2")
        bucket = storage_client.get_bucket('pose_estimation_2_dataset_mpii')

        NUM_TRAIN = 16
        NUM_TEST = 2958

        # In[ ]:

        gaussian = gaussians.Gaussians()

        load_start = time.time()

        kpt_list     = []

        torch_image = torch.zeros(NUM_TRAIN, 368, 368, 3, dtype=torch.half)
        torch_image.requires_grad = False

        # For each image, load the image
        for i in range(NUM_TRAIN):
            img_name = train_data[i]['image']

            blob = bucket.blob('MPII/images/' +  img_name)
            blob.content_type = 'image/jpeg'
            image = np.asarray(bytearray(blob.download_as_string()))
            img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

            kpt = np.asarray(train_data[i]['joints'], dtype=np.int32)

            if img.shape[0] != 368 or img.shape[1] != 368:
                kpt[:,0] = kpt[:,0] * (368/img.shape[1])
                kpt[:,1] = kpt[:,1] * (368/img.shape[0])
                img = cv2.resize(img,(368,368))
                img = np.array(img)

            kpt_list.append(kpt)
            torch_image[i,:,:,:] = torch.HalfTensor(img)

            if i % 10 == 0:
                print(f'Loaded {i+1} images')

        image_load_time = time.time()

        # construct image tensor and label tensor
        # torch_image = torch.Tensor(imagelist)
        torch_image = torch_image.permute(0, 3, 1, 2)
        expected_maps = gaussian.expected_to_gaussian(kpt_list)
        torch_image.requires_grad = False
        expected_maps.requires_grad = False

        tensor_build_time = time.time()

        print(f'Image Load:{image_load_time - load_start}, Tensor Build:{tensor_build_time - image_load_time}')
        print(f'Loaded {NUM_TRAIN} images, shape is {torch_image.shape}, kpt shape is {expected_maps.shape}')


        # In[ ]:


        from sklearn.model_selection import train_test_split

        # train_torch_image, val_torch_image = torch_image, torch_image
        train_torch_image, val_torch_image = train_test_split(torch_image, test_size = 0.25, shuffle = False)
        # train_expected_maps, val_expected_maps = expected_maps, expected_maps
        train_expected_maps, val_expected_maps = train_test_split(expected_maps, test_size = 0.25, shuffle = False)


        # In[ ]:


        

        # In[ ]:


        # # Hyperparameters

        import modules
        import modules.unipose
        import modules.criterion.distribution_difference_loss

        self.restart_needed = False



        signal.signal(signal.SIGINT, self.signal_handler)


        # In[3]:
        while True:

            self.restart_needed = False

            importlib.reload(modules)
            importlib.reload(modules.unipose)
            importlib.reload(modules.criterion)
            importlib.reload(modules.criterion.distribution_difference_loss)

            try:
                print("batch size:")
                batch_size = int(input())
                print("epochs:")
                epochs = int(input())
                print("lr:")
                learning_rate = float(input())
            except:
                batch_size, epochs, learning_rate = 16, 30, 0.0001
            optimizer_type = 'ADAM'

            hyperparam_string = f'batch_size: {batch_size}, epochs: {epochs}, lr: {learning_rate}, optimizer: {optimizer_type}'
            print(hyperparam_string)


            # # Train Loop

            # In[4]:




            model = modules.unipose.UniPose().to(device)
            criterion = modules.criterion.distribution_difference_loss.DistributionDifferenceLoss(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == 'ADAM' else None
            scheduler = ReduceLROnPlateau(optimizer, 'min')

            train_epoch_losses = []
            val_epoch_losses = []

            gc.collect()
            torch.cuda.empty_cache()

            uuid_string = str(datetime.datetime.now())

            # For each epoch
            for epoch in range(epochs):

                train_epoch_loss, is_normal = self.train(epoch, model, criterion, optimizer, train_torch_image, train_expected_maps, batch_size)
                if not is_normal:
                    break
                val_epoch_loss, is_normal = self.validation(epoch, model, criterion, val_torch_image, val_expected_maps, batch_size)
                if not is_normal:
                    break
                scheduler.step(sum(val_epoch_loss))

                print(f'Epoch: {epoch}, Training Average Batch Loss: {sum(train_epoch_loss) / len(train_epoch_loss)}')
                print(f'Epoch: {epoch}, Validation Average Batch Loss: {sum(val_epoch_loss) / len(val_epoch_loss)}')

                train_epoch_losses.append(train_epoch_loss)
                val_epoch_losses.append(val_epoch_loss)

                with open(f'epoch{epoch}.txt', 'a') as f:
                    print(hyperparam_string, file=f)
                    for e in range(epoch + 1):
                        print(f'Epoch: {e}, Training Average Batch Loss: {sum(train_epoch_losses[e]) / len(train_epoch_losses[e])}', file=f)
                        print(f'Epoch: {e}, Validation Average Batch Loss: {sum(val_epoch_losses[e]) / len(val_epoch_losses[e])}', file=f)
                    print(f'Epoch: {e}, Training Epoch Losses: {train_epoch_losses}', file=f)
                    print(f'Epoch: {e}, Validation Epoch Loss: {val_epoch_losses}', file=f)

                f_name = f'epoch{epoch}.txt'
                blob = bucket.blob(f"training_output/{uuid_string}/epoch{epoch}.txt")
                blob.upload_from_filename(f_name)
                os.remove(f_name)

                f_name = f'epoch{epoch}.pkl'
                pickle.dump(model, open(f_name, 'wb'))

                blob = bucket.blob(f"training_output/{uuid_string}/epoch{epoch}.pkl")
                blob.upload_from_filename(f_name)
                os.remove(f_name)


            # In[ ]:



            f_name = 'finalized_model.pkl'
            pickle.dump(model, open(f_name, 'wb'))

            blob = bucket.blob(f"training_output/{uuid_string}/final.pkl")
            blob.upload_from_filename(f_name)
            os.remove(f_name)

if __name__ == "__main__":
    tlu = TrainLoopUpgrade()
    tlu.main()