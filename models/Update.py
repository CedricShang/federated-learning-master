#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np


# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        #self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = dataset,idxs

    def train(self, model):
        # train and update
        
        # Note: train and log loss for image

        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # epoch_loss = []
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         net.zero_grad()
        #         log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         if self.args.verbose and batch_idx % 10 == 0:
        #             print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 iter, batch_idx * len(images), len(self.ldr_train.dataset),
        #                        100. * batch_idx / len(self.ldr_train), loss.item()))
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            print("\nStart of epoch %d" % (epoch,))
            batch_loss = []

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.ldr_train):
                
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = model.loss(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s batches" % (step + 1 ))
                batch_loss.append(loss_value)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.weights, sum(epoch_loss) / len(epoch_loss)

