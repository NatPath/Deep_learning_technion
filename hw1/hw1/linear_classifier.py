import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss

import cs236781.dataloader_utils as dataloader_utils


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        #self.weights = None  ## COMMENTED HERE
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        self.weights = torch.normal(mean = 0, std = weight_std, size=(n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.
        
        class_scores= torch.matmul(x,self.weights)
        y_pred = [torch.argmax(vec).item() for vec in class_scores]
        y_pred = torch.tensor(y_pred)
        y_pred.reshape(-1,)   #### MAKE SURE IT's OKAY
        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        # ========================
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        acc = int(torch.eq(y,y_pred).sum())/y.shape[0]
    
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            #raise NotImplementedError()
                        
            N, train_acc, train_loss = 0, 0, 0

            reg_term = self.weights.norm().pow(2) * (weight_decay / 2)
            
            for x, y in dl_train:
                
                y_pred, class_scores = self.predict(x)
                
                train_loss += loss_fn(x, y, class_scores, y_pred) * y.shape[0] + reg_term
                
                grad = loss_fn.grad()
                
                self.weights = self.weights - learn_rate * (grad + weight_decay * self.weights)
                train_acc += self.evaluate_accuracy(y, y_pred) * y.shape[0] 

                N += y.shape[0]
                
            train_res.loss.append(train_loss / N)
            train_res.accuracy.append(train_acc / N)
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            y_pred, class_scores = self.predict(x_valid)
            valid_res.loss.append(loss_fn(x_valid, y_valid, class_scores, y_pred))
            valid_res.accuracy.append(self.evaluate_accuracy(y_valid, y_pred))

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        #raise NotImplementedError()
        """       
        images = None
        if (has_bias):
            image = self.weights[1:]
        ##w_images=images.T.reshape(self.n_classes,*img_shape)
        w_images=images.T.reshape(self.n_classes,*img_shape)
        """
                
        images_new = None
        if (has_bias):
            # Omitting the bias
            images_new = self.weights[1:]
        # Convert tensor to shape  (n_classes, C, H, W)
        w_images=images_new.T.reshape(self.n_classes,*img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    hp['weight_std'] = 0.002
    hp['learn_rate'] = 0.005
    hp['weight_decay'] = 0.001
    # ========================

    return hp
