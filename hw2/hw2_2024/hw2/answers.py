r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
1.
    A. The jacobian of Y w.r.t X will reflect how each element of X affects each element of Y, so the shape of the jacobian will be (64, 1024, 64, 512).

    B. Yes, the jacobian will be sparse. Since each element $X_(i,j) affects only row i of Y, each element of the jacobian, $J_(i,j,k,l), will have some value depending on W if i=k, or 0 otherwise. this means that $63/64$ of the elements will be zero.
    
    C. There is no need to materialize the jacobian. To calculate the downstream gratdient w.r.t. to the input, we can use the chain rule.
    The forward pass is defined by  $Y = XW^T$, and the gradient of the output w.r.t. downstream scalar loss is $\delta\mat{Y}=\pderiv{L}{\mat{Y}}$. Out goal is to find $\delta\mat{X}=\pderiv{L}{\mat{X}}$. Using the chain rule, we know that $\delta\mat{X}=\delta\mat{Y}*\pderiv{\mat{Y}}{\mat{X}}$. Since $Y = XW^T$, The partial derivative of Y with respect to X is $W^T$. So: $\delta\mat{X}=\delta\mat{Y}*W^T$

2.
    A. The shape of the jacobian of Y w.r.t W will be (512, 1024, 64, 512).

    B. Yes, the jacobian will be sparse. Since each element of Y is affected by only one row of W (one column of $W^T$), each element of the jacobian, $J_(i,j,k,l), will have some value depending on X if i=l, or 0 otherwise.

    C. There is no need to materialize the jacobian. To calculate the downstream gratdient w.r.t. to the input, we can use the chain rule.
    The forward pass is defined by  $Y = XW^T$, and the gradient of the output w.r.t. downstream scalar loss is $\delta\mat{Y}=\pderiv{L}{\mat{Y}}$. Out goal is to find $\delta\mat{W}=\pderiv{L}{\mat{W}}$. Using the chain rule, we know that $\delta\mat{W}=\delta\mat{Y}*\pderiv{\mat{Y}}{\mat{W}}$. Since $Y = XW^T$, The partial derivative of Y with respect to W is X. So: $\delta\mat{W}=\delta\mat{Y}*X$

"""

part1_q2 = r"""
While back-propagation is efficient and popular, it is not not required to train neural networks. Other options that don't use back-propagation include simulated annealing and evolutoinary algorithms.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 1e-2
    reg = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1
    lr_vanilla = 0.0205
    lr_momentum = 0.015
    lr_rmsprop = 1e-3
    reg = 0.0003
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. In the graph of no dropout, we can overfitting caused the testing accuracy to be significantly lower than the training accuracy. Training accuracy reaches over 40%, but testing accuracy is just over 15%.
In the lower dropout value graph, the dropout managed to get rid of the overfitting, and the train and test results are closer to each other. Training accuracy is slightly lower than with no dropout, but training accuracy improves and reaches 20%.
In the higher dropout value graph, The model is unable to learn effectively. Both training and testing accuracies stay below 20%.

This fits our assumptions. Some level of dropout hinders the learning during training, and thus prevents overfitting, but too much overfitting (in our case, 80%) nullifies the activation of most neurons and effectively cancels almost all of the learning proccess.

2. As explained before, the lower dropout setting provides regularization and improves testing accuracy by 4% compared to no dropout. In contrast, the higher setting provides too much regularization and worsens both training and testing accuracies, compared to both no dropout and low dropout.

"""

part2_q2 = r"""
Yes, it is possible for the test loss to increase while the test accuracy also increases when training a model with the cross-entropy loss function.
The accuracy represents the amount of correct predictions, while the loss represents the confidence of the predictions, so if the model predicts more correct answers, but with less confidence, the accuracy could increase while the loss also increases.

"""

part2_q3 = r"""
1. Back-propagation computes the gradients necessary for the optimization process, while gradient descent uses these gradients to update the model's parameters.

2. GD computes the gradient of the loss function with respect to the parameters by considering the entire dataset, while SGD considers only one datapoint or a batch, a subset of the dataset. This allows SGD to achieve different results depending on the division to subsets and the order of their usage.
SGD is more efficient than GD since the computations require less memory. GD has a smoothes convergence than SGD, but SGD is able to escape local minima more effficiently.

3. Datasets used for deep learning tend to be very large, and GD requires storing the entire dataset in memory for each gradient computation, which can be either impossible or very expensive. SGD only uses a small part of the dataset for each iteration. SGD is also able to escape local minima due to its more erratic and random updates. Moreover, the noise introduced by SGD’s random sampling can act as a form of implicit regularization, helping to prevent overfitting and improving generalization.

4.
    A. Yes, this approach would produce a gradient equivalent to GD. Since calculations on the batches are independent from each other in the forward pass, it is possible to divide the forward pass into disjoint batches and acheive the same result as GD.

    B. Despite using small batch sizes, with each forward pass the input of each linear layer is saved to memory, to be used in calculating the backward pass. This means that effectively the entire data is saved regardless of division to small batches.
"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
\\
1.\\
Optimization Error is the loss difference between the returned hypothesis learned by the trained model (marked by $\bar{h}$) 
and the empirically optimal hypothesis (marked by $h_{s}^{*}$) which is an hypothesis which will classify correctly all of the test set upon prediction.
High Optimization error means that we could probably do better learning from the training set, 
for example by learning for an extended period of time in a gradient descent based method (assuming we haven't reached the minima yet).

\\
2.\\
Generalization error is the difference between $h_{s}^{*}$ and the optimal hypothesis (marked by $h_{D}^{*}$) which is the hypothesis which has the lowest population loss :
$ L_{D}(h):= E_{(X,y)\sim D}[l(y,h(X))]$ out of all possible hypotheses (hypotheses which are part of the hypotheses space defined by the model architecture).
If Generalization error is high but the training error is low, it could mean that our model is over-trained, it has too many parameters and is too biased toward data trained on, and thus can't generalize (can't manage to classify correctly for unseen examples).
Note that assesing generalization error is done empircally testing our model with data it hasn't seen upon training, but often we can't calculate it exactly as the distribution space for the data can be very large, averaging over many examples (but not all) is the empirical way, this is often called test error.
To reduce overfitting we can use regularization techniques, theses techniques essentially penalize too complex/large parameters for the model which would have enables following the trainning data too closely and preventing the model from "seeing the bigger picture".
An example for regularization is L2/L1 regularization which adds a constraint to the loss by adding to it a term proportional to the L1/L2 norm of the weights. It prevents the weights getting too large.
Another example for regularization techniuque is adapting to an optimal receptive field in a CNN senario.
When the receptive field is too large, the global context of the image analaysed might be captured, but at risk of overfitting by memorizing the entire picture.
A smaller receptive field might enable to focus our model's "understanding" to smaller/local features, but may miss broad patterns.
Which means that by "playing" with the receptive field,[by choosing increasing/deccreasing layer depth(deeper networks have larger RF),choosing different kernel sizes( larger kernels increase RF),dilated convolutions (increases RF), sub-sampling (increases RF) and more] we can achieve a sweetspot of not overfitting but also not underfitting (too simple model).

\\
3.\\
Approximation Error is the difference between $f_{D}^{*}$ (The ground truth with the minimal population loss of all existing functions[even beyond any model capabilities, let alone ours]) and $h_{D}^{*}$.
High Approximation error means that our model lakes expressivness, it might have too low number of parameters, or an unfit architecture to capture the complex nature of the problem.
To avoid this problem we should understand the problem and the data better and design an architecture which is low parameter and still capable to generalize well.



    


"""

part3_q2 = r"""
**Your answer:**
Case we expect false positive rate (FPR) to be high:
\\
Consider a case of an airport security automatic checker which is an initial checker for bags.
The system is cheaper than an experienced human checker which checks the bags only after the bag is classified possitivally by the automatic checker.
The risk of classifying wrongly by the machine is grand, so it will be sensitive to anything that can be seen as odd in the bag and thus classify it possitivally leniently.
Thus resulting with high rate of positivally classified bags, regarding the fact that most of the bags are not dangerous and thus the TN (true negatives) are negligable compared to the FP (false postivies). Thus resulting with high FPR. 

\\
Case we expect false negative rate (FNR) to be high:
\\
Consider a case of a very rare disease and a classifier returning positive if a patient has the disease.
Also, the identifying the disease implicates a very expensive and dangourous proceedures to be proceeded with.
We would expect that such system will be designed to identify as positives only patients it is extremly sure of having the disease.
The amount of TP in such senario is very small compared to the amount of FN and thus FNR is high.
\\



"""

part3_q3 = r"""
1. If people with the disease will develop non-lethal symptoms that immediately confirm the diagnosis and can then be treated, it is better to pick a higher threshold in order to get less false positives at the cost of more false negatives. A person with a false negative result will develop non-lethal symptoms and be treated. The false negative will result in a later treatment but no significant or permanent harm to the person.

2. In the second case it will be better to pick a slighty low threshold that will result in less false negatives at the cost of more false positives. A false positive would demand further testing which is expensive and risky for the patient, but a false negative has a high chance of costing the patient's life, making it a hight priority to reduce false negatives.

"""


part3_q4 = r"""
There are several downsides to MLP in regard to the task.

1. MLPs typically require a fixed size input, which can be a problem for variable-length sequences such as sentences.

2. MLPs treat input features independently and with no regard for order. The order of words in a sentence is crucial for the meaning.

3. MLPs have a limited receptive field because each layer only processes a fixed size input. For long sentences, the relationships between distant words are not captured effectively.

4. To account for longer contexts, MLPs would require exponentially more neurons and layers, making the model large, slow to train, and prone to overfitting.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn=torch.nn.CrossEntropyLoss()
    lr=0.1
    weight_decay=0.0
    momentum=0.0
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1.

Convs directly on the 256-channel input example:
Each convolution layer has 3x3x(\#input_channels=256)+1 parameters per filter (+1 because of the bias term), and 256 filters as the number of input channels for the next layer.
Thus in this example with 256 input_channels to each layer and 2 covnvolution layers overall there are:

(3x3x256+1)x256x2 = \\
1,180,160 parameters total
Bottleneck example:
first layer: (1x1x256+1)x64 parameters\\
second layer: (3x3x64+1)x64\\
third layer: (1x1x64+1)x256\\
total:70,016 parameters total

We can see that the bottleneck approach requires much less parameters, an order of magnitude less, and thus less prone to overfitting.

\\

\\
2.

A convolution with a dxd filter on an input of dimensions CxHxW, when the stride and padding are chosen so that the HxW dimensions of the result don't change
, is performing
(d^2*H*W*C*2+1) operations, the x2 factor at the end due to the multipication + summation. and +1 is due to the bias addition.
Relu is pointwise thus adds CxHxW, so does the summation operator between the skip and the main pain.

Convs directly on the 256-channel input example:
\\
Thus
(3x3x256x2xHxW+1)x256x2 + 256xHxWx3 =(neglecting terms without HxW as they are very small )
2,360,064xHxW operations total.
\\

Bottleneck example:

1x1x256x2xHxWx64 + 64xHxW + 3x3x64x2xHxWx64 + 64xHxW + 1x1x64x2xHxWx256 + 256xHxWx2 =

139,904xHxW operations total

  

The bottleneck approach requires an order of magninitude less operations thus much more computational efficient (assuming both approches are performing similarly)

\\

3.

Both approaches combine all of the feature-maps in each layer, thus are similar in their ability to combine across feature maps.

Reagarding "combining" the input spatially, the receptive field of a bottlenecked block is smaller, due to more 3x3 convolutions- its receptive field is going to be larger by a factor of 2 than the regular block.

  

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1.

The depth which resulted with the best test accuracy was L=4 and K=64. We think it is just a sweet spot between being to large of a network (which tend to overtrain due to large number of parameters or in our case, not be able to learn at all due to exploading/vanishing gradients)

and being too small (too few parameters, the expressivness of the model suffers and thus it is harder to learn).

Note that the L=4 network is slightly better than the L2 in terms of loss and accuracy when K=32. but the L4 network also shows signs of overfitting in the last iterations (test loss rises while train loss still drops) and thus is stopped earlier than the L=2 network training. In the K=64 the difference in perforemance is more significant between L=2 and L=4.

Maybe because larger K means more filters, and thus carry more information which is beneficial when the network is deeper and has more power to extract useful information from it.

2. for L in {8,16} the network wasn't trainable. We suspect it might be due to vanishing/exploding gradients due to the depth of the network.

It might be resovled using a skip connection network architecture such as resnet or using regularizatio techniques such as batch-normalisation (for exploding and vanishing) and gradient clipping (for exploding). Also, due to shortage in time we have not tuned the optimizer hyperparams for long enough, which might have resulted with trainablity for these L's.
"""

part5_q2 = r"""
**Your answer:**
The best test_acc is acchieved L4_K64, around 72% accuracy. Consistent with the previous experiment which was also the case.

For L=2 there seems to be some sort of learning, whcih is best with K=32 and gets worse as K gets larger, we can only make an educated guess that when the network is shallow it benefits smaller more concised data expressed with fewer features(which's number is set by the number of kernels for the convolutions).

For L=8 we see a non learning network, probably for the same reasons mentioned in q1.

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
1. The model did a poor job detecting the objects in the images.
In image 1, the model detected 3 objects, but mislabeled one with a high confidence (0.9), and mislabeled the other two with low confidence (0.47 and 0.37).
In image 2, the model correctly labeled one object with confidence 0.5 and mislabeled two objects with confidences 0.39 and 0.65, while not detecting another object at all.

2. There could be several reasons for the failure.
One possible reason is lack of relevant data. If the training was done without enough images of dolphins, the model would have a harder time correctly identifying dolphins. This could be resolved by making sure there are enough images that contain each possible label.
Another reason for mislabeling could be multiple objects in the same box. In image 2, a box containing a dog and a cat was labeled as "cat". To resolve this problem, we could change the model to use instance segmentation and identify every instance.

3. The idea behind an adversarial attack is to manipulate images in a small enough way as to not appear different to humans, but to fool an object detection model. The way to do that is to run an image through the model forward and backward to calculate both the loss and the gradient of the loss, and then to slightly change the image in the direction of the gradient, thereby enhancing the loss as much as possible for the least visual change.

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
Image 1 has bad illumination conditions. The light comes from directly behind the objects and so the model analyzes only shadows.
Image 2 contains many occluded objects that are misidentified and sheep instead of cats.
Image 3 is blurred due to high speed movement, and is misidentified as a bear instead of a monkey
"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
