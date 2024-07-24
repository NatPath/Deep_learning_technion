r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    wstd = 4
    lr_vanilla = 0.028
    lr_momentum = 0.015
    lr_rmsprop = 1e-3
    reg = 0.0005
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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**


"""


part3_q4 = r"""
**Your answer:**


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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
