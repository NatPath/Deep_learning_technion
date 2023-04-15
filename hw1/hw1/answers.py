r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. False - The test set does not relate to the in-sample error, because elements of this set are not trained on. in-sample error is the error on the train set.
The test set does provide an estimation to the out-of-sample/generalization error.
2. False - some splits are more useful than the others. Splits should be in such a way that the dijoint sets are sampling homogeneously from the whole data, samples should not have preffered set to be on.
3. True - cross-validation is done only on the train-set to allow learning parameters while training the model. The test set should never be trained on as it is supposed to be later used as an indicator for the generalization- 
Testing the accuired model on unseen data.
4. True - the validation set is used as a 'simulation' for the test set while tuning for parameters during the training, but it is still part of the train-set. 

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
The friend's approach FLAWED.
We shouldn't ever use the test set while training the model, as any choice made with test-set results is considered contaminated.
Tuning $\lambda$ is part of training the model, thus using the test set to select correct value is flawed.
Our friend should have used cross-validation using only the test-set to select a correct hyperparameter to restrict the models complexity and thus prevent over fitting, without contaminating the data. 

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

From the obtained results of the cv, the extermal value of the accuracy on the test set (which is a proxy to the generalization error)
was obtained for k=3. Increasing k from 1 to 3 improved generalization and increasing it further leads to a drop in accuracy.
This is because of how the data is organized, some distributions results with larger k being a more accurate parameter for predictions then others.


"""

part2_q2 = r"""
**Your answer:**

1. Training on the entire train-set will favor parameters which are biased toward the data from the trainset and will not necessarily generalize to data outside the train-set.
To eliminate this bias in the cv method, splitting of the data to folds in such a way that each fold has a different validation set to test the accuracy from the parameter obtained from training the rest of the data.
In this way the results of the trainings are being tested by data which is not trained on. 

2. Choosing of the model should never use the test-set. this is contamination of the method and the results of the accuracy of the test-set will not be a proxy to the generalization error.
k-fold CV method does not contaminate the train set thus it is better.


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
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
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
