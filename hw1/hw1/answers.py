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

The value of $\Delta$ scales the weights, and accordingly changes the model. Our aim with $\Delta$ is to guarantee a certain distance between the smaple's prediction to other alternative perdictions. Changing $\Delta$ means changing the $W$   so to reach different weights that assure a certain margin from the boundary the model created and to seprate our samples with a distance from other classes. Hence, $\Delta$ is arbitrarily chosen.

"""

part3_q2 = r"""


1. The model is learning by finding the right hyperplane that will best linearly separate the samples among other predictions. From the plot we can notice that some of the digits which look similar and so some $W$ columns are also similar according to their values and that leads to inaccurate predicitions.

2. My interpretation is similar to KNN according to the premises we make. SVM classifies sample by the argmax value of the scores, the kNN model classifies sample by the majority label of the K most similar images. One different aspect is that linear SVM takes advantage of all the data while kNN saves the entire data but takes advantage only of the neighbours of our sample.

"""

part3_q3 = r"""
1. We can deduce that the learning rate we have chosen for the training set is *good*.
If the learning rate was too low we wouldn't have seen a nice convergence in these number of epochs. Perhaps we wouldn't have seen any convergence, just constant and slow decline in the loss.
On the other hand, if the learning rate was too high, the graph might not be monotone decreasing but rather jump up and down around the minimum value.

2. I would say the model is *slightly overfitted* to our training set. After a few iterations the training accuracy is greater the test accuracy. Hence, our hypothesis class might have been too complex that we missed the "sweet spot" between overfitting and underfitting.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal residual plot is such that all of the points scatter thinly (with low variance) around y-y_hat=0 for all y_hat, that means that the predictions are close to the ground truth.
When comparing the residual plot of the 5 features and the last one obtained we can see that in the
5 features plot there are many more points which are far from the center compared to the last residual plot obtained, in which the points are thicker around 0 (but there are still some which are far from it).


"""

part4_q2 = r"""
**Your answer:**

Adding non-linear features to the data increases the expressivity of the model, allowing it to estimate more complex functions as a linear combination of different functions of the features.

1.
It is still a linear regression model, but with feature mapping to a polynomial space.
When looking at the model after the transformation, the model is just a regular linear regression.
Although, before the transformations (if we consider the transformation to be part of the model) one could argue that this is not linear regression as the prediction is not a linear function of the features (as it is only after the features are transformed in a certain way).

2.
There are a few ways we could interpret this question, so we'll give a long answer.
Any non-linear function F(X) of the original features can be expressed as a function of the features (by definition, complex as it maybe, even if it is non-analytic) which means that if only we could find the correct feature mapping such that
F(X)=X', we could use the feature X' as a data. Ofcourse this is flawed approach as this is the function we want to learn, and knowing it in advance makes learning it redundant and nonsensical.
But if there are enough features, and the desired hidden function behaves nicely enough, one could search for the representation of the function in a given basis, a basis which is a function of features.
Thus leading to an approximation which would become more and more accurate as we increase the number of basis terms, and the weight for each term could be learned with the linear regression model we know.
An example for such basis is the polynomials, we could do a feature mapping to the polynomials and if we use high enough degree for the polynomials we could estimate to an arbitrary precision any analytical function.

3.
The decision boundary will transform with the feature mapping induced by the non-linear features when viewed in the original features basis.
For non-linear feature mapping we'll get a hypersurface which is not necessarily a hyperplane.
When viewed in the mapped features basis, the decision boundary will still be defined by a hyperplane as W is a vector which defines the hyperplabe (A normal vector defined by the weights and an origin point defined by b).

"""

part4_q3 = r"""
**Your answer:**

1.
 The optimal hyperparameter value is sometimes to be searched in a large domain, searching wide range of values, as the order of magnitude of the optimal value is unknown.
Using logspace allows exploring wider range of values, Thus when observing a trend for the implication of different order of magnitudes of the parameters values to the accuracy can allow us to narrow the search in later searches more.
This is compared to linspace which searches with even skips- thus requires much more points to check results of different orders of magnitude of values. checking more points means longer computation time which can be expensive thus should be avoided.

2.
The model is fitted to the data len(hyper_paremeter1_range)*len(hyperparameter2_range)*k*2
len(hyper_paremeter1_range)*len(hyperparameter2_range) is the number of parameters sets checked,
each parameter set is checked as k times (as the number of folds) iwht the training data of the fold and k times with validation data of the fold.



"""

# ==============
