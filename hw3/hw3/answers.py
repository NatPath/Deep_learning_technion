r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT 1.\nenter SHREK\nSHREK. Somebody once told me "
    temperature = 1e-5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
There are several reasons to divide the corpus into sequences. One reason is to avoid over-fitting: dividing the text allows our model to observe different character combinations so it is less likely to just memorize the text. Another reason is that dividing the text allows the model to process the data more efficiantly. With an undivided text, the model will have only one layer and the process will be entirely serialized.
"""

part1_q2 = r"""
The text generation uses the model's hidden state, whice does not depand on the length of the sequences, so generating text of different lengths is possible.
"""

part1_q3 = r"""
The hidden state is passed between batches to improve the learning process. For the hidden state to be relevant to the new batch, the batches need to remain in order. Otherwise out model will pass irrelevant information that could worsen the learning process.
"""

part1_q4 = r"""
1. Using a higher temperature results in a more uniform distribution. We want to choose the most likely chars, and lowering the temperature results in them being chosen more often, and unlikely chars being chosen less.

2. A very high temperature results in a uniform distribution. As seen in the softmax formula:
$\text{hot_softmax}_T(y) = \frac{e^{y/T}}{\sum_k e^{y_k/T}}$,
when T (temperature) is very high, $e^{y/T}\rightarrow 1$ and the distribution becomes uniform.

3. A very low temperature results in a more erratic distribution. Chars with higher relative probabilities get even higher probabilities, and chars with lower ones become even lower.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""

When we train the GAN we train in turns the discriminator and the genrator, each has its own objective.
\\ Upon training the discriminator, the weights of the discriminators are updated using gradient descent toward a minimization of the binary cross entropy loss.
 Trying to adjust the classifier to assign the real data samples a label of 1 and to the fake samples label of 0 - it thus needs to update the weights of the discriminator only. 
The calculation of the gradients to update these weights do not depend on the calculation path of the generator and thus unneeded and we should turn off the gradients of the generator in this turn (note that the weights of the generator are not to be updated in this turn).
\\ Upon training the generator, the objective is to update its weights toward minimization of BCEloss trying to "cheat" the discriminator, and label fake data as true, thus the gradients of the generator should be turned on.
 In this turn the gradients of the discriminator should also be turned on as the calculation for the gradients of the discriminator depends on the calculation path in the discriminator by backpropagation.


"""

part2_q2 = r"""

1. A generator loss can achieve very low values and pass a given threshold while the discriminator is still untrained and perform very poorly - resulting with poor results of generation as the generator learned how to "cheat" only an untrained discriminator (underfitting). Another example for situtation such that the gen loss gets low while the results can  be  better are if the generator gets stuck predicting around some constant example, in that case a method of identifying the situation is needed (visual inspection of the results for example) (Mode collapse).Also, the loss of the generator typically oscilates alot while training GANs. The losses by themselves are not nessecerily perfect indicators for good learning, but a parameter to look at while training.
The visual inspection of the generated results can be very informational.  
\\

2. If the discriminator loss remains at a constant value while the generator loss decreases it means that the generator is fooling the discriminator faster than the discrimnator adapts. Might be because several of reasons: 
1) The generator got good at generating some samples and is still improving in a way which fools the discriminator in spots it can't seem to improve at. 
2) The discriminator got very good at the classification task and can't improve anymore, while the generator still hasn't reached its "fooling" potential and is still catching up.
3) insufficent capacity as a clasifier for the discriminator or poor choice of hyperparameters for the optimizer chosen for the generator or the discriminator.



"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

def part3_gan_hyperparams():
    gen_lr = 0.0002
    dsc_lr = 0.0001
    betas = (0.5 , 0.999)
    dsc_optim_params= { 'type' : "Adam" , 'lr' : dsc_lr, 'betas': betas }
    gen_optim_params= { 'type' : "Adam" , 'lr' : gen_lr, 'betas': betas }
    hypers= dict(
        batch_size = 128,
        z_dim = 100,
        discriminator_optimizer = dsc_optim_params ,
        generator_optimizer =  gen_optim_params ,
        data_label = 1,
        label_noise = 0.001

    )
    # Tweak them

    return hypers 


#PART3_CUSTOM_DATA_URL = "http://openminded-cat.static.domains/spongebob_pics_dataset.zip"
#PART3_CUSTOM_DATA_URL = "https://drive.google.com/uc?export=download&id=1oBkD8mtp2eJjOwhmC_IR9Df17-xnbIY7&output=/spongebob_pics_dataset.zip" 
PART3_CUSTOM_DATA_URL = "https://github.com/NatPath/spongebob_images_dataset/raw/main/spongebob_pics.zip"
#PART3_CUSTOM_DATA_URL = None



def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers["embed_dim"] = 256
    hypers["num_heads"] = 2
    hypers["num_layers"] = 3
    hypers["hidden_dim"] = 128 
    hypers["window_size"] = 32
    hypers["droupout"] = 0.1
    hypers["lr"] = 0.0002
    
    # ========================
    return hypers




part3_q1 = r"""
Stacking encoder layers increase the rception field, similarly to stacking convolution layers. Each layer of the encoder increase the distance of influence of a certain token as the output of the layer combines to the same index of the token all the tokens in that window. thus in a later layer a token on the left will see farther to the right.
Which means that as the depth increases the context broadens because the effect of all the layers accumulates.


"""

part3_q2 = r"""
SESWA -Smudged edges sliding window attention:
We propose a variation for the sliding window which enables slightly broader conext while maintaining the same running time complexity.

The variation goes as follows, given a window_size the same as in the sliding window algorithms, the window of attention for each token will be  of size (window_size-2).
Now, taking the edges of the original window and allowing them to travel randomly to another index which is not in the window (randomly by some distribution of the distance from the left/right edges- a distibution which would make sense here would be geometrical for example).
This allows the same masking scheme which enables O(n(w-1)) opperations, and taking into acount the smudges edges in O(n) in total O(nw). 

This scheme allows slightly broader context for each token as it can see farther from it in each step, the effect will also accumulate in a similar manner to that discribed in q1.

"""

part4_q1 = r"""
Compared to the model from part 3, which achieved 65% accuracy, this model is far better, with over 85% accuracy in both methods of fine-tuning. This model is more robust, was pre-trained on a larger data set, and was trained by someone with more experience than an undergrad student. It was also fine-tuned to get better results.
"""

part4_q2 = r"""
The model would not be able to fine-tune using this method. In pre-trained language models, the last layers are typically responsible for generating high-level, task-specific representations. Freezing these layers means that these representations remain fixed, and any updates made to the internal layers may not effectively align with the specific task.
"""


part4_q3= r"""
Since BERT is an encoder only model, and machine translation requires encoder-decoder (for proccessing input text and generating output text), BERT cannot be used for translation tasks.
"""

part4_q4 = r"""
The main reason to choose RNN over a transformer is its ability to Handle Long Sequences with Limited Resources.
RNNs process sequences sequentially, meaning they don’t require storing or computing pairwise attention over the entire input sequence like Transformers do. For tasks where you need to handle very long sequences with limited memory, RNNs can be more efficient since they only store hidden states across time steps rather than full attention matrices. This can be advantageous when computational resources are constrained.
"""

part4_q5 = r"""
NSP trains the model to determine whether or not two sentences are immidiately connected in a text, meaning one comes right after the other. Loss comes from the predicted certainty of both options. the more certain the model is of the correct answer (connected/not connected), the lower the loss is.
Next Sentence Prediction is a crucial part of pre-training, because it provides context for entire sentences, and not only words.
"""


# ==============

