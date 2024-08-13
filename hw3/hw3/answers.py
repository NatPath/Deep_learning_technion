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
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

"""

part1_q2 = r"""
**Your answer:**

"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


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
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
