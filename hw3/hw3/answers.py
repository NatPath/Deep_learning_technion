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
Upon training the discriminator the weights of the discriminators are updated using gradient descent toward a minimization of the binary cross entropy loss trying to adjust the classifier to assign the real data samples a label of 1 and to the fake samples label of 0 - 
it thus needs to update the weights of the discriminator only, the calculation of the gradients to update these weights do not depend on the calculation path of the generator and thus unneeded and we should turn off the gradients of the generator in this turn (note that the weights of the generator are not to be updated in this turn).
Upon training the generator, the objective is to update its weights toward minimization of BCEloss trying to "cheat" the discriminator, and label fake data as true, thus the gradients of the generator should be turned on. In this turn the gradients of the discriminator should also be turned on as the calculation for the gradients of the discriminator depends on the calculation path in the discriminator by backpropagation.


"""

part2_q2 = r"""


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
    lr = 0.0002
    betas = (0.5 , 0.999)
    dsc_optim_params= { 'type' : "Adam" , 'lr' : lr, 'betas': betas }
    gen_optim_params= { 'type' : "Adam" , 'lr' : lr, 'betas': betas }
    hypers= dict(
        batch_size = 128,
        z_dim = 100,
        discriminator_optimizer = dsc_optim_params ,
        generator_optimizer =  gen_optim_params ,
        data_label = 1,
        label_noise = 0.01

    )
    # Tweak them

    return hypers 


#PART3_CUSTOM_DATA_URL = "/home/nativ/Deep_learning_technion/hw3/spongebob_images"
PART3_CUSTOM_DATA_URL = None



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
