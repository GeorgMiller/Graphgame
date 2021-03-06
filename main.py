import agent
import logger 
import numpy as np 
import tensorflow as tf 
import keras
import tensorflow.keras.initializers as init

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

epochs = 8
mini_batch_size = 4
decay_rate = 0.999995
learning_rate = 2.5e-5
lamBda = 1e-5
weights_init = init.glorot_uniform(seed)
pretraining_steps = 0
batch_size = 1
kl_diversity = False 
cosine_diversity = False

def normal_run_learning_rate():
    # This is the config to run the tests. First, four agents are initialized and trained for different learning_rates
    learning_rates = [1e-4, 6e-5, 3e-5, 1e-5, 6e-6, 3e-6]
    path = 'experiments/normal_run_learning_rate_ppo'

    for row, learning_rate in enumerate(learning_rates):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_normal()


def hypernetwork_learning_rate():
    # Then Hypernetwork is initialized and also trained for different learning rates with batch_size 1
    learning_rates = [7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6, 2.5e-6]
    path = 'experiments/hypernetwork_learning_rate'

    for row, learning_rate in enumerate(learning_rates):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_hypernetwork()


def hypernetwork_weight_init():
    # The best learningrate is selected and different weight initializations tested
    weights_inits_num = [init.VarianceScaling(1, seed=seed), init.VarianceScaling(0.1,seed=seed), init.VarianceScaling(0.01,seed=seed), init.glorot_uniform(seed)]
    path = 'experiments/hypernetwork_weight_init'

    for row, weights_init in enumerate(weights_inits_num):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_hypernetwork()


def hypernetwork_pretraining():
    # All of them are also tested with pretraining
    pretraining_num = [100, 500, 1000]
    path = 'experiments/hypernetwork_pretraining'


    for row, pretraining_steps in enumerate(pretraining_num):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_hypernetwork()


def hypernetwork_batch_size():
    # Evaluate different batch_sizes
    batch_sizes = [4, 8, 16]
    path = 'experiments/hypernetwork_batch_size'

    for row, batch_size in enumerate(batch_sizes):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_hypernetwork()


def hypernetwork_kl_diversity():
    # When the best network type is found, it is used to evaluate the different diversity terms
    lamBdas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    path = 'experiments/hypernetwork_kl_diversity'
    kl_diversity = True

    for i, lamBda in enumerate(lamBdas):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            i)
        worker.train_hypernetwork()

def hypernetwork_cosine_diversity():
# When the best network type is found, it is used to evaluate the different diversity terms
    lamBdas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    path = 'experiments/hypernetwork_cosine_diversity'
    cosine_diversity = True

    for row, lamBda in enumerate(lamBdas):
        worker = agent.Agent(epochs,
                            mini_batch_size,
                            learning_rate,
                            lamBda,
                            decay_rate,
                            weights_init,
                            pretraining_steps,
                            batch_size,
                            kl_diversity,
                            cosine_diversity,
                            path,
                            row)
        worker.train_hypernetwork()

hypernetwork_learning_rate()
hypernetwork_weight_init()
hypernetwork_pretraining()
hypernetwork_kl_diversity()
hypernetwork_cosine_diversity()
hypernetwork_batch_size()

# Plotting
#path = 'experiments/normal_run_learning_rate_ppo'
#logger.plot_diversity(path, 'A2C PPO', 6, 'steps', 'reward')