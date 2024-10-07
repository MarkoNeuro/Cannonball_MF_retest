import numpyro

numpyro.set_host_device_count(
    4
)  # Necessary to make numpyro realise we have more than one CPU
import jax

# set jax to use cpu
jax.config.update("jax_platform_name", "cpu")
import numpy as np
import jax.numpy as jnp
from numpyro import distributions as dist
from functools import partial
from behavioural_modelling.decision_rules import softmax
from behavioural_modelling.utils import choice_from_action_p  
from typing import Tuple, Union
import matplotlib.pyplot as plt

# import colormaps as cmaps
import os, requests

# from matplotlib import font_manager, pyplot as plt

# # Some code to make figures look nicer
# url = 'https://github.com/google/fonts/blob/main/ofl/heebo/Heebo%5Bwght%5D.ttf?raw=true'
# r = requests.get(url)
# if r.status_code == 200:
#     with open('./Heebo.ttf', 'wb') as f: f.write(r.content)
# font_manager.fontManager.addfont('./Heebo.ttf')
# plt.rcParams.update({'lines.linewidth': 1, 'lines.solid_capstyle': 'butt', 'legend.fancybox': True, 'axes.facecolor': 'fafafa', 'savefig.edgecolor': 'fafafa', 'savefig.facecolor': 'fafafa', 'figure.subplot.left': 0.08, 'figure.subplot.right': 0.95, 'figure.subplot.bottom': 0.07, 'figure.facecolor': 'fafafa', 'figure.dpi': 80, 'lines.color': '383838', 'patch.edgecolor': '383838', 'text.color': '383838', 'axes.edgecolor': '383838', 'axes.labelcolor': '383838', 'xtick.color': '616161', 'ytick.color': '616161', 'font.family': 'Heebo', 'font.weight': 'regular', 'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

@jax.jit
def asymmetric_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome_chosen: Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    alpha_p: float,
    alpha_n: float,
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    """
    Updates the estimated value of a state or action using the Asymmetric Rescorla-Wagner learning rule.

    The function calculates the prediction error as the difference between the actual outcome and the current
    estimated value. It then updates the estimated value based on the prediction error and the learning rate,
    which is determined by whether the prediction error is positive or negative.

    Value estimates are only updated for chosen actions. For unchosen actions, the prediction error is set to 0.

    Args:
        value (float): The current estimated value of a state or action.
        outcome_chosen (Tuple[float, float]): A tuple containing the actual outcome and a binary value indicating
        whether the action was chosen.
        alpha_p (float): The learning rate used when the prediction error is positive.
        alpha_n (float): The learning rate used when the prediction error is negative.

    Returns:
        Tuple[float, float]: The updated value and the prediction error.
    """
    # Unpack the outcome and the chosen action
    outcome, chosen = outcome_chosen
    #chosen = jnp.expand_dims(chosen, axis=-1)# chosen has shape (10, 180) and needs to be expanded to (10, 180, 1)
  # This adds a new axis at the end

    # Calculate the prediction error
    prediction_error = outcome - value

    # Set prediction error to 0 for unchosen actions
    prediction_error = prediction_error * chosen

    # Set the learning rate based on the sign of the prediction error
    alpha_t = (alpha_p * (prediction_error > 0)) + (alpha_n * (prediction_error < 0))

    # Update the value
    updated_value = value + alpha_t * prediction_error

    return updated_value, (value, prediction_error)

def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome_key: Tuple[jax.typing.ArrayLike, jax.random.PRNGKey],
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
) -> np.ndarray:
    """
    Updates the value estimate using the asymmetric Rescorla-Wagner algorithm, and chooses an
    option based on the softmax function.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome_key (Tuple[jax.typing.ArrayLike, jax.random.PRNGKey]): A tuple containing the outcome and the PRNG key.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.

    Returns:
        Tuple[np.ndarray, Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]]:
            - updated_value (jnp.ndarray): The updated value estimate.
            - output_tuple (Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]):
                - value (jax.typing.ArrayLike): The original value estimate.
                - choice_p (jnp.ndarray): The choice probabilities.
                - choice (int): The chosen action.
                - choice_array (jnp.ndarray): The chosen action in one-hot format.
    """

    # Unpack outcome and key
    outcome, key = outcome_key

    # Get choice probabilities
    choice_p = softmax(value[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, (value, prediction_error) = asymmetric_rescorla_wagner_update(
        value,
        (outcome, choice_array),
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)

asymmetric_rescorla_wagner_update_choice = jax.jit(asymmetric_rescorla_wagner_update_choice, static_argnums=(5,))


def asymmetric_rescorla_wagner_update_choice_iterator(
    outcomes: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    key: jax.random.PRNGKey,
    n_trials: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Updates the value estimates using the asymmetric Rescorla-Wagner algorithm
    and generates choices for each trial.

    Args:
        outcomes (jax.typing.ArrayLike): The outcomes for each trial.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for the softmax
            function.
        n_actions (int): The number of actions to choose from.
        key (jax.random.PRNGKey): The random key.
        n_trials (int): The number of trials to simulate.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - values (jnp.ndarray): The value estimates.
            - choice_ps (jnp.ndarray): The choice probabilities.
            - choices (jnp.ndarray): The chosen actions.
            - prediction_errors (jnp.ndarray): The prediction errors.
    """

    # Use partial to create a function with fixed parameters
    asymmetric_rescorla_wagner_update_choice_partial = partial(
        asymmetric_rescorla_wagner_update_choice,
        alpha_p=alpha_p,
        alpha_n=alpha_n,
        temperature=temperature,
        n_actions=n_actions,
    )

    # Generate random keys using JAX
    keys = jax.random.split(key, n_trials)

    # Print shapes for debugging
    print(f"Outcomes shape: {outcomes.shape}")
    print(f"Keys shape: {keys.shape}")
    

    # Initialize the value estimates
    value = jnp.ones(2) * 0.5
    print(f"Value shape: {value.shape}")

    # Loop using scan
    _, (values, choice_ps, choices, prediction_errors) = jax.lax.scan(
        asymmetric_rescorla_wagner_update_choice_partial,
        value,
        (outcomes, keys),
    )

    return values, choice_ps, choices, prediction_errors


asymmetric_rescorla_wagner_update_choice_iterator = jax.jit(
    asymmetric_rescorla_wagner_update_choice_iterator, static_argnums=(4, 6)
)

asymmetric_rescorla_wagner_update_choice_iterator_vmap = jax.vmap(
    asymmetric_rescorla_wagner_update_choice_iterator,
    in_axes=(0, 0, 0, 0, None, None, None),
)

@jax.jit
def asymmetric_rescorla_wagner_update_iterator(
    outcomes: jax.typing.ArrayLike,
    choices: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Updates the value estimates using the asymmetric Rescorla-Wagner algorithm.

    Args:
        outcomes (jax.typing.ArrayLike): The outcomes for each trial.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - values (jnp.ndarray): The value estimates.
            - prediction_errors (jnp.ndarray): The prediction errors.
    """

    # Use partial to create a function with fixed parameters
    asymmetric_rescorla_wagner_update_partial = partial(
        asymmetric_rescorla_wagner_update,
        alpha_p=alpha_p,
        alpha_n=alpha_n,
    )

    # Initialize the value estimates
    value = jnp.ones(2) * 0.5

    # Loop using scan
    _, (values, prediction_errors) = jax.lax.scan(
        asymmetric_rescorla_wagner_update_partial,
        value,
        (outcomes, choices),
    )

    return values, prediction_errors

asymmetric_rescorla_wagner_update_iterator_vmap = jax.vmap(
    asymmetric_rescorla_wagner_update_iterator,
    in_axes=(None, 0, 0, 0),
)

# Simulate data
# Number of subjects
#N_SUBJECTS = 10

# Generate parameter values for each subject
#rng = np.random.default_rng(0)
#alpha_p = rng.beta(5, 5, size=N_SUBJECTS)
#alpha_n = rng.beta(5, 5, size=N_SUBJECTS)
#temperature = rng.beta(5, 5, size=N_SUBJECTS)

# Number of trials
N_TRIALS = 180

# Reward probabilities for each of our actions
reward_probs = jnp.array([0.7, 0.3])

# Generate rewards for each trial for each action using Numpy
# There's no need to use JAX for this
rng = np.random.default_rng(0)
rewards = rng.binomial(n=1, p=reward_probs, size=(N_TRIALS, len(reward_probs)))

# Run the model for each subject
#_, _, choices, _ = asymmetric_rescorla_wagner_update_choice_iterator_vmap(
#    rewards,
#    alpha_p,
#    alpha_n,
#    temperature,
#    2,
#    jax.random.PRNGKey(0),
#    N_TRIALS,
#)

#Model fitting
# Run the model for each subject
#values, _ = asymmetric_rescorla_wagner_update_iterator_vmap(
#    rewards,
#    choices,
#    alpha_p,
#    alpha_n,
#)
#choice_p = jax.vmap(softmax, in_axes=(0, 0))(values, temperature)

def create_subject_params(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.
    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects
    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    # Group-level mean and SD
    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 1))
    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(1))

    # Subject-level offset
    offset = numpyro.sample(
        "{0}_subject_offset".format(name), dist.Normal(0, 1), sample_shape=(n_subs,)  # One value per subject
    )

    # Calculate subject-level parameter
    subject_param = numpyro.deterministic("{0}_subject_param".format(name), jax.scipy.special.expit(group_mean + offset * group_sd))

    return subject_param

def asymmetric_rescorla_wagner_statistical_model(
    outcomes: jnp.ndarray,
    choices: jnp.ndarray,
) -> None:
    """
    Asymmetric Rescorla-Wagner model for NumPyro.

    This forms a hierarchical model using non-centred parameterisation.

    Args:
        outcomes (jnp.ndarray): The outcomes for each trial.
        choices (jnp.ndarray): The choices for each trial.

    Returns:
        None: The function does not return anything; it only samples from the model.
    """

    # Get number of subjects based on choices
    n_subs = choices.shape[0]

    # Create subject-level parameters
    alpha_p = create_subject_params("alpha_p", n_subs)
    alpha_n = create_subject_params("alpha_n", n_subs)
    temperature = create_subject_params("temperature", n_subs) * 9 + 1


    # Run the model for each subject
    values, _ = asymmetric_rescorla_wagner_update_iterator_vmap(
        rewards,
        choices,
        alpha_p,
        alpha_n,
    )

    # Get choice probabilities using inverse temperature
    choice_p = jax.vmap(softmax, in_axes=(0, 0))(values, temperature)

    # Bernoulli likelihood
    numpyro.sample(
        "observed_choices",
        dist.Bernoulli(probs=choice_p),
        obs=choices,
    )