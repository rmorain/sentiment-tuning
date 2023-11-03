import logging
import math
import random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import pudb
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

    plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ColorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_actions,
        render_mode=None,
        max_prefix_length=10,
        max_prompt_length=100,
    ):
        self.max_prefix_length = max_prefix_length
        self.max_prompt_length = max_prompt_length
        # Observation space
        self.observation_space = spaces.Dict(
            {
                "prefix": spaces.Box(
                    0, num_actions - 1, shape=(max_prefix_length,), dtype=int
                ),
                "prompt": spaces.Box(
                    0, num_actions - 1, shape=(max_prompt_length,), dtype=int
                ),
            }
        )
        # Action space
        self.action_space = spaces.Discrete(num_actions)
        self._action_to_color = {0: "red", 1: "blue"}

        # Reward model
        self._reward_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def _tokenize(self, text):
        return (
            self._tokenizer(text, return_tensors="pt")["input_ids"]
            .to(device)
            .squeeze(0)
        )

    def _get_obs(self):
        if self._prefix is not None:
            return self._tokenizer.decode(torch.cat((self._prefix, self._prompt)))
        return self._tokenizer.decode(self._prompt.squeeze(0))

    def _get_info(self):
        return {
            "target": self._target,
            "prefix": self._prefix,
            "prompt": self._prompt,
            "target_str": self._tokenizer.decode(self._target),
            "prefix_str": self._tokenizer.decode(self._prefix)
            if self._prefix is not None
            else None,
            "prompt_str": self._tokenizer.decode(self._prompt.squeeze(0)),
        }

    def _compute_reward(self):
        with torch.no_grad():
            if self._prefix is not None:
                prefix_prompt = torch.cat((self._prefix, self._prompt))
            else:
                prefix_prompt = self._prompt
            labels = torch.full(prefix_prompt.shape, -100)
            labels[-len(self._target) :] = self._target
            output = self._reward_model(prefix_prompt, labels=labels.to(device))
            loss = output.loss
        return -loss.item()

    def _terminated(self):
        if self._prefix is not None:
            if self._prefix[-1] == self._tokenizer.eos_token:
                return True
        return False

    def reset(self, prompt, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._prefix = None
        self._prompt = self._tokenize(prompt)
        if len(self._prompt) > self.max_prompt_length:
            print(
                f"Prefix too long. Prefix length: {len(self._prompt)}. Max length: "
                "{self.max_prompt_length}"
            )
            return None
        targets = [" Red", " Blue"]
        self._target = self._tokenize(random.choice(targets))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if self._prefix is not None:
            self._prefix = torch.cat((self._prefix, action))  # Append next token
        else:
            self._prefix = action
        terminated = self._terminated()
        # Add color as prefix
        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()
        truncated = len(self._prefix) >= self.max_prefix_length
        return observation, reward, terminated, truncated, info


class ColorPolicy(nn.Module):
    def __init__(self, num_colors):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained("gpt2")
        self.vocab_size = self.transformer.config.vocab_size

    def forward(self, input_ids):
        transformer_outputs = self.transformer(input_ids)
        # hidden_states = transformer_outputs[0]
        # logits = self.lm_head(hidden_states)
        logits = transformer_outputs.logits
        last_token_logits = logits[:, -1, :]
        return last_token_logits


def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the logging level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


color_policy = ColorPolicy(2)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon,
# higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

log_file = "training.log"
logger = setup_logger(log_file)

n_actions = color_policy.vocab_size

env = ColorEnv(n_actions)
# Get the number of state observations
prompt = "I have a book. Can you guess the color of the book?"
state, info = env.reset(prompt)
n_observations = len(state)

policy_net = ColorPolicy(n_actions).to(device)
target_net = ColorPolicy(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    state = tokenizer(state, return_tensors="pt")["input_ids"].to(device)
    # Include target with state
    target = env._get_info()["target_str"].strip().lower()
    metaprompt = (
        "The following examplars show you how to generate a good prompt to help"
        " a model identify the color of a book it cannot see. The correct color of"
        " the book is given to you in curly brackets. The generated prompt is in "
        "quotation marks\n\nExample:\n{Target: red}\n"
        '"The book is red."\n\nExample:\n{Target: blue}\n"The book is blue."\n\n'
        "Here is the target for the current example:\n\n"
        "{" + f"Target: {target}" + "}\n"
    )
    target = tokenizer(metaprompt, return_tensors="pt")["input_ids"].to(device)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(target).max(1)[1].view(1, 1).to(device).squeeze(0)
    else:
        return torch.tensor(
            [env.action_space.sample()], device=device, dtype=torch.long
        )


episode_durations = []
episode_rewards = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_rewards(show_result=False):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        ax.clear()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards_t.numpy())

    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    state_batch = tokenizer(batch.state, return_tensors="pt", padding=True)[
        "input_ids"
    ].to(device)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = tokenizer(
        non_final_next_states, return_tensors="pt", padding=True
    )["input_ids"].to(device)

    # state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset(prompt)
    logger.info(f"Episode {i_episode + 1}: Starting...")
    # Tokenize state string
    rewards = []
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            mean_reward = sum(rewards) / len(rewards)
            episode_rewards.append(mean_reward)
            logger.info(f"Episode {i_episode + 1}: Completed.")
            info = env._get_info()
            logger.info(f"Episode {i_episode + 1}: Info.")
            for key, value in info.items():
                logger.info(f"{key} : {value}")
            logger.info(f"Episode reward : {mean_reward}")
            # plot_durations()
            # plot_rewards()
            break

print("Complete")
# plot_durations(show_result=True)
# plot_rewards()
# plt.ioff()
# plt.show()
rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
plt.title("Result")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(rewards_t.numpy())

if len(rewards_t) >= 100:
    pu.db
    means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.full((99,), -50), means))
    plt.plot(means.numpy())
plt.savefig("fig_red_blue_256_mean.png")
torch.save(policy_net.state_dict(), "policy_red_blue_256_mean.pt")
