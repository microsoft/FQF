# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


import atari_py
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

import gin.tf
import cv2

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.




@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  print ('STICKY_ACTIONS:', sticky_actions)
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = AtariPreprocessing(env)
  return env


def nature_dqn_network(num_actions, network_type, state, aux=False, next_state=None):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  net = slim.conv2d(net, 32, [8, 8], stride=4, scope='conv2d_1')
  net = slim.conv2d(net, 64, [4, 4], stride=2, scope='conv2d_2')
  net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2d_3')
  net = slim.flatten(net)

  net = slim.fully_connected(net, 512)
  q_values = slim.fully_connected(net, num_actions, activation_fn=None)
  return network_type(q_values, None)

#@profile
def rainbow_network(num_actions, num_atoms, num_atoms_sub, support, network_type, state, runtype='run', v_support=None, a_support=None, big_z=None, big_a=None, big_qv=None, N=1, index=None, M=None, sp_a=None, unique_num=None, sortsp_a=None, v_sup_tensor=None): #run, conv, convmean
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  state_input = net
  net = slim.conv2d(
      net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  feature = slim.flatten(net)
  feature_size = int(feature.shape[-1])
  a_origin, Ea = None, None
  net = slim.fully_connected(feature, 510, weights_initializer=weights_initializer)
  net = slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)
  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities, None, None, None, a_origin, Ea, None, None, None)

#@profile
def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
  """The Implicit Quantile ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state_net = tf.cast(state, tf.float32)
  state_net = tf.div(state_net, 255.)
  state_net = slim.conv2d(
      state_net, 32, [8, 8], stride=4,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [4, 4], stride=2,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [3, 3], stride=1,
      weights_initializer=weights_initializer)
  state_net = slim.flatten(state_net)
  print ('state_net:', state_net.shape, ", num_quan:", num_quantiles)
  state_net_size = state_net.get_shape().as_list()[-1]

  state_net_tiled = tf.tile(state_net, [num_quantiles, 1])
  batch_size = state_net.get_shape().as_list()[0]
  quantiles_shape = [batch_size * num_quantiles, 1]
  quantiles = tf.random_uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer)
  net = tf.multiply(state_net_tiled, quantile_net)
  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer)
  return network_type(quantile_values=quantile_values, quantiles=quantiles)

def fqf_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles, runtype='fqf'):
  """The FQF ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state_net = tf.cast(state, tf.float32)
  state_net = tf.div(state_net, 255.)
  state_net = slim.conv2d(
      state_net, 32, [8, 8], stride=4,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [4, 4], stride=2,
      weights_initializer=weights_initializer)
  state_net = slim.conv2d(
      state_net, 64, [3, 3], stride=1,
      weights_initializer=weights_initializer)
  state_net = slim.flatten(state_net)
  print ('state_net:', state_net.shape, ", num_quan:", num_quantiles)
  state_net_size = state_net.get_shape().as_list()[-1]

  quantile_values_origin = None
  quantiles_origin = None
  Fv_diff = None
  v_diff = None
  L_tau = None
  quantile_values_mid = None
  quantiles_mid = None
  gradient_tau = None
  quantile_tau = None

  batch_size = state_net.get_shape().as_list()[0]
  state_net1 = state_net

  quantiles_right = slim.fully_connected(state_net1, num_quantiles, weights_initializer=weights_initializer, scope='fqf', reuse=False, activation_fn=None)
  quantiles_right = tf.reshape(quantiles_right, [batch_size, num_quantiles])
  quantiles_right = tf.contrib.layers.softmax(quantiles_right) * (1 - 0.00)
  zeros = tf.zeros([batch_size, 1])
  quantiles_right = tf.cumsum(quantiles_right, axis=1)  #batchsize x 32
  quantiles_all = tf.concat([zeros, quantiles_right], axis=-1)   #33
  quantiles_left = quantiles_all[:, :-1]  #32
  quantiles_center = quantiles_all[:, 1:-1] #31, delete 0&1
  quantiles = quantiles_center
  quantiles_mid = (quantiles_right + quantiles_left) / 2  #batchsize x 32
  v_diff = quantiles_right - quantiles_left  #32
  v_diff = tf.transpose(v_diff, [1, 0])  #quan x batchsize

  quantile_tau = quantiles
  quantile_tau = tf.transpose(quantile_tau, [1, 0])  #quan x batchsize
  quantile_tau = tf.reshape(quantile_tau, [num_quantiles-1, batch_size])

  quantiles = tf.transpose(quantiles, [1, 0])  #quan-1 x batchsize
  quantiles = tf.reshape(quantiles, [(num_quantiles-1) * batch_size , 1])
  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = slim.fully_connected(quantile_net, state_net_size,
                                      weights_initializer=weights_initializer, scope='quantile_net')

  quantiles_mid = tf.transpose(quantiles_mid, [1, 0])  #quan x batchsize
  quantiles_mid = tf.reshape(quantiles_mid, [(num_quantiles)*batch_size, 1])
  quantile_net_mid = tf.tile(quantiles_mid, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net_mid = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net_mid
  quantile_net_mid = tf.cos(quantile_net_mid)
  quantile_net_mid = slim.fully_connected(quantile_net_mid, state_net_size,
                                      weights_initializer=weights_initializer, scope='quantile_net', reuse=True)
  # Hadamard product.
  state_net_tiled = tf.tile(state_net, [num_quantiles - 1, 1])
  net = tf.multiply(state_net_tiled, quantile_net)
  net = slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  quantile_values = slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer,
      scope='quantile_values_net')

  state_net_tiled1 = tf.tile(state_net, [num_quantiles, 1])
  net1 = tf.multiply(state_net_tiled1, quantile_net_mid)
  net1 = slim.fully_connected(
      net1, 512, weights_initializer=weights_initializer)
  quantile_values_mid = slim.fully_connected(
      net1,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer,
      scope='quantile_values_net', reuse=True)

  quantile_values = tf.reshape(quantile_values, [num_quantiles-1, batch_size, num_actions])
  quantile_values_mid = tf.reshape(quantile_values_mid, [num_quantiles, batch_size, num_actions])
  quantile_values_mid_1 = quantile_values_mid[:-1, :, :]
  quantile_values_mid_2 = quantile_values_mid[1:, :, :]
  sum_1 = 2 * quantile_values  #31
  sum_2 = quantile_values_mid_2 + quantile_values_mid_1  #31
  L_tau = tf.square(sum_1 - sum_2)  #31 x batchsize x action
  gradient_tau = sum_1 - sum_2
  print ("sum_1:", sum_1.shape)
  print ("L_tau:", L_tau.shape)
  quantile_values_mid = tf.reshape(quantile_values_mid, [-1, num_actions])  #32 x batchsize x action
  quantile_values_mid = tf.reshape(quantile_values_mid, [-1, num_actions])  #32 x batchsize x action

  quantiles_mid = tf.reshape(quantiles_mid, [-1, 1])  #32 x batchsize x action
  #quantile_values = quantile_values_mid
  #quantile = quantile_mid
  return network_type(quantile_values=quantile_values_mid, quantiles=quantiles_mid, quantile_values_origin=quantile_values_origin, quantiles_origin=quantiles_origin, Fv_diff=Fv_diff, v_diff=v_diff, quantile_values_mid=quantile_values_mid, quantiles_mid=quantiles_mid, L_tau=L_tau, gradient_tau=gradient_tau, quantile_tau=quantile_tau)


@gin.configurable
class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if is_terminal:
        break
      # We max-pool over the last two frames, in grayscale.
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                 out=self.screen_buffer[0])

    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
