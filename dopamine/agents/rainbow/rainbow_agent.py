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
"""Compact implementation of a simplified Rainbow agent.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv, json, pickle
import multiprocessing



from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf
import random
import numpy as np

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class RainbowAgent(dqn_agent.DQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=atari_lib.rainbow_network,
               runtype='run',
               game=None,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: function expecting three parameters:
        (num_actions, network_type, state). This function will return the
        network_type object containing the tensors output by the network.
        See dopamine.discrete_domains.atari_lib.rainbow_network as
        an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.ACTION_MEANING = {
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",
        4: "LEFT",
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT",
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE",
    }
    # We need this because some tools convert round floats into ints.
    self.testing = False
    vmax = float(vmax)
    self.N = 1
    self.index = None
    self.big_z = None
    self.big_a = None
    self.big_qv = None
    self.unique_num = None
    self.v_sup_tensor = None #[tf.constant(vv, dtype=tf.float32) for vv in v_sup_]
    self.M = self.sp_a = self.sortsp_a = None
    if 'rainbow' in runtype:
        num_atoms = int(runtype.split('-')[0][7:])
        num_atoms_sub = self._num_atoms_sub = num_atoms
    elif 'c51' in runtype:
        num_atoms_sub = self._num_atoms_sub = 51
    else:
        num_atoms_sub = num_atoms
    self._num_atoms = num_atoms
    self._runtype = runtype
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self.v_sup_ = np.array(np.linspace(-vmax, vmax, num_atoms))
    self.v_support = tf.linspace(-vmax, vmax, num_atoms_sub)
    self.a_support = tf.linspace(-vmax, vmax, num_atoms_sub)  #V + k * A
    self._game = game
    self._klfactor = 0.0
    self._entfactor = 0
    self._ckfactor = 0
    self._num_actions = num_actions
    print ("NUM_ATOMS:", num_atoms)
    self.q_sup = []
    for i in range(self.N):
        self.q_sup.append(np.reshape(np.linspace(-10, 10, num_atoms), (1, num_atoms)))
    print ("RUNTYPE:", runtype)
    print ("VMAX:", vmax)
    print (">>>>>>>>", multiprocessing.cpu_count())
    self.k = 1.0
    print ("-------kkkkk =", self.k)
    self.v_support = tf.reshape(self.v_support, (1, 1, num_atoms_sub))
    self.a_support = tf.reshape(self.a_support, (1, 1, num_atoms_sub))

    self._replay_scheme = replay_scheme
    self._filename = './fout-visual/visual-%s_%s.pkl' % (self._game, self._runtype)
    self._filename_vis = './vis-%s_%s.pkl' % (self._game, self._runtype)
    print ('==========================', self._filename)
    self.dict = {"training_step": [],
                "state": [],
                "prob": [],
                "loss": [],
                "klloss": []
                }
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.vis = {"sup": [], "v": [], "s": [], "a": [], "supF": [], "F": [], "r": [], "state_input":[], "gradient":[],
            "fraction": [], 'values': []}
    self.vis['sup'] = self.q_sup #self.v_sup_
    self.vis['supF'] = self.v_sup_
    self.optimizer = optimizer

    dqn_agent.DQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)
    print (self._sess.run(self._support))
    #exit(0)

  def _get_network_type(self):
    """Returns the type of the outputs of a value distribution network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('c51_network',
                                  ['q_values', 'logits', 'probabilities', 'v_', 'a', 'q_v', 'a_origin', 'Ea', 'q_support', 'a_support', 'q_values_sub', 'state_input'])

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self.gradient = []
    for q_sub in self._net_outputs.q_values_sub:
        print (self._q_argmax.shape)
        chosen_action_q_sub = q_sub[0][self._q_argmax] ##Z_1(a), Z_2(a)
        print ("chosen_action_q_sub:", chosen_action_q_sub.shape)
        self.gradient.append(tf.gradients(chosen_action_q_sub, [self._net_outputs.state_input]))

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    return self.network(self.num_actions, self._num_atoms, self._num_atoms_sub, self._support,
                        self._get_network_type(), state, self._runtype,
                        self.v_support, self.a_support, self.big_z, self.big_a, self.big_qv, self.N, self.index, self.M, self.sp_a, self.unique_num, self.sortsp_a, self.v_sup_tensor)
                        #self.v_support, self.a_support, self.big_z, self.idx_matrix_add_onehot, self.idx_matrix_minus_onehot)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self.action = self._select_action()
    return self.action

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        if 'iqn' in self._runtype:
            self._sess.run(self._train_op)
        else:
            q_sup = None; a_sup = None
            pv, pa = None, None
            a_origin, Ea = None, None
            if 'rainbow' in self._runtype or 'c51' in self._runtype:
                _, loss, prob, targ, states, allQ = self._sess.run(self._train_op)
            if self.training_steps % 100000 == 0:
                print (np.sum(prob, -1))
                print (loss[0])
                tmp = [targ[0], prob[0], allQ[0]]
                if pv is not None:
                    tmp.extend(pv)
                    tmp.extend(self.v_sup_)
                #print (self.training_steps, states.shape, prob.shape, pv.shape, pa.shape, allQ.shape)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1


  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    # Choose the action with highest Q-value at the current state.
    #q_argmax, p = self._sess.run([self._q_argmax, self._net_outputs.probabilities], {self.state_ph: self.state})
    if self.testing:
        #print (self.subcontrol)
        if 'iqn' not in self._runtype:
            q_argmax, v_ = self._sess.run([self._q_argmax, self._net_outputs.probabilities], {self.state_ph: self.state})
            v_ = v_[0, q_argmax, :]
            self.vis['v'].append(v_)
        else:
            q_argmax = self._sess.run(self._q_argmax, {self.state_ph: self.state})
    else:
        q_argmax = self._sess.run(self._q_argmax, {self.state_ph: self.state})
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      return q_argmax


  def _build_target_distribution(self, q_support=None):
    """Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    
    if q_support is not None:
        _support = q_support
        batch_size = self._replay.batch_size

        # size of rewards: batch_size x 1
        rewards = self._replay.rewards[:, None]

        # size of tiled_support: batch_size x num_atoms
        #tiled_support = tf.tile(_support, [batch_size])
        #tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])
        tiled_support = _support

        # size of target_support: batch_size x num_atoms

        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: batch_size x 1
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = gamma_with_terminal[:, None]

        target_support = rewards + gamma_with_terminal * tiled_support

        # size of next_qt_argmax: 1 x batch_size
        next_qt_argmax = tf.argmax(
            self._replay_next_target_net_outputs.q_values, axis=1)[:, None]
        batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
        # size of next_qt_argmax: batch_size x 2
        batch_indexed_next_qt_argmax = tf.concat(
            [batch_indices, next_qt_argmax], axis=1)

        # size of next_probabilities: batch_size x num_atoms
        next_probabilities = tf.gather_nd(
            self._replay_next_target_net_outputs.probabilities,
            batch_indexed_next_qt_argmax)

        return project_distribution_1(target_support, next_probabilities,
                                    _support)
    else:
        _support = self._support
        batch_size = self._replay.batch_size

        # size of rewards: batch_size x 1
        rewards = self._replay.rewards[:, None]

        # size of tiled_support: batch_size x num_atoms
        tiled_support = tf.tile(_support, [batch_size])
        tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

        # size of target_support: batch_size x num_atoms

        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: batch_size x 1
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = gamma_with_terminal[:, None]

        target_support = rewards + gamma_with_terminal * tiled_support

        # size of next_qt_argmax: 1 x batch_size
        next_qt_argmax = tf.argmax(
            self._replay_next_target_net_outputs.q_values, axis=1)[:, None]
        batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
        # size of next_qt_argmax: batch_size x 2
        batch_indexed_next_qt_argmax = tf.concat(
            [batch_indices, next_qt_argmax], axis=1)

        # size of next_probabilities: batch_size x num_atoms
        next_probabilities = tf.gather_nd(
            self._replay_next_target_net_outputs.probabilities,
            batch_indexed_next_qt_argmax)

        return project_distribution(target_support, next_probabilities,
                                    _support)


  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    target_distribution = tf.stop_gradient(self._build_target_distribution(self._replay_net_outputs.q_support))

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.probabilities)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    print ("replay_action.shape, ", self._replay.actions.shape)
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_probabilities = tf.gather_nd(self._replay_net_outputs.probabilities,
                                        reshaped_actions)
    print ("----------------------------------------------------------")
    print (self._replay_net_outputs.probabilities.shape, reshaped_actions.shape, chosen_action_probabilities.shape)
    all_action_probabilities = self._replay_net_outputs.probabilities
    cross_entropy = -1 * target_distribution * tf.log(chosen_action_probabilities + 1e-8)
    loss = tf.reduce_sum(cross_entropy, axis=-1)
    original_loss = loss
    #loss = tf.reduce_mean(loss, axis=-1)
    print (">>>>>>>>>>>>>>loss-prob:", loss.shape)
    print (self._replay_net_outputs.a)

    if self._replay_net_outputs.a is not None:
        chosen_pa = tf.gather_nd(self._replay_net_outputs.a, reshaped_actions)
        pa = self._replay_net_outputs.a

    '''
    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    loss1 = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)
    print (">>>>>>>>>>>>>>loss-logits:", loss1.shape)
    '''

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      var = tf.trainable_variables()
      print ("all trainable var ----------------------", var)
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss, chosen_action_probabilities,\
              target_distribution, self._replay.states, all_action_probabilities

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)


def project_distribution(supports, weights, target_support,
                         validate_args=False):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the
      target_support parameter.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[1:] - target_support[:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support.shape)
  target_support.shape.assert_has_rank(1)
  if validate_args:
    # Assert that supports and weights have the same shapes.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
            [supports, weights]))
    # Assert that elements of supports and target_support have the same shape.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(
                tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
            [supports, target_support]))
    # Assert that target_support has a single dimension.
    validate_deps.append(
        tf.Assert(
            tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
    # Assert that the target_support is monotonically increasing.
    validate_deps.append(
        tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
    # Assert that the values in target_support are equally spaced.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
            [target_support]))

  with tf.control_dependencies(validate_deps):
    # Ex: `v_min, v_max = 4, 8`.
    v_min, v_max = target_support[0], target_support[-1]
    # Ex: `batch_size = 2`.
    batch_size = tf.shape(supports)[0]
    # `N` in Eq7.
    # Ex: `num_dims = 5`.
    num_dims = tf.shape(target_support)[0]
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
    #                         [[ 4.  4.  4.  5.  6.]]]`.
    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
    # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]]
    #                        [[ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]]]]`.
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
    # Ex: `reshaped_target_support = [[[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]
    #                                 [[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]]`.
    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
    # numerator = `|clipped_support - z_i|` in Eq7.
    # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
    #                     [ 1.  1.  1.  1.  3.]
    #                     [ 2.  2.  2.  0.  2.]
    #                     [ 3.  3.  3.  1.  1.]
    #                     [ 4.  4.  4.  2.  0.]]
    #                    [[ 0.  0.  0.  1.  2.]
    #                     [ 1.  1.  1.  0.  1.]
    #                     [ 2.  2.  2.  1.  0.]
    #                     [ 3.  3.  3.  2.  1.]
    #                     [ 4.  4.  4.  3.  2.]]]]`.
    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  1.]]
    #                           [[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  1.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]]]]`.
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)
    # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
    #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
    weights = weights[:, None, :]
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
    # in Eq7.
    # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0.1]]
    #                     [[ 0.1  0.2  0.5  0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0.1]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]]]]`.
    inner_prod = clipped_quotient * weights
    # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
    #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection


def project_distribution_1(supports, weights, target_support,
                         validate_args=False):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the
      target_support parameter.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[0,1:] - target_support[0,:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support[0].shape)
  target_support.shape.assert_has_rank(2)
  if validate_args:
    # Assert that supports and weights have the same shapes.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
            [supports, weights]))
    # Assert that elements of supports and target_support have the same shape.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(
                tf.equal(tf.shape(supports)[1], tf.shape(target_support[0]))),
            [supports, target_support[0]]))
    # Assert that target_support has a single dimension.
    validate_deps.append(
        tf.Assert(
            tf.equal(tf.size(tf.shape(target_support[0])), 1), [target_support[0]]))
    # Assert that the target_support is monotonically increasing.
    validate_deps.append(
        tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support[0]]))
    # Assert that the values in target_support are equally spaced.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
            [target_support[0]]))

  with tf.control_dependencies(validate_deps):
    # Ex: `v_min, v_max = 4, 8`.
    v_min, v_max = target_support[:,0], target_support[:,-1]
    # Ex: `batch_size = 2`.
    batch_size = tf.shape(supports)[0]
    # `N` in Eq7.
    # Ex: `num_dims = 5`.
    num_dims = tf.shape(target_support[0])[0]
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
    #                         [[ 4.  4.  4.  5.  6.]]]`.
    #TODO
    #clipped_support = tf.map_fn(fn=lambda inp: tf.clip_by_value(inp[0], inp[1], inp[2]), elems=[supports, v_min, v_max], dtype=tf.float32)
    v_min1 = tf.tile(tf.reshape(v_min, [-1, 1]), [1, num_dims])
    v_max1 = tf.tile(tf.reshape(v_max, [-1, 1]), [1, num_dims])
    clipped_support = tf.minimum(tf.maximum(supports, v_min1), v_max1)
    #clipped_support = tf.clip_by_value(supports, v_min[0], v_max[0])
    print ('----', clipped_support.shape)
    clipped_support = clipped_support[:, None, :]
    print ('----', clipped_support.shape)
    # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]]
    #                        [[ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]]]]`.
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
    # Ex: `reshaped_target_support = [[[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]
    #                                 [[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]]`.
    reshaped_target_support = target_support #tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
    # numerator = `|clipped_support - z_i|` in Eq7.
    # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
    #                     [ 1.  1.  1.  1.  3.]
    #                     [ 2.  2.  2.  0.  2.]
    #                     [ 3.  3.  3.  1.  1.]
    #                     [ 4.  4.  4.  2.  0.]]
    #                    [[ 0.  0.  0.  1.  2.]
    #                     [ 1.  1.  1.  0.  1.]
    #                     [ 2.  2.  2.  1.  0.]
    #                     [ 3.  3.  3.  2.  1.]
    #                     [ 4.  4.  4.  3.  2.]]]]`.
    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  1.]]
    #                           [[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  1.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]]]]`.
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)
    # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
    #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
    weights = weights[:, None, :]
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
    # in Eq7.
    # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0.1]]
    #                     [[ 0.1  0.2  0.5  0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0.1]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]]]]`.
    inner_prod = clipped_quotient * weights
    # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
    #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection
