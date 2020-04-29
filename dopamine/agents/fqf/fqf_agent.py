# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np



from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class FQFAgent(rainbow_agent.RainbowAgent):

  def __init__(self,
               sess,
               num_actions,
               network=atari_lib.fqf_network,
               kappa=1.0,
               runtype=None,
               fqf_factor=0.000001,
               fqf_ent=0.001,
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               double_dqn=False,
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      network: function expecting three parameters:
        (num_actions, network_type, state). This function will return the
        network_type object containing the tensors output by the network.
        See dopamine.discrete_domains.atari_lib.nature_dqn_network as
        an example.
      kappa: float, Huber loss cutoff.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    self._runtype= runtype
    print (self._runtype)
    self.fqf_factor = float(fqf_factor)
    self.ent = float(fqf_ent)
    self.kappa = kappa
    print ('fqf factor:', self.fqf_factor)
    # num_tau_samples = N below equation (3) in the paper.
    self.num_tau_samples = num_tau_samples
    # num_tau_prime_samples = N' below equation (3) in the paper.
    self.num_tau_prime_samples = num_tau_prime_samples
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.double_dqn = double_dqn
    if 'adam' in self._runtype:
        self.optimizer1 = tf.train.AdamOptimizer(
            learning_rate=0.00005 * self.fqf_factor,
            epsilon=0.0003125)
    else:
        self.optimizer1 = tf.train.RMSPropOptimizer(
            learning_rate=0.00005 * self.fqf_factor,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)

    super(FQFAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        network=network,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _get_network_type(self):
    return collections.namedtuple(
        'iqn_network', ['quantile_values', 'quantiles', 'quantile_values_origin', 'quantiles_origin', 'Fv_diff', 'v_diff', 'quantile_values_mid', 'quantiles_mid', 'L_tau', 'gradient_tau', 'quantile_tau'])

  def _network_template(self, state, num_quantiles):
    return self.network(self.num_actions, self.quantile_embedding_dim,
                        self._get_network_type(), state, num_quantiles, self._runtype)

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
        _, _, _, loss, loss1, quan_value, quan, vdiff = self._sess.run(self._train_op)
        if self.training_steps % 50000 == 0:
            batchsize = 32
            quan_value = np.reshape(quan_value, [batchsize, self.num_tau_samples])
            quan = np.reshape(quan, [batchsize, self.num_tau_samples])
            quan_value = quan_value[0].tolist()
            quan = quan[0].tolist()
            vdiff = vdiff[:, 0].tolist()
            print (">>> loss:", loss)
            print (">>> loss1:", loss1)
            print (">>> value:", quan_value)
            print (">>> quans:", quan)
            print (">>> vdiff:", vdiff)
            print (">>> vdiff_sum:", np.sum(vdiff))
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _build_networks(self):
    """Builds the FQF computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,
                                            self.num_quantile_samples)
    # Shape of self._net_outputs.quantile_values:
    # num_quantile_samples x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    if 'ws' in self._runtype:
        self._q_values = tf.reduce_sum(self._net_outputs.quantile_values * self._net_outputs.v_diff, axis=0)   #NOTE: quantile_values = quantile_values_mid
    else:
        self._q_values = tf.reduce_mean(self._net_outputs.quantile_values, axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)

    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   self.num_tau_samples)
    # Shape: (num_tau_samples x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    self._replay_net_quantiles = self._replay_net_outputs.quantiles

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.target_convnet(
        self._replay.next_states, self.num_tau_prime_samples)
    # Shape: (num_tau_prime_samples x batch_size) x num_actions.
    vals = self._replay_net_target_outputs.quantile_values
    self._replay_net_target_quantile_values = vals

    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    if self.double_dqn:
      outputs_action = self.online_convnet(self._replay.next_states,
                                           self.num_quantile_samples)
    else:
      outputs_action = self.target_convnet(self._replay.next_states,
                                           self.num_quantile_samples)

    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_quantile_values_action = outputs_action.quantile_values   #NOTE: quantile_values = quantile_values_mid
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                               [self.num_quantile_samples,
                                                self._replay.batch_size,
                                                self.num_actions])
    # Shape: batch_size x num_actions.
    if 'ws' in self._runtype:
        v_diff = tf.reshape(outputs_action.v_diff, [self.num_quantile_samples, self._replay.batch_size, 1])
        self._replay_net_target_q_values = tf.squeeze(tf.reduce_sum(
        target_quantile_values_action * v_diff, axis=0))
    else:
        self._replay_net_target_q_values = tf.squeeze(tf.reduce_mean(
        target_quantile_values_action, axis=0))
    self._replay_next_qt_argmax = tf.argmax(
        self._replay_net_target_q_values, axis=1)

  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]
    # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
    rewards = self._replay.rewards[:, None]
    rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

    is_terminal_multiplier = 1. - tf.to_float(self._replay.terminals)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_tau_prime_samples, 1])

    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.

    replay_next_qt_argmax = tf.tile(
        self._replay_next_qt_argmax[:, None], [self.num_tau_prime_samples, 1])

    # Shape of batch_indices: (num_tau_prime_samples x batch_size) x 1.
    batch_indices = tf.cast(tf.range(
        self.num_tau_prime_samples * batch_size)[:, None], tf.int64)

    # Shape of batch_indexed_target_values:
    # (num_tau_prime_samples x batch_size) x 2.
    batch_indexed_target_values = tf.concat(
        [batch_indices, replay_next_qt_argmax], axis=1)

    # Shape of next_target_values: (num_tau_prime_samples x batch_size) x 1.
    target_quantile_values = tf.gather_nd(
        self._replay_net_target_quantile_values,
        batch_indexed_target_values)[:, None]

    return rewards + gamma_with_terminal * target_quantile_values

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    target_quantile_values = tf.stop_gradient(
        self._build_target_quantile_values_op())
    # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_quantile_values = tf.reshape(target_quantile_values,
                                        [self.num_tau_prime_samples,
                                         batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_prime_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_tau_prime_samples x 1.
    target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])

    # Shape of indices: (num_tau_samples x batch_size) x 1.
    # Expand dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    indices = tf.range(self.num_tau_samples * batch_size)[:, None]

    # Expand the dimension by one so that it can be used to index into all the
    # quantiles when using the tf.gather_nd function (see below).
    reshaped_actions = self._replay.actions[:, None]
    reshaped_actions = tf.tile(reshaped_actions, [self.num_tau_samples, 1])
    # Shape of reshaped_actions: (num_tau_samples x batch_size) x 2.
    reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

    chosen_action_quantile_values = tf.gather_nd(
        self._replay_net_quantile_values, reshaped_actions)
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', self._replay_net_quantile_values)
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_samples x 1 to prepare for computation of
    # Bellman errors.
    # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
    # in which the quantile values are tiled.
    chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                               [self.num_tau_samples,
                                                batch_size, 1])
    # Final shape of chosen_action_quantile_values:
    # batch_size x num_tau_samples x 1.
    chosen_action_quantile_values = tf.transpose(
        chosen_action_quantile_values, [1, 0, 2])   #batchsize x quan x 1

    ##########################################################################################
    reshaped_actions1 = self._replay.actions[:, None]
    reshaped_actions1 = tf.tile(reshaped_actions1, [self.num_tau_samples-1, 1])
    # Shape of reshaped_actions1: (num_tau_samples-1 x batch_size) x 2.
    indices1 = tf.range((self.num_tau_samples-1) * batch_size)[:, None]
    reshaped_actions1 = tf.concat([indices1, reshaped_actions1], axis=1)
    gradient_tau = tf.reshape(self._replay_net_outputs.gradient_tau, (-1, self.num_actions))  #31 x 32 x 18
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', gradient_tau)
    gradient_tau = tf.gather_nd(
        gradient_tau, reshaped_actions1)
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', gradient_tau)
    chosen_action_gradient_tau = tf.reshape(gradient_tau,
                                               [self.num_tau_samples-1,
                                                batch_size, 1])
    self.chosen_action_gradient_tau = tf.transpose(
        chosen_action_gradient_tau, [1, 0, 2])   #batchsize x quan x 1 (32 x 31 x 18)
    self.chosen_action_gradient_tau = self.chosen_action_gradient_tau[:,:,0]  #(32 x 31)
    ##########################################################################################

    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = target_quantile_values[:, :, None, :] - chosen_action_quantile_values[:, None, :, :]
    #if 'fqf12' in self._runtype and 'fixbugtarg' in self._runtype:
    #    print ("============================================================= fixbug")
    #    print (bellman_errors.shape, self._replay_net_outputs.v_diff.shape, self.num_tau_samples)
    #    bellman_errors = bellman_errors * self._replay_net_outputs.v_diff[:,:,None,None] * self.num_tau_samples
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = tf.to_float(
        tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
    huber_loss_case_two = tf.to_float(
        tf.abs(bellman_errors) > self.kappa) * self.kappa * (
            tf.abs(bellman_errors) - 0.5 * self.kappa)
    huber_loss = huber_loss_case_one + huber_loss_case_two

    # Reshape replay_quantiles to batch_size x num_tau_samples x 1
    replay_quantiles = tf.reshape(
        self._replay_net_quantiles, [self.num_tau_samples, batch_size, 1])
    replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])  #batchsize x quan x 1

    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    replay_quantiles = tf.to_float(tf.tile(
        replay_quantiles[:, None, :, :], [1, self.num_tau_prime_samples, 1, 1]))
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (tf.abs(tf.stop_gradient(replay_quantiles) - tf.stop_gradient(
        tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = tf.reduce_sum(quantile_huber_loss, axis=2)
    # Shape: batch_size x 1.
    loss = tf.reduce_mean(loss, axis=1)

    chosen_action_L_tau = tf.gather_nd(self._replay_net_outputs.L_tau, reshaped_actions)
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", chosen_action_L_tau.shape)
    loss1 = tf.reduce_mean(chosen_action_L_tau, axis=0)
    print (loss1.shape)

    update_priorities_op = tf.no_op()
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
      iqn_params, fqf_params = [], []
      params = tf.trainable_variables()
      for p in params:
          if 'fqf' in p.name and 'Target' not in p.name: fqf_params.append(p)
          else: iqn_params.append(p)
      print ("fqf_params:>>>>>>", fqf_params)
      print ("iqn_params:>>>>>>", iqn_params)
      #batchsize x quan
      #batchsize x quan
      #quan x batchsize
      print ('================================================')
      quantile_tau = tf.transpose(self._replay_net_outputs.quantile_tau, (1,0))
      q_entropy = tf.reduce_sum(-quantile_tau * tf.log(quantile_tau), axis=1) * 0.001
      #print (quantile_tau)  #32x31
      print ("q_entropy:", q_entropy)
      print (self.chosen_action_gradient_tau)  #32x31
      print (fqf_params)
      grads = tf.gradients(quantile_tau, fqf_params, grad_ys=self.chosen_action_gradient_tau)
      print (grads)
      grads_and_vars = [(grads[i], fqf_params[i]) for i in range(len(grads))]
      return self.optimizer.minimize(tf.reduce_mean(loss), var_list=iqn_params), \
              self.optimizer1.apply_gradients(grads_and_vars), \
              self.optimizer1.minimize(self.ent * tf.reduce_mean(-q_entropy), var_list=fqf_params), \
              tf.reduce_mean(loss), tf.reduce_mean(loss1), \
              tf.squeeze(chosen_action_quantile_values), \
              tf.squeeze(replay_quantiles[:,0,:,:]), \
              self._replay_net_outputs.v_diff
