# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import tensorflow as tf
import open3d.ml.tf as ml3d
import numpy as np


class MyParticleNetwork(tf.keras.Model):

    def __init__(
        self,
        kernel_size=[4, 4, 4],
        radius_scale=1.5,
        coordinate_mapping='ball_to_cube_volume_preserving',
        interpolation='linear',
        use_window=True,
        particle_radius=0.025,
        timestep=1 / 50,
    ):
        super().__init__(name=type(self).__name__)
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(name=name,
                           kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)
            conv = Conv(name='conv{0}'.format(i), filters=ch, activation=None)
            self.denses.append(dense)
            self.convs.append(conv)

    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * tf.constant([0, -9.81, 0])
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.filter_extent)

        fluid_feats = [tf.ones_like(pos[:, 0:1]), vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = tf.concat(fluid_feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = tf.concat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = tf.keras.activations.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction

    def call(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs

        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected

    def init(self, feats_shape=None):
        """Runs the network with dummy data to initialize the shape of all variables"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        if feats_shape is None:
            feats = None
        else:
            feats = np.zeros(shape=feats_shape, dtype=np.float32)
        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        _ = self.__call__((pos, vel, feats, box, box_feats))
