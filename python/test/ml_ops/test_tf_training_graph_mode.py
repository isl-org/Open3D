# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""This test simulates a network training by running some ops in graph mode for
tensorflow to catch a bug observed when linking the open3d main lib.

The error is not deterministic. The most frequent message is:

2020-11-21 23:07:56.653976: E tensorflow/stream_executor/cuda/cuda_event.cc:29] Error polling for event status: failed to query event: CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
2020-11-21 23:07:56.654028: F tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc:220] Unexpected Event status: 1

followed by segfault.

We found that the bug can be reproduced with 125458ad when linking the main lib
and using cmake 3.13.2. The child commit 14c4815d does not show the problem when
linking the main lib. For cmake 3.18.2 and cmake 3.19 we cannot reproduce the
bug with 125458ad . Further the diff between both commits does not show changes
related to the problem. Since we know that the problem can be resolved by using
cmake >= 3.18.2, we think that the way cmake generates the link command may
cause the problem.

Some more info about the systems on which the problem was discovered:
    Python 3.7.4
    Tensorflow 2.3.0
    CUDA 10.1 and 10.2
    CMake 3.13.2 and 3.12.4
"""

import open3d as o3d
import numpy as np
np.set_printoptions(linewidth=600)
np.set_printoptions(threshold=np.inf)
import pytest
import mltest


@mltest.parametrize.ml
def test_training_graph_mode(ml):
    # the problem is specific to tensorflow
    if ml.module.__name__ != 'tensorflow':
        return
    # the problem is specific to CUDA
    if not 'GPU' in ml.device:
        return

    tf = ml.module

    rng = np.random.RandomState(123)
    from particle_network_tf import MyParticleNetwork
    model = MyParticleNetwork()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6)

    batch_size = 16

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b)**2, axis=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = tf.exp(-neighbor_scale * num_fluid_neighbors)
        return tf.reduce_mean(importance *
                              euclidean_distance(pr_pos, gt_pos)**gamma)

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch):
        with tf.GradientTape() as tape:
            losses = []

            for batch_i in range(batch_size):
                inputs = ([
                    batch[batch_i]['pos0'], batch[batch_i]['vel0'], None,
                    batch[batch_i]['box'], batch[batch_i]['box_normals']
                ])

                pr_pos1, pr_vel1 = model(inputs)

                l = 0.5 * loss_fn(pr_pos1, batch[batch_i]['pos1'],
                                  model.num_fluid_neighbors)

                inputs = (pr_pos1, pr_vel1, None, batch[batch_i]['box'],
                          batch[batch_i]['box_normals'])
                pr_pos2, pr_vel2 = model(inputs)

                l += 0.5 * loss_fn(pr_pos2, batch[batch_i]['pos2'],
                                   model.num_fluid_neighbors)
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    for iteration in range(100):
        batch = []
        for batch_i in range(batch_size):
            num_particles = rng.randint(1000, 2000)
            num_box_particles = rng.randint(3000, 4000)
            batch.append({
                'pos0':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_particles, 3)).astype(np.float32)
                    ),
                'vel0':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_particles, 3)).astype(np.float32)
                    ),
                'pos1':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_particles, 3)).astype(np.float32)
                    ),
                'pos2':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_particles, 3)).astype(np.float32)
                    ),
                'box':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_box_particles,
                                          3)).astype(np.float32)),
                'box_normals':
                    tf.convert_to_tensor(
                        rng.uniform(size=(num_box_particles,
                                          3)).astype(np.float32)),
            })

        current_loss = train(model, batch)

    assert (True)  # The test is successful if this line is reached
