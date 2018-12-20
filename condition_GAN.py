import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)
np.random.seed(1)


BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    labels = (a-1) > 0.5   # upper paintings (1), lower paintings (0), two classes
    labels = labels.astype(np.float32)
    return paintings, labels

art_labels = tf.placeholder(tf.float32, [None, 1])

with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])
    G_art = tf.concat((G_in, art_labels), 1)
    G_l1 = tf.layers.dense(G_art, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)

with tf.variable_scope('Discriminator'):
    real_in = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')
    real_art = tf.concat((real_in, art_labels), 1)
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='1')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')
    G_art = tf.concat((G_out, art_labels), 1)
    D_l1 = tf.layers.dense(G_art, 128, tf.nn.relu, name='1', reuse=True)
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)


D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))


train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()
for step in range(7000):
    artist_paintings, labels = artist_works()
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_paintings, pa0, D1 = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],
                                    {G_in:G_ideas, real_in:artist_paintings, art_labels:labels})[:3]
    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
        bound = [0, 0.5] if labels[0, 0] == 0 else [0.5, 1]
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score=%.2f (-1.38 for G to converge)' % -D1, fontdict={'size': 15})
        plt.text(-.5, 1.7, 'class = %i' % int(labels[0, 0]), fontdict={'size': 15})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.01)
plt.ioff()


plt.figure(2)
z = np.random.randn(1, N_IDEAS)
label = np.array([[1.]])
G_paintings = sess.run(G_out, {G_in: z, art_labels: label})
plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='G painting for upper class', )
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 1)')
plt.ylim((0, 3))
plt.legend(loc='upper right', fontsize=12)
plt.show()