from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#tf.initializer_all_variables() 
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0] < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
    
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
