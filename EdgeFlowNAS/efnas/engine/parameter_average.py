"""Optional shadow parameters; never change the optimizer's training trajectory."""
import tensorflow as tf


class ParameterAverage:
    def __init__(self, trainables, model_variables, decay):
        self.decay = float(decay)
        if not 0 < self.decay < 1:
            raise ValueError('EMA decay must lie between zero and one')
        self.original = list(trainables)
        with tf.compat.v1.variable_scope('parameter_average'):
            self.shadows = [tf.Variable(tf.zeros(v.shape, v.dtype.base_dtype), trainable=False,
                name=v.op.name.replace('/', '__')) for v in self.original]
            self.count = tf.Variable(0, trainable=False, dtype=tf.int64, name='updates')
        self.variables = self.shadows + [self.count]
        self.initialize = tf.group(*[a.assign(v) for a,v in zip(self.shadows,self.original)], self.count.assign(0))
        # Call after a completed optimizer update, in a separate sess.run.
        updates = [a.assign(self.decay*a + (1-self.decay)*v) for a,v in zip(self.shadows,self.original)]
        with tf.control_dependencies(updates):
            self.update = self.count.assign_add(1)
        mapping = {v.op.name:a for v,a in zip(self.original,self.shadows)}
        for v in model_variables:
            if v.op.name.endswith(('/moving_mean','/moving_variance')):
                mapping[v.op.name] = v
        self.export_saver = tf.compat.v1.train.Saver(var_list=mapping, max_to_keep=0)
