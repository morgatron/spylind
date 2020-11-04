import tensorflow as tf


class InterpolatorMask(tf.keras.Model):
    def __init__(self, xOrig, yOrig): #pars are parameters to the ode
        super().__init__()
        self.xOrig = tf.constant(tf.convert_to_tensor(xOrig, dtype=tf.float64))
        self.yOrig = tf.Variable(tf.convert_to_tensor(yOrig, dtype=tf.float64), name='yOrig')

        self.N = xOrig.shape[0]
        self.dx = xOrig[1]-xOrig[0]
        self.x0 = xOrig[0]
        self.xMax = xOrig[-1]
        self.zero = tf.constant(0, dtype=tf.float64)
        self.mask = tf.concat([tf.constant([0.5,0.5], dtype='float64'), tf.zeros(xOrig.shape[0]-2, dtype='float64') ], axis=0)
    
    def set_y(self, yNew):
        self.yOrig.assign(yNew)
    def call(self, x):
        #pdb.set_trace()
        #ind = tf.math.floormod((x-x0), dx)
        if x>=self.xMax or x<self.x0:
            retVal= self.zero#tf.constant(0., dtype=tf.float64);
        else:
            ind = tf.math.floordiv(x-self.x0, self.dx)
            remainder = x- ind*self.dx
            ind = tf.cast(ind, tf.int64)
            res = tf.roll(self.mask, ind, axis=0)*self.yOrig
            retVal = tf.reduce_sum(res)
        #tf.print(retVal)
        return retVal

class InterpolatorMaskArgs(tf.keras.Model):
    def __init__(self, xOrig): #pars are parameters to the ode
        super().__init__()
        self.xOrig = tf.constant(tf.convert_to_tensor(xOrig, dtype=tf.float64))
        self.N = xOrig.shape[0]
        self.dx = xOrig[1]-xOrig[0]
        self.x0 = xOrig[0]
        self.xMax = xOrig[-1]
        self.zero = tf.constant(0, dtype=tf.float64)
        self.mask = tf.concat([tf.constant([0.5,0.5], dtype='float64'), tf.zeros(xOrig.shape[0]-2, dtype='float64') ], axis=0)
    
    def call(self, x, yOrig):
        #pdb.set_trace()
        #ind = tf.math.floormod((x-x0), dx)
        if x>=self.xMax or x<self.x0:
            return self.zero#tf.constant(0., dtype=tf.float64);
        else:
            ind = tf.math.floordiv(x-self.x0, self.dx)
            remainder = x- ind*self.dx
            ind = tf.cast(ind, tf.int64)
            res = tf.roll(self.mask, ind, axis=0)*yOrig
            return tf.reduce_sum(res)
        return f

def tf_interpolator(xOrig, yOrig):
    xOrig = tf.constant(tf.convert_to_tensor(xOrig, dtype=tf.float64))
    yOrig = tf.constant(tf.convert_to_tensor(yOrig, dtype=tf.float64))
    
    N = xOrig.shape[0]
    dx = xOrig[1]-xOrig[0]
    x0 = xOrig[0]
    xMax = xOrig[-1]
    zero = tf.constant(0, dtype=tf.float64)
    #@tf.function(experimental_compile=True)
    def f(x):
        #pdb.set_trace()
        #ind = tf.math.floormod((x-x0), dx)
        
        if x>=xMax or x<x0:
            return zero#tf.constant(0., dtype=tf.float64);
        else:
            ind_f = tf.math.floordiv(x-x0, dx)
            remainder = (x/dx- ind_f)
            ind=tf.cast(ind_f, tf.int64)
            #tf.print(remainder)
            return (1.-remainder)*yOrig[ind] + remainder*yOrig[ind+1]
    return f
def tf_interpolator2(xOrig, yOrig):
    xOrig = tf.constant(tf.convert_to_tensor(xOrig, dtype=tf.float64))
    #yOrig = tf.constant(tf.convert_to_tensor(yOrig, dtype=tf.float64))
    
    N = xOrig.shape[0]
    dx = xOrig[1]-xOrig[0]
    x0 = xOrig[0]
    xMax = xOrig[-1]
    zero = tf.constant(0, dtype=tf.float64)
    mask = tf.concat([tf.constant([0.5,0.5], dtype='float64'), tf.zeros(xOrig.shape[0]-2, dtype='float64') ], axis=0)
    def f(x):
        #pdb.set_trace()
        #ind = tf.math.floormod((x-x0), dx)
        
        if x>=xMax or x<x0:
            return zero#tf.constant(0., dtype=tf.float64);
        else:
            ind = tf.math.floordiv(x-x0, dx)
            remainder = x- ind*dx
            ind=tf.cast(ind, tf.int64)
            res = tf.roll(mask, ind, axis=0)*yOrig
            return tf.reduce_sum(res)
    return f
