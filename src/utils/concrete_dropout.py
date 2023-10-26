"""
Implemenation of 'Concrete Dropout' (Gal et al.) https://doi.org/10.48550/arXiv.1705.07832.
This is the continuous analog to the usual discrete form of dropout.

"""
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dense, Lambda, Wrapper

class ConcreteDropout(Wrapper) :
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5) :
        super(ConcreteDropout, self).__init__(layer)

        self.weight_regularizer = K.cast_to_floatx(weight_regularizer)
        self.dropout_regularizer = K.cast_to_floatx(dropout_regularizer)
        self.mc_test_time = mc_test_time
        self.losses = []
        self.supports_masking = True


        def build(self, input_shape=None):
            assert len(input_shape) == 2 # TODO  test with more than two dims
            self.input_spec = InputSpec(shape = input_shape)

            if not self.layer.built:
                self.layer.build(input_shape)
                self.layer.built = True

            super(ConcreteDropout, self).build() # this is very weird .. we must call super before we add new losses

            # initialise p
            self.p_logit = self.add_weight((1 ,), initializers.RandomUniform(-2., 0.) , # ~0.1 to ~0.5 in logit space .
                                           name='p_logit', trainable=True)
            self.p = K.sigmoid(self.p_logit [0])
            # initialise regulariser / prior KL term
            input_dim = input_shape[-1] # we drop only last dim
            weight = self.layer.kernel
            # Note : we divide by (1 - p ) because we scaled layer output by (1 - p)
            kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
            dropout_regularizer = self.p * K.log(self.p)
            dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = K.sum(kernel_regularizer + dropout_regularizer)
            self.add_loss(regularizer)

            return None
        
        
        def compute_output_shape ( self , input_shape ) :
            return self . layer . compute_output_shape ( input_shape )
        
        
        def concrete_dropout (self, x):
            eps = K.cast_to_floatx(K.epsilon())
            temp = 1.0 / 10.0
            unif_noise = K.random_uniform(shape = K.shape(x))
            drop_prob = (K.log(self.p + eps) - K.log(1. - self.p + eps) + K.log(unif_noise + eps) - K.log(1. - unif_noise + eps))
            drop_prob = K.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob

            return x
        

        def call (self, inputs):
            return self.layer.call(self.concrete_dropout(inputs))

