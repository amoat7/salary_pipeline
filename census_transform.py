import tensorflow as tf 
import tensorflow_transform as tft 

NUMERIC_FEATURE_KEYS = [
    'age', 'capital-gain', 'capital-loss',
    'education-num', 'fnlwgt', 'hours-per-week'
]

VOCAB_FEATURE_DICT ={
    'education':16, 'marital-status':7, 'native-country':41,
    'occupation':15, 'race':5, 'relationship':6, 'sex':2,
    'workclass':9
}

NUM_OOV_BUCKETS = 2

LABEL_KEY = 'label'

def transformed_name(key):
    key = key.replace('-', '_')
    return key + '_xf'


def preprocessing_fn(inputs):

    outputs ={}

    for key in NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[transformed_name(key)] = tf.reshape(scaled, [-1])

    for key, vocab_size in VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets = NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + NUM_OOV_BUCKETS)
        outputs[transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + NUM_OOV_BUCKETS])

    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    return outputs