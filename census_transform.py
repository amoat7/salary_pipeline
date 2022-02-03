import tensorflow_transform as tft 
import tensorflow as tf 

VOCAB_FEATURE_DICT ={
    'education':16, 'marital_status':7, 'occupation':15, 'race':5,
    'relationship':6, 'native_country': 41, 'sex':2,  'workclass':9
}


NUMERIC_FEATURE_KEYS = ['capital-gain', 'capital-loss', 'education-num','fnlwgt', 'hours-per-week','age']

NUM_OOV_BUCKETS = 2

LABEL_KEY = 'label'

def transformed_name(key):
    key = key.replace('-','_')
    return key + 'xf'

def preprocessing_fn(inputs):

    outputs= {}
    for key in NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[transformed_name(key)] = tf.reshape(scaled, [-1])

    for key, vocab_size in VOCAB_FEATURE_DICT.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key], num_oov_buckets=NUM_OOV_BUCKETS)
        one_hot = tf.one_hot(indices, vocab_size + NUM_OOV_BUCKETS)
        outputs[transformed_name(key)] = tf.reshape(one_hot, [-1, vocab_size + NUM_OOV_BUCKETS])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs