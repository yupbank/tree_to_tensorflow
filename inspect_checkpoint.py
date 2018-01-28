from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader('./tmp/model.ckpt-211')

print reader.get_variable_to_shape_map()
import ipdb; ipdb.set_trace()
