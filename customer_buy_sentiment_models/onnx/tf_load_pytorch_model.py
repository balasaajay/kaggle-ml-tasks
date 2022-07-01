import onnx
import onnx_tf
from onnx_tf.backend import prepare
import pickle
import numpy as np

sc = pickle.load(open('sc.pickle', 'rb'))

# Load onnx model from the file
onnx_model = onnx.load('customer_buy_model.onnx')

# Create a tf model from onnx model
tf_model = prepare(onnx_model)

# Use tf model to predict
pred = tf_model.run((sc.transform(np.array([[42, 250000]])))).float32()
print(pred)
