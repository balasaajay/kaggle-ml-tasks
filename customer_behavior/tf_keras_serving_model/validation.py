import tensorflow as tf

import numpy as np
import pickle

restored_model = tf.keras.models.load_model('customer_sentiment_model/1/')
sc = pickle.load(open('sc.pickle', 'rb'))

test_predict = restored_model.predict(sc.transform((np.array([[20, 40000]]))))[:, 1]
print(test_predict)

