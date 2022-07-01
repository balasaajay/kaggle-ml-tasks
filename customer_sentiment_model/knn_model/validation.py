from knn_model import sc, model
import numpy as np

test_data1 = np.array([[40, 20000]])
test_prediction = model.predict(sc.transform(test_data1))
test_prediction_proba = model.predict_proba(sc.transform(test_data1))[:, -1]
if test_prediction.tolist()[0]:
  print("Customer may buy! Probability of buying: %s" % test_prediction_proba)
else:
  print("Customer may not buy! Probability of buying: %s" % test_prediction_proba)