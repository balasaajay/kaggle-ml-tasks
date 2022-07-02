import numpy as np
import pytest

from text_classifier_knn import vectorizer, model

def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = [1]
    
    test_data = vectorizer.transform(sample_input_data).toarray()
    test_predictions = model.predict(test_data)
    print(type(test_predictions))

    # Then
    assert isinstance(test_predictions, np.ndarray)
    assert expected_first_prediction_value[0] == test_predictions[0]

@pytest.fixture()
def sample_input_data():
    return ['great food and delicious']
