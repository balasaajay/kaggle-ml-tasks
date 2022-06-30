from typing import Any, List, Optional

from pydantic import BaseModel
from titanic_model.processing.validation import TitanicDataInputSchema

class Results(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultiplePeopleDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
      schema_extra = {
          "example": {
              "inputs": [
                  {
                      "name": "Mr. Test",
                      "pclass": 3,
                      "sex": "male",
                      "age": 50,
                      "sibsp": 30,
                      "parch": "0",
                      "fare": "24100",
                      "cabin": "C27",
                      "embarked": "S",
                      "home.dest": "Montreal, PQ / Chesterville, ON"
                  }
              ]
          }
      }