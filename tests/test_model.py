import pandas as pd
from app.model import train_model

def test_model_training():
    data ={
        "hours" : [1,2,3,],
        "marks": [2,40,60]
    }
    df = pd.DataFrame(data)

    x =  df[["hours"]]
    y = df["marks"]

    model = train_model(x,y)
    prediction = model.predict([[4]])

    assert prediction[0]> 0