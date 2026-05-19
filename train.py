import pandas as pd
import joblib
from app.model import train_model

# sample data
data ={
    "hours":[1,2,3,4,5,],
    "marks":[20,40,60,80,100]

}

df = pd.DataFrame(data)

x =  df[["hours"]]
y = df["marks"]

# train model
model = train_model(x,y)

# save model
joblib.dump(model,"models/model.pkl")

print("Model Trainea Sucessfully")
