import joblib
model = joblib.load("models/model.pkl")
hours = [[6]]


prediction = model.predict(hours)

print("Predicted Marks: ", prediction[0])