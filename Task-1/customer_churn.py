import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("customer.csv")

# Input and Output
X = data[["Age", "MonthlyCharges"]]
y = data["Churn"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new customer
age = int(input("Enter Customer Age: "))
charge = int(input("Enter Monthly Charges: "))

prediction = model.predict([[age, charge]])

if prediction[0] == 1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")