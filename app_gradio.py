# Lab 1: Linear Regression with Gradio UI
import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------- Dataset 1: Salary ----------
salary_df = pd.read_csv("salary_data.csv")
X1 = salary_df[["YearsExperience"]].values     # input
y1 = salary_df["Salary"].values                # output
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X1_train, y1_train)
print("Salary model R2 score:", model1.score(X1_test, y1_test))

# ---------- Dataset 2: House Prices ----------
house_df = pd.read_csv("house_prices.csv")
X2 = house_df[["Area_sqft", "Bedrooms", "Age_years"]].values
y2 = house_df["Price"].values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X2_train, y2_train)
print("House model R2 score:", model2.score(X2_test, y2_test))

# ---------- Prediction functions ----------
def predict_salary(years):
    result = model1.predict([[years]])[0]
    return f"Predicted Salary: {result:.2f}"

def predict_house(area, bedrooms, age):
    result = model2.predict([[area, bedrooms, age]])[0]
    return f"Predicted House Price: {result:.2f}"

# ---------- Gradio interfaces ----------
salary_ui = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(label="Years of Experience"),
    outputs="text",
    title="Salary Prediction")

house_ui = gr.Interface(
    fn=predict_house,
    inputs=[gr.Number(label="Area (sqft)"),
            gr.Number(label="Bedrooms"),
            gr.Number(label="House Age (years)")],
    outputs="text",
    title="House Price Prediction")

app = gr.TabbedInterface([salary_ui, house_ui], ["Salary", "House Price"])
app.launch()
