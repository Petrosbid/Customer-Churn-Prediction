# Customer-Churn-Prediction
a complete machine learning system for predicting the churn probability of customers for a telecommunications company.
 The system consists of two main parts:

Model Training Script (main.py): Trains and optimizes a machine learning model using historical data.

Web Application (app.py): Provides an interactive user interface using Streamlit to input new customer data and predict their churn probability in real-time.

✨ Features
Powerful XGBoost Model: Utilizes the XGBClassifier algorithm to achieve high prediction accuracy.

Hyperparameter Tuning: Uses GridSearchCV to find the best parameters and improve model performance.

Interactive User Interface: A user-friendly web application for entering inputs and displaying results clearly and understandably.

Complete Data Preprocessing: Includes handling missing values, encoding categorical variables, and standardizing data.

Automatic Visualization Generation: Creates and saves analytical charts such as churn distribution, ROC curve, and the model's decision boundary.

📂 Project Structure
.
├── inputs/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset file
├── visualizations/
│   ├── churn_distribution.png
│   ├── decision_boundary_scatter.png
│   ├── roc_curve.png
│   └── tenure_distribution.png
├── app.py                                  # Streamlit web application script
├── main.py                                 # Model training and evaluation script
├── churn_model.joblib                      # Saved trained model file
├── model_columns.joblib                    # List of columns required by the model
├── scaler.joblib                           # Saved Scaler object file
└── README.md                               # This file

⚙️ Setup and Installation
Follow the steps below to run this project.

1. Clone the Repository:

git clone [https://github.com/your-username/customer-churn-prediction.git](https://github.com/your-username/customer-churn-prediction.git)
cd customer-churn-prediction

2. Create and Activate a Virtual Environment (Recommended):

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Required Packages:

All necessary libraries are listed in the requirements.txt file (if you create one). You can install them using the following command:

pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib streamlit

4. Prepare the Dataset:

Download the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset and place it in the inputs folder.

🚀 How to Run
The project is executed in two steps:

Step 1: Train the Model

First, you need to run the main.py script to train the model, which will generate the necessary (.joblib) files and visualizations.

python main.py

After running this command, the following outputs will be generated:

churn_model.joblib, scaler.joblib, and model_columns.joblib files in the project root.

Analytical charts in the visualizations folder.

Step 2: Run the Web Application

Once the model is successfully trained, run the web application with the following command:

streamlit run app.py

This command will open the application in your browser, where you can enter customer details to predict their churn probability.

📄 Script Descriptions
main.py
This script handles the entire model training and evaluation process:

Load Data: Reads the dataset from the inputs folder.

Clean and Preprocess: Manages invalid values and converts categorical variables into a numerical format (Dummy Variables).

Prepare for Modeling: Splits the data into training and testing sets and normalizes them using StandardScaler.

Train and Optimize: Trains an XGBClassifier model using GridSearchCV to find the best combination of hyperparameters.

Evaluate: Evaluates the final model on the test data and prints metrics like accuracy and a classification_report.

Save Artifacts: Saves the optimal model, scaler, and column list into .joblib files.

Visualize: Generates important plots for data analysis and model performance and saves them in the visualizations folder.

app.py
This script creates the web user interface using the Streamlit library:

Load Model: Loads the .joblib files saved in the previous step.

Create UI: Uses various Streamlit widgets (like selectbox, slider, number_input) to create a form for receiving customer information.

Get User Input: Collects the data entered by the user.

Preprocess Input: Prepares the input data using the exact same process as in main.py.

Predict: Uses the loaded model to predict the customer's churn probability.

Display Result: Shows the result to the user as a clear message (e.g., "High Churn Risk" or "Loyal Customer") along with actionable recommendations.

📊 Sample Outputs
The web application will look like this after running:

Additionally, the following charts will be saved in the visualizations folder after running main.py:

Churn Distribution: Shows how many customers have churned versus how many have stayed.

ROC Curve: Illustrates the model's performance in distinguishing between positive and negative classes.

Decision Boundary Scatter Plot: Visualizes how the model makes decisions based on two important features (like tenure and monthly charges).

Feel free to improve this project and add new features!
