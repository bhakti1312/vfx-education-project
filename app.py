import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


human_results = None
ai_results = None


# ML Function

def run_ml(file):

    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file.name)
    else:
        df = pd.read_csv(file.name)

    le = LabelEncoder()

    for col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    models = {

        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()

    }

    results = []

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        results.append({

            "Model": name,
            "Accuracy": accuracy_score(y_test,pred)*100,
            "Precision": precision_score(y_test,pred,average="weighted")*100,
            "Recall": recall_score(y_test,pred,average="weighted")*100,
            "F1 Score": f1_score(y_test,pred,average="weighted")*100

        })

    results_df = pd.DataFrame(results)

    # Colourful Graph

    plt.figure(figsize=(12,6))

    x = np.arange(len(results_df["Model"]))

    width=0.2

    plt.bar(x,results_df["Accuracy"],width)
    plt.bar(x+width,results_df["Precision"],width)
    plt.bar(x+width*2,results_df["Recall"],width)
    plt.bar(x+width*3,results_df["F1 Score"],width)

    plt.xticks(x+width,results_df["Model"])

    plt.title("ML Performance Graph",fontsize=16,color="purple")

    plt.ylabel("Score",color="blue")

    plt.xlabel("Algorithm",color="blue")

    plt.legend(["Accuracy","Precision","Recall","F1"],fontsize=10)

    plt.grid()

    return results_df,plt.gcf()


# Upload Functions

def upload_human(file):

    global human_results

    human_results,graph=run_ml(file)

    human_results.to_csv("human_report.csv",index=False)

    return human_results,graph,"human_report.csv"


def upload_ai(file):

    global ai_results

    ai_results,graph=run_ml(file)

    ai_results.to_csv("ai_report.csv",index=False)

    return ai_results,graph,"ai_report.csv"


# Compare Graph

def compare():

    global human_results, ai_results

    # Calculate averages

    human_avg_accuracy = human_results["Accuracy"].mean()
    ai_avg_accuracy = ai_results["Accuracy"].mean()

    human_avg_precision = human_results["Precision"].mean()
    ai_avg_precision = ai_results["Precision"].mean()

    human_avg_recall = human_results["Recall"].mean()
    ai_avg_recall = ai_results["Recall"].mean()

    human_avg_f1 = human_results["F1 Score"].mean()
    ai_avg_f1 = ai_results["F1 Score"].mean()


    metrics = ["Accuracy","Precision","Recall","F1 Score"]

    human_scores = [
        human_avg_accuracy,
        human_avg_precision,
        human_avg_recall,
        human_avg_f1
    ]

    ai_scores = [
        ai_avg_accuracy,
        ai_avg_precision,
        ai_avg_recall,
        ai_avg_f1
    ]


    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(metrics))

    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(x, human_scores, width)

    plt.bar(x + width, ai_scores, width)

    plt.xticks(x + width/2, metrics)

    plt.ylabel("Score")

    plt.title("Human vs AI Performance Comparison")

    plt.legend(["Human Response","AI Generated Response"])

    plt.grid()


    # Print result

    print("Human Average Accuracy:", human_avg_accuracy)
    print("AI Average Accuracy:", ai_avg_accuracy)

    if human_avg_accuracy > ai_avg_accuracy:
        print("RESULT: Human response is higher than AI generated response")
    else:
        print("RESULT: AI response is higher than Human response")


    return plt.gcf()
    demo.launch(share=True)
