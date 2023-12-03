from model import RecommendationSystem as RS
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def evaluate(datasets_paths):
    """
    This function takes a single argument which is
    a list of tuples of two file names: train dataset and test dataset:
    [("train1.csv", "test1.csv"), ("train2.csv", "test2.csv"), ]

    For each pair, a new RS model is generated, trained, and evaluated.
    This function returns evaluation metrics for each pair and in total.
    """
    
    total_rmse = 0
    total_accuracy = 0
    total_precision = 0
    total_diversity = 0

    user_features_path = f"{os.getcwd()}/data/interim/u.user.csv"
    item_features_path = f"{os.getcwd()}/data/interim/u.item.csv"
         
    user_features = pd.read_csv(user_features_path, index_col="user_id").to_numpy()
    item_features = pd.read_csv(item_features_path, index_col="Unnamed: 0").to_numpy()
        
    best_model = None
    best_model_score = float("+inf")
   
    for X_path, y_path in datasets_paths:
        cur_dir = os.getcwd()
        X_full_path = f"{os.getcwd()}/data/interim/{X_path}"
        y_full_path = f"{os.getcwd()}/data/interim/{y_path}"

        X = pd.read_csv(X_full_path, index_col="User id").to_numpy()
        y = pd.read_csv(y_full_path, index_col="User id").to_numpy()

        print(f"\nTrain: {X_path}   Test: {y_path}")
        rs = RS(user_features, item_features, epochs=100)
        rs.fit(X)

        score = rs.get_metrics(y)
        rmse = score["rmse"]
        if rmse < best_model_score:
            best_model_score = rmse
            best_model = rs
        
        total_rmse += score["rmse"]
        total_accuracy += score["accuracy"]
        total_precision += score["precision"]
        total_diversity += score["diversity"][0]

        print(f"RMSE:      {round(score['rmse'], 5)}")
        print(f"Accuracy:  {round(score['accuracy'], 5)}%")
        print(f"Precision: {round(score['precision'], 5)}%")
        print(f"Diversity: {round(score['diversity'][0], 5)}%  {score['diversity'][1]}\n")
        
    print("\nAverage score across all splits:")
    print(f"RMSE:      {round(total_rmse/len(datasets_paths), 5)}")
    print(f"Accuracy:  {round(total_accuracy/len(datasets_paths), 5)}%")
    print(f"Precision: {round(total_precision/len(datasets_paths), 5)}%")
    print(f"Diversity: {round(total_diversity/len(datasets_paths), 5)}%\n\n")
    
    with open(f"{os.getcwd()}/model", 'wb') as f:
        pickle.dump(best_model, f)

    return best_model



try:
    with open(f"{os.getcwd()}/model", 'rb') as f:
        model = pickle.load(f)
    print("Model successfully loaded")
except:
    print("No model in root directory founded. Training the model...")
    model = evaluate([
        ("u1.base.csv", "u1.test.csv"),
        ("u2.base.csv", "u2.test.csv"),
        ("u3.base.csv", "u3.test.csv"),
        ("u4.base.csv", "u4.test.csv"),
        ("u5.base.csv", "u5.test.csv"),
        ("ua.base.csv", "ua.test.csv"),
        ("ub.base.csv", "ub.test.csv")
    ])

occupations_path = f"{os.getcwd()}/data/interim/u.occupation.csv"
item_names_path = f"{os.getcwd()}/data/interim/u.item_names.csv"
occup = pd.read_csv(occupations_path)
item_names = pd.read_csv(item_names_path)
 
age = int(input("\nEnter you age: ")) / 100
sex = int(input("Are you female(1) or male(0)? "))
occup = int(input(
    "Enter the id of your job from the list:\n"+ '\n'.join([f"{x[1]}[{x[0]}]" for x in occup.values])+": "
)) / 20

items = model.recommend_items(user_features=[age, sex, occup])
print("\nYour recommendations are:")
for i in items:
    name = item_names[item_names.item_id == i].item_name.item()
    print(name)


