import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


data = pd.read_csv("bank.csv")
accounts = data["Account No"].unique()

# clean data
data["DATE"] = pd.to_datetime(data["DATE"], format="%d-%b-%y")
data["Month"] = data["DATE"].dt.to_period("M")
data["WITHDRAWAL AMT"] = data[" WITHDRAWAL AMT "].replace({',': ''}, regex=True).astype(float) # remove whitespace and convert to float
data["DEPOSIT AMT"] = data[" DEPOSIT AMT "].replace({',': ''}, regex=True).astype(float) # remove whitespace and convert to float
data["BALANCE AMT"] = data["BALANCE AMT"].replace({',': ''}, regex=True).astype(float) # remove whitespace and convert to float

# get the rows for a certain account
def extract_data(data, target, col):
    filtered_rows = data[data[col] == target]
    return filtered_rows


## START OF ML MODEL
features = []
labels = []

for account in accounts:
    group = extract_data(data, accounts[0], "Account No")
    
    # calculate features
    monthly_summary = group.groupby("Month").agg(
        total_deposits = ("DEPOSIT AMT", "sum"),
        total_withdrawals = ("WITHDRAWAL AMT", "sum"),
        num_deposits = ("DEPOSIT AMT", "count"),
        num_withdrawals = ("WITHDRAWAL AMT", "count"),
        num_transactions = ("TRANSACTION DETAILS", "count")
    ).reset_index()
    
    # 1) monthly transactions
    monthly_transactions = []
    for _, row in monthly_summary.iterrows():
        monthly_transactions.append(row["num_transactions"])
    
    # 2) deposit-withdrawal ratio
    monthly_dw_ratio = []
    for _, row in monthly_summary.iterrows():
        deposit = row["total_deposits"]
        withdrawal = row["total_withdrawals"]
        ratio = 0.0
        
        if deposit and withdrawal:
            ratio = deposit / withdrawal
        else:
            ratio = 0.0
        monthly_dw_ratio.append(ratio)
        
    dw_ratio = np.mean(monthly_dw_ratio)
        
    # 3) average withdrawal
    avg_monthly_withdrawal = []
    for _, row in monthly_summary.iterrows():
        num_withdrawals = row["num_withdrawals"]
        avg = 0.0
        
        if num_withdrawals:
            avg = row["total_withdrawals"] / num_withdrawals
        else:
            avg = 0.0
        avg_monthly_withdrawal.append(avg)


    # 4) monthly cash flow
    monthly_cash = []
    for _, row in monthly_summary.iterrows():
        profit = row["total_deposits"] - row["total_withdrawals"]
        monthly_cash.append(profit)
    mean_monthly_cash = np.mean(monthly_cash)

    # 5) overdraft frequency
    num_overdrafts = np.where(group["BALANCE AMT"] < 0, 1, 0).sum()
    od_percentage = (num_overdrafts / len(group)) * 100

    # finds the first transaction of each month
    first_trans = {}
    
    for _, row in group.iterrows():
        date_obj = row["DATE"] # date
        month_year = date_obj.strftime("%m/%Y")  # formats to MM/YYYY
        
        # adds balance at first of each month
        if month_year not in first_trans:
            first_trans[month_year] = row
        
    coords = []
    i = 1
    for row in first_trans.values():
        balance = row["BALANCE AMT"]
        coords.append((i, balance))
        i += 1

    # check for decreasing trend in balance  
    X = np.array([coord[0] for coord in coords]).reshape(-1, 1)  # reshape for regression
    Y = np.array([coord[1] for coord in coords])

    # 6) decreasing balance
    regression = LinearRegression().fit(X,Y)
    r = regression.coef_[0]
    
    conditions = [
        np.any(od_percentage > 10), 
        np.any(r < 0), 
        np.any(dw_ratio < 1), 
        np.any(avg_monthly_withdrawal > (0.7 * mean_monthly_cash))
    ]
    
    needs_loan = 1 if sum(conditions) >= 2 else 0
    
    # add features and labels
    features.append([
        np.mean(monthly_transactions),
        np.mean(monthly_cash),
        od_percentage,
        r,
        dw_ratio,
        np.mean(avg_monthly_withdrawal)
        ])
    labels.append(needs_loan)
    
X = np.array(features)
y = np.array(labels)

# splits the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a decision tree classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

## END OF ML MODEL
