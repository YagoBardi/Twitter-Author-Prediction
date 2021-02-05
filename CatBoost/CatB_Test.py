import pandas as pd
import numpy as np

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score




##Call the data
df = pd.read_csv('23_0815_all_features_03_threshols.csv', index_col=0)
df = df.sample(frac=1,random_state=0).reset_index(drop=True)

section = round(len(df)*8/10)
train_df = df.iloc[:section]
test_df = df.iloc[section:]

train_cols_x = list(train_df.drop(['agency'], axis=1).columns)
test_cols_x = list(test_df.drop(['agency'], axis=1).columns)


train_x = train_df[train_cols_x]
test_x = test_df[test_cols_x]

train_y = train_df["agency"]
test_y = test_df["agency"]
test_y = test_y.reset_index(drop=True)
test_x = test_x.reset_index(drop=True)


train_y = np.transpose(np.array(train_y)).ravel()
test_y = np.transpose(np.array(test_y)).ravel()

#define dataset used to train the model
train_dataset = Pool(data=train_x,
                     label=train_y,
                     )

#define dataset used to test the model
eval_dataset = Pool(data=test_x,

                    )

#set model parameters
model = CatBoostClassifier(
    iterations=15000,
    random_strength=0.5, #reduce overfitting
    depth=8, #depth of the tree
    l2_leaf_reg=2,
    border_count=32,
    rsm = 1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    boosting_type = 'Plain',
)


model.fit(train_dataset, plot=True)

importances = model.feature_importances_


df_feat_imp = pd.DataFrame()
df_feat_imp['Features'] = train_cols_x
df_feat_imp['Importance'] = importances.tolist()
df_feat_imp = df_feat_imp.sort_values(by='Importance',ascending=False)
print(df_feat_imp.head(10))
# df_feat_imp.to_csv("feature_importance_combo.csv")

y_pred = model.predict(test_x)



f1 = f1_score(test_y, y_pred, average='macro')
accuracy = accuracy_score(test_y, y_pred)
print("f1 score is " + str(f1))
print("accuracy is " + str(accuracy))
#save predictions to csv file
# y_pred_df = pd.DataFrame(y_pred)
# y_pred_df.to_csv("predictions.csv")