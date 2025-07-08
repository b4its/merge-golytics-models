import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn.compose import ColumnTransformer

def load_processed(filepath):
    df = pd.read_csv(filepath)
    return df

def label_and_onehot_processing(df):
    label_features = ['Has Website', 'Social Media Presence', 'Marketplace Usage',
                      'Payment Digital Adoption', 'POS (Point of Sales) Usage',
                      'Online Ads Usage', 'E-Wallet Acceptance']

    onehot_features = ['Title', 'Active Social Media Channels', 'Social Media Posting Frequency',
                       'Year Started Digital Adoption', 'Business Size', 'Monthly Revenue',
                       'Number of Employees', 'Location Type']

    target = 'Willingness to Develop'

    X = df.drop(columns=[target])
    y = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    real_label = {target: dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))}

    for column in label_features:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        real_label[column] = dict(zip(le.transform(le.classes_), le.classes_))

    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), onehot_features)
        ],
        remainder='passthrough'
    )

    X_processed = column_transformer.fit_transform(X)

    onehot_feature_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(onehot_features)

    final_feature_names = list(onehot_feature_names) + label_features

    X_processed_df = pd.DataFrame(X_processed, columns=final_feature_names)

    return X_processed_df, y, final_feature_names, real_label

def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def check_label_distribution(df, target_column):

    print("Distribusi label : ")

    print(df[target_column].value_counts(normalize=True) * 100)

def train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy * 100:.2f}% | max_depth={max_depth} | min_samples_split={min_samples_split} | min_samples_leaf={min_samples_leaf}")
    return accuracy

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Random Forest: {accuracy * 100:.2f}% | n_estimators={n_estimators} | max_depth={max_depth} | min_samples_split={min_samples_split} | min_samples_leaf={min_samples_leaf}")
    return accuracy

def cross_validate_decision_tree(X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1, cv=5):
    clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"Cross Validation Akurasi Decision Tree (mean): {np.mean(scores) * 100:.2f}% | Per Fold: {scores}")
    return scores

def cross_validate_random_forest(X, y, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, cv=5):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    print(f"Cross Validation Akurasi Random Forest (mean): {np.mean(scores) * 100:.2f}% | Per Fold: {scores}")
    return scores

def plot_feature_importance(df, target_column='Willingness to Develop'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    feature_names = X.columns

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


def extract_decision_rules(tree, feature_names, target_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            left_rule = current_rule + [f"{name} = Tidak"]
            recurse(tree_.children_left[node], left_rule)

            right_rule = current_rule + [f"{name} = Ya"]
            recurse(tree_.children_right[node], right_rule)
        else:
            target_value = tree_.value[node]
            target_index = target_value.argmax()
            prediction = target_names[target_index]
            paths.append((current_rule, prediction))

    recurse(0, [])

    for path, prediction in paths:
        rule_text = " DAN ".join(path)
        print(f"Jika {rule_text} → Disarankan: {prediction}")

