from data.data_preprocessing import load_clear_data, save_processed
from src.classification import (load_processed, label_and_onehot_processing, split_data,
                                train_and_evaluate, train_random_forest, 
                                cross_validate_decision_tree, cross_validate_random_forest,
                                extract_decision_rules, check_label_distribution, plot_feature_importance)
from sklearn.tree import DecisionTreeClassifier

def main():
    df = load_clear_data('datasets/business_owner_dataset.csv')
    save_processed(df)

    df_processed = load_processed('datasets/processed_data.csv')

    X_encoded, y, feature_names, real_label = label_and_onehot_processing(df_processed)

    processed_df = X_encoded.copy()
    processed_df['Willingness to Develop'] = y
    processed_df.to_csv('datasets/processed_model_data.csv', index=False)

    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    check_label_distribution(df_processed, 'Willingness to Develop')

    print("\nThe Most Importance Features")
    plot_feature_importance(processed_df)

    print("\nDecision Tree Model")
    train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=6, min_samples_split=20, min_samples_leaf=5)

    print("\nRandom Forest Model")
    train_random_forest(X_train, X_test, y_train, y_test, n_estimators=150, max_depth=10, min_samples_split=10, min_samples_leaf=5)

    print("\nCross Validation Decision Tree")
    cross_validate_decision_tree(X_encoded, y, max_depth=6, min_samples_split=20, min_samples_leaf=5, cv=5)

    print("\nCross Validation Random Forest")
    cross_validate_random_forest(X_encoded, y, n_estimators=150, max_depth=10, min_samples_split=10, min_samples_leaf=5, cv=5)

    print("\nExtract Decision Rules")
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=20, min_samples_leaf=5, random_state=42)
    clf.fit(X_encoded, y)

    extract_decision_rules(clf, feature_names, ['Tidak Bersedia', 'Bersedia'])


if __name__ == "__main__":
    main()

