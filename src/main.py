from src.data_preprocessing import load_data, preprocess, split_data
from src.train_model import train_model, evaluate_model, save_model, plot_feature_importance


def main():
    # 📥 Load data
    df = load_data("data/raw/heart.csv")

    # 🧹 Preprocess
    X, y, scaler = preprocess(df)

    # ✂️ Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 🌲 Train model
    model = train_model(X_train, y_train)

    # 📊 Evaluate
    evaluate_model(model, X_test, y_test)

    # 📈 Feature Importance
    plot_feature_importance(model, df.drop("target", axis=1).columns)

    # 💾 Save model
    save_model(model, scaler)


if __name__ == "__main__":
    main()
