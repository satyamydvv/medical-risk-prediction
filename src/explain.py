import shap


def explain_prediction(model, X_sample, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For classification (class 1 = disease)
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        X_sample,
        feature_names=feature_names
    )
