import json

with open("evaluation/linear_regression_metrics.json") as f:
    lr = json.load(f)

with open("evaluation/random_forest_metrics.json") as f:
    rf = json.load(f)

print("\nModel Comparison (CV Mean)")
print("-" * 35)

print(f"Linear Regression RMSE: {lr['rmse_mean']:.4f}")
print(f"Random Forest RMSE:     {rf['rmse_mean']:.4f}\n")

print(f"Linear Regression MAE:  {lr['mae_mean']:.4f}")
print(f"Random Forest MAE:      {rf['mae_mean']:.4f}")
