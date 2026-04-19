from Model_Predictor import MonthlyKwhTableFromModel
from Formula_Predictor import MonthlyKwhTableFromFormula

model_predictor = MonthlyKwhTableFromModel()
formula_predictor = MonthlyKwhTableFromFormula()

#Example usage for January 2026 with 5 kW capacity

jan_2026_model = model_predictor.predict_table(
    kw_capacity=5.0,
    month=1,
    year=2026
)
jan_2026_formula = formula_predictor.predict_table(
    kw_capacity=5.0,
    month=1,
    year=2026
)

print("Model Predictor Output for January 2026:")
print(jan_2026_model.head())
print("\nFormula Predictor Output for January 2026:")
print(jan_2026_formula.head())