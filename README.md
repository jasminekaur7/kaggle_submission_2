
# 🌾 Yield Prediction using PyCaret

## 🧩 Problem Statement

This project aims to predict **agricultural crop yield** using a dataset that includes pollinator species activity, environmental conditions, and plant growth parameters. Accurate yield prediction supports smarter agricultural planning and resource optimization.

---

## 🧠 Machine Learning Approach

The pipeline is built using **PyCaret**, a low-code machine learning library in Python. Here's a breakdown of the steps:

1. **Data Preprocessing**
   - Feature selection (`classic`)
   - Normalization
   - Power transformation
2. **Model Comparison**
   - PyCaret auto-selects the best model using `compare_models()`
3. **Model Used**
   - `Gradient Boosting Regressor (gbr)`
4. **Prediction**
   - Predictions made on a separate test dataset
   - Results saved to `Answer.csv`

### Sample Code:
```python
from pycaret.regression import *

setup(data=mydataset, target='yield',
      feature_selection=True,
      feature_selection_method='classic',
      normalize=True,
      transformation=True)

cm = compare_models()
lightgbmModel = create_model("gbr")
predictions = predict_model(lightgbmModel, data=mynewdata)

dictionary = pd.DataFrame({
    'id': predictions['id'],
    'yield': predictions['prediction_label']
})
dictionary.to_csv("Answer.csv", index=False)
```

---

## 📊 Model Performance

Metric	Value
✅ Model Accuracy	92%
📉 MAE (My Model)	254
🥇 Lowest MAE on Leaderboard	249
Despite missing the top MAE by just 5 units, the model performed robustly across different states and conditions.
---

## 🔗 View on Google Colab

You can run the notebook in Google Colab:

👉 [Open in Google Colab](https://colab.research.google.com/)  
Then upload `Copy_of_kaggleSubmission2.ipynb` and execute cells.

---

## 💡 Key Takeaways

- PyCaret streamlines preprocessing, model selection, and evaluation.
- Automatic feature selection removed low-impact and redundant features (like `Row#`).
- Pollinator activity and plant metrics (e.g., `osmia`, `fruitmass`) were most correlated with yield.
- Output predictions were generated without needing to manually tune hyperparameters.

---

## 🤝 Connect With Me

Want to collaborate, give feedback, or ask questions?

- GitHub: https://github.com/jasminekaur7
- LinkedIn: https://www.linkedin.com/in/jasmine-kaur-463a3a324/
- Email: jkaur_be24@thapar.edu

## 🙏 Thank You

Thanks for reviewing this project!  
If you found this helpful, please ⭐️ the repository or share it with others.

Happy learning and coding! 🌱
