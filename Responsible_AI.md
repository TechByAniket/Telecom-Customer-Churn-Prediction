# Responsible AI Report

## 1. Fairness

The churn prediction model was evaluated using Fairlearn.

Measured fairness metrics on the evaluated dataset:

- Demographic Parity Difference: 0.0403
- Equalized Odds Difference: 0.0332

These values describe the observed group differences in the evaluated data. They should not be interpreted as a guarantee of fairness in future or different populations.

## 2. Group-wise Performance

| Group | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Female | 0.969444 | 0.973822 | 0.916256 | 0.944162 |
| Male | 0.965167 | 0.974194 | 0.883041 | 0.926380 |

## 3. Explainability

SHAP and LIME were used to provide explanations for model predictions.

SHAP identifies features that contribute toward or away from the predicted churn probability.

LIME provides local explanations for individual predictions.

## 4. Privacy

Customer information should be handled securely. Personally identifiable or sensitive customer information should not be unnecessarily exposed in the dashboard or public repository.

The full cleaned dataset is therefore kept outside the public GitHub repository.

## 5. Consent and Data Usage

The model should only be used with data collected and processed for legitimate purposes. Appropriate organizational policies, consent requirements, and applicable data-protection requirements should be followed.

## 6. Limitations

- Fairness metrics depend on the evaluation population and protected attribute used.
- SHAP explanations describe model behavior and do not establish causal relationships.
- Predictions are probabilistic and should support, rather than replace, appropriate business decision-making.
- Model performance may change when deployed on data that differs from the training data.

## 7. Monitoring

The deployed system should be periodically monitored for:

- Data drift
- Model performance degradation
- Changes in group-wise performance
- Unexpected prediction patterns
- Changes in the underlying customer population