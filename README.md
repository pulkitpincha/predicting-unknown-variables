# predicting-unknown-variables

Attempting to predict the outcomes from unknown variables using various models and cross-validating.

## Model Accuracy:
- KNN Model accuracy on the test data is 76.33
- LDA Model accuracy on the test data is 93.12
- QDA Model accuracy on the test data is 88.54
- SVM Model accuracy on the test data is 93.12
- Decision Tree Model accuracy on the test data is 90.07
- Naive Bayes Model accuracy on the test data is 89.31

## Cross-validation (cv=5) Accuracy:
- Cross-Validation Accuracy of the best KNN Model: 75.96
- Cross-Validation Accuracy of the best LDA Model: 90.38
- Cross-Validation Accuracy of the best QDA Model: 82.85
- Cross-Validation Accuracy of the best SVM Model: 90.38
- Cross-Validation Accuracy of the best Decision Tree Model: 84.61
- Cross-Validation Accuracy of the best Naive Bayes Model: 83.80

We can assess each model’s performance by using the test accuracies after running the models by the test dataset.

## Performance on the test data is as below:
- Test Accuracy of the best KNN Model: 76.33
- Test Accuracy of the best LDA Model: 93.12
- Test Accuracy of the best QDA Model: 88.54
- Test Accuracy of the best SVM Model: 93.12
- Test Accuracy of the best Decision Tree Model: 90.07
- Test Accuracy of the best Naive Bayes Model: 89.31

## Inference:
The models with the highest cross-validation accuracy in each category performed well on the test data. We can also see that all the models have had a better accuracy on the test dataset than the cross-validation accuracy from the train dataset.

1) KNN - The KNN model has a much lower accuracy than the other models in both, the cross-validation and the test data. This means that it may not be the best model for this as it does not generalize unseen data and in turn the performance is less robust. The model is reliant on patterns in the training data which means that it is sensitive to outliers.

2) LDA - The LDA model has high accuracy in both, cross-validation as well as the test data. This model shows good compatibility as it is fit well with the dataset overall which indicates strong generalization.

3) QDA - The QDA model shows is that it had a good accuracy not as good as the LDA model but it's not bad. 

4) SVM - The SVM model has high accuracy in both, cross-validation as well as the test data. It is a good fit for the dataset much like the LDA model; they both also share almost the same accuracies.

5) Decision Tree - The Decision Tree model has a slightly lower accuracy than the LDA and SVM models. It has a good balance of accuracy and the ability to interpret. Decision Trees are used mainly for the interpretation use case as it helps capture non-linear situations. However, we choose the models with greater accuracy as the goal is not to interpret the data.

6) Naive Bayes – Naive Bayes is used for the simplicity and computational efficiency it offers. Here, the model’s accuracy results are above average in both, cross-validation and the test dataset, but we are going to prioritise higher accuracy for choosing which model to use.

# Conclusion
Overall, the LDA and SVM models are consistent and have a higher accuracy for both the cross-validation and the test dataset than the other models. However, before making a final decision the models can be looked into further to make sure that they are the best fit for the given objective.
