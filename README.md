# Predicting Transported Passenger Survival on the Spaceship Titanic using Supervised Classification Models

## Abstract
This project explores the application of supervised classification models to predict passenger outcomes aboard the Spaceship Titanic, a binary classification task from the Kaggle competition. The objective is to determine whether a passenger was “transported” based on demographic, financial, and behavioral features. We implement and compare three models—Logistic Regression, Random Forests, and XGBoost—to evaluate their predictive performance and interpretability. Preprocessing steps include handling missing data, feature encoding, and normalization. Model performance is assessed using cross-validation and metrics such as accuracy, precision, recall, and F1-score. Results highlight the trade-offs between model complexity and interpretability, with XGBoost achieving the highest overall accuracy, while Logistic Regression offers greater transparency in feature influence. Our findings demonstrate the effectiveness of ensemble methods for complex classification tasks and emphasize the importance of model selection in data-driven decision-making.

## Documentation

### Introduction: Summarize your project report in several paragraphs.
What is the problem? For example, what are you trying to solve? Describe the motivation.
Why is this problem interesting? Is this problem helping us solve a bigger task in some way? Where would we find use cases for this problem?
What is the approach you propose to tackle the problem? What approaches make sense for this problem? Would they work well or not? Feel free to speculate here based on what we taught in class.
Why is the approach a good approach compared with other competing methods? For example, did you find any reference for solving this problem previously? If there are, how does your approach differ from theirs?
What are the key components of my approach and results? Also, include any specific limitations.

### Setup: Set up the stage for your experimental results.
In this competition our task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. From the damaged computer system, we have 8694 entries of train data to predict 4278 passengers' status.
The datasets contains information about passengers aboard the Spaceship Titanic, with each row representing an individual. It includes identifiers like a unique PassengerId, which can also hint at groupings of travelers. Passengers' origins and destinations are recorded through the HomePlanet and Destination fields, while demographic and personal details such as Age, Name, and VIP status provide insight into who they are. The data also captures whether passengers chose to be in CryoSleep during the voyage and their assigned Cabin location. Additionally, spending behavior is tracked across several onboard amenities like the Spa, Food Court, and VR Deck. The target variable, Transported, indicates whether a passenger was mysteriously transported to another dimension.

To summarize we use TotalSpending for all luxury services, split passagner id and cabinet id to analyze saprately the detailed cabin and group ids. Because in the same group the members usually are from the same family, with group only containing one id, we create new variable TravelingAlone as for if the trevelor is traveling alone or not. 

For this question we are building XGBoosting, Random Forest, and Logistic Regression and analzing with Cross Validation.

### Results: Describe the results from your experiments.
Main results: Describe the main experimental results you have; this is where you highlight the most interesting findings.
Supplementary results: Describe the parameter choices you have made while running the experiments. This part goes into justifying those choices.

### **Main Results**

We evaluated two classification models — **Logistic Regression** and **Random Forests** — to predict whether passengers were transported. Both models achieved strong and consistent performance:

- **Logistic Regression**  
  - Validation Accuracy: **77.5%**  
  - Cross-Validation Accuracy: **77.3% ± 0.0065**  
  - Consistent results indicate good calibration and generalization.

- **Random Forest**  
  - Best Cross-Validation Accuracy (tuned): **80.2%**  
  - Validation Accuracy: **80.16%**  
  - Outperformed logistic regression by capturing non-linear feature interactions.

While Random Forest achieved higher accuracy, Logistic Regression remained competitive and offers full model interpretability, making it a strong baseline choice.

---

### **Supplementary Results and Parameter Choices**

- **Logistic Regression**
  - `max_iter = 1000`
  - `C = 1.0` (default regularization)
  - Preprocessing:
    - `PowerTransformer` + `StandardScaler` for numerical features  
    - `OneHotEncoder` for categorical features

- **Random Forest (via GridSearchCV)**
  - `n_estimators = 200`
  - `max_depth = 10`
  - `min_samples_split = 5`
  - `min_samples_leaf = 1`

These parameters were selected using 5-fold cross-validation to balance performance and generalization. Limiting tree depth helped reduce overfitting while maintaining strong predictive accuracy.

### Discussion: The model with the best performance is Random Forest. A future implication of our work is on rescue missions in searching for missing people in ship wrecks or natural disasters.

### Conclusion: In several sentences, summarize what you have done in this project.

### References: Put any links, papers, blog posts, or GitHub repositories that you have borrowed from/found useful here.

