# Predicting Transported Passenger Survival on the Spaceship Titanic using Supervised Classification Models

## Abstract
This project explores the application of supervised classification models to predict passenger outcomes aboard the Spaceship Titanic, a binary classification task from the Kaggle competition. The objective is to determine whether a passenger was “transported” based on demographic, financial, and behavioral features. We implement and compare three models—Logistic Regression, Random Forests, and XGBoost—to evaluate their predictive performance and interpretability. Preprocessing steps include handling missing data, feature encoding, and normalization. Model performance is assessed using cross-validation and metrics such as accuracy, precision, recall, and F1-score. Results highlight the trade-offs between model complexity and interpretability, with XGBoost achieving the highest overall accuracy, while Logistic Regression offers greater transparency in feature influence. Our findings demonstrate the effectiveness of ensemble methods for complex classification tasks and emphasize the importance of model selection in data-driven decision-making.

## Documentation

### Introduction: Summarize your project report in several paragraphs.
In the distant future—perhaps the year 2912—the Spaceship Titanic, a luxury interstellar vessel carrying over 13,000 passengers to newly discovered exoplanets, encountered a hidden spacetime anomaly near Alpha Centauri. Though the ship itself remained intact, nearly half its passengers mysteriously vanished, seemingly transported to an alternate dimension. This project aims to solve a crucial part of that mystery: using recovered records from the ship’s damaged computer system, we seek to predict which passengers were transported. Accurately identifying these individuals would assist interstellar rescue teams in resource allocation and possibly enable the development of anomaly-avoidance protocols for future missions.

The motivation for our project lies in the broader challenge of building robust, interpretable models that can perform well under uncertainty and imperfect data conditions. The Spaceship Titanic dataset simulates real-world scenarios where data may be incomplete, noisy, or corrupted, mirroring challenges in fields such as emergency response, fraud detection, and medical diagnostics. As such, this problem provides a compelling testing ground for machine learning methods that can inform decision-making when stakes are high and data reliability is limited.

To tackle this problem, we propose a comparative modeling framework using logistic regression, random forest, and XGBoost. Each model was chosen for its unique strengths: logistic regression offers interpretability and serves as a transparent baseline; random forest handles non-linear relationships and is robust to noise and missing values—ideal for data corrupted during transmission; XGBoost is a state-of-the-art boosting algorithm known for its predictive accuracy and built-in handling of complex feature interactions. These models align well with the classification task and offer complementary trade-offs in terms of performance and interpretability.

Our approach differs from many existing solutions that often prioritize accuracy alone. While previous studies or Kaggle submissions may rely on ensemble stacking or complex pipelines to maximize leaderboard scores, we deliberately emphasize model clarity and comparative insights. We aim to understand why certain models perform better and how features influence prediction, especially in the presence of uncertainty and missing data.

The key components of our approach include careful preprocessing, meaningful feature engineering (e.g., aggregating spending patterns, extracting cabin and group structure), and standardized evaluation using cross-validation and accuracy metrics. Our results demonstrate that while XGBoost achieves the highest accuracy, logistic regression provides valuable interpretability, and random forest strikes a balance between the two. However, our approach is not without limitations, particularly in the potential for overfitting with more complex models and the challenge of dealing with high-cardinality categorical features. Despite these, our framework offers a principled, flexible methodology for predictive modeling in uncertain environments.

### Setup: Set up the stage for your experimental results.
In this competition, our task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. From the damaged computer system, we have 8694 entries of train data to predict 4278 passengers.
The datasets contain information about passengers aboard the Spaceship Titanic, with each row representing an individual. It includes identifiers like a unique PassengerId, which can also hint at groupings of travelers. Passengers' origins and destinations are recorded through the HomePlanet and Destination fields, while demographic and personal details such as Age, Name, and VIP status provide insight into who they are. The data also captures whether passengers chose to be in CryoSleep during the voyage and their assigned Cabin location. Additionally, spending behavior is tracked across several onboard amenities like the Spa, Food Court, and VR Deck. The target variable, Transported, indicates whether a passenger was mysteriously transported to another dimension.

To summarize, we use TotalSpending for all luxury services, split passenger ID and cabinet ID to analyze separately the detailed cabin and group IDs. Because in the same group, the members usually are from the same family, with the group only containing one ID, we create a new variable, TravelingAlone, as to whether the traveler is traveling alone or not. 

For this question, we are building XGBoosting, Random Forest, and Logistic Regression, and analyzing with Cross Validation.

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

### Conclusion: 
In this project, we tackled the challenge of predicting whether passengers aboard the Spaceship Titanic were transported, a binary classification problem from a Kaggle competition. Using a dataset of demographic, behavioral, and financial features, we applied and compared Logistic Regression, Random Forest, and XGBoost models. Extensive preprocessing, including missing value imputation, feature transformation, and encoding, was critical to our pipeline.

After evaluation using metrics like accuracy and cross-validation scores, Random Forest was identified as the best-performing model, balancing performance and robustness. Our findings not only highlight the importance of ensemble methods for complex datasets but also showcase how thoughtful feature engineering and model selection can lead to meaningful insights in predictive analytics. This work sets the stage for broader applications in domains requiring rapid, high-stakes decision-making based on passenger or personnel data.

### Discussion: 

The implications of our work extend beyond this specific dataset. The modeling techniques and feature engineering strategies used here can be adapted for real-world scenarios, particularly in emergency response or rescue operations. For example, similar models could assist in identifying missing individuals following natural disasters or shipwrecks by learning from structured passenger data. The ability to rapidly and accurately predict survival likelihoods could support prioritization in rescue missions, ultimately saving lives.

### References: 
Eunicl. "Spaceship Titanic with Random Forest + XGBoost." Kaggle, 2023, https://www.kaggle.com/code/eunicl/spaceship-titanic-with-random-forest-xgboost.
Dilber, Burak. "Spaceship Titanic - EDA, Preprocessing and XGBoost." Kaggle, 2023, https://www.kaggle.com/code/burakdilber/spaceship-titanic-eda-preprocessing-and-xgboost.

