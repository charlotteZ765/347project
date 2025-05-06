# Predicting Passenger Survival on the Spaceship Titanic 🚀🚢

#### QTM 347 Final Project
#### Team Members: Joyce Chen, Ryan Paik, Heyi Yang, Charlotte Zhao
#### Data Source: https://www.kaggle.com/competitions/spaceship-titanic
#### Final Presentation Slides: [Presentation Slides](https://www.figma.com/deck/KUwBVM5JqRcsdHktBC81Ek/QTM347-Project?node-id=1-303&t=g56ZEZevNkbWl1mv-1)

## Abstract
This project explores the application of supervised classification models to predict passenger outcomes aboard the Spaceship Titanic, a binary classification task from the Kaggle competition. The objective is to determine whether a passenger was “transported” based on demographic, financial, and behavioral features. We implement and compare three models—Logistic Regression, Random Forests, and XGBoost—to evaluate their predictive performance and interpretability. Preprocessing steps include handling missing data, feature encoding, and normalization. Model performance is assessed using cross-validation and metrics such as accuracy, precision, recall, and F1-score. Results highlight the trade-offs between model complexity and interpretability, with XGBoost achieving the highest overall accuracy, while Logistic Regression offers greater transparency in feature influence. Our findings demonstrate the effectiveness of ensemble methods for complex classification tasks and emphasize the importance of model selection in data-driven decision-making.

---

## Introduction
In the distant future—perhaps the year 2912—the Spaceship Titanic, a luxury interstellar vessel carrying over 13,000 passengers to newly discovered exoplanets, encountered a hidden spacetime anomaly near Alpha Centauri. Though the ship itself remained intact, nearly half its passengers mysteriously vanished, seemingly transported to an alternate dimension. This project aims to solve a crucial part of that mystery: using recovered records from the ship’s damaged computer system, we seek to predict which passengers were transported. Accurately identifying these individuals would assist interstellar rescue teams in resource allocation and possibly enable the development of anomaly-avoidance protocols for future missions.

The motivation for our project lies in the broader challenge of building robust, interpretable models that can perform well under uncertainty and imperfect data conditions. The Spaceship Titanic dataset simulates real-world scenarios where data may be incomplete, noisy, or corrupted, mirroring challenges in fields such as emergency response, fraud detection, and medical diagnostics. As such, this problem provides a compelling testing ground for machine learning methods that can inform decision-making when stakes are high and data reliability is limited.

To tackle this problem, we propose a comparative modeling framework using logistic regression, random forest, and XGBoost. Each model was chosen for its unique strengths: logistic regression offers interpretability and serves as a transparent baseline; random forest handles non-linear relationships and is robust to noise and missing values—ideal for data corrupted during transmission; XGBoost is a state-of-the-art boosting algorithm known for its predictive accuracy and built-in handling of complex feature interactions. These models align well with the classification task and offer complementary trade-offs in terms of performance and interpretability.

Our approach differs from many existing solutions that often prioritize accuracy alone. While previous studies or Kaggle submissions may rely on ensemble stacking or complex pipelines to maximize leaderboard scores, we deliberately emphasize model clarity and comparative insights. We aim to understand why certain models perform better and how features influence prediction, especially in the presence of uncertainty and missing data.

The key components of our approach include careful preprocessing, meaningful feature engineering (e.g., aggregating spending patterns, extracting cabin and group structure), and standardized evaluation using cross-validation and accuracy metrics. Our results demonstrate that while XGBoost achieves the highest accuracy, logistic regression provides valuable interpretability, and random forest strikes a balance between the two. However, our approach is not without limitations, particularly in the potential for overfitting with more complex models and the challenge of dealing with high-cardinality categorical features. Despite these, our framework offers a principled, flexible methodology for predictive modeling in uncertain environments.

---

## Setup
In this competition, our task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. From the damaged computer system (Kaggle dataset), we have two main components: a training set of 8,694 passengers whose transport status is known, and a test set of 4,278 passengers to predict. The datasets contain information about passengers aboard the Spaceship Titanic, with each row representing an individual. 

### Key Features
- Identifiers: PassengerID - Embeds group information, and Cabin, which reflects physical location on the ship (Deck/Num/Side)
- Demographics: HomePlanet, Age, and VIP status
- Behavioral features: participation in CryoSleep, as well as expenditures in RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck
- Travel route: indicated by the Destination
- Target variable: Transported, a binary label indicating whether a passenger was mysteriously transported to another dimension

### Feature Engineering
- TotalSpending: a composite measure aggregating spending across all onboard services to capture engagement level
- Cabin decomposition into CabinDeck, CabinNum, and CabinSide to account for all possible location-based effects
- Group extraction from PassengerId to define Group and NumberInGroup
- TravelingAlone: A derived binary feature indicating whether the passenger is part of a multi-person group

### Data Exploration

<p align="center">
  <img src="final_images/347img3.png" alt="Image 3" height="150"/>
  <img src="final_images/347img4.png" alt="Image 4" height="150"/>
  <img src="final_images/347img2.png" alt="Image 2" height="150"/>
</p>

Figure 1: Age Distribution of Passengers
The age distribution is right-skewed, with a concentration of passengers in the 20–30 age range. This suggests a relatively young traveler population. From a feature engineering perspective, age might correlate with passenger behavior (e.g., CryoSleep participation or spending). We may also consider creating age bands (e.g., youth, adult, senior) to capture non-linear effects and reduce model noise.

Figure 2: Total Spending Distribution Among Passengers
The total spending is heavily centered at zero, indicating many passengers spent nothing, possibly due to being in CryoSleep. The long tails reflect a few high spenders. This variable is informative but skewed; transformations (e.g., log or power) may improve model performance. We also engineered this feature as a composite of multiple spending categories, making it a valuable summary indicator of onboard activity.

Figure 3: Passenger Transport Status by HomePlanet
HomePlanet is a strong categorical feature with clear class separation. Passengers from Europa show a higher proportion of being transported than those from Earth. This suggests underlying planet-based heterogeneity in outcomes, making HomePlanet a crucial input to model classification. One-hot encoding this feature preserves this discriminatory power.

### Association Analysis

<figure style="text-align: center;">
  <img src="final_images/347img1.png" alt="Feature Association Plot" width="300">
</figure>

This plot informs our feature selection strategy by quantifying the relationship between each engineered feature and the target variable Transported. The strongest associations are observed with Group, CabinNum, and CryoSleep, validating the decision to engineer features such as group identifiers (Group, NumberInGroup), cabin decomposition (CabinDeck, CabinSide, CabinNum), and binary indicators like CryoSleep and TravelingAlone.

Features like RoomService, Spa, and VRDeck display low or even negative associations, suggesting that raw spending variables may offer limited predictive value. This insight reinforces the need to either transform these features (e.g., via total or normalized spending) or deprioritize them during feature selection.

Overall, the figure supports the inclusion of spatial, behavioral, and grouping-based features in our final model pipeline, while guiding regularization or dimensionality reduction for weaker variables.

---

## Modeling Approaches

We evaluate 3 classification models:

**1. Logistic Regression**
Serves as a simple, interpretable linear baseline.
- Parameters: max_iter=1000, C=1.0 (L2 regularization)

**2. Random Forest**
A robust ensemble method capable of capturing non-linear relationships.
- Tuned parameters: n_estimators, max_depth, min_samples_split

**3. XGBoost**
A gradient boosting model optimized for performance on tabular data.
- Handles missing values natively and includes built-in regularization
- Tuned parameters: learning_rate, n_estimators, max_depth

All models are trained within a unified scikit-learn Pipeline, which standardizes preprocessing and ensures consistent evaluation:
- Numerical features: Transformed using PowerTransformer and StandardScaler
- Categorical features: Imputed with a constant value and one-hot encoded using OneHotEncoder

### Model #1: Logistic Regression

### Model #2: Random Forest

### Model #3: XG Boost

---

## Results

### **Main Results**
We evaluated three supervised classification models — **Logistic Regression**, **Random Forest**, and **XGBoost** — to predict whether passengers aboard the Spaceship Titanic were transported to another dimension. All models demonstrated strong generalization performance under 5-fold cross-validation, with XGBoost outperforming the others on the validation set:

- **Logistic Regression**  
  - Cross-Validation Accuracy: **77.3% ± 0.0065**  
  - Validation Accuracy: **77.5%**  
  - Despite its simplicity, logistic regression achieved reliable performance and serves as a highly interpretable baseline.

- **Random Forest**  
  - Best Cross-Validation Accuracy: **80.2%**  
  - Validation Accuracy: **80.16%**  
  - Able to capture non-linearities and variable interactions, Random Forest consistently outperformed logistic regression.

- **XGBoost**
  - Best Cross-Validation Accuracy: **80.04% ± 0.00107**
  - Validation Accuracy: **84.36%**
  - Achieved the highest validation accuracy, suggesting superior generalization. XGBoost's ability to model complex feature interactions with regularization led to the most accurate predictions.
 

### **Supplementary Results and Parameter Choices**

Each model was tuned using appropriate hyperparameters and preprocessing strategies: 

- **Logistic Regression**
  - `max_iter = 1000`
  - `C = 1.0` (default L2 regularization)
  - Preprocessing:
    - Numerical: `PowerTransformer` → `StandardScaler`  
    - Categorical: `OneHotEncoder`

- **Random Forest (via GridSearchCV)**
  - `n_estimators = 200`
  - `max_depth = 10`
  - `min_samples_split = 5`
  - `min_samples_leaf = 1`

- **XGBoost (via GridSearchCV)**
  - `n_estimators = 200`
  - `max_depth = 3`
  - `learning_rate = 0.2`
  - `subsample = 1.0`

These parameters were selected through 5-fold cross-validation to balance performance (bias) and generalization (variance). Notably, limiting tree depth in XGBoost and adjusting learning rate allowed for better regularization, helping it outperform Random Forest despite similar CV scores. 

---

## Conclusion: 
In this project, we tackled the challenge of predicting whether passengers aboard the Spaceship Titanic were transported, a binary classification problem from a Kaggle competition. Using a dataset of demographic, behavioral, and financial features, we applied and compared Logistic Regression, Random Forest, and XGBoost models. Extensive preprocessing, including missing value imputation, feature transformation, and encoding, was critical to our pipeline.

After evaluation using metrics like accuracy and cross-validation scores, Random Forest was identified as the best-performing model, balancing performance and robustness. Our findings not only highlight the importance of ensemble methods for complex datasets but also showcase how thoughtful feature engineering and model selection can lead to meaningful insights in predictive analytics. This work sets the stage for broader applications in domains requiring rapid, high-stakes decision-making based on passenger or personnel data.

---

## Discussion: 

The implications of our work extend beyond this specific dataset. The modeling techniques and feature engineering strategies used here can be adapted for real-world scenarios, particularly in emergency response or rescue operations. For example, similar models could assist in identifying missing individuals following natural disasters or shipwrecks by learning from structured passenger data. The ability to rapidly and accurately predict survival likelihoods could support prioritization in rescue missions, ultimately saving lives.

---

## References: 
Eunicl. "Spaceship Titanic with Random Forest + XGBoost." Kaggle, 2023, https://www.kaggle.com/code/eunicl/spaceship-titanic-with-random-forest-xgboost.
Dilber, Burak. "Spaceship Titanic - EDA, Preprocessing and XGBoost." Kaggle, 2023, https://www.kaggle.com/code/burakdilber/spaceship-titanic-eda-preprocessing-and-xgboost.

