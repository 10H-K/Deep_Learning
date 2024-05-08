# Enhancing Venture Success Prediction with Deep Learning #


## Overview ##

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Alphabet Soup’s business team has provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization:

  - EIN and NAME: Identification columns
  - APPLICATION_TYPE: Alphabet Soup application type
  - AFFILIATION: Affiliated sector of industry
  - CLASSIFICATION: Government organization classification
  - USE_CASE: Use case for funding
  - ORGANIZATION: Organization type
  - STATUS: Active status
  - INCOME_AMT: Income classification
  - SPECIAL_CONSIDERATIONS: Special considerations for application
  - ASK_AMT: Funding amount requested
  - IS_SUCCESSFUL: Was the money used effectively

The goal of this project is to develop a binary classifier using the provided dataset to predict the success of applicants funded by Alphabet Soup. Leveraging TensorFlow, a deep learning model was crafted for binary classification, predicting the likelihood of success based on applicant features.


## Process ##

1. Preprocessing:
    - Read dataset 'charity_data.csv' using pandas
    - Drop non-beneficial ID column 'EIN' and 'NAME'from the dataset
    - Identify number of unique values in each column using nunique()
    - Explore value counts of categorical variables like 'APPLICATION_TYPE' and 'CLASSIFICATION'
    - Consolidate infrequent values in these columns into an 'Other' category to reduce dimensionality and improve model performance
2. Data Encoding:
    - Convert categorical variables into numeric format using one-hot encoding with pd.get_dummies()
3. Data Splitting:
    - Split preprocessed data into features (X) and target (y) arrays
    - Further split the data into training and testing datasets using train_test_split()
4. Data Scaling:
    - Create instances of StandardScaler and fit them to the training data to scale the features
    - Transform both the training and testing datasets using the fitted scaler
5. Model Construction:
    - Define a deep neural network (DNN) model using TensorFlow's Sequential API
    - Construct the model with multiple hidden layers and appropriate activation functions
6. Model Compilation:
    - Compile the model specifying the loss function, optimizer, and evaluation metrics using compile()
7. Model Training:
    - Train the compiled model on the scaled training data for a specified number of epochs using fit()
8. Model Evaluation:
    - Evaluate the trained model's performance on the scaled testing data, computing loss and accuracy metrics using evaluate()
9. Model Saving:
    - Save the trained model to an HDF5 file for future use or deployment using save()


## Results ##

The deep neural network model was trained and evaluated on the preprocessed dataset to predict the success of applicants funded by Alphabet Soup. After 100 epochs of training, the model achieved the following performance metrics on the testing dataset:
  - Loss: 0.563
  - Accuracy: 72.54%

![image](https://github.com/10H-K/Deep_Learning/assets/152930492/16bce186-e5f3-4ad0-9844-82d39ce5dd3f)
![image](https://github.com/10H-K/Deep_Learning/assets/152930492/de844ee1-0cbf-41c6-bab5-c0b2f301a1c8)

These results indicate that the model, based on its binary classification task, demonstrated a moderate level of accuracy in predicting venture success. The achieved accuracy suggests that the model can effectively discriminate between successful and unsuccessful applicants to a certain extent. However, to achieve the goal of at least 75% accruay, the model was optimized in the follow ways:

1. Feature Engineering:
    - Categorical variables with fewer occurrences were grouped under the 'Other' category to reduce dimensionality and enhance model generalization. This step was also performed for the 'NAME' column, which was previously removed.
2. Model Architecture:
    - An extra hidden layer was added to the neural network model to capture more complex patterns in the data.
    - Adjustments were made to the number of nodes in each hidden layer, optimizing the network's capacity to learn from the data effectively.
3. Activation Functions:
    - Different types of activation functions were experimented with, aiming to enhance the model's ability to capture nonlinear relationships within the data.

These optimizations were implemented iteratively to fine-tune the model's performance and improve its accuracy in predicting venture success.  As a result, after implementing these optimizations, the model's performance was re-evaluated on the testing dataset, yielding the following metrics:
  - Loss: 0.477
  - Accuracy: 76.22%

![image](https://github.com/10H-K/Deep_Learning/assets/152930492/23e6af61-125e-4015-9cb4-90353f49d044)
![image](https://github.com/10H-K/Deep_Learning/assets/152930492/e2ab1c13-3bd4-4cce-80fc-a78183409cf7)

These metrics represent a notable improvement over the initial performance, with the model achieving a 76.22% accuracy in predicting venture success


## Summary ##

The aim of this analysis was to develop a deep learning model to predict the success of applicants funded by Alphabet Soup. Through comprehensive data preprocessing, model construction, training, and evaluation, the project sought to optimize predictive performance.

The target variable, 'IS_SUCCESSFUL', denotes applicant success if funded, while various applicant characteristics serve as features for the model. Initially, columns like 'EIN' and 'NAME' were excluded from input data as they were neither targets nor features. The initial model comprised two hidden layers with 80 and 30 neurons respectively, utilizing 'relu' and 'sigmoid' activation functions.

Despite an initial accuracy of 72.54%, falling short of the 75% target, iterative optimization strategies were employed. These included feature engineering, adjustments to model architecture, and exploration of different activation functions. The optimized model featured three hidden layers with 10, 8, and 6 neurons respectively, utilizing 'relu', 'linear', and 'sigmoid' activation functions. Through optimization, the deep learning model demonstrated significant improvement, achieving an accuracy of 76.22%. Nonetheless, further refinement may be necessary for real-world robustness.

For future iterations, exploring alternative models such as gradient boosting or random forests, known for handling categorical data effectively, could prove beneficial. Additionally, integrating deep learning with traditional machine learning algorithms through techniques like stacking could enhance prediction accuracy, offering a comprehensive approach to addressing the classification problem.

