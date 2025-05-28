# MLflow Experiment Tracking and Model Management Homework 2

This homework assignment demonstrates essential MLflow capabilities through six progressive exercises covering experiment tracking, hyperparameter optimization, and model registry operations. The exercises build upon each other to create a complete machine learning operations workflow that mirrors industry practices.

## Learning Objectives and Context

MLflow addresses critical challenges in machine learning lifecycle management by providing systematic approaches to experiment tracking, model comparison, and production deployment. Modern machine learning teams generate hundreds of experiments during model development, making it essential to have robust systems for tracking configurations, results, and model artifacts.

This assignment progresses through the fundamental MLflow workflow used in professional machine learning environments. The exercises begin with basic experiment tracking, advance through sophisticated hyperparameter optimization, and conclude with model registry operations that prepare models for production deployment. Each exercise demonstrates core concepts while building toward a comprehensive understanding of machine learning operations best practices.

## Problem Domain: NYC Taxi Trip Duration Prediction

The exercises utilize the NYC Green Taxi Trip Records dataset to predict trip duration based on pickup and dropoff locations plus trip distance. This regression problem provides an excellent foundation for learning MLflow concepts without complex domain-specific considerations that might distract from the core learning objectives.

The dataset spans three months of 2023 data, enabling proper train-validation-test splits that reflect production machine learning workflows. January data serves as the training set, February data supports validation during hyperparameter tuning, and March data provides the final test set for unbiased model evaluation.

## Prerequisites and Environment Setup

These exercises assume solid understanding of Python programming, scikit-learn fundamentals, and core machine learning concepts including training, validation, and test set methodologies. Command-line operation familiarity and basic regression metrics knowledge, particularly RMSE (Root Mean Square Error), are also essential.

Setting up a dedicated Python environment prevents dependency conflicts and ensures reproducible results. The required packages include MLflow, scikit-learn, pandas, numpy, and hyperopt. Version management becomes crucial for reproducibility, making requirements files or environment specifications valuable for documentation purposes.

## Exercise Analysis and Expected Outcomes

### Question 1: MLflow Installation and Version Verification
*Expected Answer: 2.22.0*

Version verification establishes the foundation for reproducible experiments and debugging procedures. MLflow versions can exhibit different behaviors, making version documentation crucial for experiment reproducibility. This exercise introduces the MLflow command-line interface and confirms proper installation before proceeding to more complex operations.

### Question 2: Data Preprocessing and Pipeline Foundation
*Expected Answer: 4 files*

Data preprocessing represents the critical foundation of successful machine learning projects. The `preprocess_data.py` script handles essential tasks including raw data loading, feature vector creation, and train-validation-test split establishment. Understanding the preprocessing pipeline becomes essential for maintaining data consistency throughout the machine learning workflow.

The script generates four essential files: a DictVectorizer for feature encoding consistency, and three dataset files containing processed training, validation, and test sets. Each file serves a specific purpose in the machine learning pipeline. The DictVectorizer preservation ensures that future predictions maintain identical feature encoding to the training process, preventing subtle but critical data inconsistencies.

### Question 3: Model Training with Automatic Logging
*Expected Answer: 2 (min_samples_split parameter)*

MLflow's automatic logging capabilities significantly reduce the manual overhead of experiment tracking while ensuring comprehensive documentation of model development. Enabling autologging and wrapping training code in MLflow run contexts automatically captures model hyperparameters, training metrics, and trained model artifacts.

The `min_samples_split` parameter investigation demonstrates scikit-learn's default value handling and MLflow's comprehensive parameter logging. When hyperparameters are not explicitly specified, scikit-learn applies default values, and MLflow's autologging captures both explicit and default parameters. This comprehensive parameter documentation becomes invaluable for reproducing past experiments and understanding model behavior.

### Question 4: Production-Ready Tracking Server Configuration
*Expected Answer: default-artifact-root*

Transitioning from local experiment tracking to centralized tracking server configuration represents a crucial step toward production-ready MLOps practices. This exercise demonstrates MLflow configuration with persistent storage using SQLite for metadata management and dedicated artifact directories for large file storage.

The distinction between backend store for metadata and artifact store for large files reflects fundamental distributed systems architectural principles. The `default-artifact-root` parameter specifies storage locations for model artifacts, plots, and other files exceeding database storage limitations. This separation enables scalable storage strategies as machine learning operations expand.

### Question 5: Systematic Hyperparameter Optimization
*Expected Answer: 5.335 (validation RMSE)*

Hyperparameter optimization represents one of the most computationally intensive aspects of machine learning model development. This exercise introduces hyperopt, a sophisticated optimization library utilizing Bayesian optimization for intelligent hyperparameter space exploration.

MLflow integration transforms potentially chaotic hyperparameter experimentation into systematic, trackable scientific investigation. Each hyperparameter combination generates a separate MLflow run, creating a searchable database of successful and unsuccessful configurations. The Tree-structured Parzen Estimator algorithm learns from previous attempts to make increasingly educated guesses about promising hyperparameter regions.

The validation RMSE of approximately 5.335 minutes represents reasonable performance for taxi trip duration prediction. This metric indicates that model predictions typically err by about 5.3 minutes, which could provide useful information for passenger planning and fleet management applications.

### Question 6: Model Registry and Production Promotion
*Expected Answer: 5.567 (test RMSE)*

The final exercise represents the culmination of the machine learning development process: selecting optimal model configurations and preparing them for potential production deployment. This step introduces MLflow's Model Registry, which provides governance and lifecycle management capabilities for machine learning models.

The exercise demonstrates evaluation of top hyperparameter combinations using completely fresh test data, providing unbiased estimates of real-world performance. The slight increase from validation RMSE (5.335) to test RMSE (5.567) represents normal variation and suggests that hyperparameter optimization did not significantly overfit to the validation set.

Model registration creates permanent, versioned artifacts that can progress through staging environments to production deployment. This systematic approach to model promotion ensures that only thoroughly evaluated models reach production systems.

## Technical Implementation Patterns

Throughout these exercises, several important technical patterns emerge that reflect professional machine learning engineering practices. Consistent use of Click decorators for command-line interfaces facilitates script integration into automated pipelines. Systematic approaches to constants and configuration management prevent scattered hardcoded values that make codebases difficult to maintain.

The version-compatible approach to RMSE calculation, using MSE calculation followed by square root operation, demonstrates defensive programming practices required when working with evolving libraries. Different scikit-learn versions handle the `squared` parameter differently, making the compatible approach essential for ensuring code functionality across different environments.

## Broader MLOps Context and Applications

These exercises provide solid foundations in MLflow fundamentals while representing only the beginning of comprehensive MLOps practices. Production environments typically extend these foundations with automated model monitoring, A/B testing frameworks, continuous integration for model training, and sophisticated deployment strategies.

The patterns demonstrated here scale naturally to more complex scenarios. The same experiment tracking approaches work whether training simple linear models or complex deep learning architectures. Model registry concepts apply whether deploying models to single servers or distributed cloud environments.

This homework establishes vocabulary and basic concepts for more advanced MLOps topics including model drift detection, automated retraining pipelines, and multi-model deployment strategies. The systematic approach to experimentation and model management provides foundations for building robust, maintainable machine learning systems.

## Troubleshooting and Common Considerations

Version compatibility issues, particularly with the `squared` parameter in scikit-learn's `mean_squared_error` function, can be addressed using the version-compatible implementation patterns demonstrated in the exercise solutions. These patterns ensure code functionality across different library versions, which becomes crucial when collaborating across diverse development environments.

Attention to consistent patterns across different scripts, including constants definition, configuration management, and error handling, reflects best practices that make machine learning codebases more maintainable and collaborative rather than arbitrary stylistic choices.

The exercises are designed for sequential completion, as each builds upon artifacts and experiments created in previous steps. Maintaining MLflow tracking server operation throughout later exercises is essential, as model registry operations depend on the centralized tracking infrastructure established in Question 4.

This comprehensive approach to MLflow mastery prepares practitioners for real-world machine learning operations where systematic experiment tracking and model management distinguish professional, reproducible machine learning systems from chaotic research environments.