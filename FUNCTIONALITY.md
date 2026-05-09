# FUNCTIONALITY.md: DriftGuard AI

## Overview

DriftGuard AI is an autonomous, agent-driven MLOps solution designed to proactively manage the performance and stability of machine learning models in production environments. Its core functionality revolves around detecting various forms of model decay (data drift, concept drift) and intelligently orchestrating adaptive responses, including automated retraining or human alerting, leveraging the power of Large Language Models (LLMs).

## Architecture and Components

The system is composed of several interconnected modules, orchestrated by the main `DriftGuardAgent` class. Below is a detailed breakdown of each component and their interactions.

```mermaid
graph TD
    A[Production Data Stream] --> B{Data Collector / Monitor}
    B --> C[Feature Store / Data Warehouse]
    C -- new_production_data --> D[Drift Detector]
    C -- new_production_data --> E[Performance Tracker]
    D -- drift_report --> F[Decision Agent (LLM-powered)]
    E -- performance_report --> F
    F -- action, explanation --> G{Retraining Orchestrator}
    G -- new_model --> H[Model Registry / MLOps Platform]
    H --> A
    F -- alert_info --> I[Alerting & Reporting]
    G -- retraining_status --> I
    subgraph DriftGuard AI Core
        D
        E
        F
        G
    end
    C -- training_data_baseline --> D
    C -- training_data_baseline --> G
    E -- current_model --> F
```

### 1. Data Collector / Monitor

*   **Role**: Ingests and prepares live production data for analysis. In a real-world scenario, this would be a connector to a data pipeline (e.g., Kafka, Flink, a data warehouse query).
*   **Functionality**: Standardizes data formats, handles missing values, and ensures data types match the expected schema of the model.
*   **Data Flow**: Receives `new_production_data` (features and actual targets) from the production environment, and retrieves `training_data_baseline` from a feature store or data warehouse for comparison.
*   **Design Decisions**: For this project, it's simulated by passing `new_production_data` dicts. In a production system, this module would be robust and fault-tolerant, possibly using a streaming framework.

### 2. Drift Detector (`DriftDetector` class)

*   **Role**: Identifies statistical differences between the distribution of features and the relationship between features and targets in production data versus the training data.
*   **Functionality**:
    *   **Feature Drift**: Compares statistical properties (e.g., mean, variance, distributions) of individual features in `new_production_data` against `training_data_baseline`. Advanced implementations would use statistical tests (KS-test, PSI, Jensen-Shannon divergence) or ML-based detectors (e.g., from `Evidently AI`, `Alibi Detect`).
    *   **Concept Drift**: Assesses changes in the underlying relationship between input features and the target variable. This is often inferred by comparing model performance on `new_production_data` against expected performance, or by comparing prediction distributions. The current simulation uses accuracy drop as a proxy.
*   **Data Flow**: Takes `training_data_baseline` and `new_production_data` (features, predictions, actuals) as input. Outputs a `drift_report` (dictionary indicating which features drifted and if concept drift was detected).
*   **Design Decisions**: The current `DriftDetector` is a simplified implementation using mean differences and accuracy drops. In production, this would be replaced by more sophisticated and configurable libraries to detect various types of drift (sudden, gradual, incremental).

### 3. Performance Tracker (`PerformanceMonitor` class)

*   **Role**: Continuously evaluates the current production model's performance on the most recent `new_production_data` for which actual labels are available.
*   **Functionality**: Calculates key performance metrics (e.g., accuracy, precision, recall, F1-score for classification; RMSE, MAE for regression) and compares them against predefined thresholds or historical baselines.
*   **Data Flow**: Takes the `current_model`, `new_production_data` (features and actual targets). Outputs a `performance_report` (e.g., current accuracy score, other relevant metrics) and the `new_prod_predictions`.
*   **Design Decisions**: Modular to allow easy swapping of metrics or integration with external model evaluation services (e.g., MLflow).

### 4. Decision Agent (`DecisionAgent` class - LLM-powered)

*   **Role**: The 