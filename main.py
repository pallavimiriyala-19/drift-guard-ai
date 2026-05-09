import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from loguru import logger
import os
from openai import OpenAI # Using OpenAI for LLM interaction

class DriftDetector:
    """Simulates drift detection using basic statistical methods."""
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect_feature_drift(self, base_data, new_data):
        drift_features = {}
        for col in base_data.columns:
            if base_data[col].dtype in ['int64', 'float64']:
                # Simple mean difference as drift indicator
                mean_diff = abs(base_data[col].mean() - new_data[col].mean())
                if mean_diff > self.threshold:
                    drift_features[col] = f"Mean drift: {mean_diff:.2f}"
        return drift_features

    def detect_concept_drift(self, base_predictions, new_predictions, base_actuals, new_actuals):
        # Simulate concept drift based on accuracy drop
        base_acc = accuracy_score(base_actuals, base_predictions)
        new_acc = accuracy_score(new_actuals, new_predictions)
        if (base_acc - new_acc) > self.threshold * 2: # More sensitive for concept drift
            return f"Accuracy dropped from {base_acc:.2f} to {new_acc:.2f}"
        return None

class PerformanceMonitor:
    """Tracks model performance metrics."""
    def __init__(self):
        self.metrics = {}

    def evaluate(self, model, X_test, y_test, metric_func=accuracy_score):
        predictions = model.predict(X_test)
        score = metric_func(y_test, predictions)
        self.metrics['current_score'] = score
        logger.info(f"Current model performance: {score:.4f}")
        return score, predictions

class DecisionAgent:
    """LLM-powered agent to decide on actions based on drift and performance."""
    def __init__(self, llm_model="gpt-4o-mini", api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model

    def decide_action(self, drift_report, performance_report, model_name):
        prompt = f"""You are an MLOps autonomous agent for the model '{model_name}'.
        You have detected the following issues:

        Drift Report:
        {drift_report if drift_report else 'No significant drift detected.'}

        Performance Report:
        Current model accuracy: {performance_report.get('current_score', 'N/A')}

        Based on this information, decide the best course of action. Possible actions include:
        1. RETRAIN: The model needs to be re-trained with new data.
        2. FEATURE_ENGINEERING: Suggest specific feature engineering steps (e.g., 'log transform feature_X').
        3. ALERT_HUMAN: Critical issue requiring human intervention (e.g., data pipeline broken).
        4. NO_ACTION: No immediate action is required.

        Provide your decision as a single word (RETRAIN, FEATURE_ENGINEERING, ALERT_HUMAN, NO_ACTION) followed by a brief, comma-separated explanation or suggested action. 
        Example: 'RETRAIN, significant data drift detected, recommend full retraining'
        Example: 'FEATURE_ENGINEERING, feature 'age' distribution shifted, consider robust scaling'
        Example: 'NO_ACTION, minor fluctuations, within acceptable limits'
        """
        
        logger.debug(f"Sending prompt to LLM: {prompt}")
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a highly intelligent and pragmatic MLOps autonomous agent."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            decision_text = response.choices[0].message.content.strip()
            logger.info(f"LLM Decision: {decision_text}")
            action, *explanation = decision_text.split(',', 1)
            return action.strip().upper(), explanation[0].strip() if explanation else "No further explanation provided."
        except Exception as e:
            logger.error(f"Error calling LLM: {e}. Defaulting to ALERT_HUMAN.")
            return "ALERT_HUMAN", f"LLM failed to respond: {e}"

class RetrainingOrchestrator:
    """Manages model retraining and updates."""
    def __init__(self):
        self.retrained_models = []

    def retrain_model(self, model_class, current_model_params, training_data):
        logger.info("Triggering model retraining...")
        new_model = model_class(**current_model_params)
        X_train = training_data['features']
        y_train = training_data['target']
        new_model.fit(X_train, y_train)
        self.retrained_models.append(new_model)
        logger.success("Model successfully re-trained.")
        return new_model

class DriftGuardAgent:
    """Main autonomous agent orchestrating drift detection and model adaptation."""
    def __init__(self, initial_model, training_data, model_name="default_model", llm_model="gpt-4o-mini"):
        self.current_model = initial_model
        self.training_data = training_data # Store original training data for retraining
        self.model_name = model_name
        self.llm_model = llm_model

        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.decision_agent = DecisionAgent(llm_model=llm_model)
        self.retraining_orchestrator = RetrainingOrchestrator()
        self.action_log = []

        logger.info(f"DriftGuard Agent initialized for model: {self.model_name}")

    def monitor_and_adapt(self, new_production_data):
        logger.info(f"Starting monitoring cycle for {self.model_name}...")

        X_new_prod = pd.DataFrame(new_production_data['features'], columns=[f'feature_{i}' for i in range(new_production_data['features'].shape[1])])
        y_new_prod = new_production_data['target']

        X_train_df = pd.DataFrame(self.training_data['features'], columns=[f'feature_{i}' for i in range(self.training_data['features'].shape[1])])
        y_train_actuals = self.training_data['target']

        # 1. Evaluate current model performance on new production data
        current_score, new_prod_predictions = self.performance_monitor.evaluate(self.current_model, X_new_prod, y_new_prod)
        performance_report = {'current_score': current_score}

        # 2. Predict on original training data for baseline concept drift
        original_train_predictions = self.current_model.predict(X_train_df)

        # 3. Detect drift
        feature_drift = self.drift_detector.detect_feature_drift(X_train_df, X_new_prod)
        concept_drift = self.drift_detector.detect_concept_drift(
            original_train_predictions,
            new_prod_predictions,
            y_train_actuals,
            y_new_prod
        )

        drift_report = {
            'feature_drift': feature_drift,
            'concept_drift': concept_drift
        }

        if feature_drift or concept_drift:
            logger.warning(f"Drift detected! Feature: {feature_drift}, Concept: {concept_drift}")
        else:
            logger.info("No significant drift detected.")

        # 4. Decision Agent makes a call
        action, explanation = self.decision_agent.decide_action(drift_report, performance_report, self.model_name)
        self.action_log.append({'timestamp': pd.Timestamp.now(), 'action': action, 'explanation': explanation, 'drift_report': drift_report, 'performance_report': performance_report})
        logger.info(f"Agent decided: {action} - {explanation}")

        # 5. Execute action
        if action == "RETRAIN":
            model_class = type(self.current_model)
            current_model_params = self.current_model.get_params()
            new_model = self.retraining_orchestrator.retrain_model(
                model_class, current_model_params, self.training_data # In a real scenario, use augmented/new data
            )
            self.current_model = new_model
            logger.success(f"Model '{self.model_name}' updated with new retrained model.")
        elif action == "ALERT_HUMAN":
            logger.critical(f"Human intervention required for '{self.model_name}': {explanation}")
        elif action == "FEATURE_ENGINEERING":
            logger.warning(f"Feature engineering suggested for '{self.model_name}': {explanation}. This would typically trigger a separate pipeline.")
        else: # NO_ACTION
            logger.info("No action taken as per agent's decision.")

        logger.info(f"Monitoring cycle for {self.model_name} completed.")

if __name__ == '__main__':
    # Setup for demonstration
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>")

    # Ensure OPENAI_API_KEY is set in your environment variables
    if os.getenv("OPENAI_API_KEY") is None:
        logger.warning("OPENAI_API_KEY environment variable not set. DecisionAgent will default to ALERT_HUMAN. Please set it for full functionality.")

    # 1. Train an initial model
    np.random.seed(42)
    X_train_base = np.random.rand(1000, 5)
    y_train_base = (X_train_base[:, 0] * 2 + X_train_base[:, 1] - X_train_base[:, 2] > 1.5).astype(int)
    initial_model = LogisticRegression(solver='liblinear')
    initial_model.fit(X_train_base, y_train_base)
    logger.info("Initial model trained successfully.")

    # 2. Initialize the DriftGuard Agent
    agent = DriftGuardAgent(
        initial_model=initial_model,
        training_data={'features': X_train_base, 'target': y_train_base},
        model_name="CustomerChurnPredictor",
        llm_model="gpt-4o-mini" # Or your preferred LLM
    )

    # --- Scenario 1: No significant drift ---
    logger.info("\n--- Running Scenario 1: No significant drift ---")
    X_prod_stable = np.random.rand(200, 5)
    y_prod_stable = (X_prod_stable[:, 0] * 2 + X_prod_stable[:, 1] - X_prod_stable[:, 2] > 1.4).astype(int) # Slight variation
    agent.monitor_and_adapt(new_production_data={'features': X_prod_stable, 'target': y_prod_stable})

    # --- Scenario 2: Data drift detected ---
    logger.info("\n--- Running Scenario 2: Data drift detected ---")
    X_prod_drift = np.random.rand(200, 5)
    X_prod_drift[:, 0] = X_prod_drift[:, 0] * 3 + 1 # Introduce significant drift in feature 0
    X_prod_drift[:, 3] = X_prod_drift[:, 3] * 2.5 # Introduce significant drift in feature 3
    y_prod_drift_actual = (X_prod_drift[:, 0] * 2 + X_prod_drift[:, 1] - X_prod_drift[:, 2] > 1.5).astype(int) # Same concept, just data changed
    agent.monitor_and_adapt(new_production_data={'features': X_prod_drift, 'target': y_prod_drift_actual})

    # --- Scenario 3: Concept drift detected (model logic changes) ---
    logger.info("\n--- Running Scenario 3: Concept drift detected ---")
    X_prod_concept_drift = np.random.rand(200, 5)
    # Now, feature 4 becomes more important, and feature 0 less important (concept changes)
    y_prod_concept_drift_actual = (X_prod_concept_drift[:, 4] * 3 + X_prod_concept_drift[:, 1] * 0.5 > 2).astype(int)
    agent.monitor_and_adapt(new_production_data={'features': X_prod_concept_drift, 'target': y_prod_concept_drift_actual})

    logger.info("\n--- All scenarios completed. Final action log: ---")
    for entry in agent.action_log:
        logger.info(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] Action: {entry['action']}, Explanation: {entry['explanation']}")