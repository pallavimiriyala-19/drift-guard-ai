import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger
import os

# Assuming drift_guard_ai.agent and its dependencies are in the Python path
# In a real setup, you'd have: from drift_guard_ai.agent import DriftGuardAgent
# For this example, we'll import directly if running from the main_code directory

# This assumes the main_code.py is in the same directory as example_usage.py
# Or, if drift_guard_ai is installed as a package, use: from drift_guard_ai.agent import DriftGuardAgent
# For local testing, we'll simulate the import by defining necessary classes here

# --- Copying minimal definitions needed for demonstration from main_code.py ---
# In a real environment, you would just import these.
from sklearn.metrics import accuracy_score
from openai import OpenAI

class DriftDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect_feature_drift(self, base_data, new_data):
        drift_features = {}
        for col in base_data.columns:
            if base_data[col].dtype in ['int64', 'float64']:
                mean_diff = abs(base_data[col].mean() - new_data[col].mean())
                if mean_diff > self.threshold:
                    drift_features[col] = f"Mean drift: {mean_diff:.2f}"
        return drift_features

    def detect_concept_drift(self, base_predictions, new_predictions, base_actuals, new_actuals):
        base_acc = accuracy_score(base_actuals, base_predictions)
        new_acc = accuracy_score(new_actuals, new_predictions)
        if (base_acc - new_acc) > self.threshold * 2:
            return f"Accuracy dropped from {base_acc:.2f} to {new_acc:.2f}"
        return None

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def evaluate(self, model, X_test, y_test, metric_func=accuracy_score):
        predictions = model.predict(X_test)
        score = metric_func(y_test, predictions)
        self.metrics['current_score'] = score
        logger.info(f"Current model performance: {score:.4f}")
        return score, predictions

class DecisionAgent:
    def __init__(self, llm_model="gpt-4o-mini", api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.llm_model = llm_model

    def decide_action(self, drift_report, performance_report, model_name):
        prompt = f"""You are an MLOps autonomous agent for the model '{model_name}'.\n        You have detected the following issues:\n\n        Drift Report:\n        {drift_report if drift_report else 'No significant drift detected.'}\n\n        Performance Report:\n        Current model accuracy: {performance_report.get('current_score', 'N/A')}\n\n        Based on this information, decide the best course of action. Possible actions include:\n        1. RETRAIN: The model needs to be re-trained with new data.\n        2. FEATURE_ENGINEERING: Suggest specific feature engineering steps (e.g., 'log transform feature_X').\n        3. ALERT_HUMAN: Critical issue requiring human intervention (e.g., data pipeline broken).\n        4. NO_ACTION: No immediate action is required.\n\n        Provide your decision as a single word (RETRAIN, FEATURE_ENGINEERING, ALERT_HUMAN, NO_ACTION) followed by a brief, comma-separated explanation or suggested action. \n        Example: 'RETRAIN, significant data drift detected, recommend full retraining'\n        Example: 'FEATURE_ENGINEERING, feature 'age' distribution shifted, consider robust scaling'\n        Example: 'NO_ACTION, minor fluctuations, within acceptable limits'\n        """
        
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
    def __init__(self, initial_model, training_data, model_name="default_model", llm_model="gpt-4o-mini"):
        self.current_model = initial_model
        self.training_data = training_data
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

        current_score, new_prod_predictions = self.performance_monitor.evaluate(self.current_model, X_new_prod, y_new_prod)
        performance_report = {'current_score': current_score}

        original_train_predictions = self.current_model.predict(X_train_df)

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

        action, explanation = self.decision_agent.decide_action(drift_report, performance_report, self.model_name)
        self.action_log.append({'timestamp': pd.Timestamp.now(), 'action': action, 'explanation': explanation, 'drift_report': drift_report, 'performance_report': performance_report})
        logger.info(f"Agent decided: {action} - {explanation}")

        if action == "RETRAIN":
            model_class = type(self.current_model)
            current_model_params = self.current_model.get_params()
            new_model = self.retraining_orchestrator.retrain_model(
                model_class, current_model_params, self.training_data
            )
            self.current_model = new_model
            logger.success(f"Model '{self.model_name}' updated with new retrained model.")
        elif action == "ALERT_HUMAN":
            logger.critical(f"Human intervention required for '{self.model_name}': {explanation}")
        elif action == "FEATURE_ENGINEERING":
            logger.warning(f"Feature engineering suggested for '{self.model_name}': {explanation}. This would typically trigger a separate pipeline.")
        else:
            logger.info("No action taken as per agent's decision.")

        logger.info(f"Monitoring cycle for {self.model_name} completed.")
# --- End of minimal definitions ---


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>")

    # !!! IMPORTANT !!!
    # Set your OpenAI API key as an environment variable or uncomment and set it directly:
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
    if os.getenv("OPENAI_API_KEY") is None:
        logger.warning("OPENAI_API_KEY environment variable not set. DecisionAgent will default to ALERT_HUMAN for any LLM interaction. Please set it for full functionality.")

    # 1. Prepare initial training data and model
    np.random.seed(42)
    X_train_base = np.random.rand(1000, 5)
    y_train_base = (X_train_base[:, 0] * 2 + X_train_base[:, 1] - X_train_base[:, 2] > 1.5).astype(int)
    initial_model = LogisticRegression(solver='liblinear', random_state=42)
    initial_model.fit(X_train_base, y_train_base)
    logger.info("Initial model trained successfully.")

    # 2. Initialize the DriftGuard Agent
    agent = DriftGuardAgent(
        initial_model=initial_model,
        training_data={'features': X_train_base, 'target': y_train_base},
        model_name="CustomerChurnPredictor",
        llm_model="gpt-4o-mini" # You can change this to other OpenAI models or mock if no API key
    )

    print("\n" + "="*60 + "\n")
    logger.info("--- Scenario 1: No significant drift ---")
    print("\n" + "="*60 + "\n")
    X_prod_stable = np.random.rand(200, 5)
    y_prod_stable = (X_prod_stable[:, 0] * 2 + X_prod_stable[:, 1] - X_prod_stable[:, 2] > 1.4).astype(int) # Slight variation, but within acceptable limits
    agent.monitor_and_adapt(new_production_data={'features': X_prod_stable, 'target': y_prod_stable})

    print("\n" + "="*60 + "\n")
    logger.info("--- Scenario 2: Data drift detected ---")
    print("\n" + "="*60 + "\n")
    X_prod_drift = np.random.rand(200, 5)
    X_prod_drift[:, 0] = X_prod_drift[:, 0] * 3 + 1.5 # Introduce significant shift in feature 0
    X_prod_drift[:, 3] = X_prod_drift[:, 3] * 2.0 # Introduce significant shift in feature 3
    y_prod_drift_actual = (X_prod_drift[:, 0] * 2 + X_prod_drift[:, 1] - X_prod_drift[:, 2] > 1.5).astype(int) # Same underlying concept, just data distribution changed
    agent.monitor_and_adapt(new_production_data={'features': X_prod_drift, 'target': y_prod_drift_actual})

    print("\n" + "="*60 + "\n")
    logger.info("--- Scenario 3: Concept drift detected (underlying relationship changes) ---")
    print("\n" + "="*60 + "\n")
    X_prod_concept_drift = np.random.rand(200, 5)
    # Now, feature 4 becomes more important, and feature 0 less important -- a change in the true function
    y_prod_concept_drift_actual = (X_prod_concept_drift[:, 4] * 3 + X_prod_concept_drift[:, 1] * 0.5 - X_prod_concept_drift[:, 0] * 0.1 > 2.5).astype(int)
    agent.monitor_and_adapt(new_production_data={'features': X_prod_concept_drift, 'target': y_prod_concept_drift_actual})

    print("\n" + "="*60 + "\n")
    logger.info("--- All scenarios completed. Final action log: ---")
    print("="*60 + "\n")
    for entry in agent.action_log:
        logger.info(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] Model: {agent.model_name}, Action: {entry['action']}, Explanation: {entry['explanation']}")
        if entry['drift_report']['feature_drift']:
            logger.info(f"  Feature Drift: {entry['drift_report']['feature_drift']}")
        if entry['drift_report']['concept_drift']:
            logger.info(f"  Concept Drift: {entry['drift_report']['concept_drift']}")
        logger.info(f"  Current Performance: {entry['performance_report'].get('current_score', 'N/A'):.4f}")
    print("\n" + "="*60)
