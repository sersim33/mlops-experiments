import os
import shutil
from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# -------------------------
# PushGateway helper
# -------------------------
def push_metric_to_gateway(metric_name: str, value: float, run_id: str):
    registry = CollectorRegistry()
    gauge = Gauge(metric_name, f"Metric {metric_name}", ["run_id"], registry=registry)
    gauge.labels(run_id=run_id).set(value)
    push_to_gateway(
        os.environ["PUSHGATEWAY_URL"],
        job="mlflow_training",
        registry=registry
    )


# -------------------------
# MLflow connection
# -------------------------
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

experiment_name = "Iris Classification"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✅ Створено експеримент '{experiment_name}' (ID={experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f"ℹ️ Використовується існуючий експеримент '{experiment_name}' (ID={experiment_id})")


# -------------------------
# Дані
# -------------------------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# -------------------------
# Гіперпараметри для циклу
# -------------------------
learning_rates = [0.01, 0.05, 0.1]
epochs_list = [100, 200]

run_results = []  # збережемо (accuracy, run_id)


# -------------------------
# Цикл тренування
# -------------------------
for lr in learning_rates:
    for epochs in epochs_list:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f"\n▶ Запуск run_id={run_id}  lr={lr}, epochs={epochs}")

            # Параметри
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", epochs)

            # Модель
            model = LogisticRegression(max_iter=epochs)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)

            # Метрики
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("loss", loss)

            # Логування артефакту моделі
            mlflow.sklearn.log_model(model, "model")

            # PushGateway метрики
            push_metric_to_gateway("mlflow_accuracy", acc, run_id)
            push_metric_to_gateway("mlflow_loss", loss, run_id)

            print(f"   ℹ accuracy={acc:.4f}  loss={loss:.4f}")

            run_results.append((acc, run_id))


# -------------------------
# Пошук найкращої моделі
# -------------------------
best_run_id = max(run_results, key=lambda x: x[0])[1]
print(f"\n🎯 Найкращий run_id: {best_run_id}")

model_src = f"../mlruns/0/{best_run_id}/artifacts/model"
model_dst = "../best_model"

# очистити стару best_model
if os.path.exists(model_dst):
    shutil.rmtree(model_dst)

shutil.copytree(model_src, model_dst)

print(f"✅ Найкращу модель скопійовано в {model_dst}/")