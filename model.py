from sklearn.ensemble import RandomForestClassifier
from data import X_train, y_train, X_test, y_test

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
