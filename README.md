# Smart Grid Fault Detection using Hybrid MLP-HF

This repository implements an end-to-end workflow for detecting **faults and anomalies in Smart Grid Systems** using hybrid deep learning with feature engineering and Explainable AI.

---

# Features

- Data Cleaning & Normalization
- Feature Selection: PCA, Mutual Information, Correlation, K-Means
- Modeling: ML & DL classifiers including proposed MLP-HF
- Explainability: SHAP, LIME
- Statistical Validation: ANOVA, Friedman tests 
- Performance Metrics: Accuracy, Precision, Recall, F1, MCC, AUC
  
## Key Steps

```python
# Load and preprocess data
df = pd.read_csv("/content/drive/MyDrive/dataset.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['stabf'] = LabelEncoder().fit_transform(df['stabf'])
X, y = df.drop(['stab','stabf'], axis=1), df['stabf']
X_scaled = StandardScaler().fit_transform(X)

# Feature Selection
pca = PCA(n_components=5).fit_transform(X_scaled)
mi = mutual_info_classif(X_scaled, y)
corr = pd.Series(X.corrwith(y))
kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(X_scaled)

# Model Training
models = {
    "SVM": SVC(probability=True),
    "RF": RandomForestClassifier(),
    "XGB": XGBClassifier(eval_metric='logloss')
}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    print(name, accuracy_score(y_test, y_pred))

# Evaluation
roc_auc_score(y_test, models["RF"].predict_proba(X_test)[:,1])
shap.summary_plot(shap.TreeExplainer(models["proposed"]).shap_values(X_test)[1], X_test)


