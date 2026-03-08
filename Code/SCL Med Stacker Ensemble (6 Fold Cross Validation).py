import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("6-FOLD CROSS-VALIDATION FOR ENSEMBLE MODEL (LSTM + SNN + CNN)")
print("="*70)

# Prepare the data for cross-validation
X_cv = train_os['review_clean'].values
y_cv = train_os['Review_Sentiment'].values

# Setup stratified k-fold for 6 folds
stratified_kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# Store results
fold_results = []
fold_ensemble_accuracies = []
fold_confusion_matrices = []
fold_roc_aucs = []
fold_ensemble_predictions = []
fold_meta_learners = []

# Parameters
num_words = 50000
maxlen = 200
embedding_vector_length = 100
batch_size = 128
lstm_epochs = 5
snn_epochs = 6
cnn_epochs = 3

print(f"\n📊 Total samples: {len(X_cv):,}")
print(f"📈 Class distribution: {np.bincount(y_cv.astype(int))}")

# Perform 6-fold cross-validation
for fold_num, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_cv, y_cv), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}/6 - ENSEMBLE MODEL")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_idx):,}, Test samples: {len(test_idx):,}")

    # Split data
    X_train_fold, X_test_fold = X_cv[train_idx], X_cv[test_idx]
    y_train_fold, y_test_fold = y_cv[train_idx], y_cv[test_idx]

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=num_words, split=" ", lower=False)
    tokenizer.fit_on_texts(X_train_fold)

    X_train_seq = tokenizer.texts_to_sequences(X_train_fold)
    X_test_seq = tokenizer.texts_to_sequences(X_test_fold)

    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    # Convert labels to categorical for neural networks
    y_train_cat = pd.get_dummies(y_train_fold).values
    y_test_cat = pd.get_dummies(y_test_fold).values

    # Further split training data for meta-learner training
    X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
        X_train_pad, y_train_fold, test_size=0.2, random_state=42
    )

    # Convert validation labels to categorical for predictions
    y_val_base_cat = pd.get_dummies(y_val_base).values

    print("\n🔧 Training Base Models...")

    # ==================== LSTM MODEL ====================
    print("   Training LSTM model...")
    lstm_model = Sequential()
    lstm_model.add(Embedding(num_words, embedding_vector_length, input_length=maxlen))
    lstm_model.add(SpatialDropout1D(0.2))
    lstm_model.add(LSTM(embedding_vector_length))
    lstm_model.add(Dense(2, activation="softmax"))
    lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    lstm_model.fit(
        X_train_base, pd.get_dummies(y_train_base).values,
        epochs=lstm_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    # ==================== SNN MODEL ====================
    print("   Training SNN model...")
    snn_model = Sequential()
    snn_model.add(Embedding(num_words, embedding_vector_length, input_length=maxlen))
    snn_model.add(SpatialDropout1D(0.2))
    snn_model.add(Flatten())
    snn_model.add(Dense(128, activation="relu"))
    snn_model.add(Dense(2, activation="softmax"))
    snn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    snn_model.fit(
        X_train_base, pd.get_dummies(y_train_base).values,
        epochs=snn_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    # ==================== CNN MODEL ====================
    print("   Training CNN model...")
    cnn_model = Sequential()
    cnn_model.add(Embedding(num_words, embedding_vector_length, input_length=maxlen))
    cnn_model.add(Conv1D(64, 5, activation="relu"))
    cnn_model.add(MaxPooling1D(pool_size=4))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(2, activation="softmax"))
    cnn_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    cnn_model.fit(
        X_train_base, pd.get_dummies(y_train_base).values,
        epochs=cnn_epochs,
        batch_size=64,  # CNN uses batch_size=64 as in original
        validation_split=0.1,
        verbose=0
    )

    print("   ✅ Base models trained")

    # ==================== META-LEARNER TRAINING ====================
    print("   Training Meta-learner (Random Forest)...")

    # Get predictions from base models on validation set
    lstm_val_pred = lstm_model.predict(X_val_base, batch_size=batch_size, verbose=0)
    snn_val_pred = snn_model.predict(X_val_base, batch_size=batch_size, verbose=0)
    cnn_val_pred = cnn_model.predict(X_val_base, batch_size=batch_size, verbose=0)

    # Stack predictions for meta-learner
    stacked_val_pred = np.hstack((lstm_val_pred, snn_val_pred, cnn_val_pred))

    # Train meta-learner
    meta_learner = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    meta_learner.fit(stacked_val_pred, y_val_base)

    fold_meta_learners.append(meta_learner)
    print("   ✅ Meta-learner trained")

    # ==================== TEST SET EVALUATION ====================
    print("   Evaluating on test fold...")

    # Get predictions from base models on test set
    lstm_test_pred = lstm_model.predict(X_test_pad, batch_size=batch_size, verbose=0)
    snn_test_pred = snn_model.predict(X_test_pad, batch_size=batch_size, verbose=0)
    cnn_test_pred = cnn_model.predict(X_test_pad, batch_size=batch_size, verbose=0)

    # Stack test predictions
    stacked_test_pred = np.hstack((lstm_test_pred, snn_test_pred, cnn_test_pred))
    fold_ensemble_predictions.append(stacked_test_pred)

    # Get ensemble predictions
    ensemble_pred = meta_learner.predict(stacked_test_pred)
    ensemble_proba = meta_learner.predict_proba(stacked_test_pred)

    # Calculate metrics
    ensemble_accuracy = np.mean(ensemble_pred == y_test_fold)

    # Confusion matrix
    cm = confusion_matrix(y_test_fold, ensemble_pred)
    fold_confusion_matrices.append(cm)

    # ROC AUC
    roc_auc = roc_auc_score(y_test_fold, ensemble_proba[:, 1])
    fold_roc_aucs.append(roc_auc)

    # Store results
    fold_results.append({
        'fold': fold_num,
        'ensemble_accuracy': ensemble_accuracy,
        'roc_auc': roc_auc,
        'test_samples': len(test_idx)
    })

    # Print fold results
    print(f"\n✅ Fold {fold_num} Results:")
    print(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_fold, ensemble_pred))

    # Plot confusion matrix for this fold
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Fold {fold_num} - Ensemble Confusion Matrix (Acc: {ensemble_accuracy:.4f})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ==================== SUMMARY RESULTS ====================
print("\n" + "="*70)
print("6-FOLD CROSS-VALIDATION SUMMARY - ENSEMBLE MODEL")
print("="*70)

# Create summary dataframe
summary_df = pd.DataFrame(fold_results)
print("\n📊 Per-Fold Results:")
print(summary_df.to_string(index=False))

print(f"\n📈 Mean Ensemble Accuracy: {summary_df['ensemble_accuracy'].mean():.4f} (+/- {summary_df['ensemble_accuracy'].std()*2:.4f})")
print(f"🎯 Mean ROC AUC: {summary_df['roc_auc'].mean():.4f} (+/- {summary_df['roc_auc'].std()*2:.4f})")
print(f"   Accuracy Range: [{summary_df['ensemble_accuracy'].min():.4f}, {summary_df['ensemble_accuracy'].max():.4f}]")
print(f"   Std Deviation: {summary_df['ensemble_accuracy'].std():.4f}")

# Plot accuracy across folds
plt.figure(figsize=(10, 6))
plt.plot(summary_df['fold'], summary_df['ensemble_accuracy'], 'go-', linewidth=2, markersize=8)
plt.axhline(y=summary_df['ensemble_accuracy'].mean(), color='r', linestyle='--',
            label=f'Mean: {summary_df["ensemble_accuracy"].mean():.4f}')
plt.fill_between(summary_df['fold'],
                 summary_df['ensemble_accuracy'].mean() - summary_df['ensemble_accuracy'].std(),
                 summary_df['ensemble_accuracy'].mean() + summary_df['ensemble_accuracy'].std(),
                 alpha=0.2, color='gray', label='±1 Std Dev')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('6-Fold Cross-Validation Accuracy - Ensemble Model', fontsize=14)
plt.xticks(range(1, 7))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot all confusion matrices in a grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (cm, ax) in enumerate(zip(fold_confusion_matrices, axes)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                ax=ax)
    ax.set_title(f'Fold {i+1}\nAcc: {summary_df.iloc[i]["ensemble_accuracy"]:.4f}', fontsize=10)
    ax.set_ylabel('True', fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8)

plt.suptitle('All 6 Folds - Confusion Matrices (Ensemble Model)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Calculate and plot average confusion matrix
avg_cm = np.mean(fold_confusion_matrices, axis=0).round().astype(int)
plt.figure(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title(f'Average Confusion Matrix (6 Folds - Ensemble)\nMean Accuracy: {summary_df["ensemble_accuracy"].mean():.4f}',
          fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# ROC Curves for all folds
plt.figure(figsize=(10, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, 6))

for i in range(6):
    # Get the stored predictions for this fold
    stacked_pred = fold_ensemble_predictions[i]

    # Get meta-learner for this fold
    meta = fold_meta_learners[i]

    # Get probabilities
    proba = meta.predict_proba(stacked_pred)

    # Get test indices
    _, test_idx = list(stratified_kfold.split(X_cv, y_cv))[i]
    y_test_fold = y_cv[test_idx]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_fold, proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Fold {i+1} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for All 6 Folds - Ensemble Model', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Box plot of metrics
plt.figure(figsize=(10, 6))
summary_df[['ensemble_accuracy', 'roc_auc']].boxplot()
plt.title('Distribution of Metrics Across 6 Folds - Ensemble Model', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Individual model comparison (using first fold's models as example)
print("\n" + "="*70)
print("INDIVIDUAL MODEL PERFORMANCE (Fold 1)")
print("="*70)

# Use first fold's test data for comparison
_, test_idx = list(stratified_kfold.split(X_cv, y_cv))[0]
X_test_comp = X_cv[test_idx]
y_test_comp = y_cv[test_idx]

# Tokenize for this comparison
tokenizer_comp = Tokenizer(num_words=num_words, split=" ", lower=False)
tokenizer_comp.fit_on_texts(X_train_fold)  # Use training data from first fold

X_test_seq_comp = tokenizer_comp.texts_to_sequences(X_test_comp)
X_test_pad_comp = pad_sequences(X_test_seq_comp, maxlen=maxlen)

# Get predictions from individual models (using first fold's models)
lstm_comp_pred = lstm_model.predict(X_test_pad_comp, verbose=0)
snn_comp_pred = snn_model.predict(X_test_pad_comp, verbose=0)
cnn_comp_pred = cnn_model.predict(X_test_pad_comp, verbose=0)

lstm_acc = np.mean(np.argmax(lstm_comp_pred, axis=1) == y_test_comp)
snn_acc = np.mean(np.argmax(snn_comp_pred, axis=1) == y_test_comp)
cnn_acc = np.mean(np.argmax(cnn_comp_pred, axis=1) == y_test_comp)

print(f"LSTM Accuracy: {lstm_acc:.4f}")
print(f"SNN Accuracy: {snn_acc:.4f}")
print(f"CNN Accuracy: {cnn_acc:.4f}")
print(f"Ensemble Accuracy: {summary_df.iloc[0]['ensemble_accuracy']:.4f}")

# Improvement
ensemble_improvement = summary_df.iloc[0]['ensemble_accuracy'] - max(lstm_acc, snn_acc, cnn_acc)
print(f"\n📈 Ensemble Improvement over best individual: +{ensemble_improvement:.4f}")

print("\n" + "="*70)
print("FINAL CONCLUSION - ENSEMBLE MODEL")
print("="*70)
print(f"✅ Ensemble Model Performance across 6 folds:")
print(f"  - Average Accuracy: {summary_df['ensemble_accuracy'].mean():.4f} (±{summary_df['ensemble_accuracy'].std()*2:.4f})")
print(f"  - Average ROC AUC: {summary_df['roc_auc'].mean():.4f} (±{summary_df['roc_auc'].std()*2:.4f})")

# Stability assessment
if summary_df['ensemble_accuracy'].std() < 0.01:
    print("\n✅ Ensemble is VERY STABLE (low variance across folds)")
elif summary_df['ensemble_accuracy'].std() < 0.02:
    print("\n✅ Ensemble is STABLE (acceptable variance)")
else:
    print("\n⚠️ Ensemble shows HIGH VARIANCE across folds - consider more regularization")

# Save results
summary_df.to_csv('ensemble_6fold_cv_results.csv', index=False)
print("\n✅ Results saved to 'ensemble_6fold_cv_results.csv'")