import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Make sure train_os is defined
# If not, add your oversampling code here
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler()
# train_x, train_y = ros.fit_resample(np.array(req_df['review_clean']).reshape(-1, 1),
#                                     np.array(req_df['Review_Sentiment']).reshape(-1, 1));
# train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)),
#                        columns=['review_clean', 'Review_Sentiment']);

print(f"Data shape: {train_os.shape}")
print(f"Class distribution:\n{train_os['Review_Sentiment'].value_counts()}")

# Prepare the data for cross-validation
X_cv = train_os['review_clean'].values
y_cv = train_os['Review_Sentiment'].values

# Setup stratified k-fold for 6 folds
stratified_kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# Store results
fold_results = []
fold_histories = []
fold_confusion_matrices = []
fold_roc_aucs = []
fold_predictions = []

# Parameters
num_words = 50000
maxlen = 200
embedding_vector_length = 100
epochs = 3  # As in your original code
batch_size = 64  # As in your original code

print("\n" + "="*60)
print("STARTING 6-FOLD CROSS-VALIDATION FOR CNN MODEL")
print("="*60)

# Perform 6-fold cross-validation
for fold_num, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_cv, y_cv), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}/6 - CNN MODEL")
    print(f"{'='*60}")
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

    # Convert labels to categorical
    y_train_cat = pd.get_dummies(y_train_fold).values
    y_test_cat = pd.get_dummies(y_test_fold).values

    # Build CNN model
    cnn_model = Sequential()
    cnn_model.add(Embedding(num_words, embedding_vector_length, input_length=maxlen))
    cnn_model.add(Conv1D(64, 5, activation="relu"))
    cnn_model.add(MaxPooling1D(pool_size=4))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(2, activation="softmax"))

    cnn_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    # Train model
    history = cnn_model.fit(
        X_train_pad, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # 10% validation split as in your original code
        verbose=1
    )

    # Evaluate model
    accr = cnn_model.evaluate(X_test_pad, y_test_cat, verbose=0)
    predictions = cnn_model.predict(X_test_pad, batch_size=batch_size, verbose=0)
    fold_predictions.append(predictions)

    classes_x = np.argmax(predictions, axis=1)
    labels = np.argmax(y_test_cat, axis=1)

    # Calculate metrics
    accuracy = accr[1]
    loss = accr[0]

    # Confusion matrix
    cm = confusion_matrix(labels, classes_x)
    fold_confusion_matrices.append(cm)

    # ROC AUC
    roc_auc = roc_auc_score(y_test_cat, predictions)
    fold_roc_aucs.append(roc_auc)

    # Store results
    fold_results.append({
        'fold': fold_num,
        'accuracy': accuracy,
        'loss': loss,
        'roc_auc': roc_auc,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx)
    })
    fold_histories.append(history)

    # Print fold results
    print(f"\n✅ Fold {fold_num} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Loss: {loss:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, classes_x))

    # Plot confusion matrix for this fold
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Fold {fold_num} - CNN Confusion Matrix (Acc: {accuracy:.4f})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# After all folds, show summary results
print("\n" + "="*60)
print("6-FOLD CROSS-VALIDATION SUMMARY - CNN MODEL")
print("="*60)

# Create summary dataframe
summary_df = pd.DataFrame(fold_results)
print("\n📊 Per-Fold Results:")
print(summary_df.to_string(index=False))

print(f"\n📈 Mean Accuracy: {summary_df['accuracy'].mean():.4f} (+/- {summary_df['accuracy'].std()*2:.4f})")
print(f"📉 Mean Loss: {summary_df['loss'].mean():.4f} (+/- {summary_df['loss'].std()*2:.4f})")
print(f"🎯 Mean ROC AUC: {summary_df['roc_auc'].mean():.4f} (+/- {summary_df['roc_auc'].std()*2:.4f})")

# Plot accuracy across folds
plt.figure(figsize=(10, 6))
plt.plot(summary_df['fold'], summary_df['accuracy'], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=summary_df['accuracy'].mean(), color='r', linestyle='--',
            label=f'Mean: {summary_df["accuracy"].mean():.4f}')
plt.fill_between(summary_df['fold'],
                 summary_df['accuracy'].mean() - summary_df['accuracy'].std(),
                 summary_df['accuracy'].mean() + summary_df['accuracy'].std(),
                 alpha=0.2, color='gray', label='±1 Std Dev')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('6-Fold Cross-Validation Accuracy - CNN Model', fontsize=14)
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
    ax.set_title(f'Fold {i+1}\nAcc: {summary_df.iloc[i]["accuracy"]:.4f}', fontsize=10)
    ax.set_ylabel('True', fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8)

plt.suptitle('All 6 Folds - Confusion Matrices (CNN Model)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Calculate and plot average confusion matrix
avg_cm = np.mean(fold_confusion_matrices, axis=0).round().astype(int)
plt.figure(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title(f'Average Confusion Matrix (6 Folds - CNN)\nMean Accuracy: {summary_df["accuracy"].mean():.4f}',
          fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Plot training histories - Accuracy
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (history, ax) in enumerate(zip(fold_histories, axes)):
    ax.plot(history.history['accuracy'], label='Train')
    ax.plot(history.history['val_accuracy'], label='Validation')
    ax.set_title(f'Fold {i+1} Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training and Validation Accuracy Across 6 Folds - CNN Model', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Plot training histories - Loss
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (history, ax) in enumerate(zip(fold_histories, axes)):
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_title(f'Fold {i+1} Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training and Validation Loss Across 6 Folds - CNN Model', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ROC Curves for all folds - Using stored predictions
plt.figure(figsize=(10, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, 6))

for i, preds in enumerate(fold_predictions):
    # Get test indices for this fold
    _, test_idx = list(stratified_kfold.split(X_cv, y_cv))[i]
    y_test_fold = y_cv[test_idx]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_fold, preds[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Fold {i+1} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for All 6 Folds - CNN Model', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Box plot of metrics
metrics_df = summary_df[['accuracy', 'roc_auc']]
plt.figure(figsize=(10, 6))
metrics_df.boxplot()
plt.title('Distribution of Metrics Across 6 Folds - CNN Model', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FINAL CONCLUSION - CNN MODEL")
print("="*60)
print(f"✅ CNN Model Performance across 6 folds:")
print(f"  - Average Accuracy: {summary_df['accuracy'].mean():.4f} (±{summary_df['accuracy'].std()*2:.4f})")
print(f"  - Average Loss: {summary_df['loss'].mean():.4f} (±{summary_df['loss'].std()*2:.4f})")
print(f"  - Average ROC AUC: {summary_df['roc_auc'].mean():.4f} (±{summary_df['roc_auc'].std()*2:.4f})")
print(f"  - Accuracy Range: [{summary_df['accuracy'].min():.4f}, {summary_df['accuracy'].max():.4f}]")
print(f"  - Standard Deviation: {summary_df['accuracy'].std():.4f}")

# Check if model is stable
if summary_df['accuracy'].std() < 0.01:
    print("\n✅ Model is VERY STABLE (low variance across folds)")
elif summary_df['accuracy'].std() < 0.02:
    print("\n✅ Model is STABLE (acceptable variance)")
else:
    print("\n⚠️ Model shows HIGH VARIANCE across folds - consider regularization")

# Compare with single run
print("\n" + "="*60)
print("COMPARISON WITH SINGLE RUN")
print("="*60)
print(f"Original single run accuracy: [Insert your original accuracy here]")
print(f"Cross-validated mean accuracy: {summary_df['accuracy'].mean():.4f}")

# Summary statistics table
print("\n" + "="*60)
print("DETAILED STATISTICS BY FOLD")
print("="*60)
stats_df = summary_df.copy()
stats_df['accuracy_std'] = stats_df['accuracy'].std()
stats_df['accuracy_cv'] = stats_df['accuracy'].std() / stats_df['accuracy'].mean() * 100  # Coefficient of variation
print(stats_df[['fold', 'accuracy', 'loss', 'roc_auc']].round(4).to_string(index=False))

# Save results to CSV (optional)
# summary_df.to_csv('cnn_6fold_cv_results.csv', index=False)
# print("\n✅ Results saved to 'cnn_6fold_cv_results.csv'")