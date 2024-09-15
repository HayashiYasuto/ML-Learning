import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
#from IPython.display import Image

os.environ["PATH"] += os.pathsep + r"C:\Program Files\windows_10_cmake_Release_Graphviz-12.1.1-win32\Graphviz-12.1.1-win32\bin"

# 加载数据集
files = {
    'Finger': "手指.xlsx",
    'Ankle': "脚踝.xlsx",
    'Wrist': "手腕.xlsx",
    'Knee': "膝盖.xlsx"
}

dataframes = {name: pd.read_excel(path) for name, path in files.items()}

combined_data = []
labels = []

for label, (name, df) in enumerate(dataframes.items()):
    current_values = df['电流/A - 曲线 0'].values
    num_peaks = 50
    for i in range(0, len(current_values), num_peaks):
        peak_segment = current_values[i:i + num_peaks]
        if len(peak_segment) == num_peaks:
            combined_data.append(peak_segment)
            labels.append(label)

# 将数据和标签转换为数组
X = np.array(combined_data)
y = np.array(labels)

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 为LSTM准备数据 (samples, time_steps, features)
X_reshaped = X_scaled.reshape(-1, num_peaks, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, activation='relu', input_shape=(num_peaks, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(dataframes), activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#打印模型
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
#Image(filename='model_architecture.png')

# 训练模型
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 评估模型并输出准确率
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 预测并生成混淆矩阵和分类报告
y_pred = np.argmax(model.predict(X_test), axis=-1)

# 打印分类报告
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=list(dataframes.keys())))

# 混淆矩阵归一化
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Precision, Recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

cm_extended = np.zeros((cm.shape[0]+1, cm.shape[1]+2))


cm_extended[:-1, :-2] = cm_normalized * 100

cm_extended[-1, :-2] = recall * 100

cm_extended[:-1, -2] = precision * 100
cm_extended[:-1, -1] = f1 * 100

xticks = list(dataframes.keys()) + ["Precision", "F1-score"]
yticks = list(dataframes.keys()) + ["Recall"]

plt.figure(figsize=(10, 7))
ax = sns.heatmap(cm_extended, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=1, linecolor='black',
                 xticklabels=xticks, yticklabels=yticks)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 20, 40, 60, 80, 100])
cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
cbar.set_label('Percentage (%)', rotation=270, labelpad=20)

# 添加标题和标签
plt.title('Confusion Matrix ', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

plt.show()

# 绘制训练过程中的准确率和损失曲线
plt.figure(figsize=(12, 5))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
