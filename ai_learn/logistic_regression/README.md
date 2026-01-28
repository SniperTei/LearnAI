# 逻辑回归入门 - 分类问题实战

> 边学机器学习算法边学数学的实战教程

## 📚 项目简介

这是一个适合初学者的逻辑回归入门项目，通过学生成绩分类的实际案例，帮助理解分类问题的核心概念。不需要深厚的数学基础，边做边学！

### 学习目标

- ✅ 理解逻辑回归的基本原理
- ✅ 掌握 Sigmoid 和 Softmax 函数
- ✅ 学会处理二分类和多分类问题
- ✅ 理解交叉熵损失函数
- ✅ 完成第一个分类机器学习项目

---

## 🎯 你将学到的核心概念

| 概念 | 符号 | 应用 | 难度 |
|------|------|------|------|
| **Sigmoid函数** | σ(z) | 二分类概率映射 | ⭐⭐ |
| **Softmax函数** | σ(z)_i | 多分类概率分布 | ⭐⭐⭐ |
| **交叉熵损失** | H(p,q) | 衡量分类误差 | ⭐⭐⭐ |
| **决策边界** | Decision Boundary | 分类界限 | ⭐⭐ |
| **One-vs-Rest** | OvR | 多分类策略 | ⭐⭐ |

---

## 📁 文件说明

```
logistic_regression/
├── logistic_regression_demo.py          # 学生成绩分类（教学示例）
├── titanic_classification.py            # 泰坦尼克号生存预测（真实数据集）⭐
├── titanic_survival_data.csv            # 泰坦尼克号数据集（887条记录）📊
├── binary_classification.png             # 二分类可视化图
├── multiclass_decision_boundary.png      # 多分类决策边界图
├── titanic_eda.png                       # 泰坦尼克号数据探索分析图
├── titanic_roc_pr.png                    # 泰坦尼克号ROC和PR曲线
└── README.md                             # 本文档
```

---

## 🚀 快速开始

### 方案1：学生成绩分类（教学示例）

```bash
python logistic_regression_demo.py
```

### 方案2：泰坦尼克号生存预测（真实数据集）⭐ 推荐

```bash
python titanic_classification.py
```

这个脚本会：
- 首先检查本地是否有 `titanic_survival_data.csv`
- 如果没有，自动从网络下载并保存到本地
- 进行完整的数据探索和分析
- 训练逻辑回归模型预测乘客生存
- 生成专业的评估图表（ROC曲线、混淆矩阵等）

**注意**: 第一次运行时会自动下载并保存数据集到本地，后续运行会直接使用本地数据。

---

## 🚀 快速开始

### 1. 运行主程序

```bash
python logistic_regression_demo.py
```

### 2. 查看结果

程序会输出：
- 二分类模型训练过程
- 多分类模型训练过程
- 概率预测和决策边界
- 并生成2张可视化图表

---

## 🎯 泰坦尼克号生存预测项目（真实数据集）

### 项目简介

使用真实的泰坦尼克号乘客数据，通过逻辑回归预测乘客是否幸存。这是一个经典的机器学习二分类问题。

### 数据集信息

- **来源**：Stanford CS109 课程
- **样本数量**：887名乘客
- **特征数量**：8个字段
- **预测目标**：Survived (0=未存活, 1=存活)
- **本地文件**: `titanic_survival_data.csv`

### CSV数据字典

| 字段名 | 说明 | 数据类型 | 示例值 |
|--------|------|----------|--------|
| **Survived** | 是否存活（标签） | 整数 (0/1) | 0=未存活, 1=存活 |
| **Pclass** | 客舱等级 | 整数 (1-3) | 1=头等舱, 2=二等舱, 3=三等舱 |
| **Name** | 乘客姓名 | 字符串 | "Mr. John Doe" |
| **Sex** | 性别 | 字符串 | "male" 或 "female" |
| **Age** | 年龄 | 浮点数 | 22.0, 38.5 等 |
| **Siblings/Spouses Aboard** | 同行兄弟姐妹/配偶数量 | 整数 | 0, 1, 2, ... |
| **Parents/Children Aboard** | 同行父母/子女数量 | 整数 | 0, 1, 2, ... |
| **Fare** | 票价（英镑） | 浮点数 | 7.25, 71.28 等 |

### 使用的特征

| 特征 | 说明 | 类型 | 对生存的影响 |
|------|------|------|------------|
| **Pclass** | 客舱等级 (1/2/3等舱) | 数值 | 高等级舱位生存率更高 |
| **Sex** | 性别 | 类别（编码后） | 女性生存率远高于男性 |
| **Age** | 年龄 | 数值 | 儿童和部分老年人优先 |
| **Siblings/Spouses Aboard** | 同行配偶/兄弟姐妹数 | 数值 | 家庭规模影响 |
| **Parents/Children Aboard** | 同行父母/子女数 | 数值 | 家庭规模影响 |
| **Fare** | 票价 | 数值 | 高票价通常对应高等级舱位 |

### 关键发现

通过数据分析发现：
- ✅ **女性生存率** (~72%) >> **男性生存率** (~19%)
- ✅ **1等舱生存率** (~62%) >> **3等舱生存率** (~26%)
- ✅ **儿童** 有更高的生存概率
- ✅ **票价** 和生存率正相关

### 模型性能

训练的逻辑回归模型典型指标：
- **准确率**: ~78-80%
- **精确率**: ~75-78%
- **召回率**: ~68-72%
- **ROC-AUC**: ~0.82-0.85

### 代码亮点

```python
# 1. 数据加载和预处理
df = pd.read_csv("https://.../titanic.csv")
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())

# 2. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 模型训练
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. 查看特征重要性
coefficients = pd.DataFrame({
    '特征': feature_names,
    '系数': model.coef_[0]
})
```

### 可视化输出

脚本会生成两张专业的图表：

1. **titanic_eda.png** - 数据探索分析
   - 生存人数分布
   - 按性别的生存率
   - 按客舱等级的生存率
   - 年龄分布对比

2. **titanic_roc_pr.png** - 模型性能评估
   - ROC曲线和AUC值
   - 精确率-召回率曲线
   - 最佳阈值标记

### 预测示例

```
乘客 1:
  客舱等级: 1等舱
  性别: 女
  年龄: 25岁
  预测结果: 🟢 存活
  存活概率: 96.04%

乘客 2:
  客舱等级: 3等舱
  性别: 男
  年龄: 30岁
  预测结果: 🔴 未存活
  存活概率: 6.47%
```

### 下一步改进

1. **特征工程**
   - 创建"家庭规模"特征
   - 提取姓名中的称谓（Mr., Mrs., Miss.）
   - 年龄分段（儿童、成人、老人）

2. **模型优化**
   - 调整正则化参数C
   - 使用交叉验证
   - 尝试其他算法（随机森林、XGBoost）

3. **深入分析**
   - 分析预测错误的案例
   - 特征交互作用分析

---

## 📖 核心知识点

### 什么是逻辑回归？

逻辑回归是用于**分类问题**的算法，尽管名字里有"回归"，但它实际上输出的是**概率值**。

**与线性回归的区别：**

| 特性 | 线性回归 | 逻辑回归 |
|------|---------|---------|
| 输出 | 连续值（房价、温度） | 概率值（0-1之间） |
| 用途 | 回归问题 | 分类问题 |
| 激活函数 | 无 | Sigmoid / Softmax |
| 损失函数 | 均方误差（MSE） | 交叉熵 |

---

## 🎓 详细教程

### 第一部分：二分类逻辑回归

**问题：** 根据学习时长预测学生是否及格

**数学原理：**

#### 1. Sigmoid 函数

```
σ(z) = 1 / (1 + e^(-z))
```

**作用：**
- 将任意实数映射到 [0, 1] 区间
- 输出解释为属于正类的概率
- 例如：P(及格 | 学习时长=50小时) = 0.75

**代码实现：**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 示例
z = 0
print(sigmoid(z))  # 输出: 0.5
```

#### 2. 决策边界

通常设置阈值 0.5：
- P(y=1) ≥ 0.5 → 预测为类别 1（及格）
- P(y=1) < 0.5 → 预测为类别 0（不及格）

**为什么是 0.5？**
- σ(0) = 0.5
- 当 z = wx + b = 0 时，概率恰好为 0.5
- 这是"不确定"的中性点

#### 3. 二分类实例

```python
from sklearn.linear_model import LogisticRegression

# 准备数据
X = [[20], [30], [40], [50], [60], [70], [80]]  # 学习时长
y = [0, 0, 0, 0, 1, 1, 1]                      # 是否及格

# 训练模型
model = LogisticRegression()
model.fit(X, y)

# 预测
student_hours = [[55]]
prediction = model.predict(student_hours)      # 输出: [1]
probability = model.predict_proba(student_hours) # 输出: [[0.3, 0.7]]

print(f"预测: {'及格' if prediction[0]==1 else '不及格'}")
print(f"及格概率: {probability[0][1]:.2%}")
```

---

### 第二部分：多分类逻辑回归

**问题：** 根据学习时长和作业完成率预测成绩等级（优秀/良好/及格/不及格）

**两种策略：**

#### 策略1：One-vs-Rest (OvR)

训练多个二分类器：
```
分类器1: 优秀 vs [良好, 及格, 不及格]
分类器2: 良好 vs [优秀, 及格, 不及格]
分类器3: 及格 vs [优秀, 良好, 不及格]
分类器4: 不及格 vs [优秀, 良好, 及格]
```

选择概率最高的类别。

#### 策略2：Softmax 回归（推荐）

直接扩展到多分类，使用 Softmax 函数。

**Softmax 函数：**

```
σ(z)_i = e^z_i / Σ(e^z_j)
```

**特点：**
- 输出所有类别的概率分布
- 所有概率之和为 1
- 选择概率最大的类别

**示例：**
```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 数值稳定性
    return exp_z / exp_z.sum()

# 示例：4个类别的logits
z = np.array([2.0, 1.0, 0.1, 3.0])
probabilities = softmax(z)

print(probabilities)
# 输出: [0.21, 0.08, 0.03, 0.68]
# 解释: 最高的类别有68%的概率
```

#### 多分类实例

```python
from sklearn.linear_model import LogisticRegression

# 准备数据
X = [[20, 0.3], [40, 0.6], [60, 0.8], [80, 0.95]]  # [学习时长, 作业完成率]
y = [0, 1, 2, 3]                                    # [不及格, 及格, 良好, 优秀]

# 训练 Softmax 模型
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)

# 预测
new_student = [[55, 0.75]]
prediction = model.predict(new_student)
probabilities = model.predict_proba(new_student)

class_names = ['不及格', '及格', '良好', '优秀']
print(f"预测等级: {class_names[prediction[0]]}")
print("各等级概率:")
for name, prob in zip(class_names, probabilities[0]):
    print(f"  {name}: {prob:.2%}")
```

---

## 🔢 数学公式详解

### 交叉熵损失函数

**为什么不用 MSE？**
- MSE 适合回归问题
- 分类问题使用交叉熵更有效（梯度更大，收敛更快）

**二分类交叉熵：**

```
L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

- `y`：真实标签（0或1）
- `ŷ`：预测概率
- `log`：自然对数

**直观理解：**
- 如果 y=1，损失 = -log(ŷ)，ŷ越接近1，损失越小
- 如果 y=0，损失 = -log(1-ŷ)，ŷ越接近0，损失越小

**多分类交叉熵：**

```
L = -Σ(y_i × log(ŷ_i))
```

对所有类别求和。

---

## 💡 实用技巧

### 1. 查看概率而不仅仅是类别

```python
# 不要只这样做：
prediction = model.predict(X)

# 还要这样做：
probability = model.predict_proba(X)

# 示例输出：
# [[0.1, 0.9]]  # 10%不及格，90%及格
```

**重要性：**
- 了解模型的置信度
- 低置信度的预测可能需要人工审核
- 可以调整决策阈值

### 2. 数据标准化（重要！）

逻辑回归对特征尺度敏感，必须标准化：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model.fit(X_scaled, y)
```

**为什么需要？**
- 不同特征的量纲可能差异很大（如：时长0-100小时，完成率0-1）
- 不标准化会导致某些特征主导模型
- 标准化后收敛更快，性能更好

### 3. 调整决策阈值

默认阈值是 0.5，但可以根据需求调整：

```python
# 获取概率
y_prob = model.predict_proba(X)[:, 1]

# 使用自定义阈值（如0.3）
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

# 应用场景：
# - 医疗诊断：降低阈值（宁可误报，不可漏报）
# - 垃圾邮件：提高阈值（宁可漏报，不可误删正常邮件）
```

### 4. 处理类别不平衡

当某些类别样本很少时：

```python
# 方法1：使用类别权重
model = LogisticRegression(class_weight='balanced')

# 方法2：使用过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 方法3：调整类别权重
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

### 5. 正则化防止过拟合

```python
# L2正则化（默认）
model = LogisticRegression(C=1.0)  # C越小，正则化越强

# L1正则化（特征选择）
model = LogisticRegression(penalty='l1', solver='liblinear')

# 弹性网
model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
```

---

## 📊 模型评估

### 分类评估指标

| 指标 | 说明 | 代码 |
|------|------|------|
| **准确率** | 预测正确的比例 | `accuracy_score(y_test, y_pred)` |
| **精确率** | 预测为正的样本中真正为正的比例 | `precision_score(y_test, y_pred)` |
| **召回率** | 真正为正的样本中被预测为正的比例 | `recall_score(y_test, y_pred)` |
| **F1分数** | 精确率和召回率的调和平均 | `f1_score(y_test, y_pred)` |
| **混淆矩阵** | 详细展示各类别的预测情况 | `confusion_matrix(y_test, y_pred)` |

### 代码示例

```python
from sklearn.metrics import classification_report, confusion_matrix

# 详细的分类报告
print(classification_report(y_test, y_pred,
                          target_names=['不及格', '及格', '良好', '优秀']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)
```

---

## 🎯 扩展练习

### 练习1：修改数据

尝试修改 `logistic_regression_demo.py` 中的数据：

```python
# 添加更多噪声或边界案例
X = np.array([[30], [40], [50], [60], [70]])
y = np.array([0, 0, 1, 1, 1])
```

看看决策边界如何变化？

### 练习2：鸢尾花数据集

使用经典的多分类数据集：

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# 训练模型
model = LogisticRegression(multi_class='multinomial')
model.fit(X, y)
```

### 练习3：泰坦尼克号数据集

预测乘客是否生还：

```python
import pandas as pd

# 加载数据
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)

# 选择特征
features = ['Pclass', 'Age', 'Sex']
X = data[features]
y = data['Survived']

# 预处理...
```

### 练习4：可视化决策边界

尝试修改代码，可视化不同特征组合的决策边界：

```python
# 只选择两个特征进行可视化
X_2d = X[:, :2]  # 只用前两个特征
# 重新训练并可视化
```

### 练习5：比较 OvR 和 Softmax

```python
# OvR 策略
model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X_train, y_train)

# Softmax 策略
model_softmax = LogisticRegression(multi_class='multinomial')
model_softmax.fit(X_train, y_train)

# 比较性能
print(f"OvR 准确率: {model_ovr.score(X_test, y_test):.2%}")
print(f"Softmax 准确率: {model_softmax.score(X_test, y_test):.2%}")
```

---

## 🔗 相关资源

### 推荐阅读

- **[scikit-learn官方文档 - LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**
- **[机器学习实战](https://www.manning.com/books/machine-learning-in-action)**
- **[吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)**

### 数学基础

- **[3Blue1Brown - 神经网络](https://www.bilibili.com/video/BV1bx411M7Xg)**
- **[StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)**（强烈推荐！）
- **[Khan Academy - 概率统计](https://www.khanacademy.org/math/statistics-probability)**

### 实战项目

- Kaggle竞赛：[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- 鸢尾花数据集（sklearn内置）
- 手写数字识别（MNIST）

---

## 🤔 常见问题

### Q1: 为什么叫"逻辑"回归？

**A:** 名字来源于"Logit函数"（对数几率函数），是 Sigmoid 的反函数。实际上它用于分类，不是回归。

### Q2: 什么时候用逻辑回归？

**A:**
- 二分类问题（是/否）
- 多分类问题（类别A/B/C）
- 需要概率输出
- 特征和目标之间存在线性关系

### Q3: 逻辑回归 vs 决策树？

**A:**
| 特性 | 逻辑回归 | 决策树 |
|------|---------|--------|
| 可解释性 | 高（系数表示特征重要性） | 高（规则清晰） |
| 特征关系 | 线性 | 非线性 |
| 异常值 | 敏感 | 不敏感 |
| 输出 | 概率 | 类别 |

### Q4: 如何选择 C 参数？

**A:**
- `C` 是正则化强度的倒数（不是学习率！）
- `C` 越大，正则化越弱（可能过拟合）
- `C` 越小，正则化越强（可能欠拟合）

```python
# 使用交叉验证选择最优C
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"最佳C: {grid_search.best_params_['C']}")
```

### Q5: 模型不收敛怎么办？

**A:**
- 增加最大迭代次数：`max_iter=1000`
- 缩放特征：`StandardScaler()`
- 使用不同的求解器：`solver='saga'`
- 增大正则化参数：减小 `C`

---

## 📈 下一步学习

完成本项目后，你可以继续学习：

### 算法路线
1. ✅ 线性回归
2. ✅ 逻辑回归 ← **当前**
3. ⬜ 决策树
4. ⬜ 随机森林
5. ⬜ 支持向量机（SVM）
6. ⬜ 神经网络

### 技能提升
- ✅ 分类问题基础
- ⬜ 特征工程
- ⬜ 交叉验证
- ⬜ 超参数调优
- ⬜ 模型集成

---

## 📝 总结

### 你已经学会：
- ✅ 逻辑回归的基本原理
- ✅ Sigmoid 和 Softmax 函数
- ✅ 二分类和多分类
- ✅ 交叉熵损失函数
- ✅ 使用 scikit-learn 训练模型

### 记住：
- 🎯 **逻辑回归输出概率**，不只是类别
- 🎯 **数据标准化很重要**，影响模型性能
- 🎯 **理解比推导重要** - 先掌握概念和直觉
- 🎯 **实践比理论重要** - 多写代码，多动手

### 核心公式速查

```
# Sigmoid
σ(z) = 1 / (1 + e^(-z))

# Softmax
σ(z)_i = e^z_i / Σ(e^z_j)

# 二分类交叉熵
L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

# 多分类交叉熵
L = -Σ(y_i × log(ŷ_i))
```

---

## 📧 反馈与交流

如有问题或建议，欢迎：
- 提Issue
- 发起Pull Request
- 分享你的学习心得

---

## 📄 许可证

MIT License - 自由使用和分享

---

**Happy Learning! 🎉**

> "分类问题不只是在A和B之间选择，而是理解每个选择的确定性。"
