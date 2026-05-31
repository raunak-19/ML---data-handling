# ML---data-handling

# Handling Missing Values in Titanic Dataset

## Objective

Learn how to identify, analyze, and handle missing values in a dataset using Pandas.

Dataset used: Titanic Training Dataset

---

## Step 1: Load Dataset

```python
import pandas as pd

train = pd.read_csv("titanic_train.csv")
```

---

## Step 2: Identify Missing Values

Check the number of missing values in each column.

```python
train.isna().sum()
```

Output:

| Column | Missing Values |
|----------|----------|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

---

## Step 3: Analyze Missing Data

### Age

- Numerical feature
- 177 missing values
- Outliers may exist
- Filled using Median

```python
train['Age'] = train['Age'].fillna(
    train['Age'].median()
)
```

Reason:

Median is less affected by outliers than Mean.

---

### Embarked

- Categorical feature
- Only 2 missing values
- Filled using Mode

```python
train['Embarked'] = train['Embarked'].fillna(
    train['Embarked'].mode()[0]
)
```

Reason:

Mode represents the most frequent category.

---

### Cabin

- 687 missing values out of 891 rows
- More than 75% data missing

Dropped the column.

```python
train.drop(columns=['Cabin'], inplace=True)
```

Reason:

Too much information is missing to reliably impute values.

---

## Step 4: Verify Results

```python
train.isna().sum()
```

Output:

```text
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
```

No missing values remain.

---

## Key Learnings

### Detect Missing Values

```python
train.isna()
```

### Count Missing Values

```python
train.isna().sum()
```

### View Rows Containing Missing Values

```python
train[train.isna().any(axis=1)]
```

### View Missing Values in a Specific Column

```python
train[train['Age'].isna()]
```

---

## General Workflow for Handling Missing Values

1. Load dataset
2. Check missing values using `isna().sum()`
3. Calculate percentage of missing values
4. Decide strategy:
   - Mean
   - Median
   - Mode
   - Drop rows
   - Drop columns
5. Apply transformation
6. Verify results

---

## Conclusion

Different columns require different missing-value handling strategies.

- Numerical columns → Median or Mean
- Categorical columns → Mode
- Highly incomplete columns → Drop

The Titanic dataset was successfully cleaned by:
- Filling Age with Median
- Filling Embarked with Mode
- Dropping Cabin

Result: Dataset contains no missing values and is ready for further preprocessing and machine learning tasks.
