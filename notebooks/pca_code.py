# ===== ШАГ 1: Создание эмбеддингов для title и description =====
# Вставьте эту ячейку ПОСЛЕ создания df_with_desc_features


# ===== ШАГ 5: Train/Test Split =====

from sklearn.model_selection import train_test_split

X = df_final_pca
y = df["shouldSplit"]

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.15, random_state=45, shuffle=True, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=0.5,
    random_state=52,
    shuffle=True,
    stratify=y_holdout,
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Готово к обучению!")
