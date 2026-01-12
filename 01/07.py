# %%
import numpy as np


def solve_joint_distribution():
    # --- Входные данные ---
    # X - строки (4 значения), Y - столбцы (5 значений)
    X_vals = np.array([1, 16, 24, 27])
    Y_vals = np.array([-4, -2, -1, 2, 7])

    # Таблица вероятностей (None для пропущенного значения)
    # Строки соответствуют X, Столбцы - Y
    probs_raw = [
        [0.06, 0.02, None, 0.06, 0.01],  # X=1
        [0.07, 0.09, 0.03, 0.02, 0.07],  # X=16
        [0.07, 0.07, 0.01, 0.08, 0.04],  # X=24
        [0.06, 0.10, 0.06, 0.04, 0.04],  # X=27
    ]

    # --- 1. Поиск пропущенного значения ---
    flat_probs = []
    missing_idx = -1
    current_sum = 0

    # Превращаем в numpy array, заменяя None на np.nan
    P = np.array(probs_raw, dtype=float)

    # Считаем сумму известных элементов (nan не учитываются при nansum)
    known_sum = np.nansum(P)
    missing_val = 1.0 - known_sum

    # Заполняем пропуск (округляем, чтобы убрать float погрешности, если они есть)
    missing_val = round(missing_val, 5)

    # Вставляем значение в матрицу
    mask = np.isnan(P)
    P[mask] = missing_val

    print(f"Сумма известных вероятностей: {known_sum}")
    print(f"Восстановленное значение (NULL): {missing_val}")

    # --- 2. Маргинальные распределения ---
    # P(X) - сумма по строкам (axis=1), так как столбцы это Y
    P_X = np.sum(P, axis=1)

    # P(Y) - сумма по столбцам (axis=0)
    P_Y = np.sum(P, axis=0)

    print("\nМаргинальное распределение X:")
    for x, p in zip(X_vals, P_X):
        print(f"P(X={x}) = {p:.4f}")

    print("\nМаргинальное распределение Y:")
    for y, p in zip(Y_vals, P_Y):
        print(f"P(Y={y}) = {p:.4f}")

    # --- 3. Статистики для X и Y ---
    def calculate_stats(values, probs, name):
        # Мат ожидание
        mean = np.sum(values * probs)

        # Дисперсия и СКО
        variance = np.sum(((values - mean) ** 2) * probs)
        std_dev = np.sqrt(variance)

        # Мода (значение с макс вероятностью)
        mode = values[np.argmax(probs)]

        # Медиана (значение, где накопленная вероятность переходит через 0.5)
        cumsum = np.cumsum(probs)
        median_idx = np.searchsorted(cumsum, 0.5)
        median = values[median_idx]

        # Асимметрия и Эксцесс (Fisher's definition: normal -> 0)
        # E[(x-mu)^3] / sigma^3
        moment3 = np.sum(((values - mean) ** 3) * probs)
        skewness = moment3 / (std_dev**3)

        # E[(x-mu)^4] / sigma^4 - 3
        moment4 = np.sum(((values - mean) ** 4) * probs)
        kurt = moment4 / (std_dev**4) - 3

        return {
            "name": name,
            "mean": mean,
            "var": variance,
            "std": std_dev,
            "mode": mode,
            "median": median,
            "skew": skewness,
            "kurt": kurt,
        }

    stats_x = calculate_stats(X_vals, P_X, "X")
    stats_y = calculate_stats(Y_vals, P_Y, "Y")

    # --- 4. Ковариация и Корреляция ---
    # E[XY]
    # Создаем сетку значений
    XX, YY = np.meshgrid(X_vals, Y_vals, indexing="ij")
    # XX[i, j] = X_vals[i], YY[i, j] = Y_vals[j]

    E_XY = np.sum(XX * YY * P)
    Cov = E_XY - stats_x["mean"] * stats_y["mean"]
    Corr = Cov / (stats_x["std"] * stats_y["std"])

    # --- 5. Проверка на независимость ---
    # Матрица P(X)*P(Y)
    P_independent = np.outer(P_X, P_Y)
    # Проверка на совпадение с заданной точностью
    is_independent = np.allclose(P, P_independent, atol=1e-5)
    max_diff = np.max(np.abs(P - P_independent))

    # --- 6. Функция U = sgn(X) + sgn(Y) ---
    # Вычисляем sgn
    sgn_X = np.sign(X_vals)  # [1, 1, 1, 1] так как все X > 0
    sgn_Y = np.sign(Y_vals)  # [-1, -1, -1, 1, 1]

    U_vals_grid = np.zeros_like(XX)
    for i in range(len(X_vals)):
        for j in range(len(Y_vals)):
            U_vals_grid[i, j] = sgn_X[i] + sgn_Y[j]

    # Собираем вероятности для каждого уникального значения U
    unique_U = np.unique(U_vals_grid)
    probs_U = []

    for u in unique_U:
        mask_u = U_vals_grid == u
        p_u = np.sum(P[mask_u])
        probs_U.append(p_u)

    print("\n--- Результаты ---")
    print(f"X stats: {stats_x}")
    print(f"Y stats: {stats_y}")
    print(f"Cov: {Cov:.4f}, Corr: {Corr:.4f}")
    print(f"Независимы? {is_independent} (Макс разница: {max_diff:.4f})")

    print("\nРаспределение U = sgn(X) + sgn(Y):")
    for u, p in zip(unique_U, probs_U):
        print(f"P(U={int(u)}) = {p:.4f}")

    return (
        stats_x,
        stats_y,
        Cov,
        Corr,
        is_independent,
        unique_U,
        probs_U,
        missing_val,
    )


if __name__ == "__main__":
    solve_joint_distribution()
