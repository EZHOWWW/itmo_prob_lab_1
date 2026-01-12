# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm


def box_muller_transform(n_samples):
    """
    Генерирует n_samples пар (X, Y) стандартных нормальных величин
    из равномерных U, V методом Бокса-Мюллера.
    """
    # 1. Генерируем U и V из равномерного распределения [0, 1]
    u = np.random.uniform(0, 1, n_samples)
    v = np.random.uniform(0, 1, n_samples)

    # 2. Применяем формулы преобразования
    # R - это "радиус" в полярных координатах, распределенный по Рэлею
    radius = np.sqrt(-2 * np.log(v))
    angle = 2 * np.pi * u

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    return x, y


# %%
# --- Основной блок ---
if __name__ == "__main__":
    N = 100_000
    print(f"Генерация {N} точек методом Бокса-Мюллера...")

    x_gen, y_gen = box_muller_transform(N)

    # --- Визуализация ---
    fig = plt.figure(figsize=(15, 5))

    # 1. 2D Гистограмма (Joint distribution)
    ax1 = fig.add_subplot(131)
    ax1.hexbin(x_gen, y_gen, gridsize=50, cmap="inferno", mincnt=1)
    ax1.set_title("Совместное распределение (X, Y)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")

    # 2. Гистограмма X
    ax2 = fig.add_subplot(132)
    count, bins, ignored = ax2.hist(
        x_gen,
        bins=60,
        density=True,
        alpha=0.6,
        color="b",
        label="Гистограмма X",
    )
    # Теоретическая кривая
    theory_x = np.linspace(-4, 4, 100)
    ax2.plot(theory_x, norm.pdf(theory_x), "r-", lw=2, label="N(0, 1) PDF")
    ax2.set_title("Распределение X")
    ax2.legend()

    # 3. Гистограмма Y
    ax3 = fig.add_subplot(133)
    count, bins, ignored = ax3.hist(
        y_gen,
        bins=60,
        density=True,
        alpha=0.6,
        color="g",
        label="Гистограмма Y",
    )
    # Теоретическая кривая
    ax3.plot(theory_x, norm.pdf(theory_x), "r-", lw=2, label="N(0, 1) PDF")
    ax3.set_title("Распределение Y")
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # --- Статистическая проверка (Kolmogorov-Smirnov test) ---
    print("\n--- Проверка гипотез о нормальности ---")

    # Тест Колмогорова-Смирнова сравнивает выборку с теоретическим распределением
    ks_stat_x, p_value_x = kstest(x_gen, "norm")
    ks_stat_y, p_value_y = kstest(y_gen, "norm")

    print(f"X: KS-statistic = {ks_stat_x:.5f}, p-value = {p_value_x:.5f}")
    print(f"Y: KS-statistic = {ks_stat_y:.5f}, p-value = {p_value_y:.5f}")

    # Проверка корреляции
    corr = np.corrcoef(x_gen, y_gen)[0, 1]
    print(f"Корреляция между X и Y: {corr:.5f} (должна быть ≈ 0)")

    if p_value_x > 0.05 and p_value_y > 0.05:
        print(">> Гипотеза о нормальности НЕ отвергается (p > 0.05). Успех!")
    else:
        # Для очень больших выборок p-value часто мал даже при ничтожных отклонениях,
        # поэтому смотрим на саму статистику KS.
        if ks_stat_x < 0.01:
            print(
                ">> Статистически отклонения ничтожны. Распределение практически идеально нормальное."
            )
