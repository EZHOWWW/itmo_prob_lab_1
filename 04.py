# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rv_continuous

# --- Общие параметры и целевая функция ---


# %%
# Наша целевая плотность p(x) = 3x^2 * exp(-x^3) для x >= 0
def target_pdf(x):
    if isinstance(x, (int, float)):
        return 3 * x**2 * np.exp(-(x**3)) if x >= 0 else 0
    # Векторизованная версия для массивов numpy
    x = np.asarray(x)
    return np.where(x >= 0, 3 * x**2 * np.exp(-(x**3)), 0)


# %%
# --- Метод 1: Наследование от rv_continuous в SciPy ---


class custom_distribution(rv_continuous):
    """
    Класс для нашего распределения, унаследованный от rv_continuous.
    Мы ОБЯЗАНЫ переопределить метод _pdf.
    Все остальное (cdf, rvs, ppf) SciPy вычислит сам, численно.
    """

    def _pdf(self, x):
        # Эта функция должна работать с массивами numpy
        return 3 * x**2 * np.exp(-(x**3))


# Создаем экземпляр нашего распределения
# Указываем, что оно определено на [0, inf)
custom_dist_scipy = custom_distribution(a=0, name="custom_dist")

# %%
# --- Метод 2: Метод обратного преобразования (Inverse Transform) ---


def generate_inverse_transform(size: int):
    """
    Генерирует 'size' случайных чисел с помощью аналитически найденной
    обратной функции распределения F^-1(u).
    """
    # 1. Генерируем 'size' чисел из равномерного распределения U(0,1)
    u = np.random.uniform(0, 1, size=size)

    # 2. Применяем F^-1(u) = (-ln(1-u))^(1/3)
    samples = (-np.log(1 - u)) ** (1 / 3)

    return samples


# %%
# --- Метод 3: Метод выборки с отклонением (Rejection Sampling) ---

# В качестве "предлагающей" (proposal) функции g(x) возьмем
# экспоненциальное распределение с lambda=1, т.е. g(x) = exp(-x)
# Нам нужно найти константу M = max(p(x)/g(x))
# M = max(3*x^2*exp(-x^3) / exp(-x)) = max(3*x^2*exp(x-x^3))

# Найдем максимум этой функции численно, чтобы определить M
f_to_maximize = lambda x: 3 * x**2 * np.exp(x - x**3)
# Минимизируем отрицательную функцию
res = minimize(lambda x: -f_to_maximize(x), x0=1.0, bounds=[(0, None)])
x_max = res.x[0]
M = f_to_maximize(x_max)
M *= 1.01  # Добавим небольшой запас на всякий случай


def generate_rejection_sampling(size: int):
    """
    Генерирует 'size' случайных чисел методом выборки с отклонением.
    Использует экспоненциальное распределение в качестве g(x).
    """
    accepted_samples = []
    total_generated = 0

    while len(accepted_samples) < size:
        # Генерируем кандидата y из g(x) = Exp(1)
        # scale = 1/lambda, поэтому scale=1
        candidate_y = np.random.exponential(scale=1.0)

        # Генерируем критерий u из U(0,1)
        criterion_u = np.random.uniform(0, 1)

        total_generated += 1

        # Проверяем условие принятия: u <= p(y) / (M * g(y))
        p_y = target_pdf(candidate_y)
        g_y = np.exp(-candidate_y)  # PDF для Exp(1)

        if criterion_u <= p_y / (M * g_y):
            accepted_samples.append(candidate_y)

    # Сохраняем эффективность для анализа
    generate_rejection_sampling.efficiency = size / total_generated

    return np.array(accepted_samples)


# %%
# --- Основной блок для экспериментов и визуализации ---
if __name__ == "__main__":
    # --- 1. Сравнение производительности ---
    n_values = [1000, 5000, 20000]
    print("=" * 60)
    print("Сравнение производительности методов генерации")
    print("=" * 60)
    print(
        f"{'Кол-во чисел (n)':<20} | {'SciPy .rvs()':<15} | {'Inverse Transform':<20} | {'Rejection Sampling'}"
    )
    print("-" * 80)

    for n in n_values:
        # SciPy
        start_time = time.time()
        custom_dist_scipy.rvs(size=n)
        time_scipy = time.time() - start_time

        # Inverse Transform
        start_time = time.time()
        generate_inverse_transform(size=n)
        time_inverse = time.time() - start_time

        # Rejection Sampling
        start_time = time.time()
        generate_rejection_sampling(size=n)
        time_rejection = time.time() - start_time

        print(
            f"{n:<20} | {time_scipy:<15.4f} | {time_inverse:<20.4f} | {time_rejection:.4f}"
        )

    print("\nВывод: Метод обратного преобразования самый быстрый.")
    print(
        "Стандартный .rvs() в SciPy очень медленный из-за численных расчетов."
    )

    # --- 2. Визуальная проверка корректности ---
    n_plot = 50000
    print(f"\nГенерация {n_plot} чисел для построения гистограмм...")

    samples_scipy = custom_dist_scipy.rvs(size=n_plot)
    samples_inverse = generate_inverse_transform(size=n_plot)
    samples_rejection = generate_rejection_sampling(size=n_plot)

    print(f"Константа M для Rejection Sampling ≈ {M:.3f}")
    print(
        f"Эмпирическая эффективность Rejection Sampling ≈ {generate_rejection_sampling.efficiency:.2%}"
    )

    # Построение графиков
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(
        "Сравнение гистограмм сгенерированных выборок с истинной PDF",
        fontsize=16,
    )
    x_axis = np.linspace(0, 3, 400)
    pdf_values = target_pdf(x_axis)

    # График для SciPy .rvs()
    axes[0].hist(
        samples_scipy, bins=50, density=True, label="Гистограмма .rvs()"
    )
    axes[0].plot(x_axis, pdf_values, "r-", lw=2, label="Истинная PDF")
    axes[0].set_title("Метод 1: SciPy .rvs()")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Плотность")
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    # График для Inverse Transform
    axes[1].hist(
        samples_inverse, bins=50, density=True, label="Гистограмма Inverse"
    )
    axes[1].plot(x_axis, pdf_values, "r-", lw=2, label="Истинная PDF")
    axes[1].set_title("Метод 2: Inverse Transform")
    axes[1].set_xlabel("x")
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)

    # График для Rejection Sampling
    axes[2].hist(
        samples_rejection, bins=50, density=True, label="Гистограмма Rejection"
    )
    axes[2].plot(x_axis, pdf_values, "r-", lw=2, label="Истинная PDF")
    axes[2].set_title("Метод 3: Rejection Sampling")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
