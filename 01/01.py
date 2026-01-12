# %%
import math
import time

import numpy as np


# %%
def analytical_solution(n: int) -> float:
    """
    Вычисляет точную вероятность того, что ни один из n детей
    не получит свою пару ботинок, используя аналитическую формулу,
    выведенную через принцип включений-исключений.

    Args:
        n (int): Количество детей (и пар ботинок).

    Returns:
        float: Точная вероятность события.
    """
    if n < 0:
        raise ValueError("Количество детей не может быть отрицательным.")
    if n == 0:
        return (
            1.0  # Для 0 детей условие "никто не ушел в своей паре" выполняется.
        )

    total_prob = 0.0
    n_factorial_sq = math.factorial(n) ** 2

    # Итерация по k от 0 до n для формулы P = sum_{k=0 to n} (-1)^k * C(n,k) * ...
    for k in range(n + 1):
        # Вычисляем биномиальный коэффициент C(n, k)
        binom_coeff = math.comb(n, k)

        # Вычисляем ((n-k)!)^2 / (n!)^2
        term_numerator = math.factorial(n - k) ** 2

        term = binom_coeff * (term_numerator / n_factorial_sq)

        # Применяем знакочередование (-1)^k
        if k % 2 == 1:  # если k нечетное
            total_prob -= term
        else:  # если k четное
            total_prob += term

    return total_prob


def monte_carlo_simulation(n: int, num_trials: int = 1_000_000) -> float:
    """
    Оценивает вероятность того, что ни один из n детей не получит
    свою пару ботинок, с помощью симуляции методом Монте-Карло.

    Args:
        n (int): Количество детей.
        num_trials (int): Количество симуляций для проведения.

    Returns:
        float: Приближенная вероятность события.
    """
    if n < 0:
        raise ValueError("Количество детей не может быть отрицательным.")
    if n == 0:
        return 1.0

    no_correct_pairs_count = 0

    # Исходный порядок детей/ботинок, например [0, 1, 2, ..., n-1]
    children_ids = np.arange(n)

    for _ in range(num_trials):
        # Моделируем раздачу ботинок: создаем две случайные перестановки.
        # left_shoes_received[i] = ботинок, который получил i-й ребенок
        left_shoes_received = np.random.permutation(children_ids)
        right_shoes_received = np.random.permutation(children_ids)

        # Проверяем условие "i-й ребенок получил i-й левый И i-й правый ботинок"

        # got_left_correctly будет [True, False, ...], если 0-й ребенок получил свой левый ботинок, а 1-й - нет.
        got_left_correctly = children_ids == left_shoes_received
        got_right_correctly = children_ids == right_shoes_received

        # Логическое "И" векторов. got_pair_correctly[i] будет True,
        # только если и got_left_correctly[i], и got_right_correctly[i] равны True.
        got_pair_correctly = np.logical_and(
            got_left_correctly, got_right_correctly
        )

        # np.any() вернет True, если в массиве есть хотя бы один True.
        # Нам нужен случай, когда НИ ОДИН ребенок не получил свою пару.
        if not np.any(got_pair_correctly):
            no_correct_pairs_count += 1

    return no_correct_pairs_count / num_trials


# %%
if __name__ == "__main__":
    n_values = [2, 3, 5, 10]
    num_simulations = 2_000_000

    print("=" * 60)
    print("Решение задачи о ботинках аналитически и методом Монте-Карло")
    print("=" * 60)

    for n in n_values:
        print(f"\n--- Анализ для n = {n} детей ---\n")

        # --- Аналитическое решение ---
        analytical_prob = analytical_solution(n)
        print("Аналитическое решение:")
        print(f"  P(n={n}) = {analytical_prob:.8f}")

        # --- Решение методом Монте-Карло ---
        print(f"\nМоделирование Монте-Карло ({num_simulations:,} итераций):")
        start_time = time.time()
        mc_prob = monte_carlo_simulation(n, num_simulations)
        end_time = time.time()

        # --- Результаты и сравнение ---
        error = abs(analytical_prob - mc_prob)

        print(f"  Приближенная вероятность P ≈ {mc_prob:.8f}")
        print(f"  Абсолютная ошибка: {error:.8f}")
        print(f"  Время выполнения симуляции: {end_time - start_time:.4f} сек.")

    print("\n" + "=" * 60)

    limit_val_approx = analytical_solution(
        20
    )  # For large n, the value stabilizes
    print("\nИнтересный факт: при n -> ∞, вероятность быстро сходится.")
    print(f"P(n=20) ≈ {limit_val_approx:.8f}")
    print("Значение для n=10 уже очень близко к этому пределу.")
