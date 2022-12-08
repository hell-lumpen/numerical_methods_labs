import math
import sys

import numpy as np


def _upper_triangular_matrix(matrix: np.ndarray) -> np.array:
    try:
        _matrix = np.copy(matrix)
        for nrow, row in enumerate(_matrix):
            divider = row[nrow]
            row /= divider
            for lower_row in _matrix[nrow + 1:]:
                factor = lower_row[nrow]
                lower_row -= factor * row
        return _matrix
    except ZeroDivisionError:
        print("Деление на 0")
        return None

def _identity_matrix(matrix: np.ndarray) -> np.array:
    _matrix = np.copy(matrix)
    for nrow in range(len(_matrix) - 1, 0, -1):
        row = _matrix[nrow]
        for upper_row in _matrix[:nrow]:
            factor = upper_row[nrow]
            upper_row -= factor * row
    return _matrix

def det_gauss(matrix: np.ndarray) -> float:
    """Нахождения определителя квадратной матрицы"""
    matrix = np.copy(matrix).astype(float)
    sign = False

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise TypeError('Матрица не квадратная')

    for i in range(matrix.shape[0] - 1):
        for j in range(matrix.shape[0] - 1, i, -1):
             if matrix[j - 1][i] != 0:
                 matrix[j] -= (matrix[j][i] / matrix[j - 1][i]) * matrix[j - 1]
             else:
                 buffer = np.copy(matrix[j - 1])
                 matrix[j - 1] = matrix[j]
                 matrix[j] = buffer
                 sign = 1 - sign

    det = np.prod(np.diagonal(matrix))  # произведение диагональных элементов
    return -det if sign else det

def solve_gauss(matrix_a: np.array, matrix_b: np.array) -> np.array:
    a = np.copy(matrix_a).astype(float)
    b = np.copy(matrix_b).astype(float)

    extended_matrix = np.concatenate((a, b), axis=1)
    triangular_extended_matrix = _upper_triangular_matrix(extended_matrix)
    identity_extended_matrix = _identity_matrix(triangular_extended_matrix)

    return identity_extended_matrix[:,3]

def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    _matrix = np.copy(matrix)
    n = _matrix.shape[0]
    m = np.hstack((_matrix, np.eye(n)))

    for nrow, row in enumerate(m):
        divider = row[nrow]
        row /= divider
        for lower_row in m[nrow + 1:]:
            factor = lower_row[nrow]
            lower_row -= factor * row

    for k in range(n - 1, 0, -1):
        for row_ in range(k - 1, -1, -1):
            if m[row_, k]:
                m[row_, :] -= m[k, :] * m[row_, k]

    return m[:, n:]


def tridiagonal_matrix_algorithm(a_coefficients: list[float],
                                 b_coefficients: list[float],
                                 c_coefficients: list[float],
                                 d_coefficients: list[float]) -> list[float]:
    """https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm"""
    n = len(a_coefficients)
    x_coefficients = [None] * n

    p = []
    q = []

    p.append(-c_coefficients[0] / b_coefficients[0])
    q.append(d_coefficients[0] / b_coefficients[0])

    for i in range(1, n, 1):
        p.append(-c_coefficients[i] / (b_coefficients[i] + a_coefficients[i] * p[i - 1]))
        q.append((d_coefficients[i] - a_coefficients[i] * q[i - 1]) / (b_coefficients[i] + a_coefficients[i] * p[i - 1]))

    x_coefficients[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x_coefficients[i] = q[i] + p[i] * x_coefficients[i + 1]

    return np.expand_dims(x_coefficients, -1)


def dot(matrix_a: np.array, matrix_b: np.array) -> np.array:
    rows_matrix_a = len(matrix_a)
    columns_matrix_a = len(matrix_a[0])
    rows_matrix_b = len(matrix_b)
    columns_matrix_b = len(matrix_b[0])

    if columns_matrix_a != rows_matrix_b:
        print("Некорректные размеры матриц. Умножение невозможно")
        return None

    matrix_c = np.zeros((rows_matrix_a, columns_matrix_b))

    for i in range(rows_matrix_a):
        for j in range(columns_matrix_b):
            for k in range(columns_matrix_a):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c

def add(matrix_a: np.array, matrix_b: np.array) -> np.array:
    rows_matrix_a = len(matrix_a)
    columns_matrix_a = len(matrix_a[0])
    rows_matrix_b = len(matrix_b)
    columns_matrix_b = len(matrix_b[0])

    if rows_matrix_a != rows_matrix_b or columns_matrix_a != columns_matrix_b:
        print("Некорректные размеры матриц. Сложение невозможно")
        return None

    matrix_c = np.zeros((rows_matrix_a, columns_matrix_a))
    for i in range(rows_matrix_a):
        for j in range(columns_matrix_a):
            matrix_c[i][j] += matrix_a[i][j] + matrix_b[i][j]

    return matrix_c

def _epsilon_comparison(vector: np.array, vector_prev: np.array, epsilon: float) -> bool:
    for i in range(len(vector)):
        if abs(vector[i] - vector_prev[i]) > epsilon:
            return True

    return False


def simple_iteration_method(matrix_a: np.array, matrix_b: np.array) -> np.array:
    """https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B8%D1%82%D0%B5%D1%80%D0%B0%D1%86%D0%B8%D0%B8"""
    n = len(matrix_a[0])
    alpha_matrix = np.zeros((n, n))
    beta_matrix = np.zeros((n, 1))

    for i in range(n):
        for j in range(n):
            if i != j:
                alpha_matrix[i][j] = -matrix_a[i][j] / matrix_a[i][i]

    for i in range(n):
        beta_matrix[i] = matrix_b[i] / matrix_a[i][i]

    converge = True
    x_matrix = np.copy(beta_matrix)
    iteration = 0
    while converge:
        iteration += 1
        x_matrix_prev = np.copy(x_matrix)
        x_matrix = add(beta_matrix, dot(alpha_matrix, x_matrix_prev))
        converge = _epsilon_comparison(x_matrix, x_matrix_prev, 1.e-4)

    print(f"Количество итераций: {iteration}\n")
    return x_matrix


def zeidel_method(matrix_a: np.array, matrix_b: np.array) -> np.array:
    n = len(matrix_a)
    x_matrix_prev = np.zeros((n, 1))

    converge = True
    iteration = 0
    while converge:
        iteration += 1
        x_matrix = np.copy(x_matrix_prev)
        for i in range(n):
            s1 = sum(matrix_a[i][j] * x_matrix[j] for j in range(i))
            s2 = sum(matrix_a[i][j] * x_matrix_prev[j] for j in range(i + 1, n))
            x_matrix[i] = (matrix_b[i] - s1 - s2) / matrix_a[i][i]

        converge = _epsilon_comparison(x_matrix, x_matrix_prev, 1e-4)
        x_matrix_prev = x_matrix

    print(f"Количество итераций: {iteration}\n")
    return x_matrix_prev

def _get_abs_max_supradiagonal_element(matrix: np.array) -> tuple[float, int, int]:
    max_element = float('-inf')
    k = 0
    m = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i < j:
                if abs(matrix[i][j]) > max_element:
                    max_element = matrix[i][j]
                    k = i
                    m = j

    return max_element, k, m

def _get_rotation_matrix(rotation_angle: float, k: int, m: int, size: int) -> np.array:
    rotation_matrix = np.eye(size, dtype=float)
    rotation_matrix[k][k] = math.cos(rotation_angle)
    rotation_matrix[k][m] = - math.sin(rotation_angle)
    rotation_matrix[m][k] = math.sin(rotation_angle)
    rotation_matrix[m][m] = math.cos(rotation_angle)
    return rotation_matrix

def jacobi_eigenvalue_algorithm(matrix: np.array) -> tuple[np.array, np.array]:
    """Возвращает столбец собственных чисел и матрицу, состояющую из столбцов - собственных векторов"""
    epsilon = 1.e-7
    i = 0

    A = np.copy(matrix)
    a, k, m = _get_abs_max_supradiagonal_element(A)
    converge = abs(a) > epsilon

    V = np.eye(len(matrix), dtype=float)
    while converge:
        if abs(A[k][k] - A[m][m]) < epsilon:
            if A[k][m] > 0:
                rotation_angle = math.pi / 4
            else:
                rotation_angle = -math.pi / 4
        else:
            rotation_angle = 1/2 * math.atan(2*a / (A[k][k] - A[m][m]))

        H = _get_rotation_matrix(rotation_angle, k, m, len(matrix))

        V = dot(V, H)
        A = dot(dot(H.T, A), H)

        a, k, m = _get_abs_max_supradiagonal_element(A)
        i += 1
        converge = abs(a) > epsilon

    return A.diagonal(offset=0), V


def numerical_first_diff(discrete_function: dict, x: float, h: float):
    """Нахождение левой, правой и центральной производных"""
    try:
        y_0 = discrete_function[x]
        yl_1 = discrete_function[x - h]
        yr_1 = discrete_function[x + h]

        left_first_diff = (y_0 - yl_1) / h
        right_first_diff = (yr_1 - y_0) / h
        central_first_diff = (yr_1 - yl_1) / 2 * h

        return left_first_diff, right_first_diff, central_first_diff
    except KeyError:
        print("numerical_first_diff: x должен совпадать с узлом и не быть крайними значениями функции")
        return None, None, None
    except ZeroDivisionError:
        print("numerical_first_diff: h не может быть равен нулю")
        return None, None, None


def numerical_second_diff(discrete_function: dict, x: float, h: float):
    """Нахождение второй производной дискретной функции"""
    try:
        y_0 = discrete_function[x]
        yl_1 = discrete_function[x - h]
        yr_1 = discrete_function[x + h]
        return (yl_1 - 2 * y_0 + yr_1) / h * h
    except KeyError:
        print("numerical_second_diff: x должен совпадать с узлом и не быть крайними значениями функции")
        return None
    except ZeroDivisionError:
        print("numerical_second_diff: h не может быть равен нулю")
        return None
