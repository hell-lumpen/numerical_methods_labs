import numpy as np
import nm_math


class Labs:
    _n_g = 10.
    _n_c = 4.

    _matrix_a_lab1 = np.array([[_n_c, 5, 2],
                               [5, _n_c, -_n_g],
                               [2, -_n_g, _n_c]], dtype=float)

    _matrix_b_lab1 = np.array([[1],
                               [_n_c],
                               [-_n_g]], dtype=float)

    _a_coef = []
    _b_coef = []
    _c_coef = []
    _d_coef = []

    matrix_b_lab3 = np.array([[_n_c + 10, _n_g, 1],
                               [_n_g, _n_c + 10, 3],
                               [1, 3, _n_g + 4]], dtype=float)

    matrix_c_lab3 = np.array([[_n_g],
                               [_n_c + 10],
                               [0]], dtype=float)

    _discrete_function_dop1 = {-0.2: -0.20136, 0: 0, 0.2: 0.20136, 0.4: 0.41152, 0.6: 0.64350}

    def run_lab1(self):
        print("\nЗадача 1. Метод Гаусса\n")
        print("Матрица А:\n" + str(self._matrix_a_lab1))
        print("Матрица B:\n" + str(self._matrix_b_lab1))
        print("Определитель A = " + str(nm_math.det_gauss(self._matrix_a_lab1)))
        print("A^-1:\n" + str(nm_math.inverse_matrix(self._matrix_a_lab1)))
        print("Решение СЛАУ AX = B\nX = " + str(nm_math.solve_gauss(self._matrix_a_lab1, self._matrix_b_lab1)))

    def _generate_matrix_2lab(self):
        i = 1
        while i <= 10:
            self._a_coef.append(i * self._n_c + self._n_g)
            self._b_coef.append(self._n_c * i * i + self._n_g)
            self._c_coef.append(self._n_g - self._n_c * i)
            self._d_coef.append(self._n_c + self._n_g * i)
            i += 1

    def _print_tridiagonal_matrix(self) -> np.array:
        print(np.diag(self._a_coef[1:], -1) + np.diag(self._b_coef, 0) + np.diag(self._c_coef[:-1], 1))

    def run_lab2(self):
        self._generate_matrix_2lab()
        print("\nЗадача 2. Метод прогонки\n")
        print("Матрица A:")
        self._print_tridiagonal_matrix()
        print("Матрица В:\n" + str(np.expand_dims(self._d_coef, -1)))
        print("Решение СЛАУ AX = B\nX = " + str(nm_math.tridiagonal_matrix_algorithm(self._a_coef, self._b_coef, self._c_coef, self._d_coef)))

    def run_lab3(self):
        print("\nЗадача 3\n")
        print("Матрица А:\n" + str(self.matrix_b_lab3))
        print("Матрица В:\n" + str(self.matrix_c_lab3))
        print("Решение СЛАУ AX=B методом простых итераций")
        print(nm_math.simple_iteration_method(self.matrix_b_lab3, self.matrix_c_lab3))
        print("Решение СЛАУ AX=B методом Зейделя")
        print(nm_math.zeidel_method(self.matrix_b_lab3, self.matrix_c_lab3))

    def run_numeric_diff(self):
        print("\nЧисленное дифференцирование\n")
        print("Функция" + str(self._discrete_function_dop1))
        ld, rd, cd = nm_math.numerical_first_diff(self._discrete_function_dop1, x=0.2, h=0.2)
        sd = nm_math.numerical_second_diff(self._discrete_function_dop1, x=0.2, h=0.2)
        print("f'(x)_left = " + str(ld) + "\nf'(x)_right = " + str(rd) + "\nf'(x)_central = " + str(cd))
        print("f\"(x) = " + str(sd))