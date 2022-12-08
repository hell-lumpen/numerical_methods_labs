import nm_math
import tasks
import numpy

matrix = numpy.array([[17, 1, 1],
                      [1, 17, 2],
                      [1, 2, 4]], dtype=float)

if __name__ == '__main__':
    labs = tasks.Labs()
    labs.run_lab1()
    labs.run_lab2()
    labs.run_lab3()
    labs.run_lab4()
    labs.run_numeric_diff()
