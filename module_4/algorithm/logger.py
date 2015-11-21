__author__ = 'krekle'

import numpy


class Log:
    logtext = []

    def add_log(self, x, y):
        y_actual = None
        if y == 'left':
            y_actual = 0
        elif y == 'up':
            y_actual = 1
        elif y == 'right':
            y_actual = 2
        elif y == 'down':
            y_actual = 3

        for y in range(len(x)):
            for i in range(len(x[y])):
                if x[y][i] == ' ':
                    x[y][i] = 0

        self.logtext.append([x, y_actual])

    def write_log(self):
        print 'Saving Log ...'
        print self.logtext
        numpy.savetxt('log-gamelogic.txt', self.logtext, delimiter=" ", fmt="%s")
        print 'Saving Log [ok]'
