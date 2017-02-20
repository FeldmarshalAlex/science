import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

filename = r'160119-3_nulevaya.tiff'

lower_trashhold = 60  # порог, ниже которого локальный экстремум считаем чёрной полосой
upper_trashhold = 200  # порог, выше которого локальный экстремум считаем белой полосой
original_folder = r'original'
result_folder = r'result'


def main():
    black_lines, white_lines = trace_interference_lines_on_image(filename)

    phases = calculate_phases(filename, black_lines, white_lines)

    save_tif_image(phases, filename + '_phases.tif', 10)


def save_tif_image(data, filename, scale=255):
    print('saving', result_folder + '\\' + filename)
    height = len(data)
    width = len(data[0])
    image_data = [(0, 0, 0)] * (height*width)
    for y in range(height):
        for x in range(width):
            val = data[y][x]
            if val > scale:
                color = (0, 255, 0)
            elif val < -scale:
                color = (0, 0, 255)
            elif val >= 0:
                color = (round(val/scale * 255),)*3
            else:
                color = (round(-val/scale *255), 0, 0)
            image_data[y * width + x] = color

    phase_image = Image.new('RGBA', (width, height))
    phase_image.putdata(image_data)
    phase_image.save(filename)
    phase_image.show()


def trace_interference_lines_on_image(image_filename):
    """
    Осуществляет трассировку интерференционных линий на картинке.
    :param image_filename: имя файла исходной картинки
    :return: Возвращает два списка интерференционных линий: чёрных и белых. Причём чёрных на одну больше.
    """
    input_image = Image.open(original_folder + '\\' + image_filename)
    image_data = list(input_image.getdata())
    print('Opened image', image_filename, 'with size', input_image.size, 'and mode', input_image.mode)
    if input_image.mode == 'RGB':
        brightness = [g for r, g, b in image_data]  # Зелёный в данной картинке всегда средний по яркости - берём именно его
    elif input_image.mode == 'RGBA':
        brightness = [g for r, g, b, a in image_data]  # Зелёный в данной картинке всегда средний по яркости - берём именно его
    else:
        raise ValueError("Unknown image color mode.")
    data_2d = [brightness[i*input_image.width:(i+1)*input_image.width] for i in range(input_image.height)]

    all_black_base_points = []
    all_white_base_points = []
    for x in range(input_image.width):
        data_slice = [data_2d[y][x] for y in range(len(data_2d))]
        black_base_points = get_base_points(data_slice, 7, lambda x: x < lower_trashhold)
        white_base_points = get_base_points(data_slice, 15, lambda x: x > upper_trashhold)  
        all_black_base_points.append(black_base_points)
        all_white_base_points.append(white_base_points)

    # формируем первые опорные точки (по правой границе картинки == [-1])
    black_base_points, white_base_points = all_black_base_points[-1], all_white_base_points[-1]
    
    # удаляем верхнюю и нижние белые опорные точки, вне области чёрных линий
    white_base_points = [y for y in white_base_points
                              if black_base_points[0] < y < black_base_points[-1]]

    # Изобразим границу, на которой будут взяты первые опорные точки для трассировки, вместе с этими точками
    plot_interferogram_slice([data_2d[y][-1] for y in range(len(data_2d))], black_base_points, white_base_points)

    # проверяем, что чёрных точек на одну больше, чем белых
    assert len(black_base_points) == len(white_base_points) + 1, 'Шумовые линии на опорном срезе интерферограммы:' \
                                                                 + str(black_base_points) + str(white_base_points)
    number_of_white_lines = len(white_base_points)
    # проверяем, что в опорных точках минимумы и максимумы идут поочерёдно
    assert all(black_base_points[i] < white_base_points[i] for i in range(number_of_white_lines)), \
           'чёрные и белые точки идут не поочерёдно на опорной границе изображения'

    # Трассировка слева направо
    # black_lines[i][x] и white_lines[i][x] будут содержать значения y для каждой i-й линии в точке x
    black_lines = [[None]*(input_image.width - 1) + [black_base_points[i]] for i in range(number_of_white_lines + 1)]
    white_lines = [[None]*(input_image.width - 1) + [white_base_points[i]] for i in range(number_of_white_lines)]
    # линии перебираем по очереди: чёрные, затем белые,
    # чтобы была возможность проверки, не вылазим ли белой за соседние чёрные
    for line in black_lines:
        trace_line(line, all_black_base_points)
        
    for i in range(number_of_white_lines):
        trace_line(white_lines[i], all_white_base_points, black_lines[i], black_lines[i+1])

    for line in black_lines:
        # DEBUG        print('black', line)
        for x in range(input_image.width):
            if line[x] != None:
                image_data[line[x]*input_image.width + x] = (255, 0, 0)
    for line in white_lines:
        # DEBUG        print('white', line)
        for x in range(input_image.width):
            if line[x] != None:
                image_data[line[x]*input_image.width + x] = (0, 0, 255)

    output_image = Image.new(input_image.mode, input_image.size)
    output_image.putdata(image_data)
    output_image.save(image_filename.replace('.tif', '_trace.tif'))
    #output_image.show()

    return black_lines, white_lines


def calculate_phases(image_filename, black_lines, white_lines):
    """
    Осуществляет трассировку интерференционных линий на картинке.
    :param image_filename: имя файла исходной картинки
    :param black_lines: список линий, каждая из которых представляет из себя список y[x] точек интерференционной линии.
    :param white_lines: список линий, каждая из которых представляет из себя список y[x] точек интерференционной линии.
    :return: Возвращает два списка интерференционных линий: чёрных и белых. Причём чёрных на одну больше.
    """
    input_image = Image.open(original_folder + '\\' + image_filename)

    # сообщаем каждой точке ее фазу
    phase = [[0] * input_image.width for i in range(input_image.height)]
    for j in range(input_image.width):
        numberline = 0
        for i in range(input_image.height):
            if (i == black_lines[numberline // 2][j]) or (
                    i <= white_lines[-1][j] and i == white_lines[numberline // 2][j]):
                phase[i][j] = (numberline + 1) * math.pi  # в точках максимума-минимума pi*n
                numberline += 1
            elif i < black_lines[0][j]:
                phase[i][j] = math.pi * (0.5 + 0.5 * math.sin(-math.pi / 2 + math.pi * i / black_lines[0][j]))
            elif numberline == len(black_lines) + len(white_lines):  # после последней чёрной линии
                phase[i][j] = math.pi * (
                numberline + 0.5 + 0.5 * math.sin(-math.pi / 2 + math.pi * (i - black_lines[numberline // 2][j]) /
                                                  (input_image.height - black_lines[numberline // 2][j])))
            elif (numberline % 2 == 1):  # между ними интерполируем по синусоиде
                # DEBUG
                # print(i, j, white_lines[numberline//2][j], black_lines[numberline//2][j])
                phase[i][j] = numberline * math.pi + math.pi * (
                0.5 + 0.5 * math.sin(-math.pi / 2 + math.pi * (i - black_lines[numberline // 2][j])
                                     / (white_lines[numberline // 2][j] - black_lines[numberline // 2][j])))
            elif (numberline % 2 == 0):
                phase[i][j] = numberline * math.pi + math.pi * \
                              (0.5 + 0.5 * math.sin(-math.pi / 2 + math.pi * (i - white_lines[numberline // 2 - 1][j])/
                                     (black_lines[numberline // 2][j] - white_lines[numberline // 2 - 1][j])))
            else:
                raise Exception()

    save_tif_image(phase, image_filename.replace('.tif', '_phase.tif'), 120)

    return phase


def get_base_points(data, number_of_approximation_points, trash_filter=None):
    # приближаем значения в соседних N точках наилучшими параболами
    # и считаем хи квадрат отлонений
    # trash_filter - функция для фильтрации значений минимумов/максимумов
    N = number_of_approximation_points//2
    khi2_for_y = [None]*N
    for y in range(N, len(data)-N):
        A = 2*sum(i**4 for i in range(1, N+1))
        B = 2*sum((data[y]-data[y+i])*(i**2) for i in range(-N, N+1))
        C = sum((data[y+i]-data[y])**2 for i in range(-N, N+1))
        a_optimal = -B/2/A
        khi2 = A*a_optimal**2 + B*a_optimal + C
        #print(y, data[y], khi2, A, B, C, a_optimal, sep='\t')  # DEBUG
        khi2_for_y.append(khi2)

    base_points = []
    for y in range(N+2, len(data)-N-3):
        if (khi2_for_y[y] <= khi2_for_y[y-1] and
                khi2_for_y[y] <= khi2_for_y[y+1] and
                khi2_for_y[y] <= khi2_for_y[y-2] and
                khi2_for_y[y] <= khi2_for_y[y+2]):
            if not trash_filter or trash_filter(data[y]):
                base_points.append(y)
    return base_points


def trace_line(line, all_base_points, lower_border_line = None, upper_border_line = None):
    """ Осуществляет трассировку линии начиная от правой её точки line[-1] до левого её края
        по базовым точкам all_base_points
        line - это список координат y точек линии длиной в ширину картинки
        В момент вызова функции должна быть заполнена разумно только line[-1]
        При выходе line заполнена разумно по всей ширине:  line[x] == y(x)
        all_base_points[x] - список или множество координат y точек, похожих на точки линий данного типа
        upper_border_line и lower_border_line - пограничные линии координат y, за которые нельзя вылезать
    """
    assert len(line) == len(all_base_points), "Не совпали длины"
    width = len(line)  # input_image.width
    old_x = width-1
    old_y = line[old_x]
    for x in range(width-2, -1, -1):
        for dy in [0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, +6, -6, +7, -7]:#,
                  # +8, -8, +9, -9, +10, -10, +11, -11, +12, -12, +13, -13, +14, -14, +15, -15]:
            y = old_y + dy
            # проверяем выход за пограничные линии
            if lower_border_line and y <= lower_border_line[x] or upper_border_line and y >= upper_border_line[x]:
                continue  # пропускаем такую точку
            if y in all_base_points[x]:
                # найдена точка данной линии в координатах (x, y)
                if abs(x - old_x) == 1:
                    # точка рядом по координате x => просто её сохраняем
                    line[x] = y
                else:
                    # точка не рядом => интерполяция линии между двумя опорными точками линии
                    k = (y - old_y)/(x - old_x)
                    b = (old_y*x - old_x*y)/(x - old_x)
                    for x1 in range(old_x - 1, x - 1, -1):
                        y1 = round(k*x1 + b)
                        line[x1] = y1
                old_x = x
                old_y = y
                break
    if old_x != 0: # Для нескольких левых точек нет опорных и некуда интерполировать!
        for x in range(0, old_x):
            line[x] = old_y  # интеполируем последней найденной базовой слева

    # if any(y == None for y in line):
    #    print('ERROR: None exists in BLACK line #%d'%i, line)
    #    raise AssertionError()


def plot_interferogram_slice(data_slice, base_points1=[], base_points2=[]):
    # равномерно распределённые значения от 0 до len(data), с шагом 1
    t = np.arange(0., len(data_slice), 1.)
    dat = np.array(data_slice)
    plt.plot(t, dat, 'b')

    if base_points1:
        base_points_values = [data_slice[x] for x in base_points1]
        plt.plot(base_points1, base_points_values, 'r*')
    if base_points2:
        base_points_values = [data_slice[x] for x in base_points2]
        plt.plot(base_points2, base_points_values, 'g*')
    
    plt.show()

if __name__ == '__main__':
    main()
