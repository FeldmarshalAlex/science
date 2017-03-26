import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

filename_before = 'before_phase.txt'
filename_after = 'after_phase.txt'
filename_difference = 'phase_difference.tif'
folder = r'result'


def main():
    phases_before = load_phases(folder + '\\' + filename_before)
    phases_after = load_phases(folder + '\\' + filename_after)

    phases = phases_after
    for y in range(len(phases)):
        for x in range(len(phases[y])):
            phases[y][x] -= phases_before[y][x]

    save_tif_image(phases, folder + '\\' + filename_difference, 1)


def load_phases(filename):
    print('loading phases from:', filename)
    data = []
    with open(filename, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            data.append(row)
    return data


def save_tif_image(data, filename, scale=255):
    print('saving', filename)
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


if __name__ == '__main__':
    main()
