import numpy, cv2, click
from scipy.io.wavfile import write

def build_filters():
    filters = []
    ksize = 31
    for theta in numpy.arange(0, numpy.pi, numpy.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = numpy.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        numpy.maximum(accum, fimg, accum)
    return accum

@click.command()
@click.option('--path', default='./test.jpg', help='Image path')
@click.option('--out', default='./out.wav', help='Outfile path')
def main(path, out):
    img = cv2.imread(path)
    filters = build_filters()
    filtered = process(img, filters).swapaxes(0, 1)
    scaled = numpy.int16(filtered / numpy.max(numpy.abs(filtered)) * 32767)
    rv = numpy.ravel(scaled)
    write(out, 44100, rv)

if __name__ == '__main__':
    main()