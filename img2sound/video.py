import numpy, cv2, subprocess
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

def main():
    filters = build_filters()
    cam = cv2.VideoCapture(0)
    success, img = cam.read()
    window = 'Camera'
    cv2.namedWindow(window, cv2.CV_WINDOW_AUTOSIZE)
    n = 0

    while success:
        img = cv2.resize(img, (320, 240))
        cv2.imshow(window, img)
        
        filtered = process(img, filters).swapaxes(0, 1)
        scaled = numpy.int16(filtered / numpy.max(numpy.abs(filtered)) * 32767)
        rv = numpy.ravel(scaled)
        name = './temp/sound{}.wav'.format(n)
        write(name, 44100, rv)
        subprocess.call('afplay {}'.format(name), shell=True)
        
        n += 1
        success, img = cam.read()
        key = cv2.waitKey(20)
        if key == 27:
            cv2.destroyWindow(winName)
            break

if __name__ == '__main__':
    main()