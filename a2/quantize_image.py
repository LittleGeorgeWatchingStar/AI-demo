import numpy as np
from kmeans import Kmeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b # number of bits per pixel

    def quantize(self, img):
        """
        Quantizes an image into 2^b clusters

        Parameters
        ----------
        img : a (H,W,3) numpy array

        Returns
        -------
        quantized_img : a (H,W,1) numpy array containing cluster indices

        Stores
        ------
        colours : a (2^b, 3) numpy array, each row is a colour

        """

        H, W, _ = img.shape
        pixels = img.reshape((-1,3))
        model = Kmeans(2**self.b)
        model.fit(pixels)
        quantized_img = model.predict(pixels).reshape((H,W,1))
        self.colours = model.means

        return quantized_img

    def dequantize(self, quantized_img):
        H, W, _ = quantized_img.shape
        y = quantized_img.reshape(-1)
        img = np.ones((H*W,3))
        for i in range(H*W):
            img[i] = self.colours[y[i]]
        img = img.reshape((H,W,3))
        return img
