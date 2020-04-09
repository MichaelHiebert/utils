import numpy as np
import os
import cv2
import random

class ImageDeformer():
    def __init__(self):
        pass

    def _apply_noise(self, noise_typ, image):
        """
        https://stackoverflow.com/a/30609854/7042418
        
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        noise_typ : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        """

        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy

    def noise(self, image):
        """
            Applys noise to this image and then
            returns a copy of the image with the noise
            applied.

            Parameters
            ----------
            image : ndarray
                Input image data. Will be converted to float.
        """
        return self._apply_noise('speckle', image)

    def gaussian_blur(self, image, blur_amount=33):
        """
            Applys gaussian blur to this image and then
            returns a copy of the image with the blur
            applied.

            Parameters
            ----------
            image : ndarray
                Input image data. Will be converted to float.
        """
        return cv2.GaussianBlur(image, (blur_amount,blur_amount),0)

    def median_blur(self, image, blur_amount=13):
        """
            Applys median blur to this image and then
            returns a copy of the image with the blur
            applied.

            Parameters
            ----------
            image : ndarray
                Input image data. Will be converted to float.
        """
        return cv2.medianBlur(image, blur_amount)

    def pixelate(self, image):
        """
            Scales down and then scales up this image to its
            original size to produce a lower quality, more
            "pixely" image.

            Parameters
            ----------
            image : ndarray
                Input image data. Will be converted to float.
        """
        w,h,_ = image.shape
        return cv2.resize(cv2.resize(image, (32,32)), (w,h))

    def random_deform(self, image):
        """
            Randomly applies a deformity to this image and returns a copy of the image with the deformity
            applied.

            Parameters
            ----------
            image : ndarray
                Input image data. Will be converted to float.
        """

        choice = random.randint(0, 3)
        options = [self.gaussian_blur, self.noise, self.median_blur, self.pixelate]

        return options[choice](image)

def deform_directory(in_dir, out_dir, label_dir=1,):
    """
        TODO
    """

    imdef = ImageDeformer()

    # make the deformed directory
    try:
        os.mkdir(out_dir)
    except:
        pass

    for root, dirs, files in os.walk(in_dir, topdown=False):
        for name in files:
            if name[-4:] == '.jpg': # is an image
                def_im = imdef.random_deform(cv2.imread(os.path.join(root, name)))

                fold = root.split('/')[label_dir]

                try:
                    os.mkdir('{}/{}'.format(out_dir,fold))
                except:
                    pass

                cv2.imwrite('{}/{}/{}'.format(out_dir,fold,name), def_im)

if __name__ == "__main__":
    # imdef = ImageDeformer()

    # pic = cv2.imread('data/0000045/009.jpg')
    # cv2.imwrite('test.png', imdef.random_deform(pic))
    deform_directory('data', 'deformed', label_dir=2)