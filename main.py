# import modules and packages
import numpy as np
import imageio
import cv2
from numba import jit

from clusterring import clustFunc


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """ returns a N * N sliding window centered at (x,y)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


@jit(nopython=True, cache=True)
def get_window(m, x, y, N):
    """ returns a N * N sliding window centered at (x,y)
    x, y is the pixel position and N is the side of sliding window
    """
    h, w, c = m.shape  # dimensions of m
    halfN = N // 2
    r = np.zeros((N, N, c))  # finding boundaries of window
    xmin = max(0, x - halfN);
    xmax = min(w, x + (halfN + 1))
    ymin = max(0, y - halfN);
    ymax = min(h, y + (halfN + 1))
    pxmin = halfN - (x - xmin);
    pxmax = halfN + (xmax - x)
    pymin = halfN - (y - ymin);
    pymax = halfN + (ymax - y)

    r[pymin:pymax, pxmin:pxmax] = m[ymin:ymax, xmin:xmax]
    return r


@jit(nopython=True, cache=True)
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, sigma_C, alpha_init, maxIter, minLike):
    """
    maxIter - maximal number of iterations
    At the given pixel C, we solve for F,B and alpha that maximize the sum of log
    likelihoods.
    mu_F - means of F
    Sigma_F - covariances of F
    mu_B - means of B
    Sigma_B - covariances of B
    C - given pixel
    alpha_init - initial value for alpha
    minLike - minimal change in likelihood between consecutive iterations
    """
    I = np.eye(3)
    # 3 values for 3 channels of F
    FMax = np.zeros(3)
    # 3 values for 3 channels of B
    BMax = np.zeros(3)
    alphaMax = 0
    # initializing to -infinity
    maxlike = - np.inf
    invsgma2 = 1 / sigma_C ** 2
    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])
        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            # keeps a  track of number of iterations
            myiter = 1
            lastLike = -1.7977e+308
            while True:
                # solve for F,B
                A11 = invSigma_Fi + I * alpha ** 2 * invsgma2
                A12 = I * alpha * (1 - alpha) * invsgma2
                A22 = invSigma_Bj + I * (1 - alpha) ** 2 * invsgma2
                A = np.vstack((np.hstack((A11, A12)), np.hstack((A12, A22))))
                b1 = invSigma_Fi @ mu_Fi + C * (alpha) * invsgma2
                b2 = invSigma_Bj @ mu_Bj + C * (1 - alpha) * invsgma2
                b = np.atleast_2d(np.concatenate((b1, b2))).T

                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                # solve for alpha

                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T - B).T @ (F - B)) / np.sum((F - B) ** 2)))[
                    0, 0]
                # # calculate likelihood
                L_C = - np.sum((np.atleast_2d(C).T - alpha * F - (1 - alpha) * B) ** 2) * invsgma2
                L_F = (- ((F - np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F - np.atleast_2d(mu_Fi).T)) / 2)[0, 0]
                L_B = (- ((B - np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B - np.atleast_2d(mu_Bj).T)) / 2)[0, 0]
                like = (L_C + L_F + L_B)

                # look for maximum likelihood
                if like > maxlike:
                    alphaMax = alpha
                    maxLike = like
                    FMax = F.ravel()
                    BMax = B.ravel()
                # we break when we exceed the number of iterations or when the likelihood calculated no longer needs
                # to be maximized
                if myiter >= maxIter or abs(like - lastLike) <= minLike:
                    break

                lastLike = like
                # increment number of iterations
                myiter += 1
                # return calculated and maximized F, B and alpha
    return FMax, BMax, alphaMax


def bayesian_matte(img, trimap, N, sigma=8, minN=8):
    """ img is the input image and trimap is input trimap
        N is the dimension of sliding window
        sigma is the variance
        minN is the minimjum number of known pixels required inside the sliding window to proceed
    """
    img = img / 255
    #  get dimensions of image
    h, w, c = img.shape
    alpha = np.zeros((h, w))

    fg_mask = trimap == 255
    bg_mask = trimap == 0
    unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
    foreground = img * np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
    background = img * np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)

    alpha[fg_mask] = 1
    F = np.zeros(img.shape)
    B = np.zeros(img.shape)
    alphaRes = np.zeros(trimap.shape)
    # keeps track of number of unknown pixels
    n = 0
    alpha[unknown_mask] = np.nan
    # number of unknown pixels
    nUnknown = np.sum(unknown_mask)
    print(nUnknown)
    # decide size of sliding window based on the number of unknown pixels
    if nUnknown < 60000:
        N = 25
    else:
        N = 125

    # finding gaussain weights using N and sigma
    gaussian_weights = matlab_style_gauss2d((N, N), sigma)
    gaussian_weights = gaussian_weights / np.max(gaussian_weights)

    unkreg = unknown_mask
    kernel = np.ones((3, 3))
    while n < nUnknown - 1000:
        n = n + 1
        unkreg = cv2.erode(unkreg.astype(np.uint8), kernel, iterations=1)
        unkpixels = np.logical_and(np.logical_not(unkreg), unknown_mask)

        Y, X = np.nonzero(unkpixels)
        for i in range(Y.shape[0]):
            print(n, nUnknown)
            y, x = Y[i], X[i]
            # p is the known pixel
            p = img[y, x]
            # (x,y) is the p's coordinate
            # get the window around pixel p
            a = get_window(alpha[:, :, np.newaxis], x, y, N)[:, :, 0]

            # take the foreground pixels inside sliding window
            f_pixels = get_window(foreground, x, y, N)
            f_weights = (a ** 2 * gaussian_weights).ravel()

            f_pixels = np.reshape(f_pixels, (N * N, 3))
            posInds = np.nan_to_num(f_weights) > 0
            f_pixels = f_pixels[posInds, :]
            f_weights = f_weights[posInds]

            # Take the background pixels inside sliding window
            b_pixels = get_window(background, x, y, N)
            b_weights = ((1 - a) ** 2 * gaussian_weights).ravel()

            b_pixels = np.reshape(b_pixels, (N * N, 3))
            posInds = np.nan_to_num(b_weights) > 0
            b_pixels = b_pixels[posInds, :]
            b_weights = b_weights[posInds]

            # if enough known pixels are not present in the sliding window, we leave it for later
            if len(f_weights) < minN or len(b_weights) < minN:
                continue
            # Use clusterring algorithm
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            alpha_init = np.nanmean(a.ravel())
            # Solve for F,B for all cluster pairs and take the maximum likelihood
            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 50, 1e-6)
            foreground[y, x] = f.ravel()
            background[y, x] = b.ravel()
            alpha[y, x] = alphaT
            unknown_mask[y, x] = 0
            n += 1
            # returns the calculated alpha image
    return alpha


def main():
    # read input image
    img = cv2.imread('input_training_lowres/GT01.png')
    height, width, color = img.shape
    k = width // 801
    print(k)
    img = cv2.resize(img, (width // (k + 1), height // (k + 1)))
    print(img.shape)
    # read user defined trimap
    trimap = imageio.imread('trimap_training_lowres/Trimap1/GT01.png', as_gray=True)
    trimap = cv2.resize(trimap, (width // (k + 1), height // (k + 1)))
    # trimap = cv2.resize(trimap, (400, 323))
    # N is the size of sliding window
    # value of N will be changed later depending on number of unknown pixels
    N = 0
    # predicted alpha image
    alpha = bayesian_matte(img, trimap, N)
    alpha = cv2.resize(alpha, (width, height))
    # read actual alpha
    actual_alpha = cv2.imread('gt_training_lowres/GT01.png')
    # read Trimap
    Trimap = cv2.imread('trimap_training_lowres/Trimap1/GT01.png')
    # plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(Trimap)
    ax2.set_title('Input Trimap')
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(actual_alpha)
    ax3.set_title('Actual Alpha')
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(alpha, cmap='gray')
    ax4.set_title('Calculated Alpha')
    plt.show()
    cv2.waitKey(0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.misc

    main()
