"""

SMSI GMM with gray scale Image

"""

import time

import cv2
import numpy as np
from matplotlib import pyplot
from scipy import ndimage
from sklearn.cluster import KMeans


def initialise_parameters(features, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities
    if undefined.

    Arguments:
    features -- input features data set
    """
    if not means or not covariances:
        val = 250
        n = features.shape[0]
        # Shuffle features set
        indices = np.arange(n)
        np.random.shuffle(indices)
        features = np.array([features[i] for i in indices])

        # Split into n_components subarrays
        divs = int(np.floor(n / k))

        # Estimate means/covariances (or both)
        if not means:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features.reshape(-1, 1))
            means = kmeans.cluster_centers_

        if not covariances:
            covariances = [val * np.identity(d) for i in range(k)]

    if not weights:
        weights = [float(1 / k) for i in range(k)]

    return (means, covariances, weights)


# gaussian function
def gau(mean, var, varInv, feature, d):
    var_det = np.linalg.det(var)
    a = np.sqrt((2**d) * (np.pi ** d) * var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return b / a


def smsi_likeli(mean, var, varInv, s_value, feature, d):
    temp = []
    for x in range(k):
        temp.append(s_value * gau(mean[x], var[x], varInv[x], feature, d))
    return temp


def saliency(img):
    c = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:, :, 0] ** 2 + c[:, :, 1] ** 2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

    c[:, :, 0] = c[:, :, 0] * spectralResidual / mag
    c[:, :, 1] = c[:, :, 1] * spectralResidual / mag
    c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:, :, 0] ** 2 + c[:, :, 1] ** 2
    cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(mag)

    return mag


def neighbor_prob(img, fun):
    mask = np.ones((3, 3))
    result = ndimage.generic_filter(img, function=fun, footprint=mask, mode='constant', cval=np.NaN)
    return result


def sliding_window(im):
    rows, cols = im.shape
    final = np.zeros((rows, cols, 3, 3))
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            im1 = np.vstack((im[x:], im[:x]))
            im1 = np.column_stack((im1[:, y:], im1[:, :y]))
            final[x::3, y::3] = np.swapaxes(im1.reshape(int(rows / 3), 3, int(cols / 3), -1), 1, 2)
    return final


input_path = "../datasets/input/brainsexy.png"

img = cv2.imread(input_path, 0)

pyplot.subplot(2, 2, 1)
pyplot.imshow(img)
# no of clusters
k = 2
o_shape = img.shape


# Gray scale
d = 1

# Array of pixels
feat = img.reshape(-1)

# Total no of pixels
N = len(feat)

# s = saliency(input_path)
s = saliency(img)

means, covariances, weights = initialise_parameters(features=feat, d=d)
covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
meanPrev = [np.array([0]) for i in range(k)]
iteration = []
logLikelihoods = []
counterr = 0
smsi_init_weights = np.ones((o_shape[0] * o_shape[1], k))
# smsi_init_weights /= k
# Ri value
R_value = neighbor_prob(s, np.nansum)
R_value = R_value.reshape(-1)
s_value = s.reshape(-1)

while abs((np.asarray(meanPrev) - np.asarray(means)).sum()) > 0.01:
    # print(np.shape(means))
    start = time.time()
    smsi_resp = []
    for i, feature in enumerate(feat):
        smsi_classLikelihoods = smsi_likeli(means, covariances, covariances_Inv, s_value[i], feature, d)
        smsi_resp.append(smsi_classLikelihoods)
    # print(np.shape(smsi_resp))
    smsi_resp = np.asarray(smsi_resp)

    final_resp = []
    for cluster in range(k):
        test = smsi_resp[:, cluster]
        result = ndimage.generic_filter(test.reshape(o_shape[0], o_shape[1]), np.nansum, footprint=np.ones((3, 3)),
                                        mode='constant', cval=np.NaN).reshape(-1)
        final_resp.append(result * smsi_init_weights[:, cluster] / R_value)

    # numerator of gama
    smsi_resp_num = np.asarray(final_resp).T

    # denominator of gama
    final_resp_den = smsi_resp_num.sum(axis=1)

    # gama value of the smsi
    final_smsi_resp = []
    for cluster in range(k):
        final_smsi_resp.append(smsi_resp_num[:, cluster] / final_resp_den)

    final_smsi_resp = np.asarray(final_smsi_resp).T

    # SMSI MEAN
    ###############################################################################################
    #################################### SMSI MEAN ################################################

    smsi_mean_den = final_smsi_resp.sum(axis=0) #sigma of gamma
    smsi_mean_num_s_x = s_value * feat
    deno = s_value
    deno_S_x_P_m_Y_m = np.asarray([deno * ite for ite in smsi_resp.T]) 
    S_x_P_m_Y_m = np.asarray([smsi_mean_num_s_x * ite for ite in smsi_resp.T])
    deno_S_x_P_m_Y_m = deno_S_x_P_m_Y_m.T
    S_x_P_m_Y_m = S_x_P_m_Y_m.T
    
    smsi_mean_num = []
    deno_smsi_mean_num = []
    for i in range(k):
        gen_filter_copy = ndimage.generic_filter(deno_S_x_P_m_Y_m[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,footprint=np.ones((3, 3)), mode='constant', cval=np.NaN)
        gen_filter_copy = gen_filter_copy.reshape(-1,1)
        deno_S_x_P_m_Y_m[:, i:i+1] = gen_filter_copy
        S_x_P_m_Y_m[:, i] = ndimage.generic_filter(S_x_P_m_Y_m[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
        deno_smsi_mean_num.append(smsi_init_weights[:, i] * (deno_S_x_P_m_Y_m[:, i] / R_value))
        smsi_mean_num.append(smsi_init_weights[:, i] * (S_x_P_m_Y_m[:, i] / R_value))
    deno_smsi_mean_num = np.asarray(deno_smsi_mean_num).T
    smsi_mean_num = np.asarray(smsi_mean_num).T
    
    smsi_mean = (smsi_mean_num.sum(axis=0) / deno_smsi_mean_num.sum(axis=0)).T
    meanPrev = means.copy()
    means = smsi_mean
    means = means.reshape(-1,1)
    #################################################################################################
    ################################# CO VAR ########################################################
    # print(np.shape(feat))
    f_u = np.asarray([feat - means[i] for i in range(k)]).T
    # print(np.shape(means[0]))
    f_u = f_u ** 2
    # print(np.shape(f_u))

    covar_num_s_x_u = np.asarray([s_value * f_u[:, i] for i in range(k)]).T

    covar_num_s_x_u = np.asarray([ndimage.generic_filter(covar_num_s_x_u[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1) for i in range(k)]).T

    covar_num = np.asarray([final_smsi_resp[:, i] * (covar_num_s_x_u[:, i] / R_value) for i in range(k)]).T

    covar = covar_num.sum(axis=0) / smsi_mean_den

    covariances = [covar[i] * np.identity(d) for i in range(k)]
    covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
    # # print(np.shape(covariances))
    # feat_np = feat.reshape(-1, d)
    # print(np.shape(feat_np - np.reshape(means[i], (-1,1))), np.shape(means[i]))
#     f_u = np.asarray([feat - means[i] for i in range(k)]).T
#     # print(f_u)
#     f_u = f_u ** 2
#     # print(f_u)
#     # print(np.shape(f_u))
 
#     smsi_mean_den = final_smsi_resp.sum(axis=0) #sigma of gamma
#     smsi_mean_num_s_x = s_value
#     # print(type(s_value))
#     deno = s_value
#     deno_S_x_P_m_Y_m = np.asarray([deno * ite for ite in smsi_resp.T]) 
#     S_x_P_m_Y_m = np.asarray([smsi_mean_num_s_x * ite for ite in smsi_resp.T])
#     S_x_P_m_Y_m = S_x_P_m_Y_m.T
#     S_x_P_m_Y_m = S_x_P_m_Y_m * f_u
#     deno_S_x_P_m_Y_m = deno_S_x_P_m_Y_m.T

#     # print(feat)
    
    
#     smsi_mean_num = []
#     deno_smsi_mean_num = []
# #     smsi_mean_num_s_x = ndimage.generic_filter(smsi_mean_num_s_x.reshape(o_shape[0], o_shape[1]), np.nansum,
# #                                                footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
#     for i in range(k):
#         deno_S_x_P_m_Y_m[:, i] = ndimage.generic_filter(deno_S_x_P_m_Y_m[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
#                                                footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
#         S_x_P_m_Y_m[:, i] = ndimage.generic_filter(S_x_P_m_Y_m[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
#                                                footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
#         deno_smsi_mean_num.append(smsi_init_weights[:, i] * (deno_S_x_P_m_Y_m[:, i] / R_value))
#         smsi_mean_num.append(smsi_init_weights[:, i] * (S_x_P_m_Y_m[:, i] / R_value))
#     deno_smsi_mean_num = np.asarray(deno_smsi_mean_num).T
#     smsi_mean_num = np.asarray(smsi_mean_num).T
    
#     smsi_mean = (smsi_mean_num.sum(axis=0) / deno_smsi_mean_num.sum(axis=0)).T
#     covarPrev = covariances.copy()
#     covariances = smsi_mean.reshape(-1, 1, 1)
#     covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
    # print(covariances)
    #################################################################################################
    ################################ WEIGHTS #########################################################
    # weights for smsi
    smsi_weigths = []
    smsi_weights_num = []
    
    smsi_mean_den = final_smsi_resp.sum(axis=0) #sigma of gamma
    smsi_mean_num_s_x = s_value * feat
    deno = s_value
    deno_S_x_P_m_Y_m = np.asarray([deno * ite for ite in smsi_resp.T]) 
    
    deno_S_x_P_m_Y_m = deno_S_x_P_m_Y_m.T
    S_x_P_m_Y_m = np.asarray([smsi_mean_num_s_x * ite for ite in smsi_resp.T])
    S_x_P_m_Y_m = S_x_P_m_Y_m.T
    
    smsi_mean_num = []
    deno_smsi_mean_num = []
#     smsi_mean_num_s_x = ndimage.generic_filter(smsi_mean_num_s_x.reshape(o_shape[0], o_shape[1]), np.nansum,
#                                                footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
    for i in range(k):
        deno_S_x_P_m_Y_m[:, i] = ndimage.generic_filter(deno_S_x_P_m_Y_m[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
                                               footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
        deno_smsi_mean_num.append(smsi_init_weights[:, i] * (deno_S_x_P_m_Y_m[:, i] / R_value))
        
    deno_smsi_mean_num = np.asarray(deno_smsi_mean_num).T
    
    sumu = np.asarray([deno_smsi_mean_num.sum(axis=1) for i in range(k)]).T
    deno_smsi_mean_num /= sumu
    smsi_init_weights = deno_smsi_mean_num.copy()

    # print(smsi_init_weights)

    condition = abs((np.asarray(meanPrev) - np.asarray(means)).sum())
    print(counterr, condition, " Means: ", smsi_mean, "Time: ", time.time() - start)
    counterr += 1

segmentedImage = np.zeros((N), np.uint8)

for i, resp in enumerate(final_smsi_resp):
    maxx = resp.argmax()
    segmentedImage[i] = 255 - 255 * means[maxx]

segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])
pyplot.subplot(2, 2, 3)
pyplot.imshow(segmentedImage)
pyplot.show()
