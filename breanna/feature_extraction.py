from skimage           import io
from skimage.color     import rgb2gray
from skimage.transform import resize
from keras.models      import load_model
from scipy import ndimage
import pandas            as pd
import numpy             as np
import glob
import os
import cv2
import pytesseract

from breanna.util import to_ospath

# fixing some strange errors on my computer
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def graylevel_contrast(img):
    img_gray = rgb2gray(img)
    return np.percentile(img_gray, 95) - np.percentile(img_gray, 5)
def dominant_bins(img):
    img_gray = rgb2gray(img)
    hist, _  = np.histogram(img_gray, bins=256)
    return np.sum(hist >= 0.01 * np.max(hist))
def graylevel_std(img):
    img_gray = rgb2gray(img)
    return np.std(img_gray)
def dominant_colors(img):
    color_cube = np.zeros(shape=(8, 8, 8), dtype='int')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]//32
            color_cube[r, g, b] += 1
    return np.sum(color_cube >= 0.01 * np.max(color_cube))
def dominant_extent(img):
    color_cube = np.zeros(shape=(8, 8, 8), dtype='int')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]//32
            color_cube[r, g, b] += 1
    return np.sum(np.max(color_cube)/np.sum(color_cube))
def number_connected_components(img):
    blur_radius = 1.0
    threshold = 200
    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold)
    return nr_objects
def number_of_connected_components_saliency_map(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_map = saliency.computeSaliency(img)[1]
    threshold = 0.02
    # find connected components
    labeled, nr_objects = ndimage.label(saliency_map > threshold)
    return nr_objects
def number_of_characters(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #thresholding and blurring for preprocessing
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.medianBlur(gray, 3)
        text = pytesseract.image_to_string(gray)
        number_of_characters = len(text)
        return number_of_characters
    except Exception as e:
        print(e)
        print('if using Windows, try installing tessearact and specify the path to tesseract manually,')
        print('if using Mac, try installing tesseract using Homebrew')
        return 0
def size_of_largest_connected_component(img):
    img = img.astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity=4)
    sizes = stats[:, -1]
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_size = sizes[i]
    return max_size


VISUALFEATURE_DICT = {
        'GLC': graylevel_contrast,
        'DNB': dominant_bins,
        'GLS': graylevel_std,
        'DNC': dominant_colors,
        'DNE': dominant_extent,
        'NCC': number_connected_components,
        'SLC': size_of_largest_connected_component, #slow to compute!
        'NCS': number_of_connected_components_saliency_map,
        'NCH': number_of_characters
        }

class FeatureExtractor:
    def __init__(self):
        return
    def extract(self, image):
        return
    def get_dim(self):
        return
 
class VisualFeatureExtractor(FeatureExtractor):
    def __init__(self, features):
        for f in features:
            if f not in VISUALFEATURE_DICT.keys():
                raise ValueError(f'Unknown visual feature: {f}')
        self.features = [VISUALFEATURE_DICT[f] for f in features]
    def extract(self, image):
        return np.array([f(image) for f in self.features])
    def get_dim(self):
        return len(self.features)

class AutoEncoderExtractor(FeatureExtractor):
    def __init__(self, encoder_path, input_size, output_size):
        encoder_path = to_ospath(encoder_path)
        self.encoder = load_model(encoder_path)
        self.input_size  = input_size
        self.output_size = output_size
    def extract(self, image):
        image = resize(image, self.input_size, 
                       anti_aliasing=True, mode='reflect')
        image = image.reshape(1, *image.shape)
        return self.encoder.predict(image).flatten()
    def get_dim(self):
        return self.output_size

IMAGE_REGEXS = ['**/*.jpg']
AGGREGATE_DICT = {
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min
        }

class FeatureAggregator:
    def __init__(self, extractor, method):
        if method not in AGGREGATE_DICT.keys():
            raise ValueError(f'Unknown aggregation: {method}')
        self.method = AGGREGATE_DICT[method]
        self.extractor = extractor
    def aggregate(self, banner_path, search_for=IMAGE_REGEXS):
        image_paths = []
        for search in search_for:
            image_paths += glob.glob(banner_path+search)
        image_paths = [to_ospath(image_path) for image_path in image_paths]
        images = [io.imread(image_path) for image_path in image_paths]
        features = np.r_[[self.extractor.extract(image) for image in images]]
        return self.method(features, axis=0)
    def get_dim(self):
        return self.extractor.get_dim()

def summarize_banners(banner_root, aggregator):
    if banner_root[-1] != '/':
        banner_root = banner_root+'/'
    
    features = ['feature'+str(i) for i in range(aggregator.get_dim())]
    columns  = ['banner_name']+features
    df = pd.DataFrame(columns=columns)
    
    banner_paths = glob.glob(banner_root + '*/')
    for i, banner_path in enumerate(banner_paths):
        banner_name = banner_path.split('/')[-2] # -2 on Mac, this doesn't work on Windows, fix later.
        df.loc[i] = [banner_name, *aggregator.aggregate(banner_path)]
    return df