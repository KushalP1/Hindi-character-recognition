import os
import numpy as np
from PIL import Image
from keras.models import load_model

unicodeList=[2306, 2310, 2311, 2312, 2313, 2315, 2318, 2319, 2320, 2322, 2325, 2327, 2328, 2330, 2331, 2332, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2375, 2376, 2379, 2380, 2381, 2382, 2384, 2387, 2388, 2390, 2392, 2399, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2423, 2424, 2429]
imageDimensions = [64,64,1]

def preprocess(imageFileName, imageDimensions = [64,64,1]):
    image = Image.open(imageFileName).convert('L')
    image = image.resize((imageDimensions[0], imageDimensions[1]), Image.ANTIALIAS)
    image = np.array(image)
    return image

def predict(imageFileName):
    image = preprocess(imageFileName)
    image = np.expand_dims(image, axis = -1)
    image = np.expand_dims(image, axis = 0)

    model = load_model("./models/weights.01-0.04.hdf5")

    result = model.predict(image, verbose=1)
    result = (result>0.5)*1

    results = []

    for i in range(len(result[0])):
        if(result[0][i]==1):
            results.append(unicodeList[i])
    return results


print(predict("0.png"))
'''
for i in range(100):
    imageName = str(i) + ".png"
    print(imageName)
    result = predict("./my_test/" + imageName)
    print (result)
    for x in result:
        print(chr(int(x)))
    i = input()
    if i == 0:
        break

'''
