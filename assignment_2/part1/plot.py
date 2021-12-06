import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

RES_DIR = 'rslts'
TRANSFORMS = ['gaussian_noise_transform',
              'gaussian_blur_transform',
              'contrast_transform',
              'jpeg_transform']
SEVERETIES = np.arange(1, 6)

res18_path = os.path.join(RES_DIR, 'resnet18.pkl')
res_18_res = pickle.load(open(res18_path, "rb" ))

for transform in TRANSFORMS:
    acc = []
    for i in range(1, 6):
        acc.append(res_18_res[(transform, i)])
    plt.plot(SEVERETIES, acc, label=transform)
plt.xticks(SEVERETIES)
plt.title('Resnet-18 Accuracy Against Transformation Severity')
plt.xlabel('Transformation Severity (little->high)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.show()

# for file in os.listdir(RES_DIR):
#     # Read all result pickles
#     if file[-4:] != '.pkl' or file[-6:] == '18.pkl':
#         continue
#     result_path = os.path.join(RES_DIR, file)
#     results = pickle.load(open( result_path, "rb" ))
#     print(results)
