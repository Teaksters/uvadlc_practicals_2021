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

# Report resnet18 Accuracy in relation to transformations and severity
for transform in TRANSFORMS:
    if transform == None: continue
    acc = []
    for i in range(1, 6):
        acc.append(res_18_res[(transform, i)])
    plt.plot(SEVERETIES, acc, label=transform)
#
plt.xticks(SEVERETIES)
plt.title('Resnet-18 Accuracy Against Transformation Severity')
plt.xlabel('Transformation Severity (little->high)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.savefig('Resnet_18_acc.jpg')

# Now for the CE RCE results
for file in os.listdir(RES_DIR):
    # Read all result pickles
    if file[-4:] != '.pkl' or file[-6:] == '18.pkl':
        continue
    result_path = os.path.join(RES_DIR, file)
    results = pickle.load(open( result_path, "rb" ))
    # Do the same but now with CE and RCE measures
    fig, [ax, ax1] = plt.subplots(2, 1)
    plt.setp([ax, ax1], xticks=SEVERETIES)
    for transform in TRANSFORMS:
        ce, ce_ = 0., 0.
        rce, rce_ = 0., 0.
        for i in range(1, 6):
            a = (1 - results[(transform, i)])
            _a = (1 - results[('None', 1)])
            b = (1 - res_18_res[(transform, i)])
            _b = (1 - res_18_res[('None', 1)])
            ce +=  a
            ce_ += b
            rce += (a - _a)
            rce_ += (b - _b)
        CE = ce / ce_
        RCE = rce / rce_
        print(file[:-4], transform, ' CE: ', str(CE), ' RCE: ', str(RCE))
