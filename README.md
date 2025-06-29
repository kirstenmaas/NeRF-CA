# NeRF-CA: Dynamic reconstruction of X-ray Coronary Angiography with extremely sparse-views

## [Project Page](https://kirstenmaas.github.io/nerfca) | [Paper](https://arxiv.org/abs/2408.16355)

## About NeRF-CA
Dynamic three-dimensional (4D) reconstruction from two-dimensional X-ray coronary angiography (CA) remains a significant clinical problem. Existing CA reconstruction methods often require extensive user interaction or large training datasets. Recently, Neural Radiance Field (NeRF) has successfully reconstructed high-fidelity scenes in natural and medical contexts without these requirements. However, challenges such as sparse-views, intra-scan motion, and complex vessel morphology hinder its direct application to CA data. We introduce NeRF-CA, a first step toward a fully automatic 4D CA reconstruction that achieves reconstructions from sparse coronary angiograms. To the best of our knowledge, we are the first to address the challenges of sparse-views and cardiac motion by decoupling the scene into the moving coronary artery and the static background, effectively translating the problem of motion into a strength. NeRF-CA serves as a first stepping stone for solving the 4D CA reconstruction problem, achieving adequate 4D reconstructions from as few as four angiograms, as required by clinical practice, while significantly outperforming state-of-the-art sparse-view X-ray NeRF. We validate our approach quantitatively and qualitatively using representative 4D phantom datasets and ablation studies. 

## Method Overview
![Overview of the method](https://github.com/kirstenmaas/NeRF-CA/blob/main/imgs/overview.png)

## Repository
This repository contains the code to preprocess the 4D phantom datasets and the implementation of the PyTorch models. The 4D phantom datasets can be acquired from [XCAT](https://cvit.duke.edu/resource/xcat-phantom-program/) and [MAGIX](https://www.osirix-viewer.com/resources/dicom-image-library/). We utilize the [TIGRE](https://github.com/CERN/TIGRE?tab=readme-ov-file) repository to generate the 2D CA sequences from the 4D phantom datasets.

- <b>Preparing datasets for training:</b> The main code can be found in <i>preprocess/datatoray.py</i>. It expects pre-generated .npy files of the 3D+t volumes, similar to the [TIGRE](https://github.com/CERN/TIGRE?tab=readme-ov-file) input. These files are generated through the XCAT dataset pre-processing code <i>preprocess/xcat.py</i> or MAGIX dataset pre-processing code <i>preprocess/preprocess_ccta.py</i>.
- <b>Models</b>: The models are defined in the /model folder.
- <b>Training</b>: The training code can be found in the folder /train. Our main method can be ran through the <b>run_composite.py</b> file, for which the hyperparameters can be defined in the <i>composite.txt</i> file.

## Citation
If you use this code for your research, please cite our work.
```
@article{maas2025nerf,
  title={NeRF-CA: Dynamic Reconstruction of X-ray Coronary Angiography with Extremely Sparse-views},
  author={Maas, Kirsten WH and Ruijters, Danny and Vilanova, Anna and Pezzotti, Nicola},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```
