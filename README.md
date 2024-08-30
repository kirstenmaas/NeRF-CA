# NeRF-CA: Dynamic reconstruction of X-ray Coronary Angiography with extremely sparse-views

## [Project Page](https://kirstenmaas.github.io/nerfca) | [Paper](https://arxiv.org/abs/2408.16355)

## About NeRF-CA

Dynamic three-dimensional (4D) reconstruction from two-dimensional X-ray coronary angiography (CA) remains a significant clinical problem.
Challenges include sparse-view settings, intra-scan motion, and complex vessel morphology such as structure sparsity and background occlusion.
Existing CA reconstruction methods often require extensive user interaction or large training datasets.
On the other hand, Neural Radiance Field (NeRF), a promising deep learning technique, has successfully reconstructed high-fidelity static scenes for natural and medical scenes.
Recent work, however, identified that sparse-views, background occlusion, and dynamics still pose a challenge when applying NeRF in the X-ray angiography context.
Meanwhile, many successful works for natural scenes propose regularization for sparse-view reconstruction or scene decomposition to handle dynamics.
However, these techniques do not directly translate to the CA context, where both challenges and background occlusion are significant.

This paper introduces NeRF-CA, the first step toward a 4D CA reconstruction method that achieves reconstructions from sparse coronary angiograms with cardiac motion.
We leverage the motion of the coronary artery to decouple the scene into a dynamic coronary artery component and static background.
We combine this scene decomposition with tailored regularization techniques.
These techniques enforce the separation of the coronary artery from the background by enforcing dynamic structure sparsity and scene smoothness.
By uniquely combining these approaches, we achieve 4D reconstructions from as few as four angiogram sequences.
This setting aligns with clinical workflows while outperforming state-of-the-art X-ray sparse-view NeRF reconstruction techniques.
We validate our approach quantitatively and qualitatively using 4D phantom datasets and ablation studies.

## Method Overview
![Overview of the proposed input optimization method](https://github.com/kirstenmaas/NeRF-CA/blob/main/imgs/overview.png)

## Repository
This repository contains the code to preprocess the 4D phantom datasets and the implementation of the PyTorch models. The 4D phantom datasets can be acquired from [XCAT](https://cvit.duke.edu/resource/xcat-phantom-program/) and [MAGIX](https://www.osirix-viewer.com/resources/dicom-image-library/). We utilize the [TIGRE](https://github.com/CERN/TIGRE?tab=readme-ov-file) repository to generate the 2D CA sequences from the 4D phantom datasets.

- <b>Preparing datasets for training:</b> The main code can be found in <i>preprocess/datatoray.py</i>. It expects pre-generated .npy files of the 3D+t volumes, similar to the [TIGRE](https://github.com/CERN/TIGRE?tab=readme-ov-file) input. These files are generated through the XCAT dataset pre-processing code <i>preprocess/xcat.py</i> or MAGIX dataset pre-processing code <i>preprocess/preprocess_ccta.py</i>.
- <b>Models</b>: The models are defined in the /model folder.
- <b>Training</b>: The training code can be found in the folder /train. Our main method can be ran through the <b>run_composite.py</b> file, for which the hyperparameters can be defined in the <i>composite.txt</i> file.

## Citation
If you use this code for your research, please cite our work.
```
@article{maas2024nerfca,
    title={NeRF-CA: Dynamic Reconstruction of X-ray Coronary Angiography with Extremely Sparse-views},
    author={Kirsten W. H. Maas and Danny Ruijters and Anna Vilanova and Nicola Pezzotti},
    journal={arXiv preprint arXiv:2408.16355},
    year={2024},
}
```