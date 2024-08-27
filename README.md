## NeRF-CA: Dynamic reconstruction of X-ray Coronary Angiography with extremely sparse-views

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
