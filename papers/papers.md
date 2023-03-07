> This file was generated from papis library using papers.py script

# Papers 
This is a collection of papers from *ICCV2021*, *ICCV2019*, *CVPR2022*, *CVPR2021*, *CVPR2020* and (TODO *CVPR2019 ?*, *NeurIPS2022*, *NeurIPS2021*, *NeurIPS2020*)
regarding adversarial attack/defence.

---
- **Adversarial Robustness vs Model Compression, or Both?**, 2019 [[url](http://arxiv.org/abs/1903.12561v5)] [[code](https://github.com/yeshaokai/Robustness-Aware-Pruning-ADMM)]

Proposes a framework of concurrent adversarial training and weight pruning that enables model compression while still preserving the adversarial robustness. small model from scratch even with inherited initialization from the large model does not have accuracy nor robustness.

---
- **Relating Adversarially Robust Generalization to Flat Minima**, 2021 [[url](http://arxiv.org/abs/2104.04448v2)]

Relationship between robust generalization and flatness of the robust loss (cross-entropy loss on adversarial examples)landscape in weight space, i.e., whether robust loss changes significantly when perturbing weights. Propose metrics to measure flatness in the robust loss landscape and show correlation good robust generalization and flatness.

---
- **Adversarial Vertex Mixup: Toward Better Adversarially Robust   Generalization**, 2020 [[url](http://arxiv.org/abs/2003.02484v3)]

Soft labeling as solution to Adversarial Feature Overtting (AFO, a large gap between training accuracy and test accuracy cause by adversial training). Propose Adversarial Vertex mixup (AVmixup), a soft-labeled data augmentation approach. Test on CIFAR-10, CIFAR-100 and others. AVmixup significantly reduce trade-off std accuracy and adv rebustness.

---
- **Evaluating Robustness of Deep Image Super-Resolution against Adversarial   Attacks**, 2019 [[url](http://arxiv.org/abs/1904.06097v2)]

Investigates the robustness of deep learning-based super-resolution methods against adversarial attacks, which can significantly deteriorate the super-resolved images without noticeable distortion in the attacked low-resolution images.

---
- **Adversarial Learning with Margin-based Triplet Embedding Regularization**, 2019 [[url](http://arxiv.org/abs/1909.09481v1)] [[code](https://github.com/zhongyy/Adversarial_MTER)]

Propose to improve the local smoothness of the representation space, by integrating a margin-based triplet embedding regularization term into the classification objective. The regularization term consists of two steps optimizations which find potential perturbations and punish them by a large margin in an iterative way.

---
- **Batch Normalization Increases Adversarial Vulnerability and Decreases Adversarial Transferability: A Non-Robust Feature Perspective**, 2020 [[url](http://arxiv.org/abs/2010.03316v2)] [[code](https://github.com/phibenz/adversarial_ml.research)]

Batch Normalization (BN) is observed to increase the model accuracy while at the cost of adversarial robustness. Empirical evidence that BN make model more dependent on non-robust features (NRFs). Propose a framework for disentangling robust usefulness into robustness and usefulness.

---
- **Removing Adversarial Noise in Class Activation Feature Space**, 2021 [[url](http://arxiv.org/abs/2104.09197v1)]

Propose to remove adversarial noise by implementing a self-supervised adversarial training mechanism in a class activation feature space. Enhance adversarial robustness especially against unseen adversarial attacks.

---
- **Defending Against Universal Perturbations With Shared Adversarial Training**, 2018 [[url](http://arxiv.org/abs/1812.03705v2)]

Show that adversarial training is more effective in preventing universal perturbations, where the same perturbation needs to fool a classifier on many inputs. Investigate the trade-off between robustness against universal perturbations and performance on unperturbed data and propose an extension of adversarial training that handles this trade-off more gracefully.

---
- **Admix: Enhancing the Transferability of Adversarial Attacks**, 2021 [[url](http://arxiv.org/abs/2102.00436v3)] [[code](https://github.com/JHL-HUST/Admix)]

Propose a new input transformation based attack method called Admix that considers the input image and a set of images randomly sampled from other categories.

---
- **Meta Gradient Adversarial Attack**, 2021 [[url](http://arxiv.org/abs/2108.04204v2)]

Meta Gradient Adversarial Attack (MGAA), which can be integrated with any existing gradient based attack method for improving the cross-model transferability. It's based on iteratively simulate black and white box attacks.

---
- **Interpreting Attributions and Interactions of Adversarial Attacks**, 2021 [[url](http://arxiv.org/abs/2108.06895v1)]

Explain adversarial attacks in terms of how adversarial perturbations contribute to the attacking task. Attributions of different image regions to the decrease of the attacking cost. Quantify interactions among adversarial perturbation pixels.

---
- **Universal Adversarial Perturbation via Prior Driven Uncertainty Approximation**, 2019 [[url](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Universal_Adversarial_Perturbation_via_Prior_Driven_Uncertainty_Approximation_ICCV_2019_paper.pdf)]

New unsupervised universal adversarial perturbation method, termed as Prior Driven Uncertainty Approximation (PD-UA), to generate a robust UAP by fully exploiting the model uncertainty at each network layer.

---
- **The LogBarrier adversarial attack: making effective use of decision   boundary information**, 2019 [[url](http://arxiv.org/abs/1903.10396v1)]

New untargeted attack, based on these best practices (from the optimization literature to solve this constrained minimization problem), using the established logarithmic barrier method. Similar or better than SOTA with perturbation distance significantly smaller. Benchmarked MNIST, CIFAR10, ImageNet-1K.

---
- **Explaining Classifiers using Adversarial Perturbations on the Perceptual Ball**, 2019 [[url](http://arxiv.org/abs/1912.09405v4)] [[code](https://github.com/alan-turing-institute/perceptualBall)]

Present a simple regularization of adversarial perturbations (semi-sparse that highlight objects and regions of interest) based upon the perceptual loss. As a semantically meaningful adverse perturbations, it forms a bridge between counterfactual explanations and adversarial perturbations in the space of images.

---
- **MaxUp: Lightweight Adversarial Training with Data Augmentation Improves Neural Network Training**, 2021 [[url](https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_MaxUp_Lightweight_Adversarial_Training_With_Data_Augmentation_Improves_Neural_Network_CVPR_2021_paper.pdf)]

Embarrassingly simple, highly effective technique for improving the generalization performance: generate a set of augmented data with some random perturbations or transforms, and minimize the maximum, or worst case loss over the augmented data. Tested on CIFAR-10.

---
- **Data-free Universal Adversarial Perturbation and Black-box Attack**, 2021 [[url](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.pdf)]

Universal adversarial perturbation (UAP), i.e. a single perturbation to fool the network for most images. Explanation why most images are misclassified to a dominant label. No code available.

---
- **Adversarial Attacks are Reversible with Natural Supervision**, 2021 [[url](http://arxiv.org/abs/2103.14222v3)] [[code](https://github.com/cvlab-columbia/SelfSupDefense)]

Attack vectors cause not only image classifiers to fail, but also collaterally disrupt incidental structure in the image. Restoring the natural structure of the attacked image will reverse many types of attacks. Improved robustness for several SOTA models across CIFAR-10 and CIFAR-100. This kind of defence holds even if the attacker is aware of it.

---
- **Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**, 2022 [[url](http://arxiv.org/abs/2203.05154v3)] [[code](https://github.com/liuye6666/adaptive_auto_attack)]

Propose a parameter-free Adaptive Auto Attack (A3) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. 1st place in CVPR 2021 White-box Adversarial Attacks on Defense Models.

---
- **AGKD-BML: Defense Against Adversarial Attack by Attention Guided Knowledge Distillation and Bi-directional Metric Learning**, 2021 [[url](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_AGKD-BML_Defense_Against_Adversarial_Attack_by_Attention_Guided_Knowledge_Distillation_ICCV_2021_paper.pdf)] [[code](https://github.com/hongw579/AGKD-BML)]

Propose a novel adversarial training-based model by Attention Guided Knowledge Distillation and Bi-directional Metric Learning (AGKD-BML). Extensive adversarial robustness experiments on  with different attacks achieve SOTA.

---
- **Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better**, 2021 [[url](http://arxiv.org/abs/2108.07969v1)] [[code](https://github.com/zibojia/RSLAD)]

Revisit several SOTA Adversarial Training (AT) methods from a distillation perspective. Use of robust soft labels. Propose novel adversarial robustness distillation method: Robust Soft Label Adversarial Distillation (RSLAD). Provide a set of understandings on our RSLAD and the importance of robust soft labels for adversarial robustness distillation.

---
- **Sparse and Imperceivable Adversarial Attacks**, 2019 [[url](http://arxiv.org/abs/1909.05040v1)] [[code](https://github.com/fra31/sparse-imperceivable-attacks)]

Propose a new black-box technique to craft adversarial examples aiming at minimizing l0-distance to the original image. Allowing pixels to change only in region of high variation and avoiding changes along axis-aligned edges makes our adversarial examples almost non-perceivable.

---
- **Exploiting Explanations for Model Inversion Attacks**, 2021 [[url](http://arxiv.org/abs/2104.12669v3)]

Providing explanation (with XAI) harms privacy. We study this risk for image-based model inversion attacks and identified several attack architectures able to reconstruct private image data from model explanations. XAI-aware inversion models designed to exploit the spatial knowledge in image explanations.

---
- **Learn2Perturb: an End-to-end Feature Perturbation Learning to Improve Adversarial Robustness**, 2020 [[url](http://arxiv.org/abs/2003.01090v2)] [[code](https://github.com/Ahmadreza-Jeddi/Learn2Perturb)]

Framework for producing perturbation duning training and inference at feature level. Tested on CIFAR-10 and CIFAR-100.

---
- **AdvRush: Searching for Adversarially Robust Neural Architectures**, 2021 [[url](http://arxiv.org/abs/2108.01289v2)]

AdvRush, a novel adversarial robustness-aware neural architecture search algorithm. It search for architecture (not weight tuning) with a smoother input loss landscape. Evaluated on CIFAR-10.

---
- **Adversarial Robustness Across Representation Spaces**, 2020 [[url](http://arxiv.org/abs/2012.00802v1)] [[code](https://github.com/tensorflow/neural-structured-learning/tree/master/research/multi_representation_adversary)]

consider the problem of training of deep neural networks that can be made simultaneously robust to perturbations applied in multiple natural representations spaces. For the case of image data, examples include the standard pixel representation as well as the representation in the discrete cosine transform (DCT) basis. Tested on CIFAR-10.

---
- **Low Curvature Activations Reduce Overfitting in Adversarial Training**, 2021 [[url](http://arxiv.org/abs/2102.07861v2)] [[code](https://github.com/vasusingla/low_curvature_activations)]

Overfitting is a dominant phenomenon in adversarial training. It shows that generalization gap is closely related to the choice of the activation function.

---
- **Defending against Universal Adversarial Patches by Clipping Feature Norms**, 2021 [[url](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Defending_Against_Universal_Adversarial_Patches_by_Clipping_Feature_Norms_ICCV_2021_paper.pdf)]

Mathematical explanation why universal adversarial patches usually lead to deep feature vector with very large norms in popular CNNs. Propose a new layer: Feature Norm Clipping (FNC) It can effectively improve the robustness of different CNNs towards white-box patch attacks while maintaining a satisfactory recognition accuracy for clean samples.

---
- **Reliably fast adversarial training via latent adversarial perturbation**, 2021 [[url](http://arxiv.org/abs/2104.01575v2)]

Deviation from the existing input-space-based adversarial training regime and propose a single-step latent adversarial training method (SLAT), which leverages the gradients of latent representation as the latent adversarial perturbation. Outperforms SOTA accelerated adversarial training methods.

---
- **On the Design of Black-box Adversarial Examples by Leveraging   Gradient-free Optimization and Operator Splitting Method**, 2019 [[url](http://arxiv.org/abs/1907.11684v4)] [[code](https://github.com/LinLabNEU/Blackbox_ADMM)]

Introduce a general framework for efficient black-box attack.

---
- **When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks**, 2019 [[url](http://arxiv.org/abs/1911.10695v3)] [[code](https://github.com/gmh14/RobNets)]

Insight from Neural Architecture Search (NAS) about robustness. Produce collection of networks called RobNet. Tested on CIFAR. Notably, RobNets substantially improve the robust accuracy.

---
- **AdvDrop: Adversarial Attack to DNNs by Dropping Information**, 2021 [[url](http://arxiv.org/abs/2108.09034v1)] [[code](https://github.com/RjDuan/AdvDrop)]

Human can easily recognize visual objects with lost information: even losing most details with only contour reserved, e.g. cartoon. For DDNs this is still a challenge. Opposite to previous works, this work explores the adversarial robustness of DNN models in a novel perspective by dropping imperceptible details to craft adversarial examples. This new type of adversarial examples is more difficult to be defended by current defense systems.

---
- **Towards Understanding the Generative Capability of Adversarially Robust   Classifiers**, 2021 [[url](http://arxiv.org/abs/2108.09093v2)]

Reformulation adversarial example generation, adversarial training, and image generation in terms of an energy function. A better adversarial training method, Joint Energy Adversarial Training (JEAT) which achieve SOTA robustness in CIFAR-10 and CIFAR-100.

---
- **Enhancing Adversarial Example Transferability with an Intermediate Level   Attack**, 2019 [[url](http://arxiv.org/abs/1907.10823v3)] [[code](https://github.com/CUAI/Intermediate-Level-Attack)]

introduce the Intermediate Level Attack (ILA), which attempts to fine-tune an existing adversarial example for greater black-box transferability by increasing its perturbation on a pre-specified layer of the source model. Provide some explanatory about the effect of optimizing for adversarial examples using intermediate feature maps.

---
- **Regularizing Neural Networks via Adversarial Model Perturbation**, 2020 [[url](http://arxiv.org/abs/2010.04925v4)] [[code](https://github.com/hiyouga/AMP-Regularizer)]

Proposes a new regularization scheme, based on the understanding that the flat local minima of the empirical risk cause the model to generalize better. This scheme is referred to as adversarial model perturbation (AMP), where instead of directly minimizing the empirical risk, an alternative "AMP loss" is minimized. AMP has strong theoretical justifications.

---
- **Adversarial Defense via Learning to Generate Diverse Attacks**, 2019 [[url](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.pdf)] [[code](http://github.com/YunseokJANG/l2l-da)]

Propose to utilize the generator to learn how to create adversarial examples. Unlike the existing approaches that create a one-shot perturbation by a deterministic generator, we propose a recursive and stochastic generator that produces much stronger and diverse perturbations. experiment results on MNIST and CIFAR-10.

---
- **Robustness and Generalization via Generative Adversarial Training**, 2021 [[url](http://arxiv.org/abs/2109.02765v1)]

Present Generative Adversarial Training. Instead of altering a single pre-defined aspect of images, we generate a spectrum of low-level, mid-level and high-level changes using generative models with a disentangled latent space. More robustness to various attacks and improved performance.

---
- **Enhancing Adversarial Robustness for Deep Metric Learning**, 2022 [[url](http://arxiv.org/abs/2203.01439v1)]

Propose Hardness Manipulation to efficiently perturb the training triplet till a specified level of hardness for adversarial training. An Intra-Class Structure loss term among benign and adversarial examples further improves model robustness and efficiency. Overwhelmingly outperforms the state-of-the-art defenses in terms of robustness, training efficiency, as well as performance on benign examples.

---
- **Learnable Boundary Guided Adversarial Training**, 2020 [[url](http://arxiv.org/abs/2011.11164v2)] [[code](https://github.com/dvlab-research/LBGAT)]

We use the model logits from one clean model to guide learning of another one robust model. Constrain logits from the robust model that takes adversarial examples as input and makes it similar to those from the clean model fed with corresponding natural data. This approach preserve high natural accuracy and also benefit model robustness. Experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet. SOTA robustness on CIFAR-100.

---
- **On the Robustness of Vision Transformers to Adversarial Examples**, 2021 [[url](http://arxiv.org/abs/2104.02610v2)] [[code](https://github.com/MetaMain/ViTRobust)]

Test the transformer under standard white-box and black-box attacks. Transferability of adversarial examples between CNNs and transformers. Security of a simple ensemble defense of CNNs and transformers. This Study encompasses multiple Vision Transformers, Big Transfer Models and CNN architectures trained on CIFAR-10, CIFAR-100 and ImageNet.

---
- **Feature Importance-aware Transferable Adversarial Attacks**, 2021 [[url](http://arxiv.org/abs/2107.14185v3)] [[code](https://github.com/hcguoO0/FIA)]

Feature Importance-aware Attack (FIA) disrupts important object-aware features that dominate model decisions consistently. This make adversarial attack model agnostic.

---
- **Adversarial Defense by Restricting the Hidden Space of Deep Neural   Networks**, 2019 [[url](http://arxiv.org/abs/1904.00887v4)] [[code](https://github.com/aamir-mustafa/pcl-adversarial-defense)]

propose to class-wise disentangle the intermediate feature representations of deep networks. Specifically, we force the features for each class to lie inside a convex polytope that is maximally separated from the polytopes of other classes. In this manner, the network is forced to learn distinct and distant decision regions for each class.

---
- **Augmented Lagrangian Adversarial Attacks**, 2020 [[url](http://arxiv.org/abs/2011.11857v2)] [[code](https://github.com/jeromerony/augmented_lagrangian_adversarial_attacks)]

Propose a white-box attack algorithm to generate minimally perturbed adversarial examples based on Augmented Lagrangian principles. Compare our attack to state-of-the-art methods on three datasets (MNIST, CIFAR-10 and ImageNet) and several models with competitive performances and with similar or lower computational complexity.

---
- **Improving Adversarial Robustness via Guided Complement Entropy**, 2019 [[url](http://arxiv.org/abs/1903.09799v3)] [[code](https://github.com/henry8527/GCE)]

New training paradigm called Guided Complement Entropy (GCE) that is capable of achieving "adversarial defense for free," which involves no additional procedures in the process of improving adversarial robustness. In addition to maximizing model probabilities on the ground-truth class like cross-entropy, we neutralize its probabilities on the incorrect classes along with a "guided" term to balance between these two terms. Better model robustness with even better performance compared to the commonly used cross-entropy training objective.

---
- **Hilbert-Based Generative Defense for Adversarial Examples**, 2019 [[url](http://dx.doi.org/10.1109/iccv.2019.00488)]

Improve upon PixelDefend by flatteing 2D image into 1D vector using Hilbert curve, thus the local features in neighboring pixels can be more effectively modeled.

---
