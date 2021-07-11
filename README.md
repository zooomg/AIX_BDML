# AIX_BDML

## Methods for interpreting and understanding deep neural networks, 2018

### Interpretation
- Activation Maximize
- AM + expert
- AM in code space

### Explanation
#### pooling
- sensitivity
- simple taylor decomposition
#### filtering
- deconvolution
- extension guided back prop
#### pooling & filtering
- LRP

## Explaining the black-box model: A survey of local interpretation methods for deep neural networks, 2021.01

### data-driven
#### perturebation-based
  - LIME, "Why Should I Trust You?": Explaining the Predictions of Any Classifier (2016, SIGKDD) [https://arxiv.org/abs/1602.04938]
  - SHAP, A Unified Approach to Interpreting Model Predictions (2017, NIPS) [https://arxiv.org/abs/1705.07874]
  - CXPlain: Causal Explanations for Model Interpretation under Uncertainty (2019, NIPS) [https://arxiv.org/abs/1910.12336]
  - ZFNet, Visualizing and Understanding Convolutional Networks (2013, ECCV) [https://arxiv.org/abs/1311.2901]
  - Interpretable Explanations of Black Boxes by Meaningful Perturbation (2017, ICCV) [https://arxiv.org/abs/1704.03296]
  - Real Time Image Saliency for Black Box Classifiers (2017, NIPS) [https://arxiv.org/abs/1705.07857]
  - RISE: Randomized Input Sampling for Explanation of Black-box Models (2018, ESA) [https://arxiv.org/abs/1806.07421]
  - RMA, Regional Multi-scale Approach for Visually Pleasing Explanations of Deep Neural Networks (2018, ICCV) [https://arxiv.org/abs/1807.11720]
  - FIDO, Explaining Image Classifiers by Counterfactual Generation (2019, ICLR) [https://arxiv.org/abs/1807.08024]
  - Understanding and Visualizing Deep Visual Saliency Models (2019, ICCV) [https://arxiv.org/abs/1903.02501]
  - CEM-MAF, Generating Contrastive Explanations with Monotonic Attribute Functions (2019, ICML) [https://arxiv.org/abs/1905.12698v1]
  - CEM, Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives (2018, NIPS) [https://arxiv.org/abs/1802.07623]
  - Understanding Deep Networks via Extremal Perturbations and Smooth Masks (2019, ICCV) [https://arxiv.org/abs/1910.08485]
  - Representer Point Selection for Explaining Deep Neural Networks (2018, NIPS) [https://arxiv.org/abs/1811.09720]
  - Examining CNN Representations with respect to Dataset Bias (2018, AAAI) [https://arxiv.org/abs/1710.10577]
#### adversarial-based
  - Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples (2018, NIPS) [https://arxiv.org/abs/1810.11580]
  - Anchors: High-Precision Model-Agnostic Explanations (2018, AAAI) [https://www.semanticscholar.org/paper/Anchors%3A-High-Precision-Model-Agnostic-Explanations-Ribeiro-Singh/1d8f4f76ac6534627ef8a1c24b9937d8ab2a5c5f]
  - IF, Understanding Black-box Predictions via Influence Functions (2017, ICML) [https://arxiv.org/abs/1703.04730]
  - On the Connection Between Adversarial Robustness and Saliency Map Interpretability (2019, ICML) [https://arxiv.org/abs/1905.04172]
  - Did the Model Understand the Question? (2018, ACL) [https://arxiv.org/abs/1805.05492]
  - FGVE, Interpretable and Fine-Grained Visual Explanations for Convolutional Neural Networks (2019, CVPR) [https://arxiv.org/abs/1908.02686]
#### concept-based
  - TCAV, Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (2018, ICML) [https://arxiv.org/abs/1711.11279]
  - ND, Interpreting Deep Visual Representations via Network Dissection (2018, TPAMI) [https://arxiv.org/abs/1711.05611]
  - Interpretable Basis Decomposition for Visual Explanation (2018, ECCV) [https://openaccess.thecvf.com/content_ECCV_2018/html/Antonio_Torralba_Interpretable_Basis_Decomposition_ECCV_2018_paper.html]
  - Net2Vec: Quantifying and Explaining how Concepts are Encoded by Filters in Deep Neural Networks (2018, ICCV) [https://arxiv.org/abs/1801.03454]
  - Towards Automatic Concept-based Explanations (2019, NIPS) [https://arxiv.org/abs/1902.03129]
  - This Looks Like That: Deep Learning for Interpretable Image Recognition (2019, NIPS) [https://arxiv.org/abs/1806.10574]
### model-driven
####  gradient-based
  - Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013, CVPR) [https://arxiv.org/abs/1312.6034]
  - Striving for Simplicity: The All Convolutional Net (2014, ICLR) [https://arxiv.org/abs/1412.6806]
  - Understanding Deep Image Representations by Inverting Them (2015, ICCV) [https://arxiv.org/abs/1412.0035]
  - Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients (2018, AAAI) [https://arxiv.org/abs/1711.09404]
  - Towards Explanation of DNN-based Prediction with Guided Feature Inversion (2018, SIGKDD) [https://arxiv.org/abs/1804.00506v1]
  - Integrated Grad, Axiomatic Attribution for Deep Networks (2017, ICML) [https://arxiv.org/abs/1703.01365]
  - Full-Gradient Representation for Neural Network Visualization (2019, NIPS) [https://arxiv.org/abs/1905.00780]
  - Top-down Neural Attention by Excitation Backprop (2018, IJCV) [https://arxiv.org/abs/1608.00507]
  - Interpretable Convolutional Neural Networks (2018, ICCV) [https://arxiv.org/abs/1710.00935]
#### corrlation-score
  - LRP, On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation (2015, PLoS One) [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140]
  - LRL, Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers (2016, ICANN) [https://arxiv.org/abs/1604.00825]
  - DeepLIFT, Not Just a Black Box: Learning Important Features Through Propagating Activation Differences (2016, ICML) [https://arxiv.org/abs/1605.01713]
  - CASO, Understanding Impacts of High-Order Loss Approximations and Features in Deep Learning Interpretation (2019, ICML) [https://arxiv.org/abs/1902.00407]
  - Choose Your Neuron: Incorporating Domain Knowledge through Neuron-Importance (2018, ECCV) [https://arxiv.org/abs/1808.02861]
#### class activation map
  - CAM, Learning Deep Features for Discriminative Localization (2016, ICCV) [https://arxiv.org/abs/1512.04150]
  - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2017, ICCV) [https://arxiv.org/abs/1610.02391]
  - U-CAM: Visual Explanation using Uncertainty based Class Activation Maps (2019, ICCV) [https://arxiv.org/abs/1908.06306]
  - GFI, Towards Explanation of DNN-based Prediction with Guided Feature Inversion (2018, SIGKDD) [https://arxiv.org/abs/1804.00506]
  - Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks (2018, WACV) [https://ieeexplore.ieee.org/document/8354201]

## On Interpretability of Artificial Neural Networks: A Survey, 2021.05

### Post-Hoc
#### Feature Analysis
- inverting method
  - Inverting Visual Representations with Convolutional Networks (2016, CVPR) [https://arxiv.org/abs/1506.02753]
  - Understanding Deep Image Representations by Inverting Them (2015, CVPR) [https://arxiv.org/abs/1412.0035]
  - Striving for Simplicity: The All Convolutional Net (2014, ) [https://arxiv.org/abs/1412.6806]
  - Deconvolution, Visualizing and Understanding Convolutional Networks (2014, ECCV) [https://arxiv.org/abs/1311.2901]
- activation maximization(deep dream)
  - Visualizing Higher-Layer Features of a Deep Network (2009, University of Montreal, vol.1341) [https://www.semanticscholar.org/paper/Visualizing-Higher-Layer-Features-of-a-Deep-Network-Erhan-Bengio/65d994fb778a8d9e0f632659fb33a082949a50d3]
  - Synthesizing the preferred inputs for neurons in neural networks via deep generator networks (2016, NIPS) [https://arxiv.org/abs/1605.09304]
  - Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space (2017, CVPR) [https://arxiv.org/abs/1612.00005]
  - Intriguing properties of neural networks (2013, ) [https://arxiv.org/abs/1312.6199]
- extract from each neurons
  - Network Dissection: Quantifying Interpretability of Deep Visual Representations (2016, CVPR) [https://arxiv.org/abs/1704.05796]
  - Visualizing and Understanding Recurrent Networks (2015, ) [https://arxiv.org/abs/1506.02078]
  - Convergent Learning: Do different neural networks learn the same representations? (2016, ICLR) [https://arxiv.org/abs/1511.07543]
  - Understanding Neural Networks Through Deep Visualization (2015, ) [https://arxiv.org/abs/1506.06579]
  - Object Detectors Emerge in Deep Scene CNNs (2014, ) [https://arxiv.org/abs/1412.6856]

#### Model Inspection
- influence function
  - A comparison of some robust estimators of the variogram for use in soil survey (2000, European journal of soil science, vol.51) [https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1365-2389.2000.00280.x]
  - Understanding Black-box Predictions via Influence Functions (2017, ICML) [https://arxiv.org/abs/1703.04730]
- ?
  - Towards Transparent Systems: Semantic Characterization of Failure Modes (2014, ECCV) [https://link.springer.com/chapter/10.1007/978-3-319-10599-4_24]
  - Identifying Unknown Unknowns in the Open World: Representations and Policies for Guided Exploration (2017, AAAI) [https://arxiv.org/abs/1610.09064]
  - Examining CNN Representations with respect to Dataset Bias (2018, AAAI) [https://arxiv.org/abs/1710.10577]
- data routes
  - Interpret Neural Networks by Identifying Critical Data Routing Paths (2018, CVPR) [https://ieeexplore.ieee.org/document/8579026]
- concept-based
  - Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) (2017, ) [https://arxiv.org/abs/1711.11279]
- graph 
  - Graph Structure of Neural Networks (2020, ICML) [https://arxiv.org/abs/2007.06559]

#### Saliency
- leave-one-out
  - omission value
    - 1-cosine distance
  - shapley value
- resort to gradient
  - Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (2013, ) [https://arxiv.org/abs/1312.6034]
  - SmoothGrad: removing noise by adding noise (2017, ) [https://arxiv.org/abs/1706.03825]
  - IntegratedGrad, Axiomatic Attribution for Deep Networks (2017, ICML) [https://arxiv.org/abs/1703.01365]
- LRP
  - On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation (2015, PloS one, vol.10) [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140]
  - Explaining Recurrent Neural Network Predictions in Sentiment Analysis (2017, ) [https://arxiv.org/abs/1706.07206]
  - Explaining NonLinear Classification Decisions with Deep Taylor Decomposition (2017, Pattern Recognition, vol.65) [https://arxiv.org/abs/1512.02479]
- CAM, class activation map

#### Proxy
- direct extraction
  - decision tree
    - Extracting decision trees from trained neural networks (2019, Pattern Recognition, vol 32) [https://www.sciencedirect.com/science/article/pii/S0031320398001812]
    - Global Model Interpretation via Recursive Partitioning (2018, IEEE HPCC/SmartCity/DSS) [https://arxiv.org/abs/1802.04253]
  - rule extraction
    - decompositional method
      - Understanding Neural Networks via Rule Extraction (2017, IJCAI) [https://www.semanticscholar.org/paper/Understanding-Neural-Networks-via-Rule-Extraction-Setiono-Liu/2ba745184e671c5f32fab44429716cf136121d59]
    - pedagogical method
      - Neural network explanation using inversion (2007, Neural Networks, vol 20) [https://www.sciencedirect.com/science/article/pii/S0893608006001730]
      - Extracting rules from artificial neural networks with distributed representations (1995, NIPS) [https://dl.acm.org/doi/10.5555/2998687.2998750]
      - VIA(Validity Interval Analysis) (2000, IEEE Transaction on neural networks) [https://pubmed.ncbi.nlm.nih.gov/18249807/]
    - fuzzy logic system
      - ANFIS (1993, IEEE Transactions on Systems, Man, and Cybernetics) [https://ieeexplore.ieee.org/document/256541]
      - RBF (1994, Proc.Fuzzy) [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4907]
      - Fuzzy Logic Interpretation of Quadratic Networks (2020, Neurocomputing, vol. 374) [https://arxiv.org/abs/1807.03215]
      - A New Type of Neurons for Machine Learning (2018, International journal for numerical methods in biomedical engineering, vol.374) [https://arxiv.org/abs/1704.08362]
- knowledge-distillation
  - decision tree
    - Extracting tree-structured representations of trained networks (1995, NIPS) [https://dl.acm.org/doi/10.5555/2998828.2998832]
    - Beyond Sparsity: Tree Regularization of Deep Models for Interpretability (2018, AAAI) [https://arxiv.org/abs/1711.06178]
  - decision set
    - Interpretable & Explorable Approximations of Black Box Models (2017, ) [https://arxiv.org/abs/1707.01154]
  - global additive model
    - Learning Global Additive Explanations for Neural Nets Using Model Distillation (2018, ) [https://arxiv.org/abs/1801.08640]
  - simpler network
    - Harnessing Deep Neural Networks with Logic Rules (2016, ) [https://arxiv.org/abs/1603.06318]
- provide a local explainer
  - LIME (2016, SIGKDD) [https://arxiv.org/abs/1602.04938]
  - Anchor (2018, AAAI) [https://www.semanticscholar.org/paper/Anchors%3A-High-Precision-Model-Agnostic-Explanations-Ribeiro-Singh/1d8f4f76ac6534627ef8a1c24b9937d8ab2a5c5f]
  - LORE (2018, ) [https://arxiv.org/abs/1805.10820]

#### Advanced Mathematical/Physical Analysis

#### Explaining-by-Case
- K-Nearest Neighbor Algorithm
- Counter factual case

#### Explaining-by-Text
- Neural image captioning
  - CNN + bidirection-RNN
  - CNN + attention-RNN

### Ad-Hoc
#### Interpretable Representation
- decomposability
  - InfoGAN (NIPS, 2016) [https://arxiv.org/abs/1606.03657]
  - (CVPR, 2017) [https://arxiv.org/abs/1706.04313]
  - (IEEE Access, 2016) [https://ieeexplore.ieee.org/document/7733110]
  - (CVPR, 2018) [https://arxiv.org/abs/1710.00935]
- monotonicity (NIPS, 2017) [https://arxiv.org/abs/1709.06680]
- non-negativity (IEEE Transactions on Neural Networks and Learning Systems, 2014) [https://ieeexplore.ieee.org/document/6783731/]
- sparsity (AAAI, 2018) [https://arxiv.org/abs/1711.08792]
- human-in-the-loop prior (NIPS, 2018) [https://arxiv.org/abs/1805.11571]

#### Model Renovation
- PLNN (KDD, 2018) [https://arxiv.org/abs/1802.06259]
- Soft-AE (IEEE transaction on Computational Imaging, 2020) [https://arxiv.org/abs/1812.11675]
- L. Fan (NIPS, 2017) [https://arxiv.org/abs/1710.10328]
- D. A. Melis and T. Jaakkola (NIPS, 2018) [https://arxiv.org/abs/1806.07538]
- C. Li et al. (IEEE transaction on Computer Vision and Pattern Recognition, 2018) [https://arxiv.org/abs/1801.03399]
- T. Wang (ICML, 2019) [https://arxiv.org/abs/1802.04346]
- FA-RNN (EMNLP, 2020) [https://aclanthology.org/2020.emnlp-main.258/]

## Interpretable Deep Learning: Interpretation, Interpretability, Trustworthiness, and Beyond, 2021.05

![image](https://user-images.githubusercontent.com/11240557/125038709-b0440800-e0d0-11eb-93f8-c25ea7953647.png)

### types of models
- Model-agnostic
- Differentiable model
- Specific model

### representations of interpretation
- Feautre(Importance)
- Model Response
- Model Rationale Process
- Dataset

![image](https://user-images.githubusercontent.com/11240557/125038219-1ed49600-e0d0-11eb-8fb0-29314a305a9b.png)
### relation between the interpretation algorithm and the model
- Closed-form
- Composition
- Dependence
- Proxy
