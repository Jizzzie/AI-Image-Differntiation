# AI-Image-Differntiation

This project investigates the performance of various classification models in
the context of image classification tasks. We explore traditional approaches such as
logistic regression and linear discriminant analysis (LDA), alongside advanced
convolutional neural networks (CNNs) like MobileNetV2 and ReXNet_150. Through
extensive experimentation and evaluation on a given dataset, we compare the
accuracy and efficiency of these models. Our findings reveal the superior
performance of CNN architectures, particularly MobileNetV2 and ReXNet_150, in
achieving high accuracy in image classification tasks. Furthermore, we discuss
potential avenues for future research, including model refinement, task expansion,
and real-world deployment, to enhance the project's applicability and impact.

One of the primary challenges in image differentiation lies in the inherent
complexity and variability of visual content. AI-generated images can closely mimic
real-world scenes, making it difficult for traditional image recognition algorithms to
accurately classify them. Additionally, the diversity of subjects and styles present in
both web-scraped and AI-generated images further complicates the task of
differentiation. Furthermore, the ethical considerations surrounding the use of AI-
generated content raise questions about the reliability and authenticity of digital
media.

The dataset used in this project comprises a diverse collection of images sourced
from two distinct channels: web scraping and AI-generated content. The web-scraped
images encompass a wide range of subjects, including landscapes, portraits, paintings,
and psychedelic art. In contrast, the AI-generated images exhibit varying degrees of
realism and abstraction, covering similar topics as their real-world counterparts. By
combining images from both sources, the dataset offers a comprehensive and
representative sample of digital imagery, suitable for training and testing machine
learning models for image differentiation.

**SOURCE:** https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

**Software Requirement**
1) PyTorch
2) Tensorflow
3) GPU
4) CUDA Toolkit

**Models Implemented**
a. **Logistic Regressor:** LR models the probability of a binary outcome by fitting a logistic function to
the input features. It operates on the principle of maximizing the likelihood of the
observed data given the model parameters, making it a discriminative algorithm. LR is
well-suited for linearly separable data and offers simplicity and interpretability.
However, it assumes a linear relationship between the input features and the log-odds
of the target variable, which may limit its effectiveness in capturing complex nonlinear
relationships within the data.

b. **Linear Discriminant Analysis:** On the other hand, LDA is a generative algorithm that models the distribution
of the input features for each class and uses Bayes' theorem to compute posterior
probabilities of class membership. It assumes that the input features are normally
distributed and that the class-conditional densities have a common covariance matrix.
LDA seeks to find the linear combination of features that best separates the classes
while maximizing the between-class variance and minimizing the within-class variance.
LDA is particularly effective when the classes are well-separated and the assumptions
of normality and equal covariance matrices hold, providing robust performance in
many real-world scenarios.

However, both logistic regression and LDA may face limitations when dealing
with high-dimensional data such as images. In image classification tasks, the input
features correspond to pixel values, resulting in a very high-dimensional feature space.
Logistic regression and LDA struggle to capture complex spatial patterns and
hierarchical relationships present in images due to their linear nature and assumptions
about feature distributions.

Convolutional Neural Networks (CNNs) offer a compelling alternative by
automatically learning hierarchical representations of image features through
successive convolutional and pooling layers. CNNs can effectively capture spatial
hierarchies and patterns in images, enabling them to achieve superior performance in
image classification tasks compared to traditional linear classifiers like logistic
regression and LDA. Additionally, CNNs can handle high-dimensional image data more
efficiently through parameter sharing and local connectivity, making them well-suited
for image classification tasks in practice.

c. **Simple CNN:** A Simple Convolutional Neural Network (CNN) is a foundational architecture
widely used in image classification tasks. Convolutional layers apply learnable filters to
input images, capturing local patterns and features. Activation functions introduce
non-linearity, allowing the network to learn complex mappings between inputs and
outputs. Simple CNNs are relatively easy to implement and understand, making them
suitable for educational purposes and baseline comparisons. However, they may
struggle with large-scale datasets and complex image recognition tasks due to their
limited capacity and depth.

d. **MobileNetV2:** MobileNetV2 is a lightweight and efficient convolutional neural network
architecture designed for mobile and embedded devices with resource constraints..
MobileNetV2 achieves a good balance between accuracy and efficiency, making it
well-suited for deployment on devices with limited computational resources. It has
been widely adopted in applications such as image classification, object detection, and
semantic segmentation on mobile platforms and edge devices.

e. **ReXNet150** ReXNet150 is a state-of-the-art convolutional neural network architecture
designed to achieve high accuracy with efficient computational resource usage. It
introduces novel design principles such as Rectified Efficient Bottlenecks (ReB) and
Parametric Rectified Linear Unit (PReLU) activations. ReXNet150 leverages depthwise
separable convolutions and squeeze-and-excitation blocks to further improve model
efficiency and accuracy. With its superior performance on large-scale image
classification benchmarks, ReXNet150 is well-suited for demanding tasks such as fine-
grained recognition, medical imaging, and remote sensing where both accuracy and
computational efficiency are critical.

**Accuracy:**
The accuracy scores reveal the performance of various classification models on
a given dataset. Logistic Regression and Linear Discriminant Analysis (LDA)
achieved accuracies of 57% and 49%, respectively, showcasing their limitations in
capturing complex patterns within the data. Simple CNN attained a modest
accuracy of 61%, indicating its effectiveness for basic image classification tasks but
potentially lacking the depth to handle more challenging datasets. In contrast,
MobileNetV2 and ReXNet_150 exhibited significantly higher accuracies of 84% and
83%, respectively, underscoring the superiority of more advanced convolutional
neural network architectures for image classification.
