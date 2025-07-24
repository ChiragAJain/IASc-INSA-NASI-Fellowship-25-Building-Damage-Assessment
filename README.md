# Lightweight Convolutional Neural Network Models for Disaster Damage Assessment

## Project Context: Summer Research Fellowship Programme (SRFP)

Every year, three of India's science academies—the Indian Academy of Sciences (IASc), the Indian National Science Academy (INSA), and The National Academy of Sciences, India (NASI)—jointly conduct a paid summer fellowship programme. This fellowship selects undergraduate students and teachers based on their academic performance, projects, and recommendations. Under this programme, selected individuals work on research projects with an academy guide at various research institutes across India.

<hr>

Under this fellowship, I worked with **Dr. Venkadachalam Ramesh** (Dept. of Mathematics, Central University of Tamil Nadu) on this research project. The goal was to propose and develop a lightweight Convolutional Neural Network (CNN) model useful for embedded systems like drones and satellites. This model can assess post-disaster building damage to help notify Humanitarian Assistance and Disaster Recovery (HADR) operations, enabling them to provide quick aid to individuals in dire need.

---

## 📖 Overview

In the critical moments following a disaster, rapid and accurate assessment of building damage is paramount for effective recovery operations. This project introduces lightweight CNN models designed for real-time, post-disaster building damage classification. The models categorize building damage into four distinct classes:
* No Damage
* Minor Damage
* Major Damage
* Destroyed

Inspired by the ResNet18 architecture, three custom CNN variants were developed and benchmarked against established models, ResNet18 and MobileNetV2, to evaluate their performance, memory footprint, and inference speed.

---

## ✨ Key Features

* **Lightweight Architectures:** Custom CNN models designed for deployment on resource-constrained embedded systems.
* **Four-Class Damage Assessment:** Classifies buildings into `no-damage`, `minor-damage`, `major-damage`, and `destroyed` categories.
* **Comparative Analysis:** A comprehensive comparison of three proposed models (Standard, Attention-Based, Simplified) against ResNet18 and MobileNetV2.
* **Efficient Training:** Employs techniques like Focal Loss to handle class imbalance, along with an Adam optimizer and ReduceLROnPlateau scheduler for robust training.
* **Rigorous Evaluation:** Models were trained and evaluated using 8-fold cross-validation on the extensive xView2 dataset.

---

## 🔧 Methodology

### Dataset

The models were trained on the **xView2 dataset**, a large, publicly available collection of satellite imagery for building damage assessment.

* **Data Source:** The dataset contains pre- and post-disaster satellite images from 19 natural disasters. For this study, only post-disaster images were utilized.
* **Preprocessing:** Building-focused patches of size $128 \times 128$ were generated from the post-disaster images, resulting in an initial set of 3,04,370 patches.
* **Augmentation & Balancing:** To address class imbalance, the dataset was augmented with techniques like flips, rotations, and brightness adjustments. The dominant `no-damage` class was then undersampled to balance the dataset, which expanded to a total of 591,983 patches.

### Model Architecture

The core architecture is inspired by ResNet, featuring a stage-wise configuration with two residual blocks per stage. To enhance efficiency, the design incorporates $1 \times 1$ bottleneck convolutional blocks from ResNet-50, which reduces the parameter count and makes the models more suitable for embedded systems.

Three variants were developed:
1.  **Standard CNN:** The main proposed model, utilizing the bottleneck residual block design.
2.  **Attention-Based CNN:** This variant integrates a Squeeze-and-Excitation (SE) block to help the model focus on relevant building features and suppress background noise.
3.  **Simplified CNN:** A shallower version with only three stages, designed for maximum efficiency with the lowest memory usage and fastest inference time.

### Training

All models were trained under identical conditions to ensure a fair comparison.
* **Framework:** PyTorch
* **Validation:** 8-fold cross-validation.
* **Loss Function:** Focal Loss was used to effectively handle the inherent class imbalance in the dataset.
* **Optimizer:** Adam.
* **Scheduler:** ReduceLROnPlateau.
* **Hardware:** Training was accelerated using CUDA-enabled GPUs.

---

## 📊 Results

The models were evaluated based on classification accuracy, precision, recall, F1-score, memory usage, and inference speed.

| Metrics                   | Standard CNN | Attention-Based CNN | Simplified CNN | ResNet18  | MobileNetV2 |
| :------------------------ | :----------: | :-----------------: | :------------: | :-------: | :---------: |
| **Accuracy (%)** |    83.31     |        83.05        |     82.29      |   83.52   |  **86.09** |
| **Precision (%)** |    65.98     |        65.53        |     65.28      |   66.34   |  **69.73** |
| **Recall (%)** |    72.96     |        72.38        |     73.84      |   73.88   |  **75.42** |
| **F1-Score (%)** |    69.29     |        68.79        |     69.03      |   69.99   |  **72.47** |
| **Memory Usage (MiB)** |    28.71     |        31.08        |    **17.66** |   63.33   |    58.61    |
| **Parameters (Millions)** |     3.77     |         4.12        |     **~1** |   11.17   |    2.22     |
| **Inference Time (Batch/s)** |    37.17     |        34.67        |    **38.69** |   35.42   |    33.55    |

### Key Findings

* The **Standard CNN** offered performance very close to ResNet18 while using less than half the memory (28.71 MiB vs. 63.33 MiB) and having a faster inference time.
* The **Simplified CNN** proved to be the most efficient model, with the lowest memory footprint (17.66 MiB) and the highest inference speed (38.69 batches/sec).
* **MobileNetV2** achieved the highest overall classification accuracy and F1-score, though it was less memory- and speed-efficient than the proposed Standard and Simplified models.
* A common challenge across all models was distinguishing the `minor-damage` class from the `no-damage` class, highlighting a limitation of using only overhead satellite imagery.

---

## Set Up

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ChiragAJain/IASc-INSA-NASI-Fellowship-25-Building-Damage-Assessment.git](https://github.com/ChiragAJain/IASc-INSA-NASI-Fellowship-25-Building-Damage-Assessment.git)
    cd IASc-INSA-NASI-Fellowship-25-Building-Damage-Assessment
    ```
2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the xView2 Dataset:**
    * Visit [xView2.org](https://www.xview2.org/)
    * Sign Up for the Challenge
    * After logging in, click on download data and agree to their terms and conditions
    * Download the datasets from the links given under `Datasets from the Challenge` section

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## ✍️ Credit

If you use the code or findings from this project in your research, please appropriately credit the author as __Chirag Jain__.
