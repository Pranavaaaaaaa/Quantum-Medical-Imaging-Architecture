# ⚛️ Hybrid Quantum-Classical Diagnostic System (Case Study)

> **⚠️ Legal Disclaimer:** This project was architected and developed as part of the Unisys Innovation Program (UIP 17). The source code, proprietary datasets, and trained model weights are confidential and belong to Unisys. This repository serves strictly as a technical case study and architectural overview of the system I designed.

## 📌 Executive Summary
Standard AI models in medical imaging act as "black boxes," making them dangerous in clinical environments. This project solves the high-mortality diagnostic overlap between **Tuberculosis and Silicosis** by engineering a highly transparent, multi-stage pipeline. The system leverages classical computer vision for spatial extraction, parameterized quantum layers for complex pattern correlation, and a secondary "Teacher AI" to mathematically audit the primary model's attention in real-time.

## 🏗️ System Architecture
```mermaid 
    flowchart TD

%% ============================================================
%% EXTERNAL ENTITIES
%% ============================================================

User["👨‍⚕️ User / Physician"]
Input[("Input Chest X-Ray<br/>DICOM / JPG / PNG")]
Response[("JSON Triage Response<br/>+ Encoded Visuals")]

User --> Input

%% ============================================================
%% 1. DATA INGESTION
%% ============================================================

subgraph DI["1️⃣ Data Ingestion & Transformation"]

A["Receive Raw Image File<br/>(FastAPI UploadFile)"]

B["Convert to PyTorch RGB Tensor"]

C["Resize to 512×512<br/>Center Crop"]

D["ImageNet Normalization<br/>mean=[0.485,0.456,0.406]<br/>std=[0.229,0.224,0.225]"]

A --> B --> C --> D

end

Input --> A

%% ============================================================
%% 2. SEGMENTATION
%% ============================================================

subgraph SEG["2️⃣ Classical UNet Lung Segmentation"]

E["UNet Segmentation Model<br/>(unet.pth)"]

F["Generate Binary Lung Mask"]

G["Apply Mask<br/>(element-wise multiplication)"]

H[("Segmented Lung Image<br/>Black Background")]

E --> F --> G --> H

end

D --> E
D --> G

%% ============================================================
%% 3. HYBRID QUANTUM-CLASSICAL INFERENCE
%% ============================================================

subgraph HQC["3️⃣ Hybrid Quantum-Classical Inference<br/>Monte Carlo Interrogation"]

Loop["Loop Control<br/>i = 1 → N<br/>N = 30 Samples"]

FE["DenseNet121 Feature Extractor<br/>Bottom 120 Layers"]

Drop["Random Neuron Dropout<br/>p = 0.4"]

FV[("Classical Feature Vector<br/>Length = 512")]

Enc["Amplitude Embedding<br/>(PennyLane)"]

QC["9-Qubit Strongly Entangling<br/>Quantum Circuit<br/>n_layers = 5"]

QM["Pauli-Z Expectation Values"]

Logits[("Raw Logits<br/>4 Classes")]

Loop --> FE
FE --> Drop
Drop --> FV
FV --> Enc
Enc --> QC
QC --> QM
QM --> Logits

Logits --> Loop

end

H --> Loop

%% ============================================================
%% MONTE CARLO AGGREGATION
%% ============================================================

subgraph MC["Monte Carlo Aggregation"]

TS["Temperature Scaling<br/>T = 0.2"]

SM["Softmax"]

COL["Collate 30 Probability Vectors"]

TS --> SM --> COL

end

Loop -. "Iterate 30 stochastic passes" .-> TS

%% ============================================================
%% 4. TRIAGE DECISION MATRIX
%% ============================================================

subgraph TRIAGE["4️⃣ Statistical Triage Decision Matrix"]

AGG["Statistical Aggregation<br/>Mean + Standard Deviation"]

Mean[("Mean Probability (%)<br/>per Class")]

Var[("Monte Carlo Variance<br/>± %")]

Pred["Predicted Class<br/>argmax(mean)"]

Rule{"TRIAGE<br/>DECISION RULES"}

D1{"Predicted ≠ Normal?"}

D2{"Normal AND<br/>Confidence ≤ 70%<br/>OR<br/>Uncertainty ≥ 4% ?"}

Green["🟢 GREEN LIGHT<br/>Routine Clearance"]

Yellow["🟡 YELLOW LIGHT<br/>Manual Radiologist Review"]

Red["🔴 RED LIGHT<br/>URGENT Triage Queue"]

AGG --> Mean
AGG --> Var
AGG --> Pred

Mean --> Rule
Var --> Rule
Pred --> Rule

Rule --> D1

D1 -- YES --> Red
D1 -- NO --> D2

D2 -- YES --> Yellow
D2 -- NO --> Green

end

COL --> AGG

%% ============================================================
%% 5. EXPLAINABLE AI
%% ============================================================

subgraph XAI["5️⃣ Explainable AI<br/>Quantum Attention Grad-CAM Engine"]

Grad["Gradient Flow<br/>(Backward)"]

CAM["Grad-CAM Weighting of<br/>DenseNet Feature Maps"]

Relu["ReLU"]

Up["Upsample CAM<br/>7×7 → 512×512"]

Heat["Generate Grad-CAM Heatmap"]

Grad --> CAM
CAM --> Relu
Relu --> Up
Up --> Heat

end

Pred --> Grad
FV -. "Gradient Backpropagation" .-> Grad

%% ============================================================
%% 6. OUTPUT PACKAGING
%% ============================================================

subgraph OUT["6️⃣ Output Packaging"]

Encode["Encode UNet Mask + Grad-CAM<br/>to Base64"]

JSON["Assemble JSON Response"]

Encode --> JSON

end

Heat --> Encode
F --> Encode

Green --> JSON
Yellow --> JSON
Red --> JSON

Mean -.-> JSON
Var -.-> JSON

JSON --> Response

%% ============================================================
%% STYLING
%% ============================================================

classDef process fill:#ffffff,stroke:#2563eb,stroke-width:2px,color:#000000;
classDef data fill:#ffffff,stroke:#ca8a04,stroke-width:2px,color:#000000;
classDef decision fill:#ffffff,stroke:#be123c,stroke-width:2px,color:#000000;
classDef success fill:#ffffff,stroke:#15803d,stroke-width:3px,color:#000000;
classDef warning fill:#ffffff,stroke:#d97706,stroke-width:3px,color:#000000;
classDef danger fill:#ffffff,stroke:#b91c1c,stroke-width:3px,color:#000000;

class A,B,C,D,E,F,G,FE,Drop,Enc,QC,QM,TS,SM,COL,AGG,Grad,CAM,Relu,Up,Heat,Encode,JSON process;
class Input,Response,H,FV,Logits,Mean,Var data;
class Rule,D1,D2 decision;
class Green success;
class Yellow warning;
class Red danger;
```

## ⚙️ Core Engineering Modules

### 1. The Pre-Processing Isolation (U-Net)
* **The Problem:** Chest X-rays contain massive amounts of irrelevant noise (clavicles, ribs, pacemakers) that confuse quantum classifiers.
* **The Solution:** Implemented a PyTorch U-Net to mathematically isolate and "shrink-wrap" the lung boundaries, applying a 10-pixel padding margin to preserve edge pathologies while eliminating background noise.

### 2. The Quantum Brain (DenseNet + PennyLane)
* **The Architecture:** Deployed a DenseNet backbone to extract a compressed texture-feature vector, fed into a 9-qubit quantum node via Amplitude Embedding.
* **The Execution:** Utilized Pauli-Z operator expectation values `qml.expval(qml.PauliZ(i))` to collapse the quantum circuit into distinct medical classifications, leveraging quantum entanglement to map highly complex disease correlations.

### 3. The Clinical Triage Algorithm
* **The Problem:** AI models are notoriously overconfident, even when hallucinating. 
* **The Solution:** Engineered a safety-first routing system using Monte Carlo Dropout. By keeping dropout active during inference, the system calculates epistemic variance (standard deviation) across multiple forward passes. High-variance predictions automatically route the patient to a "Yellow" queue for manual human review.

### 4. The Dual-AI Explainability Audit (Faster R-CNN)
* **The Problem:** Doctors cannot trust a Grad-CAM heatmap alone, as it might highlight random background pixels.
* **The Solution:** Trained an independent, two-stage Faster R-CNN "Virtual Radiologist" to generate clinical bounding boxes. The system calculates the Intersection-over-Union (IoU) between the Quantum AI's Grad-CAM hotspot and the Teacher's bounding box in real-time. If the IoU drops below 5% (the Saliency vs. Localization threshold), it flags a "Hallucination Risk."

## 💻 The Practitioner Interface (React & FastAPI)
Built a production-ready Electronic Health Record (EHR) module featuring:
* An asynchronous FastAPI backend to handle deep learning tensors and base64 image generation.
* A secure React.js portal with a 3-panel automated diagnostic viewer (Raw -> U-Net -> Grad-CAM).
* A persistent sidebar logging patient session history, time-stamps, and IoU verification badges for clinical auditing.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision (DenseNet, U-Net, Faster R-CNN)
* **Quantum Computing:** PennyLane (QML)
* **Computer Vision:** OpenCV, Grad-CAM, IoU Mathematics
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** React.js, Tailwind CSS, Lucide Icons
