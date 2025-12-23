# **FlareSatFormer: A Transformer-Based Deep Learning Model for Gas Flare Segmentation in Landsat 8 Satellite Imagery**

## Introduction

**FlareSatFormer** is a deep-learning semantic segmentation project for **gas flare segmentation** using **Landsat 8 satellite imagery**, based on a transformer-driven approach. Gas flaring is commonly used in oil and gas operations for safety reasons; however, large-scale flaring contributes significantly to greenhouse gas emissions, making accurate and scalable monitoring essential for environmental assessment and mitigation.

Satellite remote sensing provides global, multispectral, and openly accessible data, yet open datasets and modern deep-learning solutions tailored for gas flare detection remain scarce. **FlareSatFormer extends the original FlareSat project**, adopting **vision transformers (ViT)** to improve contextual understanding and segmentation performance.

<p align="center">
  <img src="assets/gas_flare_araucaria.png" alt="Gas Flare" width="100%">
  <br>
  <em>Example of gas flaring at the REPAR (Refinaria Presidente Getúlio Vargas) facility in Araucária, Paraná, Brazil.</em>
</p>

This work uses **FlareSat as a reference baseline**, available at:  
https://github.com/marycamila184/flaresat/tree/main

The dataset used in **FlareSatFormer** is an **improved and expanded version** of the original FlareSat dataset. It contains **11,505 labeled image patches** (256 × 256 pixels) covering **5,508 facilities across 94 countries**, including both onshore and offshore oil and gas production sites. To enhance robustness and reduce false positives, the dataset also includes visually similar thermal sources such as **wildfires, active volcanoes, and reflective urban areas**.

<p align="center">
  <img src="assets/active_fire_1.png" alt="FlareSatFormer" width="90%">
  <br>
  <em>Example of a gas flare on Landsat 8. (a) RGB patch; (b) B7 band patch; (c) flare segmented pixels shown in red.</em>
</p>

Unlike convolutional baselines, **FlareSatFormer adopts transformer-based semantic segmentation models**, enabling improved modeling of long-range dependencies and richer feature representations in multispectral satellite imagery. This repository provides a complete and reproducible research pipeline, including **dataset preparation**, **training**, **evaluation**, and **inference**.

---

## Features

- Transformer-based gas flare detection and segmentation from Landsat 8 imagery  
- Improved and expanded dataset derived from the original **FlareSat**  
- **SegFormer-B0 transformer architecture** for semantic segmentation  
  - https://github.com/NVlabs/SegFormer  
- **Fine-tuning using Prithvi foundation models** for Earth observation tasks  
  - https://github.com/NASA-IMPACT/Prithvi-EO-Foundation-Models  
- End-to-end pipelines for **training, evaluation, and inference**  
- Spatial cross-validation across continents to reduce geographic bias  
