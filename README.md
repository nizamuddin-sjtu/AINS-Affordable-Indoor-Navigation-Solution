<h1 align="center">AINS: Affordable Indoor Navigation Solution</h1>

<p align="center">
  <a href="https://doi.org/10.1109/I2CT61223.2024.10544260"><img src="https://img.shields.io/badge/Paper-IEEE_I2CT-2f6f9f.svg" alt="Paper"></a>
  <a href="AINS.py"><img src="https://img.shields.io/badge/Project-AINS.py-555555.svg" alt="Project entry file"></a>
  <a href="https://scholar.google.com/citations?user=bvyKhaEAAAAJ&hl=en"><img src="https://img.shields.io/badge/Publications-Google_Scholar-4285F4.svg" alt="Google Scholar"></a>
  <a href="https://www.kaggle.com/nizamuddinmaitlo"><img src="https://img.shields.io/badge/Profile-Kaggle-20BEFF.svg" alt="Kaggle profile"></a>
</p>

<p align="center"><b>Nizamuddin Maitlo, Nooruddin Noonari, Kaleem Arshid, N. Ahmed, and Sathishkumar Duraisamy</b></p>

<p align="center">Mono-camera line-color identification for low-cost indoor vehicle navigation.</p>

## 🔥 Overview

AINS is a compact computer-vision pipeline for following a colored indoor path with a monocular camera. Frames are smoothed, converted to HSV, thresholded for the target line color, and reduced to the largest contour. The contour center is converted into a simple left, right, or straight steering recommendation.

## ✨ Features

- Live mono-camera capture through OpenCV.
- Gaussian filtering and HSV-based yellow-line segmentation.
- Largest-contour extraction and central-moment estimation.
- Lightweight angle-based steering guidance.

## 🧪 Method and protocol

- The current script opens the default camera with `cv2.VideoCapture(0)`.
- Yellow is detected using fixed HSV thresholds defined in `AINS.py`.
- The largest detected contour is treated as the navigation path.
- Press `q` in the OpenCV window to stop the program.

## 📁 Repository contents

| File | Purpose |
|---|---|
| `AINS.py` | OpenCV perception and steering pipeline |

## 🛠️ Setup

Install the runtime dependencies:

~~~bash
python -m pip install opencv-python numpy
~~~

## 📦 Data and inputs

| Resource | Purpose | Availability |
|---|---|---|
| Camera or local video stream | Frames for colored-line detection and navigation | User-provided input |

No external training dataset is required by the current implementation.

## 🚀 Running the project

Connect a camera, then run:

~~~bash
python AINS.py
~~~

To use a recorded video, replace `cv2.VideoCapture(0)` with the video-file path.

## ♻️ Reproducibility

- Record the Python and library versions used for each run.
- Keep preprocessing, splits, thresholds, and random seeds fixed when comparing results.
- Do not commit private input data, generated model weights, or machine-specific paths.
- Revalidate results when the dataset, sensor, operating environment, or dependency versions change.

## 📚 Paper information

This repository provides the compact vision pipeline associated with the published AINS study.

| Publication | Venue | Link |
|---|---|---|
| AINS: Affordable Indoor Navigation Solution via Line Color Identification Using Mono-Camera for Autonomous Vehicles | 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), 1–7 | [DOI](https://doi.org/10.1109/I2CT61223.2024.10544260) |

## ⭐ Citation

~~~bibtex
@inproceedings{maitlo2024ains,
  title     = {AINS: Affordable Indoor Navigation Solution via Line Color Identification Using Mono-Camera for Autonomous Vehicles},
  author    = {Maitlo, Nizamuddin and Noonari, Nooruddin and Arshid, Kaleem and Ahmed, N. and Duraisamy, Sathishkumar},
  booktitle = {2024 IEEE 9th International Conference for Convergence in Technology (I2CT)},
  pages     = {1--7},
  year      = {2024},
  doi       = {10.1109/I2CT61223.2024.10544260}
}
~~~

A machine-readable [CITATION.cff](CITATION.cff) file is included for GitHub's citation interface.



## ⚠️ Scope and limitations

The fixed HSV thresholds assume a visible yellow path and may require retuning for different cameras, exposure settings, floor materials, shadows, or line colors. The script prints steering recommendations but does not interface with a physical motor controller or provide obstacle avoidance.

## 📄 License

No standalone code-license file is currently included in this repository. The publication remains subject to the publisher terms.

## 🤝 Acknowledgements

This project uses open-source Python libraries and the data or inputs described above. We thank the original dataset, framework, and software contributors.
