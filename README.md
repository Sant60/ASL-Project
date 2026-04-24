# American Sign Language Recognition System

An accessibility-focused computer vision project that recognises American Sign Language (ASL) hand signs from a webcam feed and converts them into text in real time.

This repository is a strong fit for GSSoC contributors who want to work on Python, Flask, computer vision, machine learning, UI improvements, documentation, and accessibility-driven open source.

## Overview

The project uses a webcam, hand detection, and a trained classification model to identify ASL alphabet gestures. The current application serves a Flask interface, streams webcam frames, predicts letters, and builds a word progressively from recognized signs.

## Why This Project Matters

Communication barriers still exist for people who rely on sign language in classrooms, workplaces, and everyday interactions. This project aims to reduce that gap by building a practical ASL-to-text interface that is approachable, extendable, and open to community collaboration.

## Current Features

- Real-time ASL alphabet recognition using a webcam
- Flask-based web interface
- Hand detection with `cvzone` and OpenCV
- Model inference using a saved Keras classifier
- Automatic letter accumulation into a word buffer
- Manual controls to add, undo, and reset letters
- Dataset collection script for new samples
- Model training script for custom experiments

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Backend | Flask |
| Computer Vision | OpenCV, cvzone |
| ML / Deep Learning | TensorFlow, Keras |
| Data Handling | NumPy |
| Deployment Support | Procfile, runtime.txt |

## How It Works

The application follows this pipeline:

1. Capture live webcam frames.
2. Detect a hand region using `HandDetector`.
3. Normalize the cropped hand image onto a white canvas.
4. Run inference through the trained classifier in `Model/`.
5. Display the predicted letter and confidence.
6. Append letters into a word after the same sign remains stable for a few seconds.

## Repository Structure

```text
ASL-Project/
|-- app.py                 # Flask app and real-time prediction pipeline
|-- dataCollection.py      # Utility to capture hand sign images for a class
|-- train_model.py         # Model training script
|-- requirements.txt       # Python dependencies
|-- Procfile               # Deployment process file
|-- runtime.txt            # Python runtime version
|-- Data/                  # Collected sign image data
|-- Model/                 # Trained model files and labels
|-- static/                # CSS, images, frontend static assets
|-- templates/             # HTML templates for Flask UI
|-- README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sant60/ASL-Project.git
cd ASL-Project
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Then open your browser to:

```text
http://127.0.0.1:5000/
```

## Dataset Collection

Use the collector script to capture hand-sign samples for a target class.

```bash
python dataCollection.py
```

Notes:

- The current script saves images into `Data/Y`.
- Press `s` to save a processed frame.
- Make sure the target class folder exists before collecting data.
- A webcam is required.

## Model Training

Train a classification model on your dataset:

```bash
python train_model.py
```

Current training behavior:

- The script expects a dataset directory configured in `train_model.py`.
- It saves `keras_model.h5` and `labels.txt` after training.
- If you want the Flask app to use the trained files directly, place them inside the `Model/` directory or update the paths in `app.py`.

## Prerequisites

- Python 3.8 or above
- Working webcam
- Basic familiarity with Python and Git for contributors

## Known Gaps and Improvement Opportunities

This repo already works as a useful base, but there are several places where contributors can make high-impact improvements:

- Improve prediction accuracy and robustness
- Expand from alphabet recognition to words and sentences
- Add better error handling around webcam and prediction failures
- Improve UI/UX and accessibility of the frontend
- Add tests, linting, and CI workflows
- Refactor scripts for configurability and maintainability
- Improve dataset organization and augmentation
- Add deployment and containerization support
- Document model performance and evaluation metrics

## GSSoC Contribution Guide

We warmly welcome GSSoC contributors of all experience levels.

### Good First Contributions

- Improve project documentation
- Fix typos, broken instructions, or unclear setup steps
- Add code comments in complex areas
- Clean up Python naming, formatting, or structure
- Improve templates and frontend styling
- Add helpful issue templates or pull request templates

### Intermediate Contributions

- Add modular utility functions to reduce duplication
- Improve prediction stability logic
- Add keyboard/UI controls for text editing actions
- Improve model-loading and configuration management
- Add dataset validation or augmentation scripts
- Add responsiveness and accessibility improvements to the frontend

### Advanced Contributions

- Redesign the recognition pipeline for better accuracy
- Add sentence building, autocorrect, or language modeling
- Support dynamic gestures and not just static alphabets
- Add speech synthesis from predicted text
- Build an evaluation dashboard or benchmark workflow
- Dockerize the project and prepare production-ready deployment

## Contribution Workflow

1. Fork the repository.
2. Clone your fork locally.
3. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

4. Make your changes.
5. Test your work locally.
6. Commit with a clear message:

```bash
git commit -m "feat: improve webcam prediction stability"
```

7. Push your branch:

```bash
git push origin feature/your-feature-name
```

8. Open a pull request with a clear summary, screenshots if relevant, and testing notes.

## Pull Request Checklist

Before opening a PR, please make sure:

- Your code runs locally
- You tested the affected flow where possible
- Your changes are focused and easy to review
- Documentation is updated if behavior changed
- Commit messages are meaningful
- New contributors keep PRs small and well-scoped when possible

## Suggested Issue Labels

Maintainers may want to use labels like:

- `gssoc`
- `good first issue`
- `beginner friendly`
- `documentation`
- `bug`
- `enhancement`
- `help wanted`
- `ml`
- `frontend`
- `backend`

## Coding Expectations

- Follow readable and consistent Python style
- Prefer small, focused functions
- Avoid unnecessary breaking changes
- Keep accessibility and usability in mind
- Document assumptions when changing model or data behaviour

## Testing Suggestions for Contributors

The project does not yet have a formal automated test suite, so contributors should include manual verification notes such as:

- App launches successfully with `python app.py`
- Webcam feed renders correctly
- Prediction overlay appears without crashing
- Word buffer controls behave as expected
- Training script runs with the intended dataset path

Adding automated tests is a valuable contribution in itself.

## Roadmap Ideas

- Better model versioning inside `Model/`
- Config-driven dataset and model paths
- Safer exception handling in the frame generator
- Prediction smoothing and confidence thresholds
- Multi-hand or multi-sign support
- Sentence suggestions and text-to-speech
- Hosted demo or packaged desktop app

## Screenshots and Demo

- UI screenshots
- Sample prediction GIFs
- Demo video links
- Model accuracy snapshots

These help contributors understand the expected behavior faster.

## Maintainer Notes

If you are preparing this repository for GSSoC, adding the following will make contributions smoother:

- A `LICENSE` file
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- issue templates
- pull request template
- project board or roadmap
- labels for contributor-friendly tasks

## Acknowledgements

This project builds on the open-source ecosystem around:

- Flask
- OpenCV
- TensorFlow / Keras
- cvzone
- the broader accessibility and sign-language-tech community

// ## License

// No license file is currently added. If you want outside contributors to reuse and contribute confidently, it is strongly recommended that you 
// add an explicit open-source license.

## Support

If you want to contribute but are not sure where to begin:

- Open an issue with your question
- ask to be assigned a beginner-friendly task
- propose documentation, testing, or UI improvements

Contributions that improve usability, accessibility, reliability, and documentation are just as valuable as model improvements.
