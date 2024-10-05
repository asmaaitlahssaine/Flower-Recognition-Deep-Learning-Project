# 🌸 Flower Recognition Deep Learning Project

## Project Overview
This project focuses on recognizing five types of flowers—**daisy, tulip, rose, sunflower, and dandelion**—from a dataset of 4242 images. Utilizing deep learning techniques, the goal is to classify images accurately into their respective categories based on visual features. The dataset is drawn from various image sources, including Flickr, Google Images, and Yandex Images.


## 🌱 Context
The dataset contains images of flowers that vary in resolution (around 320x240 pixels) and proportion, presenting a challenge for recognition models due to the lack of uniformity. This project uses PyTorch's `ImageFolder` class to handle the dataset and applies deep learning models to automate flower classification.

---

## 🎯 Objectives
- **Dataset Handling**: Efficiently load and preprocess flower images.
- **Deep Learning**: Train a neural network for flower classification.
- **Evaluation**: Test the model’s performance and refine it using accuracy and loss metrics.
- **Deployment**: Provide insights into using the model in real-world applications, such as plant recognition tools.

---

## 📁 Dataset Information
- **Source**: [Kaggle - Flower Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition)
- **Classes**: Five classes of flowers—**Chamomile, Tulip, Rose, Sunflower, Dandelion**
- **Number of Images**: About 800 images per class

The images are divided into their respective folders, each corresponding to a specific flower type.

---

## 🚀 Model Architecture
The model is built using a **Convolutional Neural Network (CNN)**, leveraging the power of deep learning to recognize patterns in flower images. Key components of the model:
- **Convolutional Layers**: For feature extraction from images.
- **Pooling Layers**: To reduce the spatial dimensions.
- **Fully Connected Layers**: For classification into one of the five flower classes.

### Tools & Libraries Used:
- **PyTorch**: For building and training the deep learning model.
- **OpenDatasets**: To import datasets directly from Kaggle.
- **TorchVision**: To preprocess and load image datasets.

---

## 📊 Results
After training the model, the accuracy achieved on the test dataset was **X%** (replace with actual accuracy), demonstrating the model’s potential for real-world flower recognition tasks. Further fine-tuning and data augmentation can be applied to improve the accuracy.

---

## 🛠 How to Run This Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/flower-recognition.git
   ```
2. **Install Dependencies**:
   Install the necessary libraries such as PyTorch, TorchVision, and OpenDatasets:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Download the dataset directly from Kaggle:
   ```bash
   !pip install opendatasets
   od.download('https://www.kaggle.com/alxmamaev/flowers-recognition')
   ```

4. **Run the Notebook**:
   Open and run the Jupyter notebook for the project:
   ```bash
   jupyter notebook deep_learning_project_live.ipynb
   ```

---

## 📚 Project Structure
```bash
├── flowers-recognition/          # Flower image dataset
├── deep_learning_project_live.ipynb  # Jupyter notebook for model training
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 👩‍💻 Future Improvements
- **Increase Dataset Size**: Adding more diverse images will improve model robustness.
- **Data Augmentation**: Apply transformations like rotation, scaling, and flipping to generalize the model better.
- **Hyperparameter Tuning**: Experiment with different optimizers and learning rates to enhance accuracy.

---

## 🏅 Acknowledgments
- **Dataset**: The flower images are from the Kaggle dataset: [Flower Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition).
- **Inspiration**: Plant identification tools such as PlantSnap and Google Lens inspired this project.

---

## 🙌 Let's Connect
If you have any questions or would like to collaborate on future projects, feel free to reach out!

- **GitHub**: [asmaaitlahssaine](https://github.com/asmaaitlahssaine)
- **LinkedIn**: [https://www.linkedin.com/in/asma-ait-lahssaine-157273260/)

