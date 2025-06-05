# PL Klasyfikacja Chorób Oczu
## Opis projektu 

Ten projekt został opracowany w ramach kursu Biologically Inspired Artificial Intelligence na Politechnice Śląskiej. 
Celem jest automatyczna klasyfikacja chorób oczu (retinopatia cukrzycowa, zaćma, jaskra, zdrowe) na podstawie obrazów siatkówki przy użyciu konwolucyjnej sieci neuronowej (CNN) inspirowanej ludzką korą wzrokową. 
Model, zaimplementowany w Pythonie z użyciem PyTorch, wykorzystuje mechanizm uwagi do skupiania się na kluczowych obszarach obrazu. 
Projekt obejmuje preprocessing danych, trening modelu, analizę wyników (dokładność, precyzja, czułość, F1-score) oraz wizualizacje, 
takie jak mapy cieplne uwagi i macierz pomyłek.

## Instrukcja instalacji i uruchomienia

1. **Sklonuj repozytorium**
   - Przejdź do repozytorium na GitHub i sklonuj je:
     ```bash
     git clone https://github.com/Maciek600/CNN-Eye-Diseases-Classification.git
     ```
   - Alternatywnie pobierz archiwum ZIP i rozpakuj do wybranego folderu (np. `EyeDiseaseClassification`).

2. **Zainstaluj wymagane biblioteki**
   - Otwórz terminal, przejdź do folderu projektu i uruchom:
     ```bash
     pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn
     ```

3. **Pobierz dataset z Kaggle**
   - Wejdź na [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data).
   - Zaloguj się na Kaggle, pobierz dataset i rozpakuj plik ZIP.

4. **Rozpakuj dataset**
   - Umieść rozpakowany folder w projekcie, np. `eye_diseases_dataset`, tak aby struktura wyglądała tak:
     ```
     EyeDiseaseClassification/
     ├── eye_diseases_dataset/
     │   ├── normal/
     │   ├── diabetic_retinopathy/
     │   ├── cataract/
     │   ├── glaucoma/
     ├── eye_disease_classification.py
     ├── generate_diagrams.py
     ├── eye_disease_app.py (opcjonalne)
     ├── requirements.txt (opcjonalne)
     └── ...
     ```

5. **Zaktualizuj ścieżkę w kodzie**
   - Otwórz plik `eye_disease_classification.py` w edytorze (np. PyCharm).
   - Znajdź linię `data_dir` w funkcji `main()` i zmień na ścieżkę do datasetu, np.:
     ```python
     data_dir = 'eye_diseases_dataset'
     ```
   - Zapisz plik.

6. **Trenowanie modelu i generowanie wyników**
   - W terminalu uruchom główny skrypt:
     ```bash
     python eye_disease_classification.py
     ```
   - Skrypt wytrenuje model, zapisze go jako `best_model.pth`, a także wygeneruje:
     - Statystyki datasetu (`dataset_stats.txt`),
     - Metryki testowe (`test_metrics.txt`),
     - Wizualizacje (`training_history.png`, `confusion_matrix.png`).

7. **Uruchomienie aplikacji GUI**
   - Uruchom aplikację:
     ```bash
     python eye_disease_app.py
     ```
   - Otworzy się okno, gdzie możesz wgrać zdjęcie oka. Aplikacja pokaże przewidywanie i heatmapę uwagi. Upewnij się, że plik `best_model.pth` istnieje (powinien powstać po treningu).


# ENG Eye Disease Classification
## Project Description 

This project was developed as part of the Biologically Inspired Artificial Intelligence course at the Silesian University of Technology. 
The goal is to automatically classify eye diseases (diabetic retinopathy, cataract, glaucoma, normal) from retinal images using a convolutional neural network (CNN) inspired by the human visual cortex. 
The model, implemented in Python using PyTorch, incorporates an attention mechanism to focus on critical image regions. 
The project includes data preprocessing, model training, performance analysis (accuracy, precision, recall, F1-score), and visualizations such as attention heatmaps and confusion matrix.

## Installation and Setup Steps
1. **Clone the repository**
   - Go to the GitHub repository and clone it:
     ```bash
     git clone https://github.com/Maciek600/CNN-Eye-Diseases-Classification.git
     ```
   - Alternatively, download the ZIP file and extract it to a folder (e.g., `EyeDiseaseClassification`).

2. **Install required libraries**
   - Open a terminal, navigate to the project folder, and run:
     ```bash
     pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn
     ```

3. **Download dataset from Kaggle**
   - Visit [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data).
   - Log in to Kaggle, download the dataset, and extract the ZIP file.

4. **Extract the dataset**
   - Place the extracted folder in the project directory, e.g., `eye_diseases_dataset`, so the structure looks like this:
     ```
     EyeDiseaseClassification/
     ├── eye_diseases_dataset/
     │   ├── normal/
     │   ├── diabetic_retinopathy/
     │   ├── cataract/
     │   ├── glaucoma/
     ├── eye_disease_classification.py
     ├── generate_diagrams.py
     ├── eye_disease_app.py (optional)
     ├── requirements.txt (optional)
     └── ...
     ```

5. **Update the path in the code**
   - Open `eye_disease_classification.py` in an editor (e.g., PyCharm).
   - Find the `data_dir` line in the `main()` function and update it to the dataset location, e.g.:
     ```python
     data_dir = 'eye_diseases_dataset'
     ```
   - Save the file.

6. **Train the model and generate results**
   - In the terminal, run the main script:
     ```bash
     python eye_disease_classification.py
     ```
   - The script will train the model, save it as `best_model.pth`, and generate:
     - Dataset statistics (`dataset_stats.txt`),
     - Test metrics (`test_metrics.txt`),
     - Visualizations (`training_history.png`, `confusion_matrix.png`).

7. **Run the GUI application**
   - Launch the application:
     ```bash
     python eye_disease_app.py
     ```
   - A window will open where you can upload an eye image. The app will display a prediction and attention heatmap. Ensure `best_model.pth` exists (it should be created after training).

## Dataset
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data


