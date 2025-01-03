# Custom Medical NER Model Training with Spark NLP

This project demonstrates how to train a custom Named Entity Recognition (NER) model for medical text using Spark NLP. The pipeline includes data preprocessing, model training, and evaluation capabilities.

## Prerequisites

- Python 3.7+
- PySpark 3.4.1
- Spark NLP Healthcare License
- Required Python packages:
  ```
  pyspark==3.4.1
  spark-nlp==[PUBLIC_VERSION]
  spark-nlp-jsl==[JSL_VERSION]
  spark-nlp-display
  tensorflow==2.12.0
  tensorflow-addons
  ```

## Setup Instructions

1. Clone this repository and navigate to the project directory

2. Install the required dependencies:
```bash
pip install --upgrade -q pyspark==3.4.1 spark-nlp==[PUBLIC_VERSION]
pip install --upgrade -q spark-nlp-jsl==[JSL_VERSION] --extra-index-url https://pypi.johnsnowlabs.com/[SECRET]
pip install -q spark-nlp-display
pip install -q tensorflow==2.12.0 tensorflow-addons
```

3. Place your Spark NLP Healthcare license file (`spark_nlp_for_healthcare_spark_ocr_*.json`) in the project root directory

4. Configure Spark environment:
```python
params = {
    "spark.driver.memory": "10G",
    "spark.kryoserializer.buffer.max": "2000M",
    "spark.driver.maxResultSize": "2000M"
}
```

## Dataset

The project uses oncology clinical notes for training. The dataset should be organized as follows:
- Text files (.txt) containing clinical notes in the `/content/onc_notes` directory
- Each note contains medical information including problems, treatments, tests, and other clinical entities

## Pipeline Steps

### 1. Data Preparation
```python
# Load text data
mt_samples_df = spark.read.text("onc_notes/*.txt", wholetext=True)
mt_samples_df = mt_samples_df.withColumnRenamed("value", "text")
```

### 2. NER Pipeline Execution
The pipeline includes:
- Document Assembly
- Sentence Detection
- Tokenization
- Word Embeddings
- Multiple NER Models (Clinical, De-identification, Posology)
- NER Conversion
- Chunk Merging

### 3. CoNLL File Creation
The pipeline converts the annotated text into CoNLL format for training:
1. Creates entity and text CSV files
2. Transforms the data into CoNLL format
3. Saves the CoNLL file for training

### 4. Custom NER Model Training
```python
# Initialize embeddings
clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")

# Train-test split
(train_data, test_data) = full_data.randomSplit([0.8, 0.2], seed=100)

# Train the model
ner_pipeline = Pipeline(stages=[
    clinical_embeddings,
    ner_graph_builder,
    nerTagger
])
```

## Model Training Configuration

The NER model is trained with the following parameters:
- Maximum epochs: 100
- Learning rate: 0.003
- Batch size: 8
- Hidden units: 24
- Random seed: 0

## Evaluation

The model's performance is evaluated using:
- Precision, recall, and F1 score for each entity type
- Confusion matrix
- Classification report
- Entity-wise performance metrics

## Saving and Loading the Model

```python
# Save the model
ner_model.stages[2].write().overwrite().save('models/medical_NER_A_100_epoch')

# Load the model
loaded_model = MedicalNerModel.load("models/medical_NER_A_100_epoch")
```

## Visualization

The project includes visualization capabilities using `sparknlp_display`:
```python
from sparknlp_display import NerVisualizer
visualiser = NerVisualizer()
visualiser.display(result, label_col='ner_chunks', document_col='document', save_path="display_result.html")
```

## Model Performance

The custom trained model demonstrates improved performance compared to pretrained models, particularly in:
- Identifying medical problems and their anatomical locations
- Recognizing tests and treatments
- Understanding medical context
- Entity boundary detection

## Troubleshooting

Common issues and solutions:
1. Memory errors: Adjust Spark parameters in the configuration
2. License issues: Ensure the license file is properly placed and loaded
3. Pipeline errors: Check the input data format and column names
4. Training errors: Monitor the training logs in the `ner_logs` directory

## License

This project requires a valid Spark NLP Healthcare license. Please contact John Snow Labs for licensing information.
