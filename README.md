# Data Science and Machine Learning Portfolio

Six projects covering data analysis, machine learning, computer vision, NLP, audio, and generative AI. Each has its own folder with a notebook and a detailed README.

## Projects

### Appliance Energy Consumption Analysis
Exploratory analysis of household energy use, combining indoor sensor readings with weather data across 19,735 records.

Usage peaks at 18:00 and bottoms out at 03:00. No single sensor predicts consumption well, but the full sensor network together lifts model R squared from 0.10 to 0.17.

`pandas` `matplotlib` `scikit-learn`

### EuroSAT Land Use Classification
Ten class classification of satellite imagery, comparing a baseline CNN, an improved CNN, and a fine tuned ResNet50 across 27,000 images.

Transfer learning gives the biggest gain, reaching about 0.95 accuracy. Grad-CAM confirms the model looks at the right things: road lines for Highway, canopy texture for Forest.

`TensorFlow` `Keras` `ResNet50` `Grad-CAM`

### Customer Support Ticket Classification
Routing support tickets into four departments using classical NLP, built on 55,000 customer tweets labelled through keyword based weak supervision.

Linear SVM reaches 0.937 accuracy. The more useful finding: the weak labels are only about 80% accurate, so the score mostly measures agreement with the keyword rule, not real correctness.

`NLTK` `spaCy` `TF-IDF` `scikit-learn`

### Credit Card Fraud Detection
Fraud classification on 10,000 transactions with a 1.5% fraud rate, comparing three models with grid search.

Random Forest catches 17 of 30 frauds with zero false alarms. Logistic Regression catches all 30 but flags four legitimate transactions for each real fraud. Which is better depends on what a missed fraud costs versus a wasted review.

`scikit-learn` `GridSearchCV`

### AI vs Human Voice Classification
Telling synthesised speech from human speech using acoustic features rather than a deep model, across 200 clips.

Random Forest hits 0.975 accuracy. Statistical tests show a clear pattern: AI speech is spectrally darker, flatter in pitch, and faster than human speech.

`librosa` `scikit-learn` `scipy`

### Text Generation with the Groq API
Exploring LLM prompting: PDF extraction, few shot prompts, and comparing three open weight models on the same input.

Best moment came from a deliberately mislabelled few shot example. The model refused to copy the pattern, flagged the mistake, and returned corrected labels instead.

`Groq API` `PyMuPDF` `prompt engineering`

## Skills

Exploratory analysis · classical ML pipelines · deep learning and transfer learning · NLP · audio signal processing · generative AI · model explainability · hypothesis testing · handling class imbalance

## Approach

Preprocessing is always fitted on training data only, so no test information leaks into training. Imbalanced problems are scored on macro F1 rather than accuracy. Where a result relies on a shaky assumption, the README says so instead of just reporting the better looking number.
