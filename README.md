# Assignment: Spam vs. Ham Classifier (Heuristic-Based)

## Objective
Your task is to generate the Python code necessary to classify messages as **spam** or **ham**. The program must:
- Load a dataset from a CSV file.
- Use a spam word list.
- Analyze punctuation in each message.
- Compute a score and compare it to a threshold.
- Report results and evaluation metrics when labels are provided.

---

## Project Structure

```
.
├── spam_classifier.py      # Main CLI entry point
├── config.py               # Configuration (weights, thresholds, punctuation)
├── requirements.txt        # Dependencies (empty - uses only stdlib)
├── README.md               # This file
├── lists/
│   └── spam_words.txt      # List of spam indicator words/phrases
└── data/
    └── sample.csv          # Sample input data for testing
```

---

## Input CSV Format

The classifier expects a CSV file with two columns:

- **`text`** (required): The message content to classify.
- **`label`** (optional): The true classification for evaluation (`spam`/`ham`, `1`/`0`, `yes`/`no`, etc.).

Example:
```csv
text,label
"This is a test message",ham
"CLICK HERE NOW FOR FREE MONEY!!!",spam
```

If the label column is missing, the classifier runs in predict-only mode (no accuracy metrics).

---

## Spam Word List Format

The spam word list (`lists/spam_words.txt`) contains one word or phrase per line. Lines starting with `#` are treated as comments and ignored.

Example:
```
click here
buy now
limited time
act now
# This is a comment
free money
```

---

## Scoring and Classification (Overview)

The classifier produces a numeric **score** for each message and compares it to a configurable **threshold**. Messages at or above the threshold are labeled **SPAM**; others are labeled **HAM**.

You can adjust scoring behavior and thresholds in `config.py`.

---

## How the Current Code Works (High-Level)

### `spam_classifier.py`
- Parses command-line arguments (input file, column names, threshold, spam list).
- Loads the spam word list.
- Loads the CSV and extracts text and optional labels.
- Computes a score for each message and applies a threshold.
- If labels exist, calculates accuracy, precision, recall, and F1.

### `config.py`
- Stores:
  - `SPAM_THRESHOLD`
  - `FEATURE_WEIGHTS`
  - `SPAM_PUNCTUATION`
  - Label normalization mappings

---

## Usage

### Basic Usage

```bash
python spam_classifier.py --input data/sample.csv
```

### With Custom Column Names

```bash
python spam_classifier.py --input data/sample.csv --text-column message --label-column classification
```

### With Evaluation (if labels available)

```bash
python spam_classifier.py --input data/sample.csv --text-column text --label-column label
```

### With Custom Threshold

```bash
python spam_classifier.py --input data/sample.csv --threshold 0.4
```

### With Custom Spam Word List

```bash
python spam_classifier.py --input data/sample.csv --spam-words my_words.txt
```

### Help

```bash
python spam_classifier.py --help
```

---

## Example Output (What It Means)

```
==================================================
SPAM CLASSIFIER RESULTS
==================================================
Total messages: 16
Predicted as SPAM: 8
Predicted as HAM: 8

Accuracy: 100.00%
Precision (SPAM): 100.00%
Recall (SPAM): 100.00%
F1-Score (SPAM): 100.00%
==================================================
```

**Explanation:**
- **Total messages** = number of rows in the CSV.
- **Predicted as SPAM/HAM** = how many were labeled by the classifier.
- **Accuracy** = % of total predictions that were correct.
- **Precision (SPAM)** = % of predicted spam that was actually spam.
- **Recall (SPAM)** = % of actual spam correctly identified.
- **F1-score** = balance between precision and recall.

---

## Running the Sample

```bash
python spam_classifier.py --input data/sample.csv --text-column text --label-column label
```

Expected output: All 16 sample messages classified correctly (100% accuracy).

---

## Limitations and Notes

- **English-only**: Word matching is case-insensitive but language-specific.
- **Heuristic-based**: No machine learning; relies on manual rules.
- **Quality of Word List**: Classifier performance depends on curating a good spam word list.
- **No context**: Does not understand semantics or conversation flow.
- **False positives/negatives**: May misclassify legitimate messages with excitement (`!!!`) or spam keywords used innocently.

---

## Dependencies

- **Python 3.6+**
- Standard library only: `argparse`, `csv`, `re`, `sys`

No external packages required.

---

## License

MIT (or your choice)