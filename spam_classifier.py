#!/usr/bin/env python3
"""
Spam Classifier: Heuristic-based spam detection using word lists, punctuation analysis, and thresholds.
Loads CSV data, computes features, applies threshold formula, and prints classification stats.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from config import (
    SPAM_THRESHOLD, FEATURE_WEIGHTS, SPAM_PUNCTUATION,
    PUNCTUATION_RATIO_THRESHOLD, ALL_CAPS_THRESHOLD,
    LABEL_SPAM, LABEL_HAM, LABEL_ALTERNATIVES
)


def load_spam_words(filepath):
    """Load spam word list from file (one word/phrase per line)."""
    spam_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith('#'):  # Skip empty lines and comments
                    spam_words.add(word)
    except FileNotFoundError:
        print(f"Warning: Spam word list not found at {filepath}", file=sys.stderr)
    return spam_words


def normalize_text(text):
    """Normalize text for analysis."""
    return text.lower()


def compute_features(message, spam_words):
    """
    Compute feature signals for a message.
    Returns a dict with feature scores (0.0 to 1.0).
    """
    features = {}
    
    if not message.strip():
        return features
    
    normalized = normalize_text(message)
    
    # Tokenize the message into words
    words = re.findall(r'\b\w+\b', normalized)
    total_words = len(words)
    
    if total_words == 0:
        return features
    
    # Feature 1: Spam word hits
    # Count how many spam words/phrases are present in the message
    spam_word_count = 0
    for spam_word in spam_words:
        if spam_word in normalized:
            spam_word_count += 1
    
    # Normalize by total words (cap at 1.0)
    features['spam_words'] = min(spam_word_count / max(total_words / 3, 1), 1.0)
    
    # Feature 2: Punctuation analysis
    # Count excessive spam punctuation (!, ?, etc.)
    spam_punct_count = sum(message.count(char) for char in SPAM_PUNCTUATION)
    total_chars = len(message)
    features['punctuation'] = min(spam_punct_count / total_chars, 1.0) if total_chars > 0 else 0.0
    
    # Feature 3: ALL CAPS words
    # Count words that are all uppercase (longer than 2 chars to avoid acronyms)
    all_caps_words = sum(1 for word in words if len(word) > 2 and word.isupper())
    features['all_caps'] = min(all_caps_words / total_words, 1.0)
    
    # Feature 4: Urgency indicators
    # Look for multiple exclamation marks, urgency keywords, etc.
    urgency_score = 0.0
    
    # Multiple exclamation or question marks
    if '!!' in message or '???' in message:
        urgency_score += 0.4
    
    # Check for urgency keywords in uppercase
    urgency_keywords = ['urgent', 'now', 'immediately', 'act', 'limited', 'expires', 'final']
    for keyword in urgency_keywords:
        if keyword.upper() in message:
            urgency_score += 0.15
    
    features['urgency'] = min(urgency_score, 1.0)
    
    return features


def compute_score(features):
    """
    Compute weighted spam score from features.
    Returns a score between 0.0 and 1.0.
    """
    score = 0.0
    for feature_name, weight in FEATURE_WEIGHTS.items():
        if feature_name in features:
            score += features[feature_name] * weight
    return min(score, 1.0)


def is_spam(message, spam_words, threshold=SPAM_THRESHOLD):
    """Determine if a message is spam based on threshold formula."""
    features = compute_features(message, spam_words)
    score = compute_score(features)
    return score >= threshold, score


def normalize_label(label):
    """Normalize label to standard format (spam or ham)."""
    label_lower = label.lower().strip()
    if label_lower in LABEL_ALTERNATIVES:
        return LABEL_ALTERNATIVES[label_lower]
    elif label_lower == LABEL_SPAM:
        return LABEL_SPAM
    elif label_lower == LABEL_HAM:
        return LABEL_HAM
    else:
        return None


def load_csv(filepath, text_column, label_column=None):
    """Load CSV file and return list of (text, label) tuples."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Error: CSV file is empty or malformed.", file=sys.stderr)
                return []
            
            if text_column not in reader.fieldnames:
                print(f"Error: Column '{text_column}' not found in CSV.", file=sys.stderr)
                print(f"Available columns: {', '.join(reader.fieldnames)}", file=sys.stderr)
                return []
            
            if label_column and label_column not in reader.fieldnames:
                print(f"Warning: Label column '{label_column}' not found. Running in predict-only mode.", file=sys.stderr)
                label_column = None
            
            for row in reader:
                text = row.get(text_column, '').strip()
                label = None
                if label_column:
                    label = normalize_label(row.get(label_column, ''))
                data.append((text, label))
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
    
    return data


def compute_stats(predictions, labels):
    """
    Compute classification stats: accuracy, precision, recall.
    predictions: list of (predicted_label, score) tuples
    labels: list of true labels or None if predict-only
    """
    stats = {
        'total': len(predictions),
        'spam_predicted': sum(1 for pred, _ in predictions if pred == LABEL_SPAM),
        'ham_predicted': sum(1 for pred, _ in predictions if pred == LABEL_HAM),
        'accuracy': None,
        'precision_spam': None,
        'recall_spam': None,
        'f1_spam': None,
    }
    
    if labels and any(label is not None for label in labels):
        # Compute metrics when labels available
        true_positives = sum(1 for (pred, _), true_label in zip(predictions, labels)
                 if pred == LABEL_SPAM and true_label == LABEL_SPAM)
        false_positives = sum(1 for (pred, _), true_label in zip(predictions, labels)
                 if pred == LABEL_SPAM and true_label == LABEL_HAM)
        false_negatives = sum(1 for (pred, _), true_label in zip(predictions, labels)
                 if pred == LABEL_HAM and true_label == LABEL_SPAM)
        true_negatives = sum(1 for (pred, _), true_label in zip(predictions, labels)
                 if pred == LABEL_HAM and true_label == LABEL_HAM)
        
        if len(predictions) > 0:
            stats['accuracy'] = (true_positives + true_negatives) / len(predictions)
        
        if (true_positives + false_positives) > 0:
            stats['precision_spam'] = true_positives / (true_positives + false_positives)
        
        if (true_positives + false_negatives) > 0:
            stats['recall_spam'] = true_positives / (true_positives + false_negatives)
        
        if stats['precision_spam'] and stats['recall_spam']:
            precision = stats['precision_spam']
            recall = stats['recall_spam']
            if (precision + recall) > 0:
                stats['f1_spam'] = 2 * (precision * recall) / (precision + recall)
    
    return stats


def print_stats(stats):
    """Print classification statistics."""
    print("\n" + "="*50)
    print("SPAM CLASSIFIER RESULTS")
    print("="*50)
    print(f"Total messages: {stats['total']}")
    print(f"Predicted as SPAM: {stats['spam_predicted']}")
    print(f"Predicted as HAM: {stats['ham_predicted']}")
    
    if stats['accuracy'] is not None:
        print(f"\nAccuracy: {stats['accuracy']:.2%}")
        if stats['precision_spam'] is not None:
            print(f"Precision (SPAM): {stats['precision_spam']:.2%}")
        if stats['recall_spam'] is not None:
            print(f"Recall (SPAM): {stats['recall_spam']:.2%}")
        if stats['f1_spam'] is not None:
            print(f"F1-Score (SPAM): {stats['f1_spam']:.2%}")
    else:
        print("\n(No labels provided; running in predict-only mode)")
    
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Spam Classifier: Detect spam using heuristic features.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--text-column',
        default='text',
        help='Name of the column containing message text (default: text)'
    )
    parser.add_argument(
        '--label-column',
        default="label",
        help='Name of the column containing true labels (optional, for evaluation)'
    )
    parser.add_argument(
        '--spam-words',
        default='lists/spam_words.txt',
        help='Path to spam word list file (default: lists/spam_words.txt)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=SPAM_THRESHOLD,
        help=f'Spam threshold (0.0-1.0, default: {SPAM_THRESHOLD})'
    )
    
    args = parser.parse_args()
    
    # Load spam word list
    spam_words = load_spam_words(args.spam_words)
    if not spam_words:
        print("Warning: Spam word list is empty or not found.", file=sys.stderr)
    
    # Load CSV data
    data = load_csv(args.input, args.text_column, args.label_column)
    if not data:
        print("Error: No data loaded from CSV.", file=sys.stderr)
        sys.exit(1)
    
    # Classify messages
    predictions = []
    true_labels = []
    for text, label in data:
        if text.strip():
            spam_pred, score = is_spam(text, spam_words, threshold=args.threshold)
            pred_label = LABEL_SPAM if spam_pred else LABEL_HAM
            predictions.append((pred_label, score))
            true_labels.append(label)
        else:
            predictions.append((LABEL_HAM, 0.0))  # Empty messages are ham
            true_labels.append(label)
    
    # Compute and print stats
    # print(true_labels)
    stats = compute_stats(predictions, true_labels)
    print_stats(stats)


if __name__ == '__main__':
    main()
