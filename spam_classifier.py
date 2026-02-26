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
    
    if not message or not message.strip():
        return features
    
    # Keep original message for punctuation / caps checks, and use a lowercase copy for matching
    normalized = message.lower()
    
    # Tokenize both original and lowercase words
    words_orig = re.findall(r'\b\w+\b', message)
    words_lower = [w.lower() for w in words_orig]
    total_words = len(words_lower)
    
    if total_words == 0:
        return features
    
    # Feature 1: Spam word hits
    # Match full words or exact phrases to avoid substring false positives
    spam_word_count = 0
    for spam_word in spam_words:
        spam_word = spam_word.lower().strip()
        if not spam_word:
            continue
        if ' ' in spam_word:
            # phrase match with word boundaries on the normalized text
            if re.search(r'\b' + re.escape(spam_word) + r'\b', normalized):
                spam_word_count += 1
        else:
            # single-word exact token match
            if spam_word in words_lower:
                spam_word_count += 1
    
    # Normalize by total words (use a smaller divisor so single hits matter less)
    features['spam_words'] = min(spam_word_count / max(total_words / 2, 1), 1.0)
    
    # Feature 2: Punctuation analysis
    # Count excessive spam punctuation (!, ?, etc.) relative to non-space chars
    spam_punct_count = sum(message.count(char) for char in SPAM_PUNCTUATION)
    non_space_chars = len(message.replace(' ', ''))
    features['punctuation'] = min(spam_punct_count / non_space_chars, 1.0) if non_space_chars > 0 else 0.0
    
    # Feature 3: ALL CAPS words
    # Count words that are all uppercase (longer than 2 chars to avoid acronyms)
    all_caps_words = sum(1 for word in words_orig if len(word) > 2 and word.isupper())
    features['all_caps'] = min(all_caps_words / total_words, 1.0)
    
    # Feature 4: Urgency indicators
    urgency_score = 0.0
    
    # Multiple exclamation or question marks in the original message
    if '!!' in message or '???' in message:
        urgency_score += 0.4
    
    # Check for urgency keywords in the normalized (lowercase) text
    urgency_keywords = ['urgent', 'now', 'immediately', 'act', 'limited', 'expires', 'final']
    for keyword in urgency_keywords:
        if keyword in normalized:
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


def optimize_threshold(data, spam_words, start=0.0, end=1.0, step=0.003):
    """
    Find the optimal threshold that maximizes F1-Score.
    Tests thresholds from start to end in given step increments.
    Returns (best_threshold, best_stats, all_results).
    """
    best_threshold = start
    best_f1 = 0.0
    best_stats = None
    all_results = []
    
    # Test each threshold value
    threshold = start
    while threshold <= end + 1e-9:
        # Classify all messages with this threshold
        predictions = []
        true_labels = []
        
        for text, label in data:
            if text.strip():
                spam_pred, score = is_spam(text, spam_words, threshold=threshold)
                pred_label = LABEL_SPAM if spam_pred else LABEL_HAM
                predictions.append((pred_label, score))
                true_labels.append(label)
            else:
                predictions.append((LABEL_HAM, 0.0))
                true_labels.append(label)
        
        # Compute stats for this threshold
        stats = compute_stats(predictions, true_labels)
        
        # Track this result
        result = {
            'threshold': threshold,
            'f1': stats['f1_spam'] if stats['f1_spam'] is not None else 0.0,
            'accuracy': stats['accuracy'] if stats['accuracy'] is not None else 0.0,
            'precision': stats['precision_spam'] if stats['precision_spam'] is not None else 0.0,
            'recall': stats['recall_spam'] if stats['recall_spam'] is not None else 0.0,
            'stats': stats
        }
        all_results.append(result)
        
        # Update best if this is better
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_threshold = threshold
            best_stats = stats
        
        threshold = round(threshold + step, 4)  # Round to keep step precision stable
    
    return best_threshold, best_stats, all_results


def print_optimization_results(best_threshold, best_stats, all_results):
    """Print threshold optimization results."""
    print("\n" + "="*50)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*50)
    # determine step from results if possible
    step = round(all_results[1]['threshold'] - all_results[0]['threshold'], 4) if len(all_results) > 1 else 0.0
    print(f"Tested {len(all_results)} threshold values from 0.0 to 1.0 (step: {step})\n")
    
    print("Top 5 thresholds by F1-Score:")
    print("-" * 50)
    # Sort by F1-score descending
    sorted_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. Threshold: {result['threshold']:.4f}")
        print(f"   F1-Score: {result['f1']:.2%} | Accuracy: {result['accuracy']:.2%}")
        print(f"   Precision: {result['precision']:.2%} | Recall: {result['recall']:.2%}")
        print()
    
    print("="*50)
    print(f"OPTIMAL THRESHOLD: {best_threshold:.4f}")
    print("="*50)
    print(f"F1-Score: {best_stats['f1_spam']:.2%}")
    print(f"Accuracy: {best_stats['accuracy']:.2%}")
    print(f"Precision: {best_stats['precision_spam']:.2%}")
    print(f"Recall: {best_stats['recall_spam']:.2%}")
    print(f"Predicted as SPAM: {best_stats['spam_predicted']}")
    print(f"Predicted as HAM: {best_stats['ham_predicted']}")
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
    parser.add_argument(
        '--optimize-threshold',
        action='store_true',
        help='Find the optimal threshold by testing values from 0.0 to 1.0 in steps of 0.05 to maximize F1-Score'
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
    
    # Check if optimization mode is enabled
    if args.optimize_threshold:
        # Verify labels exist for optimization
        if not any(label is not None for _, label in data):
            print("Error: Cannot optimize threshold without labels in the dataset.", file=sys.stderr)
            sys.exit(1)
        
        # Run threshold optimization
        print("Optimizing threshold to maximize F1-Score...")
        best_threshold, best_stats, all_results = optimize_threshold(data, spam_words)
        print_optimization_results(best_threshold, best_stats, all_results)
    else:
        # Normal classification with specified threshold
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
        stats = compute_stats(predictions, true_labels)
        print_stats(stats)


if __name__ == '__main__':
    main()
