"""
Configuration for spam classifier.
Weights, thresholds, and feature parameters.
"""

# Threshold for classifying a message as spam (0.0 to 1.0)
SPAM_THRESHOLD = 0.5

# Feature weights (sum should not exceed 1.0 for interpretability)
FEATURE_WEIGHTS = {
   ## YOUR CODE HERE ##
   # e.g. 'spam_words': 0.4,
}

# Punctuation marks that indicate spam when excessive
SPAM_PUNCTUATION = {}

# Threshold for what counts as "excessive" punctuation (ratio)
PUNCTUATION_RATIO_THRESHOLD = 0.15

# Threshold for ALL CAPS ratio
ALL_CAPS_THRESHOLD = 0.5

# Label mappings for classification
LABEL_SPAM = 'spam'
LABEL_HAM = 'ham'
LABEL_ALTERNATIVES = {
    '1': LABEL_SPAM,
    '0': LABEL_HAM,
    'yes': LABEL_SPAM,
    'no': LABEL_HAM,
    'true': LABEL_SPAM,
    'false': LABEL_HAM,
}
