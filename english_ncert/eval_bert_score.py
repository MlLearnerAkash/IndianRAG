# Install the bert-score package if not already installed
# pip install bert-score


from bert_score import score

# Example texts
candidate = "The cat sat on the mat."
reference = "A cat rested on a mat."

# Compute BERTScore
P, R, F1 = score([candidate], [reference], lang="en", verbose=True)

# Print results
print(f"Precision: {P.item():.4f}")
print(f"Recall:    {R.item():.4f}")
print(f"F1:        {F1.item():.4f}")

