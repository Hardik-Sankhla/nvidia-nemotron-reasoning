
# Dataset Description

This dataset comprises a collection of logical reasoning puzzles requiring the identification and application of underlying transformation rules. The puzzles cover various domains, such as bit manipulation and algebraic equations.

## File and Field Information

### train.csv
The training set containing puzzles and their corresponding solutions.

- **id** - A unique identifier for each puzzle.
- **prompt** - The puzzle description, including input-output examples and the specific instance to be solved.
- **answer** - The ground truth solution for the puzzle.

### test.csv
A sample test set to help you author your submissions. When your submission is scored, this will be replaced by a test set of several hundred problems.

- **id** - A unique identifier for each puzzle.
- **prompt** - As in train.csv.

> **Note:** Your submission must be a file `submission.zip` containing a LoRA adapter. See the Evaluation page for details.

## Dataset Summary

| Property | Value |
|----------|-------|
| Files | 2 files |
| Size | 3.07 MB |
| Type | CSV |
| License | Attribution 4.0 International (CC BY 4.0) |

### test.csv Details
- **Size:** 1.46 kB
- **Columns:** 2 of 2 columns
- **Unique values:** 3 unique values each for id and prompt

### Sample Data
| id | prompt |
|----|--------|
| 00066667 | In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transform... |
| 000b53cf | In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transform... |
| 00189f6a | In Alice's Wonderland, secret encryption rules are used on text. Here are some examples: ucoov pwgtf... |

## Data Explorer
- **Total Size:** 3.07 MB
- **Files:** test.csv, train.csv
- **Summary:** 2 files, 5 columns total

## Download

### Using Kaggle CLI
```bash
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge
```

### Using kagglehub
```python
kagglehub.competition_download('nvidia-nemotron-model-reasoning-challenge')
```