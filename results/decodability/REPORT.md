# Decodability sweep — results

`L*` = earliest read point within 1 SE of that curve's own maximum. Read points: 0 = embeddings, i = output of block i-1.


## Family A — scalar readouts (is the preference EXTRACTABLE from h_L?)

| model | dataset | family | read | rung | L0 (tie) | L* | L*/D | peak | top | shuffled | floor grp | floor rnd | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | brit_culture | culture | last | linear | 0.500 (1.00) | 4 | 0.14 | 0.994 | 0.975 | 0.475 | 0.857 | 0.951 | 157 |
| qwen3-0.6b | brit_culture | culture | last | mlp | 0.500 (1.00) | 5 | 0.18 | 1.000 | 0.983 | 0.486 | 0.857 | 0.951 | 157 |
| qwen3-0.6b | brit_culture | culture | mean | linear | 0.909 (0.00) | 4 | 0.14 | 0.992 | 0.985 | 0.470 | 0.857 | 0.951 | 157 |
| qwen3-0.6b | brit_culture | culture | mean | mlp | 0.915 (0.00) | 1 | 0.04 | 0.994 | 0.987 | 0.468 | 0.857 | 0.951 | 157 |
| qwen3-1.7b | brit_culture | culture | last | linear | 0.500 (1.00) | 4 | 0.14 | 1.000 | 0.996 | 0.477 | 0.857 | 0.951 | 157 |
| qwen3-1.7b | brit_culture | culture | last | mlp | 0.500 (1.00) | 4 | 0.14 | 1.000 | 0.996 | 0.471 | 0.857 | 0.951 | 157 |
| qwen3-1.7b | brit_culture | culture | mean | linear | 0.902 (0.00) | 3 | 0.11 | 1.000 | 0.983 | 0.447 | 0.857 | 0.951 | 157 |
| qwen3-1.7b | brit_culture | culture | mean | mlp | 0.902 (0.00) | 3 | 0.11 | 1.000 | 0.998 | 0.441 | 0.857 | 0.951 | 157 |
| qwen3-4b | brit_culture | culture | last | linear | 0.500 (1.00) | 4 | 0.11 | 1.000 | 0.989 | 0.500 | 0.857 | 0.951 | 157 |
| qwen3-4b | brit_culture | culture | last | mlp | 0.500 (1.00) | 4 | 0.11 | 1.000 | 0.994 | 0.494 | 0.857 | 0.951 | 157 |
| qwen3-4b | brit_culture | culture | mean | linear | 0.898 (0.00) | 1 | 0.03 | 1.000 | 0.996 | 0.461 | 0.857 | 0.951 | 157 |
| qwen3-4b | brit_culture | culture | mean | mlp | 0.896 (0.00) | 1 | 0.03 | 1.000 | 1.000 | 0.452 | 0.857 | 0.951 | 157 |
| qwen3-8b | brit_culture | culture | last | linear | 0.500 (1.00) | 3 | 0.08 | 1.000 | 0.985 | 0.497 | 0.857 | 0.951 | 157 |
| qwen3-8b | brit_culture | culture | last | mlp | 0.500 (1.00) | 4 | 0.11 | 1.000 | 0.987 | 0.484 | 0.857 | 0.951 | 157 |
| qwen3-8b | brit_culture | culture | mean | linear | 0.904 (0.00) | 1 | 0.03 | 1.000 | 0.994 | 0.460 | 0.857 | 0.951 | 157 |
| qwen3-8b | brit_culture | culture | mean | mlp | 0.892 (0.00) | 1 | 0.03 | 1.000 | 0.994 | 0.452 | 0.857 | 0.951 | 157 |
| qwen3-0.6b | brit_language | language | last | linear | 0.500 (1.00) | 1 | 0.04 | 0.979 | 0.968 | 0.529 | 0.784 | 0.985 | 146 |
| qwen3-0.6b | brit_language | language | last | mlp | 0.500 (1.00) | 1 | 0.04 | 0.982 | 0.975 | 0.526 | 0.784 | 0.985 | 146 |
| qwen3-0.6b | brit_language | language | mean | linear | 0.966 (0.00) | 0 | 0.00 | 0.986 | 0.959 | 0.512 | 0.784 | 0.985 | 146 |
| qwen3-0.6b | brit_language | language | mean | mlp | 0.970 (0.00) | 0 | 0.00 | 0.991 | 0.991 | 0.522 | 0.784 | 0.985 | 146 |
| qwen3-1.7b | brit_language | language | last | linear | 0.500 (1.00) | 2 | 0.07 | 0.995 | 0.979 | 0.526 | 0.784 | 0.985 | 146 |
| qwen3-1.7b | brit_language | language | last | mlp | 0.500 (1.00) | 2 | 0.07 | 0.993 | 0.984 | 0.526 | 0.784 | 0.985 | 146 |
| qwen3-1.7b | brit_language | language | mean | linear | 0.977 (0.00) | 0 | 0.00 | 0.993 | 0.984 | 0.513 | 0.784 | 0.985 | 146 |
| qwen3-1.7b | brit_language | language | mean | mlp | 0.979 (0.00) | 0 | 0.00 | 0.991 | 0.982 | 0.522 | 0.784 | 0.985 | 146 |
| qwen3-4b | brit_language | language | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.986 | 0.520 | 0.784 | 0.985 | 146 |
| qwen3-4b | brit_language | language | last | mlp | 0.500 (1.00) | 4 | 0.11 | 1.000 | 0.991 | 0.512 | 0.784 | 0.985 | 146 |
| qwen3-4b | brit_language | language | mean | linear | 0.957 (0.00) | 1 | 0.03 | 1.000 | 0.993 | 0.512 | 0.784 | 0.985 | 146 |
| qwen3-4b | brit_language | language | mean | mlp | 0.966 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.520 | 0.784 | 0.985 | 146 |
| qwen3-8b | brit_language | language | last | linear | 0.500 (1.00) | 4 | 0.11 | 1.000 | 1.000 | 0.484 | 0.784 | 0.985 | 146 |
| qwen3-8b | brit_language | language | last | mlp | 0.500 (1.00) | 2 | 0.06 | 0.998 | 0.998 | 0.512 | 0.784 | 0.985 | 146 |
| qwen3-8b | brit_language | language | mean | linear | 0.975 (0.00) | 0 | 0.00 | 0.998 | 0.993 | 0.501 | 0.784 | 0.985 | 146 |
| qwen3-8b | brit_language | language | mean | mlp | 0.975 (0.00) | 0 | 0.00 | 1.000 | 0.991 | 0.512 | 0.784 | 0.985 | 146 |
| qwen3-0.6b | brit_truth | dialect_to_guard* | last | linear | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.009 | 0.523 | -- | -- | 36 |
| qwen3-0.6b | brit_truth | dialect_to_guard* | last | mlp | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.046 | 0.485 | -- | -- | 36 |
| qwen3-0.6b | brit_truth | dialect_to_guard* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.056 | 0.000 | 0.544 | -- | -- | 36 |
| qwen3-0.6b | brit_truth | dialect_to_guard* | mean | mlp | 0.056 (0.00) | 9 | 0.32 | 0.259 | 0.102 | 0.508 | -- | -- | 36 |
| qwen3-1.7b | brit_truth | dialect_to_guard* | last | linear | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.037 | 0.538 | -- | -- | 36 |
| qwen3-1.7b | brit_truth | dialect_to_guard* | last | mlp | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.009 | 0.523 | -- | -- | 36 |
| qwen3-1.7b | brit_truth | dialect_to_guard* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.028 | 0.000 | 0.506 | -- | -- | 36 |
| qwen3-1.7b | brit_truth | dialect_to_guard* | mean | mlp | 0.056 (0.00) | 0 | 0.00 | 0.083 | 0.037 | 0.499 | -- | -- | 36 |
| qwen3-4b | brit_truth | dialect_to_guard* | last | linear | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.157 | 0.527 | -- | -- | 36 |
| qwen3-4b | brit_truth | dialect_to_guard* | last | mlp | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.046 | 0.512 | -- | -- | 36 |
| qwen3-4b | brit_truth | dialect_to_guard* | mean | linear | 0.019 (0.00) | 0 | 0.00 | 0.019 | 0.000 | 0.524 | -- | -- | 36 |
| qwen3-4b | brit_truth | dialect_to_guard* | mean | mlp | 0.056 (0.00) | 0 | 0.00 | 0.056 | 0.037 | 0.492 | -- | -- | 36 |
| qwen3-8b | brit_truth | dialect_to_guard* | last | linear | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.000 | 0.552 | -- | -- | 36 |
| qwen3-8b | brit_truth | dialect_to_guard* | last | mlp | 0.500 (1.00) | 0 | 0.00 | 0.500 | 0.009 | 0.527 | -- | -- | 36 |
| qwen3-8b | brit_truth | dialect_to_guard* | mean | linear | 0.028 (0.00) | 0 | 0.00 | 0.028 | 0.000 | 0.505 | -- | -- | 36 |
| qwen3-8b | brit_truth | dialect_to_guard* | mean | mlp | 0.083 (0.00) | 0 | 0.00 | 0.083 | 0.000 | 0.489 | -- | -- | 36 |
| qwen3-0.6b | brit_truth | false_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.507 | 0.917 | 0.949 | 36 |
| qwen3-0.6b | brit_truth | false_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.450 | 0.917 | 0.949 | 36 |
| qwen3-0.6b | brit_truth | false_british_over_american | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.485 | 0.917 | 0.949 | 36 |
| qwen3-0.6b | brit_truth | false_british_over_american | mean | mlp | 0.981 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.499 | 0.917 | 0.949 | 36 |
| qwen3-1.7b | brit_truth | false_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.447 | 0.917 | 0.949 | 36 |
| qwen3-1.7b | brit_truth | false_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.464 | 0.917 | 0.949 | 36 |
| qwen3-1.7b | brit_truth | false_british_over_american | mean | linear | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.506 | 0.917 | 0.949 | 36 |
| qwen3-1.7b | brit_truth | false_british_over_american | mean | mlp | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.491 | 0.917 | 0.949 | 36 |
| qwen3-4b | brit_truth | false_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 1.000 | 0.532 | 0.917 | 0.949 | 36 |
| qwen3-4b | brit_truth | false_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.03 | 1.000 | 1.000 | 0.514 | 0.917 | 0.949 | 36 |
| qwen3-4b | brit_truth | false_british_over_american | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.475 | 0.917 | 0.949 | 36 |
| qwen3-4b | brit_truth | false_british_over_american | mean | mlp | 0.981 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.495 | 0.917 | 0.949 | 36 |
| qwen3-8b | brit_truth | false_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.972 | 0.527 | 0.917 | 0.949 | 36 |
| qwen3-8b | brit_truth | false_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.981 | 0.494 | 0.917 | 0.949 | 36 |
| qwen3-8b | brit_truth | false_british_over_american | mean | linear | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.486 | 0.917 | 0.949 | 36 |
| qwen3-8b | brit_truth | false_british_over_american | mean | mlp | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.493 | 0.917 | 0.949 | 36 |
| qwen3-0.6b | brit_truth | true_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.503 | 0.917 | 0.859 | 36 |
| qwen3-0.6b | brit_truth | true_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.04 | 1.000 | 0.972 | 0.509 | 0.917 | 0.859 | 36 |
| qwen3-0.6b | brit_truth | true_british_over_american | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.500 | 0.917 | 0.859 | 36 |
| qwen3-0.6b | brit_truth | true_british_over_american | mean | mlp | 0.981 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.499 | 0.917 | 0.859 | 36 |
| qwen3-1.7b | brit_truth | true_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.513 | 0.917 | 0.859 | 36 |
| qwen3-1.7b | brit_truth | true_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.04 | 1.000 | 1.000 | 0.511 | 0.917 | 0.859 | 36 |
| qwen3-1.7b | brit_truth | true_british_over_american | mean | linear | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.512 | 0.917 | 0.859 | 36 |
| qwen3-1.7b | brit_truth | true_british_over_american | mean | mlp | 0.972 (0.00) | 0 | 0.00 | 1.000 | 0.972 | 0.505 | 0.917 | 0.859 | 36 |
| qwen3-4b | brit_truth | true_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.972 | 0.580 | 0.917 | 0.859 | 36 |
| qwen3-4b | brit_truth | true_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.972 | 0.565 | 0.917 | 0.859 | 36 |
| qwen3-4b | brit_truth | true_british_over_american | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.496 | 0.917 | 0.859 | 36 |
| qwen3-4b | brit_truth | true_british_over_american | mean | mlp | 0.981 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.502 | 0.917 | 0.859 | 36 |
| qwen3-8b | brit_truth | true_british_over_american | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.972 | 0.526 | 0.917 | 0.859 | 36 |
| qwen3-8b | brit_truth | true_british_over_american | last | mlp | 0.500 (1.00) | 1 | 0.03 | 1.000 | 0.972 | 0.526 | 0.917 | 0.859 | 36 |
| qwen3-8b | brit_truth | true_british_over_american | mean | linear | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.522 | 0.917 | 0.859 | 36 |
| qwen3-8b | brit_truth | true_british_over_american | mean | mlp | 0.972 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.478 | 0.917 | 0.859 | 36 |
| qwen3-0.6b | brit_truth | truth_over_british | last | linear | 0.500 (1.00) | 3 | 0.11 | 0.972 | 0.972 | 0.505 | 0.917 | 0.958 | 36 |
| qwen3-0.6b | brit_truth | truth_over_british | last | mlp | 0.500 (1.00) | 4 | 0.14 | 0.972 | 0.972 | 0.539 | 0.917 | 0.958 | 36 |
| qwen3-0.6b | brit_truth | truth_over_british | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.495 | 0.917 | 0.958 | 36 |
| qwen3-0.6b | brit_truth | truth_over_british | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 0.972 | 0.518 | 0.917 | 0.958 | 36 |
| qwen3-1.7b | brit_truth | truth_over_british | last | linear | 0.500 (1.00) | 2 | 0.07 | 1.000 | 1.000 | 0.502 | 0.917 | 0.958 | 36 |
| qwen3-1.7b | brit_truth | truth_over_british | last | mlp | 0.500 (1.00) | 2 | 0.07 | 1.000 | 1.000 | 0.522 | 0.917 | 0.958 | 36 |
| qwen3-1.7b | brit_truth | truth_over_british | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.474 | 0.917 | 0.958 | 36 |
| qwen3-1.7b | brit_truth | truth_over_british | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.524 | 0.917 | 0.958 | 36 |
| qwen3-4b | brit_truth | truth_over_british | last | linear | 0.500 (1.00) | 1 | 0.03 | 1.000 | 1.000 | 0.515 | 0.917 | 0.958 | 36 |
| qwen3-4b | brit_truth | truth_over_british | last | mlp | 0.500 (1.00) | 1 | 0.03 | 1.000 | 1.000 | 0.508 | 0.917 | 0.958 | 36 |
| qwen3-4b | brit_truth | truth_over_british | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.471 | 0.917 | 0.958 | 36 |
| qwen3-4b | brit_truth | truth_over_british | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 0.981 | 0.525 | 0.917 | 0.958 | 36 |
| qwen3-8b | brit_truth | truth_over_british | last | linear | 0.500 (1.00) | 3 | 0.08 | 1.000 | 1.000 | 0.520 | 0.917 | 0.958 | 36 |
| qwen3-8b | brit_truth | truth_over_british | last | mlp | 0.500 (1.00) | 2 | 0.06 | 1.000 | 1.000 | 0.518 | 0.917 | 0.958 | 36 |
| qwen3-8b | brit_truth | truth_over_british | mean | linear | 0.944 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.394 | 0.917 | 0.958 | 36 |
| qwen3-8b | brit_truth | truth_over_british | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.480 | 0.917 | 0.958 | 36 |
| qwen3-0.6b | offsetbias | debiased | last | linear | 0.584 (0.75) | 4 | 0.14 | 0.949 | 0.940 | 0.499 | 0.907 | 0.915 | 798 |
| qwen3-0.6b | offsetbias | debiased | last | mlp | 0.581 (0.75) | 4 | 0.14 | 0.949 | 0.936 | 0.496 | 0.907 | 0.915 | 798 |
| qwen3-0.6b | offsetbias | debiased | mean | linear | 0.905 (0.00) | 2 | 0.07 | 0.959 | 0.931 | 0.524 | 0.907 | 0.915 | 798 |
| qwen3-0.6b | offsetbias | debiased | mean | mlp | 0.898 (0.00) | 2 | 0.07 | 0.958 | 0.925 | 0.497 | 0.907 | 0.915 | 798 |
| qwen3-1.7b | offsetbias | debiased | last | linear | 0.585 (0.75) | 4 | 0.14 | 0.956 | 0.949 | 0.505 | 0.907 | 0.915 | 798 |
| qwen3-1.7b | offsetbias | debiased | last | mlp | 0.580 (0.75) | 6 | 0.21 | 0.959 | 0.953 | 0.504 | 0.907 | 0.915 | 798 |
| qwen3-1.7b | offsetbias | debiased | mean | linear | 0.895 (0.00) | 3 | 0.11 | 0.965 | 0.940 | 0.523 | 0.907 | 0.915 | 798 |
| qwen3-1.7b | offsetbias | debiased | mean | mlp | 0.878 (0.00) | 3 | 0.11 | 0.965 | 0.945 | 0.501 | 0.907 | 0.915 | 798 |
| qwen3-4b | offsetbias | debiased | last | linear | 0.586 (0.75) | 13 | 0.36 | 0.967 | 0.955 | 0.505 | 0.907 | 0.915 | 798 |
| qwen3-4b | offsetbias | debiased | last | mlp | 0.576 (0.75) | 17 | 0.47 | 0.968 | 0.962 | 0.501 | 0.907 | 0.915 | 798 |
| qwen3-4b | offsetbias | debiased | mean | linear | 0.901 (0.00) | 12 | 0.33 | 0.974 | 0.956 | 0.526 | 0.907 | 0.915 | 798 |
| qwen3-4b | offsetbias | debiased | mean | mlp | 0.883 (0.00) | 10 | 0.28 | 0.975 | 0.959 | 0.499 | 0.907 | 0.915 | 798 |
| qwen3-8b | offsetbias | debiased | last | linear | 0.579 (0.75) | 11 | 0.31 | 0.975 | 0.959 | 0.495 | 0.907 | 0.915 | 798 |
| qwen3-8b | offsetbias | debiased | last | mlp | 0.580 (0.75) | 12 | 0.33 | 0.977 | 0.973 | 0.507 | 0.907 | 0.915 | 798 |
| qwen3-8b | offsetbias | debiased | mean | linear | 0.894 (0.00) | 12 | 0.33 | 0.984 | 0.962 | 0.513 | 0.907 | 0.915 | 798 |
| qwen3-8b | offsetbias | debiased | mean | mlp | 0.879 (0.00) | 14 | 0.39 | 0.985 | 0.963 | 0.501 | 0.907 | 0.915 | 798 |
| qwen3-0.6b | rewardbench2 | Factuality | last | linear | 0.569 (0.71) | 4 | 0.14 | 0.794 | 0.741 | 0.534 | 0.649 | 0.688 | 94 |
| qwen3-0.6b | rewardbench2 | Factuality | last | mlp | 0.555 (0.71) | 1 | 0.04 | 0.773 | 0.773 | 0.503 | 0.649 | 0.688 | 94 |
| qwen3-0.6b | rewardbench2 | Factuality | mean | linear | 0.702 (0.00) | 1 | 0.04 | 0.812 | 0.773 | 0.589 | 0.649 | 0.688 | 94 |
| qwen3-0.6b | rewardbench2 | Factuality | mean | mlp | 0.674 (0.00) | 2 | 0.07 | 0.816 | 0.773 | 0.500 | 0.649 | 0.688 | 94 |
| qwen3-1.7b | rewardbench2 | Factuality | last | linear | 0.573 (0.71) | 10 | 0.36 | 0.823 | 0.773 | 0.514 | 0.649 | 0.688 | 94 |
| qwen3-1.7b | rewardbench2 | Factuality | last | mlp | 0.551 (0.71) | 8 | 0.29 | 0.812 | 0.741 | 0.497 | 0.649 | 0.688 | 94 |
| qwen3-1.7b | rewardbench2 | Factuality | mean | linear | 0.699 (0.00) | 3 | 0.11 | 0.858 | 0.805 | 0.581 | 0.649 | 0.688 | 94 |
| qwen3-1.7b | rewardbench2 | Factuality | mean | mlp | 0.681 (0.00) | 9 | 0.32 | 0.809 | 0.798 | 0.502 | 0.649 | 0.688 | 94 |
| qwen3-4b | rewardbench2 | Factuality | last | linear | 0.562 (0.71) | 17 | 0.47 | 0.858 | 0.798 | 0.517 | 0.649 | 0.688 | 94 |
| qwen3-4b | rewardbench2 | Factuality | last | mlp | 0.566 (0.71) | 17 | 0.47 | 0.848 | 0.784 | 0.493 | 0.649 | 0.688 | 94 |
| qwen3-4b | rewardbench2 | Factuality | mean | linear | 0.727 (0.00) | 5 | 0.14 | 0.848 | 0.812 | 0.572 | 0.649 | 0.688 | 94 |
| qwen3-4b | rewardbench2 | Factuality | mean | mlp | 0.667 (0.00) | 16 | 0.44 | 0.851 | 0.791 | 0.530 | 0.649 | 0.688 | 94 |
| qwen3-8b | rewardbench2 | Factuality | last | linear | 0.551 (0.71) | 19 | 0.53 | 0.883 | 0.794 | 0.537 | 0.649 | 0.688 | 94 |
| qwen3-8b | rewardbench2 | Factuality | last | mlp | 0.555 (0.71) | 16 | 0.44 | 0.862 | 0.805 | 0.504 | 0.649 | 0.688 | 94 |
| qwen3-8b | rewardbench2 | Factuality | mean | linear | 0.695 (0.00) | 11 | 0.31 | 0.876 | 0.837 | 0.563 | 0.649 | 0.688 | 94 |
| qwen3-8b | rewardbench2 | Factuality | mean | mlp | 0.720 (0.00) | 18 | 0.50 | 0.887 | 0.816 | 0.515 | 0.649 | 0.688 | 94 |
| qwen3-0.6b | rewardbench2 | Focus | last | linear | 0.570 (0.78) | 2 | 0.07 | 0.867 | 0.797 | 0.564 | 0.830 | 0.784 | 100 |
| qwen3-0.6b | rewardbench2 | Focus | last | mlp | 0.567 (0.78) | 1 | 0.04 | 0.840 | 0.813 | 0.527 | 0.830 | 0.784 | 100 |
| qwen3-0.6b | rewardbench2 | Focus | mean | linear | 0.773 (0.00) | 6 | 0.21 | 0.900 | 0.853 | 0.480 | 0.830 | 0.784 | 100 |
| qwen3-0.6b | rewardbench2 | Focus | mean | mlp | 0.773 (0.00) | 9 | 0.32 | 0.890 | 0.780 | 0.493 | 0.830 | 0.784 | 100 |
| qwen3-1.7b | rewardbench2 | Focus | last | linear | 0.570 (0.78) | 2 | 0.07 | 0.887 | 0.820 | 0.566 | 0.830 | 0.784 | 100 |
| qwen3-1.7b | rewardbench2 | Focus | last | mlp | 0.567 (0.78) | 6 | 0.21 | 0.880 | 0.837 | 0.535 | 0.830 | 0.784 | 100 |
| qwen3-1.7b | rewardbench2 | Focus | mean | linear | 0.800 (0.00) | 6 | 0.21 | 0.910 | 0.847 | 0.502 | 0.830 | 0.784 | 100 |
| qwen3-1.7b | rewardbench2 | Focus | mean | mlp | 0.763 (0.00) | 5 | 0.18 | 0.897 | 0.847 | 0.484 | 0.830 | 0.784 | 100 |
| qwen3-4b | rewardbench2 | Focus | last | linear | 0.570 (0.78) | 18 | 0.50 | 0.940 | 0.913 | 0.559 | 0.830 | 0.784 | 100 |
| qwen3-4b | rewardbench2 | Focus | last | mlp | 0.560 (0.78) | 18 | 0.50 | 0.953 | 0.923 | 0.533 | 0.830 | 0.784 | 100 |
| qwen3-4b | rewardbench2 | Focus | mean | linear | 0.803 (0.00) | 16 | 0.44 | 0.950 | 0.893 | 0.488 | 0.830 | 0.784 | 100 |
| qwen3-4b | rewardbench2 | Focus | mean | mlp | 0.850 (0.00) | 14 | 0.39 | 0.933 | 0.920 | 0.475 | 0.830 | 0.784 | 100 |
| qwen3-8b | rewardbench2 | Focus | last | linear | 0.570 (0.78) | 18 | 0.50 | 0.940 | 0.907 | 0.550 | 0.830 | 0.784 | 100 |
| qwen3-8b | rewardbench2 | Focus | last | mlp | 0.570 (0.78) | 20 | 0.56 | 0.960 | 0.933 | 0.530 | 0.830 | 0.784 | 100 |
| qwen3-8b | rewardbench2 | Focus | mean | linear | 0.797 (0.00) | 8 | 0.22 | 0.937 | 0.880 | 0.472 | 0.830 | 0.784 | 100 |
| qwen3-8b | rewardbench2 | Focus | mean | mlp | 0.793 (0.00) | 13 | 0.36 | 0.930 | 0.873 | 0.471 | 0.830 | 0.784 | 100 |
| qwen3-0.6b | rewardbench2 | Math | last | linear | 0.679 (0.13) | 4 | 0.14 | 0.846 | 0.667 | 0.378 | 0.615 | 0.690 | 39 |
| qwen3-0.6b | rewardbench2 | Math | last | mlp | 0.645 (0.13) | 1 | 0.04 | 0.752 | 0.675 | 0.415 | 0.615 | 0.690 | 39 |
| qwen3-0.6b | rewardbench2 | Math | mean | linear | 0.718 (0.00) | 2 | 0.07 | 0.846 | 0.795 | 0.439 | 0.615 | 0.690 | 39 |
| qwen3-0.6b | rewardbench2 | Math | mean | mlp | 0.650 (0.00) | 7 | 0.25 | 0.795 | 0.701 | 0.393 | 0.615 | 0.690 | 39 |
| qwen3-1.7b | rewardbench2 | Math | last | linear | 0.705 (0.13) | 18 | 0.64 | 0.846 | 0.718 | 0.420 | 0.615 | 0.690 | 39 |
| qwen3-1.7b | rewardbench2 | Math | last | mlp | 0.628 (0.13) | 1 | 0.04 | 0.769 | 0.692 | 0.468 | 0.615 | 0.690 | 39 |
| qwen3-1.7b | rewardbench2 | Math | mean | linear | 0.718 (0.00) | 1 | 0.04 | 0.846 | 0.821 | 0.399 | 0.615 | 0.690 | 39 |
| qwen3-1.7b | rewardbench2 | Math | mean | mlp | 0.615 (0.00) | 6 | 0.21 | 0.786 | 0.752 | 0.419 | 0.615 | 0.690 | 39 |
| qwen3-4b | rewardbench2 | Math | last | linear | 0.654 (0.13) | 3 | 0.08 | 0.846 | 0.795 | 0.384 | 0.615 | 0.690 | 39 |
| qwen3-4b | rewardbench2 | Math | last | mlp | 0.628 (0.13) | 3 | 0.08 | 0.821 | 0.701 | 0.431 | 0.615 | 0.690 | 39 |
| qwen3-4b | rewardbench2 | Math | mean | linear | 0.718 (0.00) | 6 | 0.17 | 0.846 | 0.795 | 0.432 | 0.615 | 0.690 | 39 |
| qwen3-4b | rewardbench2 | Math | mean | mlp | 0.641 (0.00) | 6 | 0.17 | 0.812 | 0.718 | 0.425 | 0.615 | 0.690 | 39 |
| qwen3-8b | rewardbench2 | Math | last | linear | 0.628 (0.13) | 19 | 0.53 | 0.821 | 0.769 | 0.424 | 0.615 | 0.690 | 39 |
| qwen3-8b | rewardbench2 | Math | last | mlp | 0.654 (0.13) | 1 | 0.03 | 0.752 | 0.726 | 0.468 | 0.615 | 0.690 | 39 |
| qwen3-8b | rewardbench2 | Math | mean | linear | 0.718 (0.00) | 11 | 0.31 | 0.872 | 0.744 | 0.392 | 0.615 | 0.690 | 39 |
| qwen3-8b | rewardbench2 | Math | mean | mlp | 0.598 (0.00) | 2 | 0.06 | 0.795 | 0.726 | 0.414 | 0.615 | 0.690 | 39 |
| qwen3-0.6b | rewardbench2 | Precise IF | last | linear | 0.500 (0.76) | 1 | 0.04 | 0.720 | 0.440 | 0.548 | 0.480 | 0.621 | 25 |
| qwen3-0.6b | rewardbench2 | Precise IF | last | mlp | 0.540 (0.76) | 1 | 0.04 | 0.640 | 0.627 | 0.536 | 0.480 | 0.621 | 25 |
| qwen3-0.6b | rewardbench2 | Precise IF | mean | linear | 0.640 (0.00) | 0 | 0.00 | 0.720 | 0.480 | 0.491 | 0.480 | 0.621 | 25 |
| qwen3-0.6b | rewardbench2 | Precise IF | mean | mlp | 0.547 (0.00) | 3 | 0.11 | 0.707 | 0.493 | 0.476 | 0.480 | 0.621 | 25 |
| qwen3-1.7b | rewardbench2 | Precise IF | last | linear | 0.580 (0.76) | 2 | 0.07 | 0.680 | 0.440 | 0.530 | 0.480 | 0.621 | 25 |
| qwen3-1.7b | rewardbench2 | Precise IF | last | mlp | 0.487 (0.76) | 2 | 0.07 | 0.680 | 0.587 | 0.523 | 0.480 | 0.621 | 25 |
| qwen3-1.7b | rewardbench2 | Precise IF | mean | linear | 0.680 (0.00) | 0 | 0.00 | 0.680 | 0.680 | 0.488 | 0.480 | 0.621 | 25 |
| qwen3-1.7b | rewardbench2 | Precise IF | mean | mlp | 0.467 (0.00) | 2 | 0.07 | 0.613 | 0.373 | 0.501 | 0.480 | 0.621 | 25 |
| qwen3-4b | rewardbench2 | Precise IF | last | linear | 0.580 (0.76) | 2 | 0.06 | 0.720 | 0.400 | 0.501 | 0.480 | 0.621 | 25 |
| qwen3-4b | rewardbench2 | Precise IF | last | mlp | 0.553 (0.76) | 0 | 0.00 | 0.640 | 0.453 | 0.508 | 0.480 | 0.621 | 25 |
| qwen3-4b | rewardbench2 | Precise IF | mean | linear | 0.640 (0.00) | 0 | 0.00 | 0.720 | 0.720 | 0.478 | 0.480 | 0.621 | 25 |
| qwen3-4b | rewardbench2 | Precise IF | mean | mlp | 0.547 (0.00) | 0 | 0.00 | 0.600 | 0.507 | 0.451 | 0.480 | 0.621 | 25 |
| qwen3-8b | rewardbench2 | Precise IF | last | linear | 0.580 (0.76) | 1 | 0.03 | 0.720 | 0.400 | 0.557 | 0.480 | 0.621 | 25 |
| qwen3-8b | rewardbench2 | Precise IF | last | mlp | 0.540 (0.76) | 2 | 0.06 | 0.640 | 0.480 | 0.536 | 0.480 | 0.621 | 25 |
| qwen3-8b | rewardbench2 | Precise IF | mean | linear | 0.640 (0.00) | 0 | 0.00 | 0.720 | 0.560 | 0.464 | 0.480 | 0.621 | 25 |
| qwen3-8b | rewardbench2 | Precise IF | mean | mlp | 0.547 (0.00) | 0 | 0.00 | 0.613 | 0.453 | 0.462 | 0.480 | 0.621 | 25 |
| qwen3-0.6b | rewardbench2 | Safety | last | linear | 0.614 (0.64) | 4 | 0.14 | 0.951 | 0.943 | 0.537 | 0.875 | 0.889 | 88 |
| qwen3-0.6b | rewardbench2 | Safety | last | mlp | 0.583 (0.64) | 4 | 0.14 | 0.951 | 0.902 | 0.514 | 0.875 | 0.889 | 88 |
| qwen3-0.6b | rewardbench2 | Safety | mean | linear | 0.909 (0.00) | 1 | 0.04 | 0.966 | 0.947 | 0.586 | 0.875 | 0.889 | 88 |
| qwen3-0.6b | rewardbench2 | Safety | mean | mlp | 0.902 (0.00) | 2 | 0.07 | 0.973 | 0.947 | 0.549 | 0.875 | 0.889 | 88 |
| qwen3-1.7b | rewardbench2 | Safety | last | linear | 0.606 (0.64) | 4 | 0.14 | 0.966 | 0.958 | 0.554 | 0.875 | 0.889 | 88 |
| qwen3-1.7b | rewardbench2 | Safety | last | mlp | 0.625 (0.64) | 4 | 0.14 | 0.955 | 0.943 | 0.538 | 0.875 | 0.889 | 88 |
| qwen3-1.7b | rewardbench2 | Safety | mean | linear | 0.902 (0.00) | 2 | 0.07 | 0.970 | 0.955 | 0.575 | 0.875 | 0.889 | 88 |
| qwen3-1.7b | rewardbench2 | Safety | mean | mlp | 0.902 (0.00) | 2 | 0.07 | 0.977 | 0.962 | 0.561 | 0.875 | 0.889 | 88 |
| qwen3-4b | rewardbench2 | Safety | last | linear | 0.610 (0.64) | 10 | 0.28 | 0.977 | 0.947 | 0.558 | 0.875 | 0.889 | 88 |
| qwen3-4b | rewardbench2 | Safety | last | mlp | 0.617 (0.64) | 5 | 0.14 | 0.977 | 0.932 | 0.518 | 0.875 | 0.889 | 88 |
| qwen3-4b | rewardbench2 | Safety | mean | linear | 0.909 (0.00) | 2 | 0.06 | 0.989 | 0.951 | 0.578 | 0.875 | 0.889 | 88 |
| qwen3-4b | rewardbench2 | Safety | mean | mlp | 0.902 (0.00) | 3 | 0.08 | 0.989 | 0.917 | 0.557 | 0.875 | 0.889 | 88 |
| qwen3-8b | rewardbench2 | Safety | last | linear | 0.629 (0.64) | 5 | 0.14 | 0.977 | 0.951 | 0.567 | 0.875 | 0.889 | 88 |
| qwen3-8b | rewardbench2 | Safety | last | mlp | 0.625 (0.64) | 5 | 0.14 | 0.966 | 0.909 | 0.528 | 0.875 | 0.889 | 88 |
| qwen3-8b | rewardbench2 | Safety | mean | linear | 0.894 (0.00) | 1 | 0.03 | 0.989 | 0.966 | 0.585 | 0.875 | 0.889 | 88 |
| qwen3-8b | rewardbench2 | Safety | mean | mlp | 0.879 (0.00) | 2 | 0.06 | 0.992 | 0.962 | 0.549 | 0.875 | 0.889 | 88 |
| qwen3-0.6b | rewardbench2 | Ties | last | linear | 0.750 (0.05) | 1 | 0.04 | 0.909 | 0.818 | 0.433 | 0.705 | 0.694 | 22 |
| qwen3-0.6b | rewardbench2 | Ties | last | mlp | 0.598 (0.05) | 9 | 0.32 | 0.848 | 0.727 | 0.469 | 0.705 | 0.694 | 22 |
| qwen3-0.6b | rewardbench2 | Ties | mean | linear | 0.477 (0.05) | 1 | 0.04 | 0.818 | 0.682 | 0.504 | 0.705 | 0.694 | 22 |
| qwen3-0.6b | rewardbench2 | Ties | mean | mlp | 0.492 (0.05) | 11 | 0.39 | 0.803 | 0.652 | 0.488 | 0.705 | 0.694 | 22 |
| qwen3-1.7b | rewardbench2 | Ties | last | linear | 0.841 (0.05) | 12 | 0.43 | 1.000 | 0.955 | 0.437 | 0.705 | 0.694 | 22 |
| qwen3-1.7b | rewardbench2 | Ties | last | mlp | 0.705 (0.05) | 14 | 0.50 | 1.000 | 0.955 | 0.495 | 0.705 | 0.694 | 22 |
| qwen3-1.7b | rewardbench2 | Ties | mean | linear | 0.705 (0.05) | 13 | 0.46 | 1.000 | 0.909 | 0.523 | 0.705 | 0.694 | 22 |
| qwen3-1.7b | rewardbench2 | Ties | mean | mlp | 0.553 (0.05) | 14 | 0.50 | 1.000 | 0.924 | 0.493 | 0.705 | 0.694 | 22 |
| qwen3-4b | rewardbench2 | Ties | last | linear | 0.841 (0.05) | 10 | 0.28 | 1.000 | 1.000 | 0.475 | 0.705 | 0.694 | 22 |
| qwen3-4b | rewardbench2 | Ties | last | mlp | 0.705 (0.05) | 15 | 0.42 | 1.000 | 0.955 | 0.488 | 0.705 | 0.694 | 22 |
| qwen3-4b | rewardbench2 | Ties | mean | linear | 0.659 (0.05) | 16 | 0.44 | 1.000 | 0.909 | 0.528 | 0.705 | 0.694 | 22 |
| qwen3-4b | rewardbench2 | Ties | mean | mlp | 0.614 (0.05) | 17 | 0.47 | 1.000 | 0.985 | 0.504 | 0.705 | 0.694 | 22 |
| qwen3-8b | rewardbench2 | Ties | last | linear | 0.841 (0.05) | 16 | 0.44 | 1.000 | 1.000 | 0.485 | 0.705 | 0.694 | 22 |
| qwen3-8b | rewardbench2 | Ties | last | mlp | 0.720 (0.05) | 14 | 0.39 | 1.000 | 0.985 | 0.480 | 0.705 | 0.694 | 22 |
| qwen3-8b | rewardbench2 | Ties | mean | linear | 0.659 (0.05) | 17 | 0.47 | 1.000 | 0.955 | 0.515 | 0.705 | 0.694 | 22 |
| qwen3-8b | rewardbench2 | Ties | mean | mlp | 0.705 (0.05) | 16 | 0.44 | 1.000 | 0.955 | 0.485 | 0.705 | 0.694 | 22 |
| qwen3-0.6b | styc | aligned | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.522 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | aligned | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.547 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | aligned | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.527 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | aligned | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.536 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | aligned | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.533 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | aligned | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.539 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | aligned | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.525 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | aligned | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.539 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | aligned | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.516 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | aligned | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.527 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | aligned | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.539 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | aligned | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.543 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | aligned | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.531 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | aligned | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.527 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | aligned | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.531 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | aligned | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.543 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | conflict | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.560 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | conflict | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.567 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | conflict | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.543 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | conflict | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.547 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | conflict | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.578 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | conflict | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.583 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | conflict | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.574 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | conflict | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.590 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | conflict | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.537 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | conflict | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.538 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | conflict | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.570 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | conflict | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.570 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | conflict | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.550 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | conflict | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.553 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | conflict | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.564 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | conflict | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.563 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | corr_e | last | linear | 0.500 (1.00) | 26 | 0.93 | 0.944 | 0.944 | 0.480 | 0.504 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_e | last | mlp | 0.500 (1.00) | 26 | 0.93 | 0.958 | 0.958 | 0.474 | 0.504 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_e | mean | linear | 0.514 (0.00) | 26 | 0.93 | 0.950 | 0.950 | 0.477 | 0.504 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_e | mean | mlp | 0.508 (0.00) | 24 | 0.86 | 0.950 | 0.950 | 0.477 | 0.504 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_e | last | linear | 0.500 (1.00) | 28 | 1.00 | 0.986 | 0.986 | 0.516 | 0.504 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_e | last | mlp | 0.500 (1.00) | 28 | 1.00 | 0.989 | 0.989 | 0.497 | 0.504 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_e | mean | linear | 0.506 (0.00) | 24 | 0.86 | 0.983 | 0.975 | 0.477 | 0.504 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_e | mean | mlp | 0.506 (0.00) | 24 | 0.86 | 0.992 | 0.992 | 0.485 | 0.504 | 0.465 | 120 |
| qwen3-4b | styc | corr_e | last | linear | 0.500 (1.00) | 28 | 0.78 | 0.983 | 0.983 | 0.487 | 0.504 | 0.465 | 120 |
| qwen3-4b | styc | corr_e | last | mlp | 0.500 (1.00) | 28 | 0.78 | 0.992 | 0.983 | 0.477 | 0.504 | 0.465 | 120 |
| qwen3-4b | styc | corr_e | mean | linear | 0.508 (0.00) | 27 | 0.75 | 0.994 | 0.992 | 0.488 | 0.504 | 0.465 | 120 |
| qwen3-4b | styc | corr_e | mean | mlp | 0.536 (0.00) | 27 | 0.75 | 0.992 | 0.992 | 0.467 | 0.504 | 0.465 | 120 |
| qwen3-8b | styc | corr_e | last | linear | 0.500 (1.00) | 28 | 0.78 | 1.000 | 0.992 | 0.498 | 0.504 | 0.465 | 120 |
| qwen3-8b | styc | corr_e | last | mlp | 0.500 (1.00) | 28 | 0.78 | 1.000 | 0.994 | 0.482 | 0.504 | 0.465 | 120 |
| qwen3-8b | styc | corr_e | mean | linear | 0.506 (0.00) | 27 | 0.75 | 1.000 | 1.000 | 0.505 | 0.504 | 0.465 | 120 |
| qwen3-8b | styc | corr_e | mean | mlp | 0.506 (0.00) | 27 | 0.75 | 1.000 | 1.000 | 0.490 | 0.504 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_t | last | linear | 0.519 (0.62) | 24 | 0.86 | 0.858 | 0.853 | 0.499 | 0.500 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_t | last | mlp | 0.500 (0.62) | 24 | 0.86 | 0.861 | 0.856 | 0.494 | 0.500 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_t | mean | linear | 0.514 (0.00) | 24 | 0.86 | 0.869 | 0.858 | 0.468 | 0.500 | 0.465 | 120 |
| qwen3-0.6b | styc | corr_t | mean | mlp | 0.492 (0.00) | 24 | 0.86 | 0.872 | 0.869 | 0.495 | 0.500 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_t | last | linear | 0.492 (0.62) | 27 | 0.96 | 0.964 | 0.964 | 0.526 | 0.500 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_t | last | mlp | 0.525 (0.62) | 27 | 0.96 | 0.964 | 0.964 | 0.510 | 0.500 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_t | mean | linear | 0.492 (0.00) | 24 | 0.86 | 0.975 | 0.975 | 0.489 | 0.500 | 0.465 | 120 |
| qwen3-1.7b | styc | corr_t | mean | mlp | 0.483 (0.00) | 24 | 0.86 | 0.969 | 0.969 | 0.499 | 0.500 | 0.465 | 120 |
| qwen3-4b | styc | corr_t | last | linear | 0.528 (0.62) | 29 | 0.81 | 0.981 | 0.981 | 0.528 | 0.500 | 0.465 | 120 |
| qwen3-4b | styc | corr_t | last | mlp | 0.508 (0.62) | 28 | 0.78 | 0.969 | 0.969 | 0.497 | 0.500 | 0.465 | 120 |
| qwen3-4b | styc | corr_t | mean | linear | 0.481 (0.00) | 27 | 0.75 | 0.978 | 0.972 | 0.516 | 0.500 | 0.465 | 120 |
| qwen3-4b | styc | corr_t | mean | mlp | 0.506 (0.00) | 27 | 0.75 | 0.983 | 0.969 | 0.496 | 0.500 | 0.465 | 120 |
| qwen3-8b | styc | corr_t | last | linear | 0.503 (0.62) | 28 | 0.78 | 0.989 | 0.989 | 0.544 | 0.500 | 0.465 | 120 |
| qwen3-8b | styc | corr_t | last | mlp | 0.511 (0.62) | 28 | 0.78 | 0.997 | 0.997 | 0.518 | 0.500 | 0.465 | 120 |
| qwen3-8b | styc | corr_t | mean | linear | 0.492 (0.00) | 27 | 0.75 | 0.992 | 0.983 | 0.502 | 0.500 | 0.465 | 120 |
| qwen3-8b | styc | corr_t | mean | mlp | 0.503 (0.00) | 27 | 0.75 | 0.992 | 0.992 | 0.500 | 0.500 | 0.465 | 120 |
| qwen3-0.6b | styc | diet_to_conflict* | last | linear | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.000 | 0.505 | -- | -- | 120 |
| qwen3-0.6b | styc | diet_to_conflict* | last | mlp | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.011 | 0.516 | -- | -- | 120 |
| qwen3-0.6b | styc | diet_to_conflict* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.000 | 0.000 | 0.536 | -- | -- | 120 |
| qwen3-0.6b | styc | diet_to_conflict* | mean | mlp | 0.003 (0.00) | 0 | 0.00 | 0.019 | 0.008 | 0.505 | -- | -- | 120 |
| qwen3-1.7b | styc | diet_to_conflict* | last | linear | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.000 | 0.501 | -- | -- | 120 |
| qwen3-1.7b | styc | diet_to_conflict* | last | mlp | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.042 | 0.496 | -- | -- | 120 |
| qwen3-1.7b | styc | diet_to_conflict* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.000 | 0.000 | 0.513 | -- | -- | 120 |
| qwen3-1.7b | styc | diet_to_conflict* | mean | mlp | 0.000 (0.00) | 7 | 0.25 | 0.056 | 0.044 | 0.510 | -- | -- | 120 |
| qwen3-4b | styc | diet_to_conflict* | last | linear | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.019 | 0.509 | -- | -- | 120 |
| qwen3-4b | styc | diet_to_conflict* | last | mlp | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.092 | 0.492 | -- | -- | 120 |
| qwen3-4b | styc | diet_to_conflict* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.044 | 0.044 | 0.516 | -- | -- | 120 |
| qwen3-4b | styc | diet_to_conflict* | mean | mlp | 0.008 (0.00) | 36 | 1.00 | 0.147 | 0.147 | 0.499 | -- | -- | 120 |
| qwen3-8b | styc | diet_to_conflict* | last | linear | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.014 | 0.511 | -- | -- | 120 |
| qwen3-8b | styc | diet_to_conflict* | last | mlp | 0.267 (0.53) | 0 | 0.00 | 0.267 | 0.097 | 0.495 | -- | -- | 120 |
| qwen3-8b | styc | diet_to_conflict* | mean | linear | 0.000 (0.00) | 0 | 0.00 | 0.039 | 0.000 | 0.490 | -- | -- | 120 |
| qwen3-8b | styc | diet_to_conflict* | mean | mlp | 0.000 (0.00) | 22 | 0.61 | 0.144 | 0.144 | 0.501 | -- | -- | 120 |
| qwen3-0.6b | styc | style_c | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.571 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_c | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.574 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_c | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.554 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_c | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.557 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_c | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.569 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_c | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.580 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_c | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.574 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_c | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.590 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_c | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.537 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_c | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.544 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_c | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.584 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_c | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.582 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_c | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.553 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_c | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.557 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_c | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.567 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_c | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.566 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_w | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.530 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_w | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.548 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_w | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.526 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | styc | style_w | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.528 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_w | last | linear | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.530 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_w | last | mlp | 0.733 (0.53) | 1 | 0.04 | 1.000 | 1.000 | 0.544 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_w | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.527 | 1.000 | 1.000 | 120 |
| qwen3-1.7b | styc | style_w | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.545 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_w | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.526 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_w | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.526 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_w | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.533 | 1.000 | 1.000 | 120 |
| qwen3-4b | styc | style_w | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.545 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_w | last | linear | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.538 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_w | last | mlp | 0.733 (0.53) | 1 | 0.03 | 1.000 | 1.000 | 0.544 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_w | mean | linear | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.528 | 1.000 | 1.000 | 120 |
| qwen3-8b | styc | style_w | mean | mlp | 1.000 (0.00) | 0 | 0.00 | 1.000 | 1.000 | 0.545 | 1.000 | 1.000 | 120 |
| qwen3-0.6b | uf | quality | last | linear | 0.546 (0.44) | 15 | 0.54 | 0.793 | 0.721 | 0.517 | 0.729 | 0.666 | 269 |
| qwen3-0.6b | uf | quality | last | mlp | 0.575 (0.44) | 16 | 0.57 | 0.767 | 0.699 | 0.508 | 0.729 | 0.666 | 269 |
| qwen3-0.6b | uf | quality | mean | linear | 0.706 (0.00) | 10 | 0.36 | 0.825 | 0.761 | 0.525 | 0.729 | 0.666 | 269 |
| qwen3-0.6b | uf | quality | mean | mlp | 0.656 (0.00) | 13 | 0.46 | 0.797 | 0.717 | 0.509 | 0.729 | 0.666 | 269 |
| qwen3-1.7b | uf | quality | last | linear | 0.548 (0.44) | 13 | 0.46 | 0.792 | 0.772 | 0.517 | 0.729 | 0.666 | 269 |
| qwen3-1.7b | uf | quality | last | mlp | 0.545 (0.44) | 12 | 0.43 | 0.767 | 0.730 | 0.511 | 0.729 | 0.666 | 269 |
| qwen3-1.7b | uf | quality | mean | linear | 0.714 (0.00) | 13 | 0.46 | 0.833 | 0.810 | 0.532 | 0.729 | 0.666 | 269 |
| qwen3-1.7b | uf | quality | mean | mlp | 0.665 (0.00) | 11 | 0.39 | 0.807 | 0.773 | 0.512 | 0.729 | 0.666 | 269 |
| qwen3-4b | uf | quality | last | linear | 0.559 (0.44) | 12 | 0.33 | 0.820 | 0.782 | 0.502 | 0.729 | 0.666 | 269 |
| qwen3-4b | uf | quality | last | mlp | 0.559 (0.44) | 19 | 0.53 | 0.803 | 0.739 | 0.492 | 0.729 | 0.666 | 269 |
| qwen3-4b | uf | quality | mean | linear | 0.700 (0.00) | 17 | 0.47 | 0.851 | 0.809 | 0.537 | 0.729 | 0.666 | 269 |
| qwen3-4b | uf | quality | mean | mlp | 0.654 (0.00) | 19 | 0.53 | 0.843 | 0.762 | 0.521 | 0.729 | 0.666 | 269 |
| qwen3-8b | uf | quality | last | linear | 0.566 (0.44) | 17 | 0.47 | 0.815 | 0.798 | 0.496 | 0.729 | 0.666 | 269 |
| qwen3-8b | uf | quality | last | mlp | 0.572 (0.44) | 17 | 0.47 | 0.796 | 0.776 | 0.503 | 0.729 | 0.666 | 269 |
| qwen3-8b | uf | quality | mean | linear | 0.701 (0.00) | 20 | 0.56 | 0.867 | 0.825 | 0.536 | 0.729 | 0.666 | 269 |
| qwen3-8b | uf | quality | mean | mlp | 0.664 (0.00) | 21 | 0.58 | 0.849 | 0.797 | 0.521 | 0.729 | 0.666 | 269 |

`*` = cross-family transfer cell (fit on one family, scored on another).


## Family B — through-head likelihood (is it EXPRESSIBLE through the frozen unembedding?)

Heads distilled on generative replay only; zero preference fitting. `KL` and `agree` are the head-competence covariate — a depth curve that is really a competence curve must show it here.

**Read `agrees-with-base`, not `pref`.** `pref` is the sign of the summed-logp gap: it is dominated by completion length wherever the two sides differ in length (styc `style_c` reads 0.000 and `conflict` 1.000 at every layer *and* at the full base model), and it encodes the model's prior rather than its decodability (brit reads ~0.15 because the base is American-default). `agrees-with-base` asks whether layer L ranks the pair the way the full stack does — the prior and the length term are shared with the reference, so what remains is how much of the model's own ordering is already expressible at L.

| model | arch | dataset | family | layer | agrees-base | pref | pref/tok | corr | KL(base‖head) | agree | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | eagle-2l | brit_culture | culture | 0 | **0.682** | 0.318 | 0.420 | 0.30 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 0 | **0.605** | 0.299 | 0.376 | 0.19 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 0 | **0.516** | 0.338 | 0.389 | 0.19 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 0 | **0.516** | 0.325 | 0.420 | 0.12 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 4 | **0.675** | 0.287 | 0.344 | 0.35 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 4 | **0.561** | 0.331 | 0.439 | 0.15 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 5 | **0.554** | 0.312 | 0.420 | 0.19 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 5 | **0.497** | 0.318 | 0.446 | 0.08 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 8 | **0.662** | 0.274 | 0.389 | 0.41 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 8 | **0.592** | 0.350 | 0.490 | 0.12 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 11 | **0.561** | 0.268 | 0.331 | 0.26 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 11 | **0.503** | 0.312 | 0.535 | 0.13 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 13 | **0.662** | 0.274 | 0.401 | 0.41 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 13 | **0.561** | 0.306 | 0.503 | 0.18 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 16 | **0.592** | 0.299 | 0.389 | 0.26 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 16 | **0.522** | 0.280 | 0.325 | 0.10 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 17 | **0.675** | 0.299 | 0.363 | 0.46 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 17 | **0.561** | 0.318 | 0.471 | 0.24 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 21 | **0.739** | 0.312 | 0.427 | 0.52 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 21 | **0.637** | 0.357 | 0.459 | 0.38 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 22 | **0.554** | 0.274 | 0.389 | 0.25 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 22 | **0.516** | 0.363 | 0.541 | 0.15 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 25 | **0.771** | 0.242 | 0.389 | 0.70 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 25 | **0.739** | 0.318 | 0.459 | 0.69 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 27 | **0.605** | 0.299 | 0.414 | 0.40 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 27 | **0.605** | 0.325 | 0.401 | 0.19 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | 28 | **0.994** | 0.325 | 0.471 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | 28 | **0.987** | 0.401 | 0.541 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | 32 | **0.637** | 0.268 | 0.401 | 0.54 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 32 | **0.662** | 0.395 | 0.541 | 0.57 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | brit_culture | culture | 36 | **0.885** | 0.389 | 0.535 | 0.95 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | 36 | **0.866** | 0.369 | 0.522 | 0.92 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_culture | culture | base | **1.000** | 0.331 | 0.471 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | brit_culture | culture | base | **1.000** | 0.389 | 0.535 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | brit_culture | culture | base | **1.000** | 0.439 | 0.573 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | brit_culture | culture | base | **1.000** | 0.452 | 0.586 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 0 | **0.656** | 0.331 | 0.414 | 0.35 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 0 | **0.554** | 0.312 | 0.376 | 0.10 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 0 | **0.535** | 0.306 | 0.363 | 0.23 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 0 | **0.529** | 0.248 | 0.318 | 0.06 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 4 | **0.656** | 0.306 | 0.357 | 0.34 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 4 | **0.573** | 0.331 | 0.420 | 0.14 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 5 | **0.561** | 0.318 | 0.401 | 0.23 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 5 | **0.548** | 0.293 | 0.433 | 0.10 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 8 | **0.656** | 0.268 | 0.369 | 0.41 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 8 | **0.605** | 0.299 | 0.465 | 0.21 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 11 | **0.586** | 0.293 | 0.331 | 0.24 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 11 | **0.554** | 0.274 | 0.376 | 0.14 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 13 | **0.694** | 0.331 | 0.459 | 0.44 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 13 | **0.586** | 0.293 | 0.459 | 0.20 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 16 | **0.573** | 0.293 | 0.318 | 0.26 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 16 | **0.554** | 0.274 | 0.401 | 0.15 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 17 | **0.662** | 0.287 | 0.382 | 0.45 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 17 | **0.580** | 0.287 | 0.433 | 0.19 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 21 | **0.726** | 0.325 | 0.414 | 0.52 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 21 | **0.605** | 0.376 | 0.459 | 0.34 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 22 | **0.592** | 0.299 | 0.414 | 0.24 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 22 | **0.529** | 0.287 | 0.408 | 0.15 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 25 | **0.745** | 0.255 | 0.414 | 0.68 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 25 | **0.732** | 0.312 | 0.446 | 0.65 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 27 | **0.618** | 0.325 | 0.414 | 0.32 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 27 | **0.605** | 0.338 | 0.465 | 0.34 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | 28 | **1.000** | 0.331 | 0.471 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | 28 | **0.968** | 0.395 | 0.541 | 0.98 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | 32 | **0.611** | 0.318 | 0.427 | 0.43 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 32 | **0.694** | 0.414 | 0.580 | 0.53 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | brit_culture | culture | 36 | **0.904** | 0.420 | 0.561 | 0.97 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | 36 | **0.936** | 0.401 | 0.567 | 0.97 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_culture | culture | base | **1.000** | 0.331 | 0.471 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | brit_culture | culture | base | **1.000** | 0.389 | 0.535 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | brit_culture | culture | base | **1.000** | 0.439 | 0.573 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | brit_culture | culture | base | **1.000** | 0.452 | 0.586 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 0 | **0.637** | 0.350 | 0.401 | 0.32 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 0 | **0.618** | 0.299 | 0.382 | 0.22 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 0 | **0.548** | 0.280 | 0.293 | 0.23 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 0 | **0.497** | 0.293 | 0.357 | 0.09 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 4 | **0.650** | 0.287 | 0.344 | 0.34 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 4 | **0.554** | 0.299 | 0.357 | 0.14 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 5 | **0.541** | 0.287 | 0.318 | 0.21 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 5 | **0.516** | 0.299 | 0.420 | 0.10 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 8 | **0.669** | 0.280 | 0.395 | 0.41 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 8 | **0.580** | 0.287 | 0.439 | 0.19 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 11 | **0.554** | 0.261 | 0.318 | 0.27 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 11 | **0.554** | 0.274 | 0.325 | 0.17 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 13 | **0.682** | 0.293 | 0.414 | 0.41 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 13 | **0.548** | 0.306 | 0.420 | 0.18 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 16 | **0.567** | 0.274 | 0.299 | 0.21 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 16 | **0.541** | 0.274 | 0.376 | 0.16 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 17 | **0.682** | 0.280 | 0.350 | 0.44 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 17 | **0.573** | 0.318 | 0.427 | 0.25 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 21 | **0.713** | 0.312 | 0.408 | 0.54 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 21 | **0.637** | 0.331 | 0.408 | 0.37 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 22 | **0.573** | 0.306 | 0.389 | 0.25 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 22 | **0.522** | 0.280 | 0.376 | 0.17 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 25 | **0.739** | 0.261 | 0.389 | 0.69 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 25 | **0.720** | 0.274 | 0.427 | 0.66 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 27 | **0.624** | 0.344 | 0.459 | 0.37 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 27 | **0.611** | 0.318 | 0.439 | 0.31 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | 28 | **1.000** | 0.331 | 0.471 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | 28 | **0.962** | 0.389 | 0.516 | 0.99 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 32 | **0.637** | 0.306 | 0.427 | 0.55 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 32 | **0.669** | 0.389 | 0.548 | 0.59 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | brit_culture | culture | 36 | **0.949** | 0.427 | 0.554 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | 36 | **0.987** | 0.465 | 0.580 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_culture | culture | base | **1.000** | 0.331 | 0.471 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_culture | culture | base | **1.000** | 0.389 | 0.535 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | brit_culture | culture | base | **1.000** | 0.439 | 0.573 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | brit_culture | culture | base | **1.000** | 0.452 | 0.586 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 0 | **0.675** | 0.350 | 0.401 | 0.40 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 0 | **0.605** | 0.287 | 0.420 | 0.17 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 0 | **0.573** | 0.306 | 0.363 | 0.25 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 0 | **0.516** | 0.287 | 0.382 | 0.11 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 4 | **0.656** | 0.306 | 0.357 | 0.35 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 4 | **0.573** | 0.280 | 0.369 | 0.15 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 5 | **0.567** | 0.287 | 0.344 | 0.21 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 5 | **0.510** | 0.293 | 0.452 | 0.14 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 8 | **0.656** | 0.280 | 0.401 | 0.40 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 8 | **0.567** | 0.325 | 0.452 | 0.18 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 11 | **0.592** | 0.248 | 0.318 | 0.29 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 11 | **0.548** | 0.331 | 0.408 | 0.13 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 13 | **0.688** | 0.274 | 0.414 | 0.41 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 13 | **0.561** | 0.293 | 0.395 | 0.18 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 16 | **0.592** | 0.287 | 0.344 | 0.24 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 16 | **0.484** | 0.318 | 0.465 | 0.09 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 17 | **0.688** | 0.312 | 0.401 | 0.45 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 17 | **0.573** | 0.293 | 0.414 | 0.25 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 21 | **0.726** | 0.312 | 0.420 | 0.53 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 21 | **0.631** | 0.389 | 0.465 | 0.41 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 22 | **0.561** | 0.306 | 0.382 | 0.27 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 22 | **0.522** | 0.280 | 0.376 | 0.16 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 25 | **0.758** | 0.255 | 0.408 | 0.70 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 25 | **0.764** | 0.306 | 0.452 | 0.68 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 27 | **0.599** | 0.293 | 0.408 | 0.34 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 27 | **0.611** | 0.318 | 0.420 | 0.30 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | 28 | **1.000** | 0.331 | 0.471 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | 28 | **0.975** | 0.389 | 0.529 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | 32 | **0.637** | 0.306 | 0.433 | 0.57 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 32 | **0.688** | 0.446 | 0.554 | 0.61 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | brit_culture | culture | 36 | **0.943** | 0.433 | 0.561 | 0.98 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | 36 | **0.924** | 0.452 | 0.580 | 0.97 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_culture | culture | base | **1.000** | 0.331 | 0.471 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | brit_culture | culture | base | **1.000** | 0.389 | 0.535 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | brit_culture | culture | base | **1.000** | 0.439 | 0.573 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | brit_culture | culture | base | **1.000** | 0.452 | 0.586 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | brit_language | language | 0 | **0.795** | 0.212 | 0.301 | 0.14 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 0 | **0.767** | 0.103 | 0.301 | 0.03 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 0 | **0.836** | 0.116 | 0.212 | -0.03 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 0 | **0.747** | 0.123 | 0.240 | -0.31 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 4 | **0.918** | 0.089 | 0.171 | 0.18 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 4 | **0.788** | 0.082 | 0.240 | -0.01 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 5 | **0.829** | 0.110 | 0.192 | -0.01 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 5 | **0.774** | 0.151 | 0.288 | -0.24 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 8 | **0.904** | 0.116 | 0.219 | 0.35 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 8 | **0.760** | 0.110 | 0.205 | -0.01 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 11 | **0.822** | 0.144 | 0.288 | 0.17 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 11 | **0.781** | 0.116 | 0.267 | -0.20 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 13 | **0.911** | 0.110 | 0.219 | 0.37 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 13 | **0.788** | 0.123 | 0.260 | 0.06 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 16 | **0.747** | 0.178 | 0.267 | -0.03 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 16 | **0.774** | 0.151 | 0.260 | -0.26 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 17 | **0.932** | 0.089 | 0.185 | 0.44 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 17 | **0.801** | 0.123 | 0.240 | 0.23 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | brit_language | language | 21 | **0.938** | 0.096 | 0.178 | 0.48 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 21 | **0.808** | 0.158 | 0.322 | 0.31 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 22 | **0.849** | 0.062 | 0.205 | 0.14 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 22 | **0.740** | 0.158 | 0.260 | -0.22 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 25 | **0.945** | 0.116 | 0.267 | 0.71 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 25 | **0.842** | 0.151 | 0.363 | 0.66 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 27 | **0.842** | 0.110 | 0.212 | 0.29 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 27 | **0.781** | 0.144 | 0.226 | -0.12 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | 28 | **1.000** | 0.075 | 0.301 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | 28 | **1.000** | 0.158 | 0.370 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | 32 | **0.877** | 0.144 | 0.295 | 0.55 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 32 | **0.829** | 0.151 | 0.349 | 0.48 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | brit_language | language | 36 | **0.945** | 0.158 | 0.377 | 0.94 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | 36 | **0.918** | 0.185 | 0.377 | 0.91 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_language | language | base | **1.000** | 0.075 | 0.308 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | brit_language | language | base | **1.000** | 0.158 | 0.390 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | brit_language | language | base | **1.000** | 0.158 | 0.377 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | brit_language | language | base | **1.000** | 0.185 | 0.397 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | brit_language | language | 0 | **0.897** | 0.096 | 0.192 | 0.15 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 0 | **0.788** | 0.123 | 0.212 | -0.03 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 0 | **0.849** | 0.075 | 0.164 | 0.02 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 0 | **0.788** | 0.164 | 0.260 | -0.20 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 4 | **0.918** | 0.089 | 0.151 | 0.20 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 4 | **0.788** | 0.082 | 0.219 | 0.05 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 5 | **0.836** | 0.089 | 0.178 | 0.05 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 5 | **0.726** | 0.199 | 0.315 | -0.31 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 8 | **0.911** | 0.096 | 0.199 | 0.35 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 8 | **0.801** | 0.110 | 0.212 | 0.14 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 11 | **0.842** | 0.096 | 0.199 | 0.10 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 11 | **0.774** | 0.137 | 0.281 | -0.14 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 13 | **0.904** | 0.103 | 0.219 | 0.40 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 13 | **0.795** | 0.116 | 0.240 | 0.11 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 16 | **0.842** | 0.110 | 0.247 | 0.21 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 16 | **0.767** | 0.130 | 0.199 | -0.14 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 17 | **0.918** | 0.103 | 0.192 | 0.39 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 17 | **0.801** | 0.123 | 0.240 | 0.18 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | brit_language | language | 21 | **0.938** | 0.082 | 0.199 | 0.50 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 21 | **0.822** | 0.144 | 0.267 | 0.32 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 22 | **0.753** | 0.171 | 0.301 | 0.17 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 22 | **0.788** | 0.110 | 0.260 | -0.05 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 25 | **0.952** | 0.096 | 0.226 | 0.65 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 25 | **0.842** | 0.137 | 0.349 | 0.66 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 27 | **0.801** | 0.151 | 0.301 | 0.25 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 27 | **0.767** | 0.212 | 0.322 | 0.09 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | 28 | **1.000** | 0.075 | 0.301 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | 28 | **0.973** | 0.158 | 0.356 | 0.98 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | 32 | **0.849** | 0.171 | 0.301 | 0.37 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 32 | **0.822** | 0.158 | 0.363 | 0.38 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | brit_language | language | 36 | **0.952** | 0.164 | 0.363 | 0.96 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | 36 | **0.938** | 0.123 | 0.370 | 0.97 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_language | language | base | **1.000** | 0.075 | 0.308 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | brit_language | language | base | **1.000** | 0.158 | 0.390 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | brit_language | language | base | **1.000** | 0.158 | 0.377 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | brit_language | language | base | **1.000** | 0.185 | 0.397 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 0 | **0.897** | 0.123 | 0.260 | 0.17 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 0 | **0.774** | 0.096 | 0.192 | -0.02 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 0 | **0.849** | 0.075 | 0.144 | 0.07 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 0 | **0.740** | 0.130 | 0.260 | -0.30 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 4 | **0.925** | 0.096 | 0.212 | 0.28 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 4 | **0.795** | 0.103 | 0.164 | 0.06 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 5 | **0.836** | 0.089 | 0.247 | 0.09 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 5 | **0.801** | 0.123 | 0.260 | -0.16 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 8 | **0.918** | 0.103 | 0.253 | 0.37 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 8 | **0.747** | 0.137 | 0.205 | 0.12 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 11 | **0.829** | 0.123 | 0.226 | 0.18 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 11 | **0.781** | 0.116 | 0.240 | -0.08 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 13 | **0.911** | 0.110 | 0.219 | 0.42 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 13 | **0.781** | 0.116 | 0.226 | 0.12 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 16 | **0.842** | 0.110 | 0.226 | 0.23 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 16 | **0.760** | 0.110 | 0.226 | -0.10 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 17 | **0.918** | 0.103 | 0.178 | 0.44 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 17 | **0.801** | 0.110 | 0.233 | 0.22 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 21 | **0.945** | 0.075 | 0.151 | 0.51 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 21 | **0.808** | 0.130 | 0.281 | 0.29 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 22 | **0.822** | 0.130 | 0.226 | 0.23 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 22 | **0.795** | 0.116 | 0.274 | 0.05 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 25 | **0.952** | 0.082 | 0.247 | 0.73 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 25 | **0.836** | 0.158 | 0.363 | 0.67 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 27 | **0.856** | 0.110 | 0.219 | 0.23 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 27 | **0.808** | 0.144 | 0.281 | 0.16 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | 28 | **1.000** | 0.075 | 0.308 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | 28 | **0.986** | 0.144 | 0.370 | 0.99 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | 32 | **0.849** | 0.144 | 0.301 | 0.49 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 32 | **0.842** | 0.137 | 0.336 | 0.45 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | brit_language | language | 36 | **0.973** | 0.158 | 0.363 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | 36 | **1.000** | 0.185 | 0.397 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_language | language | base | **1.000** | 0.075 | 0.308 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_language | language | base | **1.000** | 0.158 | 0.390 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | brit_language | language | base | **1.000** | 0.158 | 0.377 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | brit_language | language | base | **1.000** | 0.185 | 0.397 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | brit_language | language | 0 | **0.904** | 0.089 | 0.199 | 0.14 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 0 | **0.781** | 0.116 | 0.212 | 0.02 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 0 | **0.836** | 0.075 | 0.171 | 0.06 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 0 | **0.767** | 0.103 | 0.219 | -0.30 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 4 | **0.911** | 0.096 | 0.178 | 0.25 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 4 | **0.795** | 0.089 | 0.205 | 0.02 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 5 | **0.822** | 0.116 | 0.226 | 0.01 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 5 | **0.781** | 0.116 | 0.226 | -0.22 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 8 | **0.911** | 0.110 | 0.226 | 0.34 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 8 | **0.774** | 0.110 | 0.185 | 0.10 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 11 | **0.815** | 0.123 | 0.199 | 0.05 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 11 | **0.747** | 0.151 | 0.267 | -0.24 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 13 | **0.925** | 0.082 | 0.199 | 0.37 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 13 | **0.808** | 0.103 | 0.219 | 0.16 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 16 | **0.842** | 0.123 | 0.253 | 0.24 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 16 | **0.781** | 0.130 | 0.253 | -0.22 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 17 | **0.911** | 0.096 | 0.171 | 0.44 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 17 | **0.822** | 0.116 | 0.205 | 0.21 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | brit_language | language | 21 | **0.945** | 0.075 | 0.151 | 0.52 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 21 | **0.808** | 0.158 | 0.322 | 0.31 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 22 | **0.842** | 0.110 | 0.219 | 0.25 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 22 | **0.815** | 0.096 | 0.247 | 0.05 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 25 | **0.945** | 0.089 | 0.226 | 0.69 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 25 | **0.842** | 0.151 | 0.349 | 0.66 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 27 | **0.863** | 0.116 | 0.226 | 0.27 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 27 | **0.781** | 0.212 | 0.342 | 0.19 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | 28 | **1.000** | 0.075 | 0.301 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | 28 | **0.966** | 0.164 | 0.377 | 0.99 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | 32 | **0.829** | 0.151 | 0.336 | 0.45 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 32 | **0.815** | 0.151 | 0.356 | 0.51 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | brit_language | language | 36 | **0.945** | 0.144 | 0.370 | 0.98 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | 36 | **0.945** | 0.171 | 0.377 | 0.97 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_language | language | base | **1.000** | 0.075 | 0.308 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | brit_language | language | base | **1.000** | 0.158 | 0.390 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | brit_language | language | base | **1.000** | 0.158 | 0.377 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | brit_language | language | base | **1.000** | 0.185 | 0.397 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 0 | **0.889** | 0.083 | 0.111 | 0.23 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 0 | **0.889** | 0.056 | 0.139 | 0.29 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 0 | **0.889** | 0.083 | 0.167 | -0.11 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 0 | **0.778** | 0.139 | 0.194 | 0.02 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 4 | **0.889** | 0.139 | 0.222 | 0.37 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 4 | **0.861** | 0.083 | 0.111 | 0.21 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 5 | **0.917** | 0.111 | 0.167 | 0.02 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 5 | **0.694** | 0.222 | 0.250 | -0.04 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 8 | **0.861** | 0.111 | 0.167 | 0.48 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 8 | **0.722** | 0.222 | 0.278 | 0.04 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 11 | **0.917** | 0.111 | 0.167 | 0.08 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 11 | **0.722** | 0.194 | 0.194 | -0.01 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 13 | **0.889** | 0.139 | 0.194 | 0.54 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 13 | **0.806** | 0.139 | 0.222 | 0.29 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 16 | **0.861** | 0.167 | 0.250 | 0.06 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 16 | **0.778** | 0.139 | 0.194 | 0.06 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 17 | **0.917** | 0.111 | 0.167 | 0.51 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 17 | **0.861** | 0.139 | 0.167 | 0.43 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 21 | **0.917** | 0.111 | 0.167 | 0.65 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 21 | **0.833** | 0.167 | 0.222 | 0.44 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 22 | **0.944** | 0.083 | 0.167 | 0.03 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 22 | **0.667** | 0.250 | 0.278 | 0.07 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 25 | **0.972** | 0.056 | 0.111 | 0.69 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 25 | **0.917** | 0.139 | 0.167 | 0.67 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 27 | **0.889** | 0.139 | 0.167 | 0.27 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 27 | **0.833** | 0.083 | 0.111 | 0.23 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | 28 | **1.000** | 0.083 | 0.167 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | 28 | **1.000** | 0.111 | 0.194 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 32 | **0.861** | 0.167 | 0.222 | 0.28 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 32 | **0.861** | 0.056 | 0.056 | 0.45 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | 36 | **0.972** | 0.056 | 0.167 | 0.92 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | 36 | **0.889** | 0.139 | 0.250 | 0.88 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | false_british_over_american | base | **1.000** | 0.083 | 0.167 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | false_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | false_british_over_american | base | **1.000** | 0.028 | 0.167 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | false_british_over_american | base | **1.000** | 0.139 | 0.222 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 0 | **0.861** | 0.111 | 0.167 | 0.44 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 0 | **0.917** | 0.083 | 0.139 | 0.15 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 0 | **0.833** | 0.139 | 0.222 | 0.01 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 0 | **0.750** | 0.111 | 0.167 | -0.05 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 4 | **0.861** | 0.111 | 0.139 | 0.45 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 4 | **0.889** | 0.111 | 0.139 | 0.26 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 5 | **0.833** | 0.139 | 0.222 | -0.01 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 5 | **0.806** | 0.056 | 0.111 | -0.06 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 8 | **0.861** | 0.167 | 0.222 | 0.47 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 8 | **0.861** | 0.139 | 0.139 | 0.26 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 11 | **0.889** | 0.139 | 0.222 | 0.14 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 11 | **0.806** | 0.056 | 0.083 | 0.13 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 13 | **0.889** | 0.139 | 0.194 | 0.54 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 13 | **0.833** | 0.111 | 0.167 | 0.32 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 16 | **0.917** | 0.111 | 0.167 | 0.13 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 16 | **0.778** | 0.139 | 0.139 | 0.11 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 17 | **0.889** | 0.139 | 0.194 | 0.51 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 17 | **0.889** | 0.111 | 0.139 | 0.36 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 21 | **0.944** | 0.083 | 0.139 | 0.62 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 21 | **0.806** | 0.139 | 0.167 | 0.33 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 22 | **0.917** | 0.056 | 0.139 | 0.11 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 22 | **0.806** | 0.056 | 0.111 | 0.03 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 25 | **0.972** | 0.056 | 0.083 | 0.66 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 25 | **0.917** | 0.083 | 0.111 | 0.63 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 27 | **0.833** | 0.139 | 0.167 | 0.21 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 27 | **0.750** | 0.167 | 0.167 | 0.26 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | 28 | **1.000** | 0.083 | 0.167 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | 28 | **0.972** | 0.083 | 0.139 | 0.99 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 32 | **0.889** | 0.083 | 0.139 | 0.21 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 32 | **0.806** | 0.111 | 0.111 | 0.44 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | 36 | **1.000** | 0.028 | 0.167 | 0.97 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | 36 | **0.972** | 0.167 | 0.306 | 0.95 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | false_british_over_american | base | **1.000** | 0.083 | 0.167 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | false_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | false_british_over_american | base | **1.000** | 0.028 | 0.167 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | false_british_over_american | base | **1.000** | 0.139 | 0.222 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 0 | **0.917** | 0.056 | 0.167 | 0.43 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 0 | **0.889** | 0.056 | 0.111 | 0.13 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 0 | **0.889** | 0.083 | 0.194 | 0.01 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 0 | **0.806** | 0.111 | 0.194 | 0.08 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 4 | **0.833** | 0.139 | 0.194 | 0.44 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 4 | **0.917** | 0.083 | 0.139 | 0.28 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 5 | **0.917** | 0.111 | 0.167 | 0.08 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 5 | **0.750** | 0.111 | 0.139 | 0.05 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 8 | **0.833** | 0.139 | 0.194 | 0.54 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 8 | **0.833** | 0.111 | 0.167 | 0.29 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 11 | **0.944** | 0.083 | 0.167 | 0.17 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 11 | **0.833** | 0.083 | 0.139 | 0.08 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 13 | **0.861** | 0.111 | 0.167 | 0.51 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 13 | **0.833** | 0.111 | 0.167 | 0.39 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 16 | **0.944** | 0.028 | 0.083 | 0.09 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 16 | **0.806** | 0.056 | 0.083 | 0.16 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 17 | **0.833** | 0.139 | 0.194 | 0.54 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 17 | **0.806** | 0.194 | 0.222 | 0.47 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 21 | **0.917** | 0.111 | 0.167 | 0.66 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 21 | **0.833** | 0.111 | 0.167 | 0.42 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 22 | **0.917** | 0.056 | 0.111 | 0.10 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 22 | **0.778** | 0.083 | 0.139 | 0.10 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 25 | **1.000** | 0.083 | 0.111 | 0.72 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 25 | **0.917** | 0.139 | 0.167 | 0.65 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 27 | **0.889** | 0.139 | 0.194 | 0.24 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 27 | **0.750** | 0.167 | 0.167 | 0.36 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | 28 | **1.000** | 0.083 | 0.167 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 0.97 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 32 | **0.861** | 0.167 | 0.222 | 0.26 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 32 | **0.833** | 0.083 | 0.083 | 0.44 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | 36 | **1.000** | 0.028 | 0.194 | 0.97 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | 36 | **1.000** | 0.139 | 0.278 | 0.96 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | false_british_over_american | base | **1.000** | 0.083 | 0.167 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | false_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | false_british_over_american | base | **1.000** | 0.028 | 0.167 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | false_british_over_american | base | **1.000** | 0.139 | 0.222 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 0 | **0.917** | 0.000 | 0.083 | 0.44 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 0 | **0.917** | 0.083 | 0.139 | 0.31 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 0 | **0.833** | 0.139 | 0.222 | 0.01 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 0 | **0.750** | 0.167 | 0.222 | 0.02 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 4 | **0.833** | 0.139 | 0.194 | 0.40 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 4 | **0.861** | 0.083 | 0.139 | 0.23 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 5 | **0.944** | 0.083 | 0.167 | 0.00 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 5 | **0.778** | 0.139 | 0.194 | 0.13 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 8 | **0.889** | 0.139 | 0.194 | 0.47 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 8 | **0.861** | 0.139 | 0.167 | 0.30 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 11 | **0.917** | 0.056 | 0.139 | -0.03 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 11 | **0.694** | 0.222 | 0.278 | 0.04 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 13 | **0.861** | 0.111 | 0.167 | 0.53 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 13 | **0.861** | 0.139 | 0.194 | 0.40 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 16 | **0.972** | 0.000 | 0.111 | 0.08 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 16 | **0.833** | 0.139 | 0.194 | 0.19 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 17 | **0.889** | 0.139 | 0.194 | 0.55 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 17 | **0.861** | 0.139 | 0.167 | 0.38 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 21 | **0.917** | 0.111 | 0.167 | 0.63 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 21 | **0.861** | 0.139 | 0.194 | 0.43 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 22 | **0.861** | 0.111 | 0.167 | 0.16 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 22 | **0.806** | 0.056 | 0.111 | 0.11 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 25 | **0.972** | 0.056 | 0.111 | 0.72 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 25 | **0.944** | 0.111 | 0.139 | 0.67 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 27 | **0.889** | 0.139 | 0.194 | 0.22 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 27 | **0.806** | 0.111 | 0.111 | 0.27 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | 28 | **1.000** | 0.083 | 0.167 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 32 | **0.917** | 0.111 | 0.167 | 0.28 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 32 | **0.806** | 0.111 | 0.139 | 0.51 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | 36 | **0.972** | 0.056 | 0.167 | 0.97 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | 36 | **0.972** | 0.111 | 0.250 | 0.97 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | false_british_over_american | base | **1.000** | 0.083 | 0.167 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | false_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | false_british_over_american | base | **1.000** | 0.028 | 0.167 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | false_british_over_american | base | **1.000** | 0.139 | 0.222 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 0 | **0.861** | 0.083 | 0.111 | 0.25 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 0 | **0.861** | 0.056 | 0.139 | 0.11 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 0 | **0.833** | 0.111 | 0.194 | 0.02 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 0 | **0.833** | 0.139 | 0.194 | 0.21 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 4 | **0.806** | 0.194 | 0.250 | 0.33 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 4 | **0.889** | 0.083 | 0.111 | -0.02 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 5 | **0.861** | 0.083 | 0.139 | 0.08 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 5 | **0.639** | 0.222 | 0.250 | 0.10 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 8 | **0.806** | 0.139 | 0.194 | 0.49 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 8 | **0.750** | 0.167 | 0.222 | -0.04 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 11 | **0.861** | 0.139 | 0.194 | 0.19 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 11 | **0.722** | 0.139 | 0.139 | 0.09 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 13 | **0.861** | 0.139 | 0.194 | 0.55 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 13 | **0.806** | 0.222 | 0.306 | 0.05 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 16 | **0.833** | 0.167 | 0.278 | 0.13 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 16 | **0.806** | 0.111 | 0.167 | 0.23 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 17 | **0.917** | 0.083 | 0.139 | 0.53 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 17 | **0.806** | 0.167 | 0.194 | 0.23 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 21 | **0.861** | 0.083 | 0.139 | 0.66 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 21 | **0.806** | 0.167 | 0.222 | 0.26 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 22 | **0.861** | 0.083 | 0.194 | 0.07 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 22 | **0.722** | 0.139 | 0.139 | 0.15 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 25 | **0.917** | 0.083 | 0.167 | 0.71 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 25 | **0.972** | 0.167 | 0.167 | 0.57 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 27 | **0.833** | 0.167 | 0.167 | 0.31 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 27 | **0.833** | 0.139 | 0.139 | 0.38 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | 28 | **1.000** | 0.139 | 0.250 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 32 | **0.833** | 0.222 | 0.222 | 0.42 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 32 | **0.889** | 0.083 | 0.083 | 0.58 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | 36 | **0.972** | 0.083 | 0.222 | 0.93 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | 36 | **1.000** | 0.139 | 0.194 | 0.96 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | true_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.250 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | true_british_over_american | base | **1.000** | 0.056 | 0.167 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.194 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 0 | **0.861** | 0.139 | 0.194 | 0.44 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 0 | **0.889** | 0.083 | 0.139 | -0.04 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 0 | **0.806** | 0.139 | 0.222 | 0.04 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 0 | **0.806** | 0.111 | 0.167 | 0.19 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 4 | **0.833** | 0.111 | 0.167 | 0.35 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 4 | **0.861** | 0.111 | 0.139 | 0.05 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 5 | **0.833** | 0.111 | 0.194 | 0.11 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 5 | **0.861** | 0.056 | 0.111 | 0.05 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 8 | **0.806** | 0.194 | 0.250 | 0.50 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 8 | **0.833** | 0.139 | 0.139 | 0.03 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 11 | **0.861** | 0.139 | 0.222 | 0.18 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 11 | **0.861** | 0.056 | 0.083 | 0.32 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 13 | **0.806** | 0.139 | 0.194 | 0.53 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 13 | **0.833** | 0.139 | 0.167 | 0.03 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 16 | **0.861** | 0.139 | 0.194 | 0.25 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 16 | **0.861** | 0.167 | 0.167 | 0.33 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 17 | **0.833** | 0.167 | 0.222 | 0.55 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 17 | **0.806** | 0.167 | 0.222 | 0.14 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 21 | **0.889** | 0.111 | 0.194 | 0.63 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 21 | **0.806** | 0.167 | 0.222 | 0.17 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 22 | **0.917** | 0.083 | 0.167 | 0.17 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 22 | **0.861** | 0.056 | 0.111 | 0.31 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 25 | **0.944** | 0.056 | 0.111 | 0.66 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 25 | **0.917** | 0.111 | 0.139 | 0.50 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 27 | **0.833** | 0.222 | 0.250 | 0.27 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 27 | **0.833** | 0.194 | 0.222 | 0.35 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | 28 | **1.000** | 0.139 | 0.250 | 0.99 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 32 | **0.861** | 0.139 | 0.167 | 0.46 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 32 | **0.861** | 0.111 | 0.111 | 0.52 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | 36 | **1.000** | 0.056 | 0.194 | 0.94 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | 36 | **1.000** | 0.139 | 0.194 | 0.96 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | true_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.250 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | true_british_over_american | base | **1.000** | 0.056 | 0.167 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.194 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 0 | **0.889** | 0.056 | 0.167 | 0.43 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 0 | **0.861** | 0.056 | 0.111 | -0.03 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 0 | **0.861** | 0.083 | 0.194 | 0.08 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 0 | **0.806** | 0.111 | 0.194 | 0.24 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 4 | **0.806** | 0.139 | 0.194 | 0.38 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 4 | **0.889** | 0.083 | 0.139 | 0.04 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 5 | **0.889** | 0.111 | 0.167 | 0.12 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 5 | **0.861** | 0.111 | 0.139 | 0.25 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 8 | **0.861** | 0.083 | 0.139 | 0.53 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 8 | **0.806** | 0.111 | 0.139 | 0.04 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 11 | **0.944** | 0.056 | 0.139 | 0.21 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 11 | **0.806** | 0.056 | 0.111 | 0.24 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 13 | **0.833** | 0.111 | 0.167 | 0.52 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 13 | **0.833** | 0.083 | 0.139 | 0.10 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 16 | **0.917** | 0.028 | 0.083 | 0.18 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 16 | **0.889** | 0.083 | 0.111 | 0.32 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 17 | **0.833** | 0.167 | 0.222 | 0.57 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 17 | **0.861** | 0.167 | 0.194 | 0.24 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 21 | **0.889** | 0.111 | 0.167 | 0.67 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 21 | **0.861** | 0.111 | 0.167 | 0.28 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 22 | **0.889** | 0.056 | 0.111 | 0.21 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 22 | **0.861** | 0.056 | 0.111 | 0.32 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 25 | **0.972** | 0.083 | 0.139 | 0.73 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 25 | **0.944** | 0.139 | 0.167 | 0.54 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 27 | **0.833** | 0.167 | 0.194 | 0.31 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 27 | **0.861** | 0.111 | 0.111 | 0.48 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | 28 | **1.000** | 0.139 | 0.250 | 0.99 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 32 | **0.861** | 0.194 | 0.194 | 0.44 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 32 | **0.917** | 0.111 | 0.139 | 0.64 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | 36 | **1.000** | 0.056 | 0.167 | 0.97 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | 36 | **1.000** | 0.139 | 0.194 | 0.99 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | true_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.250 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | true_british_over_american | base | **1.000** | 0.056 | 0.167 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.194 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 0 | **0.889** | 0.000 | 0.083 | 0.48 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 0 | **0.889** | 0.083 | 0.139 | 0.12 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 0 | **0.833** | 0.111 | 0.194 | 0.04 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 0 | **0.722** | 0.139 | 0.222 | 0.24 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 4 | **0.861** | 0.139 | 0.194 | 0.34 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 4 | **0.833** | 0.083 | 0.139 | 0.01 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 5 | **0.889** | 0.111 | 0.194 | 0.06 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 5 | **0.806** | 0.167 | 0.222 | 0.30 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 8 | **0.889** | 0.111 | 0.167 | 0.46 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 8 | **0.833** | 0.139 | 0.167 | 0.02 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 11 | **0.861** | 0.083 | 0.167 | 0.07 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 11 | **0.833** | 0.194 | 0.222 | 0.26 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 13 | **0.889** | 0.111 | 0.167 | 0.53 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 13 | **0.861** | 0.111 | 0.194 | 0.13 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 16 | **0.944** | 0.056 | 0.111 | 0.18 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 16 | **0.806** | 0.167 | 0.222 | 0.33 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 17 | **0.833** | 0.167 | 0.222 | 0.56 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 17 | **0.750** | 0.167 | 0.222 | 0.14 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 21 | **0.861** | 0.083 | 0.139 | 0.66 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 21 | **0.833** | 0.194 | 0.250 | 0.24 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 22 | **0.861** | 0.083 | 0.139 | 0.20 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 22 | **0.833** | 0.083 | 0.111 | 0.29 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 25 | **0.972** | 0.083 | 0.139 | 0.73 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 25 | **0.972** | 0.167 | 0.167 | 0.53 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 27 | **0.778** | 0.222 | 0.222 | 0.33 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 27 | **0.861** | 0.167 | 0.194 | 0.42 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | 28 | **1.000** | 0.111 | 0.167 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | 28 | **1.000** | 0.139 | 0.250 | 0.99 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 32 | **0.861** | 0.194 | 0.194 | 0.45 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 32 | **0.889** | 0.139 | 0.139 | 0.68 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | 36 | **0.972** | 0.028 | 0.139 | 0.97 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | 36 | **0.944** | 0.083 | 0.139 | 0.98 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | true_british_over_american | base | **1.000** | 0.111 | 0.167 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.250 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | true_british_over_american | base | **1.000** | 0.056 | 0.167 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | true_british_over_american | base | **1.000** | 0.139 | 0.194 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 0 | **0.750** | 0.833 | 0.833 | 0.06 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 0 | **0.917** | 0.917 | 0.861 | 0.14 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 0 | **0.694** | 0.722 | 0.722 | -0.05 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 0 | **0.889** | 0.889 | 0.917 | -0.08 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 4 | **0.722** | 0.806 | 0.750 | 0.06 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 4 | **0.806** | 0.806 | 0.833 | 0.07 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 5 | **0.861** | 0.889 | 0.861 | -0.09 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 5 | **0.694** | 0.694 | 0.694 | 0.09 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 8 | **0.722** | 0.750 | 0.722 | 0.20 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 8 | **0.722** | 0.722 | 0.750 | -0.18 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 11 | **0.778** | 0.806 | 0.833 | 0.00 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 11 | **0.806** | 0.806 | 0.833 | -0.02 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 13 | **0.778** | 0.750 | 0.722 | 0.21 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 13 | **0.889** | 0.889 | 0.861 | 0.04 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 16 | **0.750** | 0.778 | 0.778 | -0.00 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 16 | **0.861** | 0.861 | 0.861 | 0.06 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 17 | **0.806** | 0.778 | 0.778 | 0.28 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 17 | **0.861** | 0.806 | 0.778 | 0.19 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 21 | **0.778** | 0.750 | 0.750 | 0.27 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 21 | **0.833** | 0.778 | 0.778 | 0.33 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 22 | **0.833** | 0.861 | 0.833 | 0.10 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 22 | **0.806** | 0.806 | 0.806 | 0.05 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 25 | **0.889** | 0.861 | 0.833 | 0.61 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 25 | **0.972** | 0.972 | 0.944 | 0.62 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 27 | **0.833** | 0.861 | 0.861 | 0.11 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 27 | **0.833** | 0.833 | 0.833 | 0.02 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | 28 | **1.000** | 0.917 | 0.889 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | 28 | **1.000** | 0.944 | 0.861 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 32 | **0.944** | 0.917 | 0.917 | 0.54 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 32 | **0.944** | 0.944 | 0.944 | 0.26 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | 36 | **1.000** | 0.972 | 0.917 | 0.97 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | 36 | **1.000** | 1.000 | 0.972 | 0.96 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | brit_truth | truth_over_british | base | **1.000** | 0.917 | 0.889 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | brit_truth | truth_over_british | base | **1.000** | 0.944 | 0.861 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | brit_truth | truth_over_british | base | **1.000** | 0.972 | 0.917 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | brit_truth | truth_over_british | base | **1.000** | 1.000 | 0.944 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 0 | **0.833** | 0.861 | 0.861 | 0.23 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 0 | **0.917** | 0.917 | 0.861 | 0.19 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 0 | **0.806** | 0.833 | 0.806 | -0.13 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 0 | **0.778** | 0.778 | 0.722 | -0.09 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 4 | **0.722** | 0.806 | 0.750 | 0.03 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 4 | **0.833** | 0.833 | 0.833 | 0.05 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 5 | **0.806** | 0.833 | 0.833 | -0.17 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 5 | **0.861** | 0.861 | 0.806 | 0.04 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 8 | **0.694** | 0.722 | 0.667 | 0.17 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 8 | **0.778** | 0.833 | 0.861 | 0.03 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 11 | **0.778** | 0.806 | 0.778 | -0.02 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 11 | **0.861** | 0.861 | 0.889 | 0.04 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 13 | **0.778** | 0.750 | 0.750 | 0.20 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 13 | **0.861** | 0.861 | 0.833 | 0.07 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 16 | **0.806** | 0.833 | 0.861 | -0.01 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 16 | **0.778** | 0.778 | 0.722 | -0.04 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 17 | **0.806** | 0.778 | 0.750 | 0.21 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 17 | **0.750** | 0.750 | 0.750 | 0.11 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 21 | **0.778** | 0.750 | 0.750 | 0.29 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 21 | **0.778** | 0.722 | 0.750 | 0.30 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 22 | **0.861** | 0.889 | 0.889 | 0.12 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 22 | **0.861** | 0.861 | 0.917 | 0.03 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 25 | **0.889** | 0.861 | 0.861 | 0.58 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 25 | **0.972** | 0.972 | 0.917 | 0.50 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 27 | **0.722** | 0.750 | 0.778 | 0.18 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 27 | **0.833** | 0.833 | 0.861 | 0.11 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | 28 | **1.000** | 0.917 | 0.889 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | 28 | **0.972** | 0.917 | 0.861 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 32 | **0.972** | 0.944 | 0.944 | 0.61 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 32 | **0.972** | 0.972 | 0.972 | 0.33 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | 36 | **0.917** | 0.944 | 0.944 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | 36 | **0.972** | 0.972 | 0.944 | 0.97 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | brit_truth | truth_over_british | base | **1.000** | 0.917 | 0.889 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | brit_truth | truth_over_british | base | **1.000** | 0.944 | 0.861 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | brit_truth | truth_over_british | base | **1.000** | 0.972 | 0.917 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | brit_truth | truth_over_british | base | **1.000** | 1.000 | 0.944 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 0 | **0.806** | 0.833 | 0.778 | 0.10 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 0 | **0.889** | 0.944 | 0.889 | -0.01 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 0 | **0.778** | 0.806 | 0.833 | -0.06 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 0 | **0.806** | 0.806 | 0.778 | 0.00 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 4 | **0.722** | 0.806 | 0.750 | 0.16 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 4 | **0.861** | 0.861 | 0.806 | 0.11 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 5 | **0.778** | 0.806 | 0.861 | -0.00 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 5 | **0.889** | 0.889 | 0.861 | -0.09 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 8 | **0.722** | 0.750 | 0.722 | 0.28 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 8 | **0.806** | 0.806 | 0.833 | 0.03 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 11 | **0.750** | 0.778 | 0.806 | -0.01 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 11 | **0.889** | 0.889 | 0.861 | 0.10 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 13 | **0.750** | 0.722 | 0.722 | 0.24 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 13 | **0.833** | 0.833 | 0.806 | 0.06 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 16 | **0.778** | 0.806 | 0.861 | 0.12 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 16 | **0.833** | 0.833 | 0.806 | 0.09 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 17 | **0.750** | 0.778 | 0.806 | 0.23 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 17 | **0.833** | 0.833 | 0.778 | 0.17 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 21 | **0.778** | 0.750 | 0.750 | 0.26 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 21 | **0.861** | 0.806 | 0.806 | 0.37 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 22 | **0.833** | 0.861 | 0.861 | 0.15 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 22 | **0.861** | 0.861 | 0.861 | 0.03 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 25 | **0.889** | 0.861 | 0.833 | 0.64 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 25 | **0.972** | 0.972 | 0.944 | 0.53 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 27 | **0.833** | 0.806 | 0.806 | 0.16 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 27 | **0.917** | 0.917 | 0.944 | 0.23 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | 28 | **1.000** | 0.917 | 0.889 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | 28 | **0.944** | 0.889 | 0.833 | 0.98 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 32 | **0.944** | 0.917 | 0.889 | 0.56 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 32 | **1.000** | 1.000 | 1.000 | 0.25 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | 36 | **1.000** | 0.972 | 0.917 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | 36 | **0.972** | 0.972 | 0.917 | 0.99 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | brit_truth | truth_over_british | base | **1.000** | 0.917 | 0.889 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | brit_truth | truth_over_british | base | **1.000** | 0.944 | 0.861 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | brit_truth | truth_over_british | base | **1.000** | 0.972 | 0.917 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | brit_truth | truth_over_british | base | **1.000** | 1.000 | 0.944 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 0 | **0.833** | 0.861 | 0.889 | 0.28 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 0 | **0.917** | 0.917 | 0.889 | 0.07 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 0 | **0.806** | 0.833 | 0.750 | -0.12 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 0 | **0.806** | 0.806 | 0.806 | -0.06 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 4 | **0.722** | 0.806 | 0.750 | 0.05 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 4 | **0.889** | 0.889 | 0.889 | -0.04 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 5 | **0.806** | 0.833 | 0.861 | -0.04 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 5 | **0.778** | 0.778 | 0.750 | 0.15 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 8 | **0.722** | 0.750 | 0.750 | 0.17 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 8 | **0.750** | 0.806 | 0.833 | 0.01 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 11 | **0.778** | 0.806 | 0.806 | 0.01 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 11 | **0.722** | 0.722 | 0.750 | -0.19 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 13 | **0.778** | 0.750 | 0.722 | 0.24 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 13 | **0.861** | 0.861 | 0.806 | 0.07 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 16 | **0.806** | 0.833 | 0.778 | -0.03 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 16 | **0.778** | 0.778 | 0.806 | 0.16 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 17 | **0.778** | 0.806 | 0.778 | 0.19 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 17 | **0.833** | 0.833 | 0.806 | 0.12 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 21 | **0.778** | 0.750 | 0.750 | 0.24 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 21 | **0.861** | 0.806 | 0.833 | 0.30 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 22 | **0.861** | 0.889 | 0.917 | 0.09 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 22 | **0.889** | 0.889 | 0.944 | 0.05 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 25 | **0.861** | 0.833 | 0.861 | 0.64 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 25 | **0.917** | 0.917 | 0.917 | 0.53 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 27 | **0.861** | 0.833 | 0.833 | 0.13 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 27 | **0.833** | 0.833 | 0.889 | 0.20 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | 28 | **1.000** | 0.917 | 0.889 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | 28 | **1.000** | 0.944 | 0.861 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 32 | **0.917** | 0.944 | 0.944 | 0.60 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 32 | **1.000** | 1.000 | 1.000 | 0.27 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | 36 | **1.000** | 0.972 | 0.917 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | 36 | **1.000** | 1.000 | 0.972 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | brit_truth | truth_over_british | base | **1.000** | 0.917 | 0.889 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | brit_truth | truth_over_british | base | **1.000** | 0.944 | 0.861 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | brit_truth | truth_over_british | base | **1.000** | 0.972 | 0.917 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | brit_truth | truth_over_british | base | **1.000** | 1.000 | 0.944 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 0 | **0.906** | 0.858 | 0.576 | 0.90 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 0 | **0.895** | 0.860 | 0.554 | 0.90 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 5 | **0.896** | 0.863 | 0.571 | 0.91 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 5 | **0.893** | 0.863 | 0.580 | 0.90 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 11 | **0.907** | 0.862 | 0.607 | 0.91 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 11 | **0.897** | 0.857 | 0.586 | 0.90 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 16 | **0.904** | 0.863 | 0.619 | 0.90 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 16 | **0.898** | 0.856 | 0.555 | 0.90 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 22 | **0.907** | 0.865 | 0.629 | 0.91 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 22 | **0.901** | 0.858 | 0.602 | 0.90 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 27 | **0.929** | 0.863 | 0.619 | 0.96 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 27 | **0.910** | 0.860 | 0.576 | 0.93 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 32 | **0.954** | 0.851 | 0.623 | 0.94 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 32 | **0.956** | 0.848 | 0.596 | 0.98 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | 36 | **0.979** | 0.841 | 0.487 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | 36 | **0.982** | 0.857 | 0.516 | 1.00 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | offsetbias | debiased | base | **1.000** | 0.842 | 0.481 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | offsetbias | debiased | base | **1.000** | 0.845 | 0.489 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 0 | **0.907** | 0.857 | 0.556 | 0.90 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 0 | **0.898** | 0.856 | 0.559 | 0.90 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 5 | **0.902** | 0.862 | 0.576 | 0.91 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 5 | **0.891** | 0.861 | 0.580 | 0.91 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 11 | **0.904** | 0.868 | 0.628 | 0.91 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 11 | **0.888** | 0.858 | 0.561 | 0.90 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 16 | **0.900** | 0.862 | 0.632 | 0.92 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 16 | **0.902** | 0.860 | 0.579 | 0.90 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 22 | **0.905** | 0.860 | 0.581 | 0.93 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 22 | **0.895** | 0.850 | 0.580 | 0.92 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 27 | **0.926** | 0.863 | 0.648 | 0.67 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 27 | **0.920** | 0.865 | 0.588 | 0.79 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 32 | **0.954** | 0.856 | 0.623 | 0.98 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 32 | **0.955** | 0.860 | 0.603 | 0.98 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | 36 | **0.989** | 0.843 | 0.481 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | 36 | **0.991** | 0.851 | 0.496 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | offsetbias | debiased | base | **1.000** | 0.842 | 0.481 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | offsetbias | debiased | base | **1.000** | 0.845 | 0.489 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 0 | **0.649** | 0.468 | 0.479 | 0.44 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 0 | **0.681** | 0.489 | 0.596 | 0.40 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 5 | **0.660** | 0.479 | 0.521 | 0.48 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 5 | **0.660** | 0.468 | 0.585 | 0.41 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 11 | **0.660** | 0.479 | 0.489 | 0.47 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 11 | **0.691** | 0.479 | 0.574 | 0.42 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 16 | **0.660** | 0.479 | 0.489 | 0.46 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 16 | **0.691** | 0.479 | 0.574 | 0.43 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 22 | **0.670** | 0.489 | 0.543 | 0.46 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 22 | **0.670** | 0.457 | 0.574 | 0.42 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 27 | **0.734** | 0.511 | 0.681 | 0.63 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 27 | **0.691** | 0.479 | 0.628 | 0.44 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 32 | **0.862** | 0.596 | 0.755 | 0.74 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 32 | **0.819** | 0.606 | 0.723 | 0.76 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | 36 | **0.957** | 0.691 | 0.745 | 0.99 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | 36 | **0.936** | 0.660 | 0.766 | 0.99 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Factuality | base | **1.000** | 0.734 | 0.734 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Factuality | base | **1.000** | 0.681 | 0.777 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 0 | **0.649** | 0.468 | 0.553 | 0.45 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 0 | **0.670** | 0.479 | 0.596 | 0.42 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 5 | **0.649** | 0.468 | 0.521 | 0.47 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 5 | **0.681** | 0.468 | 0.574 | 0.44 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 11 | **0.660** | 0.479 | 0.553 | 0.47 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 11 | **0.681** | 0.468 | 0.543 | 0.43 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 16 | **0.670** | 0.489 | 0.543 | 0.46 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 16 | **0.660** | 0.468 | 0.596 | 0.43 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 22 | **0.681** | 0.500 | 0.617 | 0.45 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 22 | **0.702** | 0.489 | 0.585 | 0.44 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 27 | **0.628** | 0.447 | 0.553 | 0.32 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 27 | **0.745** | 0.489 | 0.638 | 0.04 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 32 | **0.851** | 0.585 | 0.777 | 0.75 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 32 | **0.830** | 0.596 | 0.734 | 0.80 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | 36 | **0.979** | 0.734 | 0.734 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | 36 | **0.989** | 0.691 | 0.766 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Factuality | base | **1.000** | 0.734 | 0.734 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Factuality | base | **1.000** | 0.681 | 0.777 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 0 | **0.830** | 0.670 | 0.640 | 0.87 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 0 | **0.900** | 0.670 | 0.640 | 0.88 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 5 | **0.850** | 0.670 | 0.550 | 0.88 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 5 | **0.870** | 0.660 | 0.620 | 0.88 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 11 | **0.840** | 0.660 | 0.600 | 0.88 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 11 | **0.910** | 0.680 | 0.600 | 0.88 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 16 | **0.830** | 0.690 | 0.680 | 0.87 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 16 | **0.900** | 0.690 | 0.610 | 0.88 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 22 | **0.840** | 0.680 | 0.630 | 0.87 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 22 | **0.880** | 0.650 | 0.620 | 0.88 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 27 | **0.830** | 0.670 | 0.620 | 0.89 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 27 | **0.850** | 0.640 | 0.540 | 0.88 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 32 | **0.830** | 0.710 | 0.620 | 0.91 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 32 | **0.890** | 0.680 | 0.560 | 0.90 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | 36 | **0.980** | 0.720 | 0.450 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | 36 | **0.950** | 0.680 | 0.400 | 0.99 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Focus | base | **1.000** | 0.740 | 0.450 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Focus | base | **1.000** | 0.690 | 0.460 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 0 | **0.830** | 0.670 | 0.660 | 0.87 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 0 | **0.890** | 0.680 | 0.640 | 0.88 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 5 | **0.850** | 0.670 | 0.590 | 0.88 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 5 | **0.890** | 0.660 | 0.620 | 0.88 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 11 | **0.840** | 0.660 | 0.620 | 0.88 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 11 | **0.880** | 0.650 | 0.650 | 0.88 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 16 | **0.820** | 0.660 | 0.640 | 0.87 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 16 | **0.880** | 0.650 | 0.590 | 0.88 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 22 | **0.810** | 0.650 | 0.540 | 0.87 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 22 | **0.840** | 0.630 | 0.540 | 0.87 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 27 | **0.790** | 0.730 | 0.730 | 0.58 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 27 | **0.870** | 0.700 | 0.600 | 0.37 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 32 | **0.830** | 0.710 | 0.630 | 0.90 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 32 | **0.910** | 0.680 | 0.570 | 0.92 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | 36 | **0.980** | 0.720 | 0.440 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | 36 | **0.970** | 0.660 | 0.440 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Focus | base | **1.000** | 0.740 | 0.450 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Focus | base | **1.000** | 0.690 | 0.460 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 0 | **0.744** | 0.615 | 0.333 | 0.67 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 0 | **0.769** | 0.615 | 0.359 | 0.76 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 5 | **0.769** | 0.641 | 0.462 | 0.73 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 5 | **0.769** | 0.615 | 0.308 | 0.73 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 11 | **0.769** | 0.641 | 0.462 | 0.69 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 11 | **0.769** | 0.615 | 0.359 | 0.72 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 16 | **0.769** | 0.641 | 0.359 | 0.69 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 16 | **0.769** | 0.615 | 0.333 | 0.77 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 22 | **0.744** | 0.615 | 0.462 | 0.69 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 22 | **0.769** | 0.615 | 0.333 | 0.72 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 27 | **0.769** | 0.641 | 0.615 | 0.78 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 27 | **0.821** | 0.615 | 0.462 | 0.78 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 32 | **0.821** | 0.641 | 0.410 | 0.91 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 32 | **0.872** | 0.615 | 0.462 | 0.92 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | 36 | **0.974** | 0.692 | 0.513 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | 36 | **0.974** | 0.667 | 0.538 | 1.00 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Math | base | **1.000** | 0.667 | 0.513 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Math | base | **1.000** | 0.641 | 0.487 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 0 | **0.744** | 0.615 | 0.333 | 0.66 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 0 | **0.744** | 0.590 | 0.333 | 0.76 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 5 | **0.769** | 0.641 | 0.462 | 0.71 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 5 | **0.769** | 0.615 | 0.359 | 0.70 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 11 | **0.769** | 0.641 | 0.487 | 0.71 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 11 | **0.795** | 0.641 | 0.359 | 0.73 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 16 | **0.769** | 0.641 | 0.462 | 0.68 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 16 | **0.769** | 0.615 | 0.282 | 0.72 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 22 | **0.769** | 0.641 | 0.513 | 0.70 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 22 | **0.795** | 0.641 | 0.462 | 0.74 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 27 | **0.744** | 0.564 | 0.513 | 0.63 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 27 | **0.821** | 0.615 | 0.615 | 0.82 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 32 | **0.795** | 0.615 | 0.487 | 0.91 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 32 | **0.872** | 0.564 | 0.538 | 0.83 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | 36 | **1.000** | 0.667 | 0.487 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | 36 | **0.974** | 0.667 | 0.538 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Math | base | **1.000** | 0.667 | 0.513 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Math | base | **1.000** | 0.641 | 0.487 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 0 | **0.760** | 0.280 | 0.440 | 0.78 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 0 | **0.880** | 0.280 | 0.320 | 0.88 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 5 | **0.760** | 0.280 | 0.400 | 0.82 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 5 | **0.920** | 0.240 | 0.440 | 0.88 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 11 | **0.800** | 0.240 | 0.400 | 0.81 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 11 | **0.920** | 0.240 | 0.440 | 0.89 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 16 | **0.800** | 0.240 | 0.440 | 0.79 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 16 | **0.880** | 0.280 | 0.360 | 0.86 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 22 | **0.800** | 0.320 | 0.440 | 0.80 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 22 | **0.880** | 0.280 | 0.480 | 0.88 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 27 | **0.840** | 0.280 | 0.480 | 0.86 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 27 | **0.920** | 0.320 | 0.360 | 0.85 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 32 | **0.880** | 0.320 | 0.360 | 0.90 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 32 | **0.800** | 0.360 | 0.480 | 0.94 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | 36 | **0.920** | 0.360 | 0.480 | 0.99 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | 36 | **1.000** | 0.320 | 0.440 | 0.98 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Precise IF | base | **1.000** | 0.440 | 0.560 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Precise IF | base | **1.000** | 0.320 | 0.520 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 0 | **0.760** | 0.280 | 0.440 | 0.79 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 0 | **0.920** | 0.240 | 0.400 | 0.88 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 5 | **0.800** | 0.240 | 0.400 | 0.84 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 5 | **0.920** | 0.240 | 0.480 | 0.88 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 11 | **0.800** | 0.240 | 0.440 | 0.82 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 11 | **0.920** | 0.240 | 0.400 | 0.89 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 16 | **0.840** | 0.280 | 0.400 | 0.81 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 16 | **0.880** | 0.280 | 0.480 | 0.90 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 22 | **0.760** | 0.280 | 0.400 | 0.85 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 22 | **0.840** | 0.320 | 0.440 | 0.89 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 27 | **0.800** | 0.240 | 0.360 | 0.80 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 27 | **0.880** | 0.360 | 0.520 | 0.90 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 32 | **0.880** | 0.320 | 0.400 | 0.91 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 32 | **0.800** | 0.360 | 0.480 | 0.90 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | 36 | **1.000** | 0.440 | 0.640 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | 36 | **0.960** | 0.360 | 0.520 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Precise IF | base | **1.000** | 0.440 | 0.560 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Precise IF | base | **1.000** | 0.320 | 0.520 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 0 | **0.841** | 0.557 | 0.648 | 0.78 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 0 | **0.830** | 0.568 | 0.682 | 0.78 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 5 | **0.864** | 0.580 | 0.648 | 0.79 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 5 | **0.841** | 0.580 | 0.636 | 0.77 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 11 | **0.864** | 0.580 | 0.636 | 0.78 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 11 | **0.830** | 0.568 | 0.625 | 0.76 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 16 | **0.841** | 0.557 | 0.648 | 0.77 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 16 | **0.830** | 0.568 | 0.557 | 0.75 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 22 | **0.852** | 0.568 | 0.682 | 0.76 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 22 | **0.830** | 0.568 | 0.625 | 0.76 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 27 | **0.920** | 0.636 | 0.795 | 0.76 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 27 | **0.875** | 0.614 | 0.807 | 0.62 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 32 | **0.955** | 0.693 | 0.784 | 0.91 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 32 | **0.943** | 0.682 | 0.807 | 0.89 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | 36 | **0.966** | 0.636 | 0.648 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | 36 | **0.989** | 0.682 | 0.705 | 1.00 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Safety | base | **1.000** | 0.670 | 0.648 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Safety | base | **1.000** | 0.693 | 0.705 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 0 | **0.841** | 0.557 | 0.659 | 0.78 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 0 | **0.830** | 0.568 | 0.670 | 0.77 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 5 | **0.864** | 0.580 | 0.670 | 0.77 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 5 | **0.841** | 0.580 | 0.591 | 0.75 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 11 | **0.864** | 0.580 | 0.580 | 0.78 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 11 | **0.841** | 0.580 | 0.636 | 0.77 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 16 | **0.864** | 0.580 | 0.602 | 0.75 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 16 | **0.841** | 0.580 | 0.625 | 0.73 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 22 | **0.875** | 0.591 | 0.739 | 0.70 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 22 | **0.830** | 0.568 | 0.739 | 0.60 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 27 | **0.932** | 0.625 | 0.761 | 0.70 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 27 | **0.875** | 0.614 | 0.795 | 0.66 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 32 | **0.955** | 0.693 | 0.773 | 0.92 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 32 | **0.943** | 0.682 | 0.784 | 0.93 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | 36 | **0.989** | 0.659 | 0.659 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | 36 | **0.989** | 0.705 | 0.705 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Safety | base | **1.000** | 0.670 | 0.648 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Safety | base | **1.000** | 0.693 | 0.705 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 0 | **0.591** | 0.545 | 0.545 | 0.48 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 0 | **0.500** | 0.409 | 0.455 | 0.25 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 5 | **0.455** | 0.409 | 0.455 | 0.44 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 5 | **0.591** | 0.500 | 0.545 | 0.17 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 11 | **0.500** | 0.455 | 0.455 | 0.36 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 11 | **0.636** | 0.545 | 0.636 | 0.31 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 16 | **0.455** | 0.409 | 0.409 | 0.43 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 16 | **0.591** | 0.500 | 0.591 | 0.31 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 22 | **0.545** | 0.500 | 0.500 | 0.40 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 22 | **0.636** | 0.545 | 0.591 | 0.26 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 27 | **0.545** | 0.500 | 0.682 | 0.87 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 27 | **0.591** | 0.500 | 0.636 | 0.47 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 32 | **0.636** | 0.682 | 0.727 | 0.88 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 32 | **0.682** | 0.682 | 0.682 | 0.86 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | 36 | **0.864** | 0.818 | 0.773 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | 36 | **1.000** | 0.818 | 0.818 | 0.99 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | rewardbench2 | Ties | base | **1.000** | 0.773 | 0.682 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | rewardbench2 | Ties | base | **1.000** | 0.818 | 0.818 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 0 | **0.455** | 0.409 | 0.364 | 0.59 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 0 | **0.500** | 0.409 | 0.455 | 0.22 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 5 | **0.500** | 0.455 | 0.455 | 0.52 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 5 | **0.500** | 0.409 | 0.455 | 0.25 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 11 | **0.500** | 0.455 | 0.545 | 0.45 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 11 | **0.545** | 0.455 | 0.591 | 0.31 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 16 | **0.545** | 0.500 | 0.545 | 0.50 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 16 | **0.591** | 0.500 | 0.500 | 0.25 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 22 | **0.591** | 0.545 | 0.682 | 0.52 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 22 | **0.591** | 0.500 | 0.545 | 0.18 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 27 | **0.545** | 0.500 | 0.636 | 0.82 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 27 | **0.727** | 0.545 | 0.636 | 0.85 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 32 | **0.591** | 0.545 | 0.636 | 0.87 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 32 | **0.773** | 0.682 | 0.636 | 0.88 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | 36 | **1.000** | 0.773 | 0.682 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | 36 | **1.000** | 0.818 | 0.818 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | rewardbench2 | Ties | base | **1.000** | 0.773 | 0.682 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | rewardbench2 | Ties | base | **1.000** | 0.818 | 0.818 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | aligned | 0 | **0.983** | 0.000 | 0.200 | -0.22 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 0 | **0.983** | 0.000 | 0.333 | 0.01 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 0 | **1.000** | 0.000 | 0.300 | -0.20 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 0 | **0.850** | 0.000 | 0.467 | -0.46 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 4 | **0.983** | 0.000 | 0.975 | -0.05 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 4 | **0.983** | 0.000 | 0.283 | 0.02 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 5 | **1.000** | 0.000 | 0.992 | 0.21 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 5 | **0.850** | 0.000 | 0.917 | -0.40 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 8 | **0.983** | 0.000 | 0.767 | -0.01 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 8 | **0.983** | 0.000 | 0.392 | 0.13 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 11 | **1.000** | 0.000 | 0.800 | 0.09 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 11 | **0.850** | 0.000 | 0.850 | -0.13 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 13 | **0.983** | 0.000 | 0.900 | 0.26 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 13 | **0.983** | 0.000 | 0.767 | 0.20 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 16 | **1.000** | 0.000 | 0.950 | 0.12 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 16 | **0.850** | 0.000 | 0.983 | -0.42 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 17 | **0.983** | 0.000 | 0.950 | 0.37 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 17 | **0.983** | 0.000 | 0.800 | 0.44 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | aligned | 21 | **0.983** | 0.000 | 0.975 | 0.61 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 21 | **0.983** | 0.000 | 0.908 | 0.71 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 22 | **1.000** | 0.000 | 0.833 | -0.02 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 22 | **0.850** | 0.000 | 0.458 | -0.11 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 25 | **0.983** | 0.000 | 0.983 | 0.94 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 25 | **0.983** | 0.000 | 0.942 | 0.89 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 27 | **1.000** | 0.000 | 0.958 | 0.74 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 27 | **0.842** | 0.008 | 0.892 | 0.64 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | 28 | **1.000** | 0.017 | 0.958 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | 28 | **1.000** | 0.017 | 0.942 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | 32 | **1.000** | 0.000 | 0.967 | 0.86 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 32 | **0.833** | 0.017 | 0.967 | 0.83 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | aligned | 36 | **0.983** | 0.017 | 0.983 | 0.94 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | 36 | **0.867** | 0.067 | 0.983 | 0.97 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | aligned | base | **1.000** | 0.017 | 0.958 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | aligned | base | **1.000** | 0.017 | 0.942 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | aligned | base | **1.000** | 0.000 | 0.975 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | aligned | base | **1.000** | 0.150 | 1.000 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | aligned | 0 | **0.983** | 0.000 | 0.575 | -0.37 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 0 | **0.983** | 0.000 | 0.200 | -0.12 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 0 | **1.000** | 0.000 | 0.283 | -0.23 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 0 | **0.850** | 0.000 | 0.083 | -0.42 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 4 | **0.983** | 0.000 | 0.992 | 0.20 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 4 | **0.983** | 0.000 | 0.792 | 0.34 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 5 | **1.000** | 0.000 | 0.208 | -0.02 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 5 | **0.850** | 0.000 | 0.958 | -0.48 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 8 | **0.983** | 0.000 | 0.875 | 0.22 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 8 | **0.983** | 0.000 | 1.000 | 0.17 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 11 | **1.000** | 0.000 | 0.958 | 0.03 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 11 | **0.850** | 0.000 | 0.792 | 0.14 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 13 | **0.983** | 0.000 | 0.917 | 0.41 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 13 | **0.983** | 0.000 | 0.825 | 0.18 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 16 | **1.000** | 0.000 | 1.000 | 0.25 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 16 | **0.850** | 0.000 | 0.933 | -0.00 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 17 | **0.983** | 0.000 | 0.983 | 0.42 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 17 | **0.983** | 0.000 | 0.750 | 0.22 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | aligned | 21 | **0.983** | 0.000 | 0.933 | 0.64 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 21 | **0.975** | 0.008 | 0.967 | 0.62 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 22 | **1.000** | 0.000 | 0.950 | 0.18 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 22 | **0.850** | 0.000 | 0.883 | -0.55 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 25 | **0.983** | 0.000 | 0.967 | 0.91 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 25 | **0.983** | 0.000 | 0.950 | 0.79 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 27 | **0.983** | 0.017 | 0.867 | 0.40 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 27 | **0.833** | 0.017 | 0.950 | 0.40 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | 28 | **1.000** | 0.017 | 0.958 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | 28 | **1.000** | 0.017 | 0.942 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | 32 | **0.908** | 0.092 | 0.950 | 0.81 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 32 | **0.850** | 0.000 | 0.975 | 0.87 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | aligned | 36 | **1.000** | 0.000 | 0.975 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | 36 | **0.908** | 0.058 | 1.000 | 0.98 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | aligned | base | **1.000** | 0.017 | 0.958 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | aligned | base | **1.000** | 0.017 | 0.942 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | aligned | base | **1.000** | 0.000 | 0.975 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | aligned | base | **1.000** | 0.150 | 1.000 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 0 | **0.983** | 0.000 | 0.208 | -0.20 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 0 | **0.983** | 0.000 | 0.217 | 0.04 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 0 | **1.000** | 0.000 | 0.217 | -0.27 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 0 | **0.850** | 0.000 | 0.450 | -0.32 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 4 | **0.983** | 0.000 | 0.933 | 0.20 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 4 | **0.983** | 0.000 | 0.983 | 0.30 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 5 | **1.000** | 0.000 | 0.908 | 0.29 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 5 | **0.850** | 0.000 | 0.808 | -0.20 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 8 | **0.983** | 0.000 | 0.858 | 0.12 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 8 | **0.983** | 0.000 | 1.000 | 0.35 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 11 | **1.000** | 0.000 | 1.000 | 0.11 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 11 | **0.850** | 0.000 | 0.700 | -0.26 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 13 | **0.983** | 0.000 | 0.925 | 0.36 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 13 | **0.983** | 0.000 | 0.908 | 0.30 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 16 | **1.000** | 0.000 | 1.000 | 0.27 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 16 | **0.850** | 0.000 | 0.983 | 0.15 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 17 | **0.983** | 0.000 | 0.883 | 0.26 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 17 | **0.983** | 0.000 | 0.842 | 0.44 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 21 | **0.983** | 0.000 | 0.975 | 0.66 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 21 | **0.975** | 0.008 | 0.917 | 0.65 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 22 | **1.000** | 0.000 | 1.000 | 0.30 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 22 | **0.850** | 0.000 | 0.950 | -0.35 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 25 | **0.983** | 0.000 | 0.925 | 0.93 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 25 | **0.983** | 0.000 | 0.950 | 0.86 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 27 | **1.000** | 0.000 | 0.892 | 0.72 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 27 | **0.842** | 0.008 | 0.958 | 0.81 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | 28 | **1.000** | 0.017 | 0.958 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | 28 | **0.867** | 0.150 | 0.942 | 0.95 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | 32 | **0.942** | 0.058 | 0.925 | 0.88 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 32 | **0.800** | 0.067 | 0.958 | 0.83 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | aligned | 36 | **1.000** | 0.000 | 0.975 | 0.96 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | 36 | **1.000** | 0.150 | 1.000 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | aligned | base | **1.000** | 0.017 | 0.958 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | aligned | base | **1.000** | 0.017 | 0.942 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | aligned | base | **1.000** | 0.000 | 0.975 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | aligned | base | **1.000** | 0.150 | 1.000 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | aligned | 0 | **0.983** | 0.000 | 0.517 | -0.24 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 0 | **0.983** | 0.000 | 0.583 | -0.01 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 0 | **1.000** | 0.000 | 0.517 | -0.23 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 0 | **0.850** | 0.000 | 0.392 | -0.26 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 4 | **0.983** | 0.000 | 0.817 | 0.11 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 4 | **0.983** | 0.000 | 0.983 | 0.27 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 5 | **1.000** | 0.000 | 0.950 | 0.30 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 5 | **0.850** | 0.000 | 0.917 | -0.26 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 8 | **0.983** | 0.000 | 0.942 | -0.09 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 8 | **0.983** | 0.000 | 0.833 | 0.20 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 11 | **1.000** | 0.000 | 1.000 | 0.07 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 11 | **0.850** | 0.000 | 0.950 | -0.08 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 13 | **0.983** | 0.000 | 0.858 | 0.26 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 13 | **0.983** | 0.000 | 0.858 | 0.30 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 16 | **1.000** | 0.000 | 1.000 | 0.28 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 16 | **0.850** | 0.000 | 0.908 | -0.01 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 17 | **0.983** | 0.000 | 0.983 | 0.55 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 17 | **0.983** | 0.000 | 0.842 | 0.46 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | aligned | 21 | **0.983** | 0.000 | 0.975 | 0.66 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 21 | **0.975** | 0.008 | 0.917 | 0.67 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 22 | **1.000** | 0.000 | 0.983 | 0.28 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 22 | **0.850** | 0.000 | 0.958 | -0.14 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 25 | **0.983** | 0.000 | 0.967 | 0.94 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 25 | **0.942** | 0.042 | 0.967 | 0.76 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 27 | **1.000** | 0.000 | 0.933 | 0.70 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 27 | **0.833** | 0.017 | 0.992 | 0.80 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | 28 | **1.000** | 0.017 | 0.958 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | 28 | **1.000** | 0.017 | 0.942 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | 32 | **0.958** | 0.042 | 0.900 | 0.87 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 32 | **0.850** | 0.000 | 0.917 | 0.81 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | aligned | 36 | **1.000** | 0.000 | 0.983 | 0.99 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | 36 | **0.933** | 0.100 | 1.000 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | aligned | base | **1.000** | 0.017 | 0.958 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | aligned | base | **1.000** | 0.017 | 0.942 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | aligned | base | **1.000** | 0.000 | 0.975 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | aligned | base | **1.000** | 0.150 | 1.000 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | conflict | 0 | **1.000** | 1.000 | 0.817 | -0.03 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 0 | **1.000** | 1.000 | 0.650 | 0.49 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 0 | **1.000** | 1.000 | 0.733 | 0.24 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 0 | **1.000** | 1.000 | 0.525 | -0.17 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 4 | **1.000** | 1.000 | 0.008 | 0.11 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 4 | **1.000** | 1.000 | 0.683 | 0.44 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 5 | **1.000** | 1.000 | 0.000 | 0.55 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 5 | **1.000** | 1.000 | 0.058 | -0.10 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 8 | **1.000** | 1.000 | 0.175 | 0.13 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 8 | **1.000** | 1.000 | 0.617 | 0.58 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 11 | **1.000** | 1.000 | 0.175 | 0.55 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 11 | **1.000** | 1.000 | 0.183 | 0.13 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 13 | **1.000** | 1.000 | 0.075 | 0.36 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 13 | **1.000** | 1.000 | 0.208 | 0.67 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 16 | **1.000** | 1.000 | 0.042 | 0.59 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 16 | **1.000** | 1.000 | 0.033 | -0.19 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 17 | **1.000** | 1.000 | 0.033 | 0.45 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 17 | **1.000** | 1.000 | 0.258 | 0.81 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | conflict | 21 | **1.000** | 1.000 | 0.042 | 0.72 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 21 | **1.000** | 1.000 | 0.108 | 0.78 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 22 | **1.000** | 1.000 | 0.175 | 0.39 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 22 | **1.000** | 1.000 | 0.542 | 0.19 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 25 | **1.000** | 1.000 | 0.092 | 0.96 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 25 | **1.000** | 1.000 | 0.175 | 0.80 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 27 | **1.000** | 1.000 | 0.225 | 0.60 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 27 | **1.000** | 1.000 | 0.258 | 0.48 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | 28 | **1.000** | 1.000 | 0.083 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | 28 | **1.000** | 1.000 | 0.217 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | 32 | **1.000** | 1.000 | 0.167 | 0.77 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 32 | **1.000** | 1.000 | 0.233 | 0.87 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | conflict | 36 | **1.000** | 1.000 | 0.017 | 0.93 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | 36 | **1.000** | 1.000 | 0.033 | 0.96 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | conflict | base | **1.000** | 1.000 | 0.075 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | conflict | base | **1.000** | 1.000 | 0.208 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | conflict | base | **1.000** | 1.000 | 0.033 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | conflict | base | **1.000** | 1.000 | 0.017 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | conflict | 0 | **1.000** | 1.000 | 0.433 | -0.23 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 0 | **1.000** | 1.000 | 0.783 | 0.37 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 0 | **1.000** | 1.000 | 0.708 | 0.20 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 0 | **1.000** | 1.000 | 0.883 | -0.18 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 4 | **1.000** | 1.000 | 0.000 | 0.36 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 4 | **1.000** | 1.000 | 0.167 | 0.71 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 5 | **1.000** | 1.000 | 0.825 | 0.43 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 5 | **1.000** | 1.000 | 0.025 | -0.16 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 8 | **1.000** | 1.000 | 0.100 | 0.35 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 8 | **1.000** | 1.000 | 0.008 | 0.54 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 11 | **1.000** | 1.000 | 0.042 | 0.51 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 11 | **1.000** | 1.000 | 0.183 | 0.39 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 13 | **1.000** | 1.000 | 0.067 | 0.56 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 13 | **1.000** | 1.000 | 0.167 | 0.61 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 16 | **1.000** | 1.000 | 0.000 | 0.56 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 16 | **1.000** | 1.000 | 0.067 | 0.21 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 17 | **1.000** | 1.000 | 0.017 | 0.60 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 17 | **1.000** | 1.000 | 0.300 | 0.67 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | conflict | 21 | **1.000** | 1.000 | 0.083 | 0.77 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 21 | **1.000** | 1.000 | 0.033 | 0.69 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 22 | **1.000** | 1.000 | 0.050 | 0.62 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 22 | **1.000** | 1.000 | 0.167 | -0.42 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 25 | **1.000** | 1.000 | 0.050 | 0.95 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 25 | **1.000** | 1.000 | 0.192 | 0.66 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 27 | **1.000** | 1.000 | 0.350 | 0.43 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 27 | **1.000** | 1.000 | 0.283 | 0.24 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | 28 | **1.000** | 1.000 | 0.083 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | 28 | **1.000** | 1.000 | 0.258 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | 32 | **1.000** | 1.000 | 0.250 | 0.71 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 32 | **1.000** | 1.000 | 0.233 | 0.89 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | conflict | 36 | **1.000** | 1.000 | 0.067 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | 36 | **1.000** | 1.000 | 0.025 | 0.97 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | conflict | base | **1.000** | 1.000 | 0.075 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | conflict | base | **1.000** | 1.000 | 0.208 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | conflict | base | **1.000** | 1.000 | 0.033 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | conflict | base | **1.000** | 1.000 | 0.017 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 0 | **1.000** | 1.000 | 0.758 | -0.04 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 0 | **1.000** | 1.000 | 0.750 | 0.51 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 0 | **1.000** | 1.000 | 0.775 | 0.21 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 0 | **1.000** | 1.000 | 0.533 | -0.03 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 4 | **1.000** | 1.000 | 0.067 | 0.34 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 4 | **1.000** | 1.000 | 0.008 | 0.68 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 5 | **1.000** | 1.000 | 0.075 | 0.53 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 5 | **1.000** | 1.000 | 0.208 | -0.05 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 8 | **1.000** | 1.000 | 0.092 | 0.24 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 8 | **1.000** | 1.000 | 0.000 | 0.67 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 11 | **1.000** | 1.000 | 0.000 | 0.47 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 11 | **1.000** | 1.000 | 0.367 | -0.04 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 13 | **1.000** | 1.000 | 0.058 | 0.48 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 13 | **1.000** | 1.000 | 0.092 | 0.71 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 16 | **1.000** | 1.000 | 0.000 | 0.55 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 16 | **1.000** | 1.000 | 0.008 | 0.24 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 17 | **1.000** | 1.000 | 0.075 | 0.41 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 17 | **1.000** | 1.000 | 0.142 | 0.78 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 21 | **1.000** | 1.000 | 0.033 | 0.76 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 21 | **1.000** | 1.000 | 0.067 | 0.73 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 22 | **1.000** | 1.000 | 0.000 | 0.50 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 22 | **1.000** | 1.000 | 0.017 | -0.27 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 25 | **1.000** | 1.000 | 0.150 | 0.95 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 25 | **1.000** | 1.000 | 0.150 | 0.75 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 27 | **1.000** | 1.000 | 0.392 | 0.58 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 27 | **1.000** | 1.000 | 0.200 | 0.62 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | 28 | **1.000** | 1.000 | 0.075 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | 28 | **1.000** | 1.000 | 0.458 | 0.96 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | 32 | **1.000** | 1.000 | 0.292 | 0.83 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 32 | **1.000** | 1.000 | 0.292 | 0.85 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | conflict | 36 | **1.000** | 1.000 | 0.025 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | 36 | **1.000** | 1.000 | 0.017 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | conflict | base | **1.000** | 1.000 | 0.075 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | conflict | base | **1.000** | 1.000 | 0.208 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | conflict | base | **1.000** | 1.000 | 0.033 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | conflict | base | **1.000** | 1.000 | 0.017 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | conflict | 0 | **1.000** | 1.000 | 0.483 | -0.09 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 0 | **1.000** | 1.000 | 0.417 | 0.48 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 0 | **1.000** | 1.000 | 0.458 | 0.25 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 0 | **1.000** | 1.000 | 0.575 | 0.02 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 4 | **1.000** | 1.000 | 0.133 | 0.21 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 4 | **1.000** | 1.000 | 0.000 | 0.69 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 5 | **1.000** | 1.000 | 0.050 | 0.57 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 5 | **1.000** | 1.000 | 0.100 | -0.05 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 8 | **1.000** | 1.000 | 0.058 | 0.04 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 8 | **1.000** | 1.000 | 0.150 | 0.69 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 11 | **1.000** | 1.000 | 0.008 | 0.47 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 11 | **1.000** | 1.000 | 0.067 | 0.24 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 13 | **1.000** | 1.000 | 0.108 | 0.40 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 13 | **1.000** | 1.000 | 0.133 | 0.71 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 16 | **1.000** | 1.000 | 0.000 | 0.48 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 16 | **1.000** | 1.000 | 0.050 | 0.20 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 17 | **1.000** | 1.000 | 0.017 | 0.64 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 17 | **1.000** | 1.000 | 0.167 | 0.84 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | conflict | 21 | **1.000** | 1.000 | 0.042 | 0.76 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 21 | **1.000** | 1.000 | 0.075 | 0.71 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 22 | **1.000** | 1.000 | 0.008 | 0.68 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 22 | **1.000** | 1.000 | 0.025 | 0.07 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 25 | **1.000** | 1.000 | 0.075 | 0.95 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 25 | **1.000** | 1.000 | 0.158 | 0.63 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 27 | **1.000** | 1.000 | 0.250 | 0.54 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 27 | **0.992** | 0.992 | 0.025 | 0.49 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | 28 | **1.000** | 1.000 | 0.083 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | 28 | **1.000** | 1.000 | 0.225 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | 32 | **1.000** | 1.000 | 0.375 | 0.81 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 32 | **1.000** | 1.000 | 0.467 | 0.82 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | conflict | 36 | **1.000** | 1.000 | 0.025 | 0.99 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | 36 | **1.000** | 1.000 | 0.025 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | conflict | base | **1.000** | 1.000 | 0.075 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | conflict | base | **1.000** | 1.000 | 0.208 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | conflict | base | **1.000** | 1.000 | 0.033 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | conflict | base | **1.000** | 1.000 | 0.017 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 0 | **0.467** | 0.408 | 0.417 | 0.01 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 0 | **0.433** | 0.417 | 0.408 | -0.38 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 0 | **0.492** | 0.500 | 0.508 | -0.18 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 0 | **0.467** | 0.500 | 0.500 | 0.02 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 4 | **0.508** | 0.500 | 0.500 | 0.07 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 4 | **0.433** | 0.433 | 0.433 | -0.31 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 5 | **0.442** | 0.483 | 0.483 | -0.29 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 5 | **0.450** | 0.483 | 0.483 | -0.03 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 8 | **0.550** | 0.492 | 0.500 | 0.13 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 8 | **0.483** | 0.500 | 0.483 | -0.41 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 11 | **0.483** | 0.492 | 0.517 | -0.24 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 11 | **0.442** | 0.492 | 0.475 | -0.04 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 13 | **0.533** | 0.492 | 0.492 | 0.06 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 13 | **0.475** | 0.492 | 0.483 | -0.31 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 16 | **0.458** | 0.500 | 0.483 | -0.20 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 16 | **0.467** | 0.517 | 0.508 | -0.11 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 17 | **0.525** | 0.467 | 0.450 | 0.22 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 17 | **0.517** | 0.517 | 0.517 | -0.09 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 21 | **0.617** | 0.575 | 0.567 | 0.20 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 21 | **0.550** | 0.583 | 0.583 | 0.04 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 22 | **0.500** | 0.508 | 0.500 | -0.10 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 22 | **0.492** | 0.558 | 0.550 | 0.02 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 25 | **0.942** | 0.917 | 0.917 | 0.53 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 25 | **0.900** | 0.917 | 0.900 | 0.59 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 27 | **0.900** | 0.892 | 0.892 | 0.52 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 27 | **0.833** | 0.867 | 0.850 | 0.21 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | 28 | **1.000** | 0.925 | 0.908 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | 28 | **0.992** | 0.908 | 0.917 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | 32 | **0.950** | 0.892 | 0.892 | 0.65 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 32 | **0.908** | 0.925 | 0.925 | 0.64 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | corr_e | 36 | **0.992** | 0.883 | 0.900 | 0.98 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | 36 | **0.983** | 0.900 | 0.883 | 0.99 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_e | base | **1.000** | 0.925 | 0.908 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_e | base | **1.000** | 0.917 | 0.917 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_e | base | **1.000** | 0.875 | 0.883 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_e | base | **1.000** | 0.883 | 0.883 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 0 | **0.533** | 0.525 | 0.517 | -0.07 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 0 | **0.525** | 0.508 | 0.508 | -0.33 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 0 | **0.483** | 0.475 | 0.483 | -0.13 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 0 | **0.500** | 0.467 | 0.475 | 0.10 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 4 | **0.533** | 0.508 | 0.508 | 0.06 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 4 | **0.517** | 0.517 | 0.508 | -0.29 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 5 | **0.458** | 0.517 | 0.500 | -0.21 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 5 | **0.475** | 0.508 | 0.508 | -0.06 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 8 | **0.517** | 0.492 | 0.475 | 0.06 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 8 | **0.475** | 0.475 | 0.483 | -0.30 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 11 | **0.442** | 0.483 | 0.475 | -0.14 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 11 | **0.450** | 0.517 | 0.492 | 0.04 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 13 | **0.542** | 0.517 | 0.500 | -0.02 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 13 | **0.517** | 0.533 | 0.508 | -0.24 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 16 | **0.508** | 0.500 | 0.475 | -0.02 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 16 | **0.492** | 0.525 | 0.525 | 0.06 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 17 | **0.550** | 0.492 | 0.492 | 0.17 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 17 | **0.558** | 0.558 | 0.550 | -0.13 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 21 | **0.550** | 0.525 | 0.525 | 0.18 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 21 | **0.558** | 0.575 | 0.567 | 0.01 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 22 | **0.525** | 0.533 | 0.533 | -0.14 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 22 | **0.508** | 0.508 | 0.500 | 0.02 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 25 | **0.967** | 0.908 | 0.900 | 0.57 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 25 | **0.933** | 0.933 | 0.925 | 0.57 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 27 | **0.850** | 0.808 | 0.817 | 0.46 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 27 | **0.867** | 0.933 | 0.925 | 0.33 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | 28 | **1.000** | 0.925 | 0.908 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | 28 | **0.992** | 0.908 | 0.917 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | 32 | **0.917** | 0.875 | 0.883 | 0.63 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 32 | **0.917** | 0.917 | 0.917 | 0.52 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | corr_e | 36 | **0.983** | 0.875 | 0.883 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | 36 | **0.967** | 0.900 | 0.892 | 0.99 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_e | base | **1.000** | 0.925 | 0.908 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_e | base | **1.000** | 0.917 | 0.917 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_e | base | **1.000** | 0.875 | 0.883 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_e | base | **1.000** | 0.883 | 0.883 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 0 | **0.483** | 0.475 | 0.467 | -0.04 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 0 | **0.508** | 0.508 | 0.492 | -0.20 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 0 | **0.483** | 0.475 | 0.483 | -0.17 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 0 | **0.475** | 0.492 | 0.475 | 0.03 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 4 | **0.467** | 0.458 | 0.450 | 0.04 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 4 | **0.483** | 0.517 | 0.525 | -0.35 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 5 | **0.475** | 0.467 | 0.458 | -0.07 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 5 | **0.458** | 0.475 | 0.467 | -0.04 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 8 | **0.525** | 0.467 | 0.450 | 0.17 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 8 | **0.517** | 0.550 | 0.542 | -0.33 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 11 | **0.450** | 0.492 | 0.483 | -0.03 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 11 | **0.458** | 0.475 | 0.475 | 0.04 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 13 | **0.567** | 0.525 | 0.500 | 0.12 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 13 | **0.483** | 0.517 | 0.500 | -0.27 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 16 | **0.483** | 0.492 | 0.483 | 0.04 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 16 | **0.525** | 0.525 | 0.525 | 0.07 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 17 | **0.508** | 0.467 | 0.475 | 0.16 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 17 | **0.525** | 0.525 | 0.550 | -0.08 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 21 | **0.608** | 0.567 | 0.583 | 0.18 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 21 | **0.542** | 0.575 | 0.567 | 0.05 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 22 | **0.433** | 0.475 | 0.442 | -0.02 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 22 | **0.517** | 0.450 | 0.450 | 0.20 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 25 | **0.933** | 0.908 | 0.900 | 0.53 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 25 | **0.900** | 0.917 | 0.908 | 0.51 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 27 | **0.925** | 0.883 | 0.892 | 0.52 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 27 | **0.858** | 0.908 | 0.908 | 0.47 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | 28 | **1.000** | 0.925 | 0.908 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | 28 | **0.975** | 0.925 | 0.933 | 0.99 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | 32 | **0.925** | 0.900 | 0.892 | 0.71 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 32 | **0.950** | 0.917 | 0.908 | 0.67 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | corr_e | 36 | **0.992** | 0.867 | 0.858 | 0.95 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | 36 | **1.000** | 0.883 | 0.883 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_e | base | **1.000** | 0.925 | 0.908 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_e | base | **1.000** | 0.917 | 0.917 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_e | base | **1.000** | 0.875 | 0.883 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_e | base | **1.000** | 0.883 | 0.883 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 0 | **0.492** | 0.467 | 0.467 | -0.02 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 0 | **0.500** | 0.517 | 0.517 | -0.29 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 0 | **0.542** | 0.550 | 0.550 | -0.10 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 0 | **0.475** | 0.492 | 0.492 | -0.01 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 4 | **0.533** | 0.492 | 0.483 | 0.17 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 4 | **0.483** | 0.500 | 0.500 | -0.38 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 5 | **0.475** | 0.517 | 0.525 | -0.22 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 5 | **0.500** | 0.533 | 0.517 | 0.01 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 8 | **0.508** | 0.467 | 0.475 | 0.10 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 8 | **0.500** | 0.517 | 0.517 | -0.32 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 11 | **0.517** | 0.558 | 0.567 | -0.21 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 11 | **0.492** | 0.525 | 0.517 | -0.02 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 13 | **0.558** | 0.500 | 0.508 | 0.14 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 13 | **0.542** | 0.558 | 0.567 | -0.23 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 16 | **0.492** | 0.517 | 0.492 | -0.03 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 16 | **0.500** | 0.483 | 0.483 | 0.09 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 17 | **0.550** | 0.492 | 0.483 | 0.17 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 17 | **0.542** | 0.558 | 0.567 | -0.05 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 21 | **0.583** | 0.575 | 0.567 | 0.17 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 21 | **0.575** | 0.608 | 0.592 | 0.07 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 22 | **0.533** | 0.525 | 0.508 | 0.02 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 22 | **0.450** | 0.450 | 0.458 | 0.08 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 25 | **0.958** | 0.917 | 0.908 | 0.61 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 25 | **0.900** | 0.917 | 0.900 | 0.57 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 27 | **0.917** | 0.908 | 0.900 | 0.52 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 27 | **0.883** | 0.917 | 0.908 | 0.53 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | 28 | **1.000** | 0.925 | 0.908 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | 28 | **0.983** | 0.900 | 0.908 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | 32 | **0.942** | 0.900 | 0.900 | 0.68 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 32 | **0.925** | 0.925 | 0.917 | 0.66 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | corr_e | 36 | **1.000** | 0.875 | 0.883 | 0.99 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | 36 | **0.975** | 0.908 | 0.900 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_e | base | **1.000** | 0.925 | 0.908 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_e | base | **1.000** | 0.917 | 0.917 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_e | base | **1.000** | 0.875 | 0.883 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_e | base | **1.000** | 0.883 | 0.883 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 0 | **0.508** | 0.442 | 0.442 | 0.16 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 0 | **0.463** | 0.429 | 0.429 | 0.00 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 0 | **0.500** | 0.517 | 0.500 | 0.01 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 0 | **0.463** | 0.487 | 0.471 | 0.08 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 4 | **0.592** | 0.508 | 0.500 | 0.23 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 4 | **0.500** | 0.450 | 0.442 | 0.05 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 5 | **0.487** | 0.471 | 0.463 | -0.09 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 5 | **0.467** | 0.492 | 0.475 | 0.06 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 8 | **0.504** | 0.438 | 0.429 | 0.14 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 8 | **0.454** | 0.421 | 0.412 | -0.03 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 11 | **0.458** | 0.458 | 0.450 | -0.06 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 11 | **0.417** | 0.458 | 0.442 | 0.10 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 13 | **0.504** | 0.487 | 0.479 | 0.10 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 13 | **0.492** | 0.492 | 0.492 | -0.01 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 16 | **0.429** | 0.463 | 0.438 | -0.04 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 16 | **0.450** | 0.458 | 0.442 | 0.04 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 17 | **0.562** | 0.496 | 0.479 | 0.22 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 17 | **0.500** | 0.483 | 0.483 | 0.05 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 21 | **0.658** | 0.592 | 0.608 | 0.26 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 21 | **0.571** | 0.588 | 0.588 | 0.20 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 22 | **0.412** | 0.412 | 0.412 | -0.04 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 22 | **0.392** | 0.417 | 0.408 | 0.23 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 25 | **0.925** | 0.858 | 0.858 | 0.56 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 25 | **0.942** | 0.908 | 0.892 | 0.67 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 27 | **0.875** | 0.892 | 0.867 | 0.60 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 27 | **0.817** | 0.858 | 0.850 | 0.31 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | 28 | **0.992** | 0.875 | 0.875 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | 28 | **0.975** | 0.908 | 0.875 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | 32 | **0.933** | 0.933 | 0.925 | 0.70 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 32 | **0.908** | 0.917 | 0.917 | 0.68 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | corr_t | 36 | **0.963** | 0.929 | 0.896 | 0.99 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | 36 | **0.983** | 0.908 | 0.908 | 0.98 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | corr_t | base | **1.000** | 0.867 | 0.867 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | corr_t | base | **1.000** | 0.900 | 0.875 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | corr_t | base | **1.000** | 0.933 | 0.908 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | corr_t | base | **1.000** | 0.925 | 0.917 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 0 | **0.487** | 0.454 | 0.429 | 0.14 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 0 | **0.496** | 0.479 | 0.479 | 0.12 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 0 | **0.458** | 0.492 | 0.467 | 0.09 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 0 | **0.425** | 0.417 | 0.433 | 0.23 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 4 | **0.517** | 0.483 | 0.467 | 0.17 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 4 | **0.487** | 0.504 | 0.496 | 0.04 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 5 | **0.500** | 0.483 | 0.475 | 0.04 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 5 | **0.433** | 0.442 | 0.433 | 0.19 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 8 | **0.492** | 0.458 | 0.450 | 0.19 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 8 | **0.433** | 0.467 | 0.458 | -0.12 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 11 | **0.508** | 0.525 | 0.508 | -0.03 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 11 | **0.450** | 0.475 | 0.458 | 0.12 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 13 | **0.492** | 0.508 | 0.500 | 0.07 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 13 | **0.512** | 0.512 | 0.512 | 0.06 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 16 | **0.487** | 0.521 | 0.512 | -0.03 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 16 | **0.450** | 0.475 | 0.467 | 0.17 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 17 | **0.467** | 0.483 | 0.483 | 0.22 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 17 | **0.508** | 0.492 | 0.500 | 0.06 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 21 | **0.629** | 0.546 | 0.562 | 0.25 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 21 | **0.575** | 0.608 | 0.600 | 0.19 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 22 | **0.471** | 0.487 | 0.479 | -0.08 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 22 | **0.517** | 0.542 | 0.525 | 0.15 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 25 | **0.950** | 0.867 | 0.875 | 0.58 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 25 | **0.933** | 0.900 | 0.883 | 0.67 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 27 | **0.833** | 0.817 | 0.808 | 0.38 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 27 | **0.875** | 0.883 | 0.875 | 0.43 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | 28 | **1.000** | 0.867 | 0.867 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | 28 | **0.983** | 0.900 | 0.875 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | 32 | **0.900** | 0.933 | 0.925 | 0.65 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 32 | **0.942** | 0.933 | 0.933 | 0.70 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | corr_t | 36 | **0.967** | 0.933 | 0.900 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | 36 | **0.992** | 0.917 | 0.917 | 0.99 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | corr_t | base | **1.000** | 0.867 | 0.867 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | corr_t | base | **1.000** | 0.900 | 0.875 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | corr_t | base | **1.000** | 0.933 | 0.908 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | corr_t | base | **1.000** | 0.925 | 0.917 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 0 | **0.475** | 0.425 | 0.417 | 0.10 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 0 | **0.483** | 0.483 | 0.483 | 0.10 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 0 | **0.463** | 0.479 | 0.471 | 0.06 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 0 | **0.500** | 0.492 | 0.483 | 0.18 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 4 | **0.479** | 0.446 | 0.438 | 0.16 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 4 | **0.450** | 0.450 | 0.442 | -0.03 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 5 | **0.471** | 0.487 | 0.479 | 0.02 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 5 | **0.417** | 0.442 | 0.425 | 0.13 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 8 | **0.517** | 0.467 | 0.475 | 0.22 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 8 | **0.442** | 0.475 | 0.467 | -0.08 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 11 | **0.483** | 0.517 | 0.517 | 0.09 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 11 | **0.475** | 0.483 | 0.475 | 0.12 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 13 | **0.504** | 0.504 | 0.496 | 0.18 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 13 | **0.454** | 0.471 | 0.454 | 0.05 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 16 | **0.467** | 0.500 | 0.483 | 0.12 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 16 | **0.500** | 0.492 | 0.475 | 0.24 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 17 | **0.537** | 0.487 | 0.471 | 0.17 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 17 | **0.562** | 0.562 | 0.554 | 0.01 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 21 | **0.650** | 0.567 | 0.583 | 0.25 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 21 | **0.592** | 0.575 | 0.575 | 0.26 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 22 | **0.446** | 0.479 | 0.454 | 0.01 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 22 | **0.450** | 0.442 | 0.442 | 0.22 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 25 | **0.883** | 0.850 | 0.842 | 0.59 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 25 | **0.933** | 0.900 | 0.883 | 0.64 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 27 | **0.875** | 0.875 | 0.858 | 0.65 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 27 | **0.858** | 0.867 | 0.858 | 0.45 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | 28 | **0.992** | 0.875 | 0.875 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | 28 | **0.963** | 0.904 | 0.879 | 1.00 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | 32 | **0.917** | 0.933 | 0.917 | 0.72 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 32 | **0.950** | 0.925 | 0.925 | 0.71 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | corr_t | 36 | **0.975** | 0.925 | 0.900 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | 36 | **1.000** | 0.925 | 0.917 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | corr_t | base | **1.000** | 0.867 | 0.867 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | corr_t | base | **1.000** | 0.900 | 0.875 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | corr_t | base | **1.000** | 0.933 | 0.908 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | corr_t | base | **1.000** | 0.925 | 0.917 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 0 | **0.529** | 0.463 | 0.446 | 0.14 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 0 | **0.446** | 0.463 | 0.463 | 0.04 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 0 | **0.467** | 0.467 | 0.450 | 0.01 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 0 | **0.442** | 0.467 | 0.450 | 0.14 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 4 | **0.542** | 0.475 | 0.458 | 0.27 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 4 | **0.496** | 0.496 | 0.487 | -0.06 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 5 | **0.458** | 0.458 | 0.442 | -0.06 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 5 | **0.442** | 0.467 | 0.458 | 0.13 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 8 | **0.471** | 0.438 | 0.446 | 0.14 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 8 | **0.496** | 0.529 | 0.529 | -0.03 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 11 | **0.442** | 0.458 | 0.442 | -0.11 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 11 | **0.433** | 0.475 | 0.458 | 0.10 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 13 | **0.496** | 0.463 | 0.454 | 0.12 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 13 | **0.500** | 0.500 | 0.500 | 0.08 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 16 | **0.446** | 0.463 | 0.446 | -0.01 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 16 | **0.417** | 0.442 | 0.433 | 0.14 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 17 | **0.517** | 0.450 | 0.450 | 0.20 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 17 | **0.483** | 0.483 | 0.475 | 0.04 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 21 | **0.617** | 0.583 | 0.592 | 0.20 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 21 | **0.592** | 0.625 | 0.625 | 0.19 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 22 | **0.450** | 0.467 | 0.450 | -0.00 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 22 | **0.483** | 0.475 | 0.475 | 0.11 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 25 | **0.942** | 0.858 | 0.850 | 0.60 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 25 | **0.925** | 0.892 | 0.867 | 0.67 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 27 | **0.867** | 0.883 | 0.875 | 0.60 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 27 | **0.875** | 0.883 | 0.867 | 0.56 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | 28 | **1.000** | 0.867 | 0.867 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | 28 | **0.975** | 0.908 | 0.875 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | 32 | **0.908** | 0.942 | 0.917 | 0.69 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 32 | **0.917** | 0.925 | 0.925 | 0.71 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | corr_t | 36 | **0.992** | 0.942 | 0.908 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | 36 | **0.983** | 0.908 | 0.908 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | corr_t | base | **1.000** | 0.867 | 0.867 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | corr_t | base | **1.000** | 0.900 | 0.875 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | corr_t | base | **1.000** | 0.933 | 0.908 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | corr_t | base | **1.000** | 0.925 | 0.917 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | style_c | 0 | **1.000** | 0.000 | 0.175 | -0.22 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 0 | **1.000** | 0.000 | 0.350 | 0.22 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 0 | **1.000** | 0.000 | 0.250 | -0.14 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 0 | **1.000** | 0.000 | 0.467 | -0.52 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 4 | **1.000** | 0.000 | 0.992 | -0.04 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 4 | **1.000** | 0.000 | 0.283 | 0.19 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 5 | **1.000** | 0.000 | 1.000 | 0.33 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 5 | **1.000** | 0.000 | 0.967 | -0.47 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 8 | **1.000** | 0.000 | 0.825 | -0.00 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 8 | **1.000** | 0.000 | 0.408 | 0.33 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 11 | **1.000** | 0.000 | 0.817 | 0.20 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 11 | **1.000** | 0.000 | 0.825 | -0.16 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 13 | **1.000** | 0.000 | 0.925 | 0.29 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 13 | **1.000** | 0.000 | 0.817 | 0.45 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 16 | **1.000** | 0.000 | 0.950 | 0.20 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 16 | **1.000** | 0.000 | 1.000 | -0.48 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 17 | **1.000** | 0.000 | 0.967 | 0.42 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 17 | **1.000** | 0.000 | 0.758 | 0.69 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | style_c | 21 | **1.000** | 0.000 | 0.967 | 0.65 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 21 | **1.000** | 0.000 | 0.917 | 0.85 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 22 | **1.000** | 0.000 | 0.825 | 0.08 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 22 | **1.000** | 0.000 | 0.467 | -0.19 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 25 | **1.000** | 0.000 | 0.925 | 0.97 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 25 | **1.000** | 0.000 | 0.850 | 0.91 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 27 | **1.000** | 0.000 | 0.825 | 0.76 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 27 | **1.000** | 0.000 | 0.767 | 0.62 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | 28 | **1.000** | 0.000 | 0.925 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | 28 | **1.000** | 0.000 | 0.842 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | 32 | **1.000** | 0.000 | 0.867 | 0.90 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 32 | **1.000** | 0.000 | 0.825 | 0.82 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | style_c | 36 | **1.000** | 0.000 | 0.967 | 0.93 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | 36 | **1.000** | 0.000 | 0.975 | 0.97 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_c | base | **1.000** | 0.000 | 0.933 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_c | base | **1.000** | 0.000 | 0.842 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | style_c | base | **1.000** | 0.000 | 0.967 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | style_c | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | style_c | 0 | **1.000** | 0.000 | 0.567 | -0.37 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 0 | **1.000** | 0.000 | 0.183 | 0.07 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 0 | **1.000** | 0.000 | 0.275 | -0.18 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 0 | **1.000** | 0.000 | 0.100 | -0.48 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 4 | **1.000** | 0.000 | 1.000 | 0.24 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 4 | **1.000** | 0.000 | 0.842 | 0.53 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 5 | **1.000** | 0.000 | 0.158 | 0.04 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 5 | **1.000** | 0.000 | 0.983 | -0.54 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 8 | **1.000** | 0.000 | 0.892 | 0.24 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 8 | **1.000** | 0.000 | 1.000 | 0.41 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 11 | **1.000** | 0.000 | 0.958 | 0.19 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 11 | **1.000** | 0.000 | 0.825 | 0.12 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 13 | **1.000** | 0.000 | 0.933 | 0.47 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 13 | **1.000** | 0.000 | 0.842 | 0.39 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 16 | **1.000** | 0.000 | 1.000 | 0.36 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 16 | **1.000** | 0.000 | 0.950 | -0.02 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 17 | **1.000** | 0.000 | 0.983 | 0.50 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 17 | **1.000** | 0.000 | 0.717 | 0.44 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | style_c | 21 | **1.000** | 0.000 | 0.933 | 0.69 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 21 | **1.000** | 0.000 | 0.967 | 0.77 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 22 | **1.000** | 0.000 | 0.950 | 0.31 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 22 | **1.000** | 0.000 | 0.833 | -0.60 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 25 | **1.000** | 0.000 | 0.950 | 0.94 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 25 | **1.000** | 0.000 | 0.825 | 0.79 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 27 | **1.000** | 0.000 | 0.658 | 0.47 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 27 | **0.992** | 0.008 | 0.808 | 0.36 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | 28 | **1.000** | 0.000 | 0.933 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | 28 | **1.000** | 0.000 | 0.842 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | 32 | **1.000** | 0.000 | 0.792 | 0.86 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 32 | **1.000** | 0.000 | 0.850 | 0.85 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | style_c | 36 | **1.000** | 0.000 | 0.967 | 1.00 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | 36 | **1.000** | 0.000 | 0.983 | 0.98 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_c | base | **1.000** | 0.000 | 0.933 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_c | base | **1.000** | 0.000 | 0.842 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | style_c | base | **1.000** | 0.000 | 0.967 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | style_c | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 0 | **1.000** | 0.000 | 0.208 | -0.21 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 0 | **1.000** | 0.000 | 0.233 | 0.24 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 0 | **1.000** | 0.000 | 0.192 | -0.22 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 0 | **1.000** | 0.000 | 0.483 | -0.37 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 4 | **1.000** | 0.000 | 0.942 | 0.23 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 4 | **1.000** | 0.000 | 0.992 | 0.52 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 5 | **1.000** | 0.000 | 0.908 | 0.39 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 5 | **1.000** | 0.000 | 0.783 | -0.25 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 8 | **1.000** | 0.000 | 0.908 | 0.12 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 8 | **1.000** | 0.000 | 1.000 | 0.58 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 11 | **1.000** | 0.000 | 1.000 | 0.19 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 11 | **1.000** | 0.000 | 0.633 | -0.30 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 13 | **1.000** | 0.000 | 0.942 | 0.38 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 13 | **1.000** | 0.000 | 0.900 | 0.55 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 16 | **1.000** | 0.000 | 1.000 | 0.37 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 16 | **1.000** | 0.000 | 0.992 | 0.13 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 17 | **1.000** | 0.000 | 0.917 | 0.31 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 17 | **1.000** | 0.000 | 0.858 | 0.68 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 21 | **1.000** | 0.000 | 0.975 | 0.71 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 21 | **1.000** | 0.000 | 0.958 | 0.79 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 22 | **1.000** | 0.000 | 1.000 | 0.45 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 22 | **1.000** | 0.000 | 1.000 | -0.41 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 25 | **1.000** | 0.000 | 0.858 | 0.97 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 25 | **1.000** | 0.000 | 0.842 | 0.88 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 27 | **1.000** | 0.000 | 0.692 | 0.73 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 27 | **0.992** | 0.008 | 0.875 | 0.83 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | 28 | **1.000** | 0.000 | 0.933 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | 28 | **1.000** | 0.000 | 0.717 | 0.94 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | 32 | **1.000** | 0.000 | 0.783 | 0.92 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 32 | **1.000** | 0.000 | 0.792 | 0.81 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | style_c | 36 | **1.000** | 0.000 | 0.967 | 0.95 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | 36 | **1.000** | 0.000 | 0.983 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_c | base | **1.000** | 0.000 | 0.933 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_c | base | **1.000** | 0.000 | 0.842 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_c | base | **1.000** | 0.000 | 0.967 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_c | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | style_c | 0 | **1.000** | 0.000 | 0.517 | -0.23 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 0 | **1.000** | 0.000 | 0.575 | 0.21 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 0 | **1.000** | 0.000 | 0.533 | -0.15 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 0 | **1.000** | 0.000 | 0.408 | -0.33 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 4 | **1.000** | 0.000 | 0.867 | 0.12 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 4 | **1.000** | 0.000 | 1.000 | 0.52 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 5 | **1.000** | 0.000 | 0.950 | 0.41 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 5 | **1.000** | 0.000 | 0.933 | -0.33 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 8 | **1.000** | 0.000 | 0.942 | -0.08 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 8 | **1.000** | 0.000 | 0.842 | 0.44 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 11 | **1.000** | 0.000 | 0.992 | 0.23 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 11 | **1.000** | 0.000 | 0.950 | -0.11 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 13 | **1.000** | 0.000 | 0.875 | 0.30 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 13 | **1.000** | 0.000 | 0.867 | 0.54 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 16 | **1.000** | 0.000 | 1.000 | 0.46 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 16 | **1.000** | 0.000 | 0.942 | -0.05 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 17 | **1.000** | 0.000 | 0.983 | 0.62 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 17 | **1.000** | 0.000 | 0.842 | 0.70 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | style_c | 21 | **1.000** | 0.000 | 0.958 | 0.72 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 21 | **0.992** | 0.008 | 0.933 | 0.81 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 22 | **1.000** | 0.000 | 1.000 | 0.43 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 22 | **1.000** | 0.000 | 0.975 | -0.18 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 25 | **1.000** | 0.000 | 0.942 | 0.97 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 25 | **1.000** | 0.000 | 0.858 | 0.75 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 27 | **1.000** | 0.000 | 0.808 | 0.75 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 27 | **0.992** | 0.008 | 0.975 | 0.79 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | 28 | **1.000** | 0.000 | 0.933 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | 28 | **1.000** | 0.000 | 0.842 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | 32 | **1.000** | 0.000 | 0.775 | 0.91 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 32 | **1.000** | 0.000 | 0.642 | 0.78 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | style_c | 36 | **1.000** | 0.000 | 0.967 | 0.99 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | 36 | **1.000** | 0.000 | 0.983 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_c | base | **1.000** | 0.000 | 0.933 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_c | base | **1.000** | 0.000 | 0.842 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | style_c | base | **1.000** | 0.000 | 0.967 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | style_c | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 100.7M |
| qwen3-0.6b | eagle-2l | styc | style_w | 0 | **1.000** | 0.000 | 0.183 | -0.05 | 5.272 | 0.09 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 0 | **1.000** | 0.000 | 0.342 | 0.32 | 5.428 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 0 | **1.000** | 0.000 | 0.283 | 0.15 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 0 | **1.000** | 0.000 | 0.475 | -0.16 | 4.787 | 0.20 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 4 | **1.000** | 0.000 | 0.975 | 0.07 | 2.724 | 0.27 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 4 | **1.000** | 0.000 | 0.250 | 0.30 | 5.226 | 0.12 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 5 | **1.000** | 0.000 | 0.992 | 0.49 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 5 | **1.000** | 0.000 | 0.967 | -0.08 | 5.004 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 8 | **1.000** | 0.000 | 0.792 | 0.10 | 2.377 | 0.32 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 8 | **1.000** | 0.000 | 0.408 | 0.45 | 4.488 | 0.16 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 11 | **1.000** | 0.000 | 0.833 | 0.47 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 11 | **1.000** | 0.000 | 0.850 | 0.14 | 4.341 | 0.23 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 13 | **1.000** | 0.000 | 0.925 | 0.34 | 2.212 | 0.34 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 13 | **1.000** | 0.000 | 0.792 | 0.52 | 2.877 | 0.30 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 16 | **1.000** | 0.000 | 0.950 | 0.55 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 16 | **1.000** | 0.000 | 0.983 | -0.17 | 4.858 | 0.18 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 17 | **1.000** | 0.000 | 0.950 | 0.41 | 1.922 | 0.39 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 17 | **1.000** | 0.000 | 0.800 | 0.70 | 2.100 | 0.42 | 50.3M |
| qwen3-0.6b | eagle-2l | styc | style_w | 21 | **1.000** | 0.000 | 0.975 | 0.74 | 1.215 | 0.53 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 21 | **1.000** | 0.000 | 0.917 | 0.84 | 1.111 | 0.59 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 22 | **1.000** | 0.000 | 0.825 | 0.30 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 22 | **1.000** | 0.000 | 0.458 | 0.22 | 4.832 | 0.22 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 25 | **1.000** | 0.000 | 0.983 | 0.96 | 0.601 | 0.73 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 25 | **1.000** | 0.000 | 0.958 | 0.88 | 0.369 | 0.78 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 27 | **1.000** | 0.000 | 0.933 | 0.58 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 27 | **1.000** | 0.000 | 0.858 | 0.55 | 3.118 | 0.39 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.001 | 0.99 | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.004 | 0.97 | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | 32 | **1.000** | 0.000 | 0.967 | 0.76 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 32 | **1.000** | 0.000 | 0.958 | 0.89 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | styc | style_w | 36 | **1.000** | 0.000 | 0.992 | 0.94 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | 36 | **1.000** | 0.000 | 0.983 | 0.96 | 0.208 | 0.83 | 201.4M |
| qwen3-0.6b | eagle-2l | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 12.6M |
| qwen3-1.7b | eagle-2l | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 50.3M |
| qwen3-4b | eagle-2l | styc | style_w | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | styc | style_w | base | **1.000** | 0.000 | 1.000 | 1.00 | -- | -- | 201.4M |
| qwen3-0.6b | eagle-attn | styc | style_w | 0 | **1.000** | 0.000 | 0.567 | -0.27 | 3.603 | 0.20 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 0 | **1.000** | 0.000 | 0.167 | 0.17 | 4.984 | 0.16 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 0 | **1.000** | 0.000 | 0.283 | 0.10 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 0 | **1.000** | 0.000 | 0.083 | -0.17 | 7.815 | 0.07 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 4 | **1.000** | 0.000 | 0.992 | 0.33 | 2.543 | 0.30 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 4 | **1.000** | 0.000 | 0.792 | 0.63 | 3.506 | 0.25 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 5 | **1.000** | 0.000 | 0.200 | 0.36 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 5 | **1.000** | 0.000 | 0.983 | -0.18 | 5.756 | 0.18 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 8 | **1.000** | 0.000 | 0.867 | 0.34 | 2.447 | 0.31 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 8 | **1.000** | 0.000 | 1.000 | 0.45 | 2.978 | 0.30 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 11 | **1.000** | 0.000 | 0.958 | 0.37 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 11 | **1.000** | 0.000 | 0.817 | 0.41 | 5.013 | 0.21 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 13 | **1.000** | 0.000 | 0.942 | 0.54 | 2.342 | 0.32 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 13 | **1.000** | 0.000 | 0.817 | 0.48 | 2.905 | 0.31 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 16 | **1.000** | 0.000 | 1.000 | 0.48 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 16 | **1.000** | 0.000 | 0.950 | 0.21 | 4.134 | 0.25 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 17 | **1.000** | 0.000 | 0.983 | 0.53 | 2.036 | 0.37 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 17 | **1.000** | 0.000 | 0.742 | 0.53 | 2.620 | 0.33 | 16.8M |
| qwen3-0.6b | eagle-attn | styc | style_w | 21 | **1.000** | 0.000 | 0.933 | 0.78 | 1.360 | 0.49 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 21 | **1.000** | 0.000 | 0.967 | 0.73 | 1.386 | 0.53 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 22 | **1.000** | 0.000 | 0.967 | 0.52 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 22 | **1.000** | 0.000 | 0.867 | -0.45 | 4.145 | 0.30 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 25 | **1.000** | 0.000 | 0.967 | 0.95 | 0.507 | 0.71 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 25 | **1.000** | 0.000 | 0.967 | 0.74 | 0.460 | 0.74 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 27 | **0.992** | 0.008 | 0.842 | 0.36 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 27 | **1.000** | 0.000 | 0.950 | 0.27 | 2.801 | 0.43 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.000 | 1.00 | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.035 | 0.92 | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | 32 | **0.992** | 0.008 | 0.950 | 0.67 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 32 | **1.000** | 0.000 | 0.967 | 0.91 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | styc | style_w | 36 | **1.000** | 0.000 | 0.983 | 0.99 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | 36 | **1.000** | 0.000 | 1.000 | 0.97 | 0.064 | 0.91 | 67.1M |
| qwen3-0.6b | eagle-attn | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 4.2M |
| qwen3-1.7b | eagle-attn | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 16.8M |
| qwen3-4b | eagle-attn | styc | style_w | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | styc | style_w | base | **1.000** | 0.000 | 1.000 | 1.00 | -- | -- | 67.1M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 0 | **1.000** | 0.000 | 0.208 | -0.06 | 4.795 | 0.14 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 0 | **1.000** | 0.000 | 0.233 | 0.35 | 5.458 | 0.14 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 0 | **1.000** | 0.000 | 0.175 | 0.10 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 0 | **1.000** | 0.000 | 0.483 | -0.02 | 4.961 | 0.18 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 4 | **1.000** | 0.000 | 0.950 | 0.31 | 2.455 | 0.31 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 4 | **1.000** | 0.000 | 0.992 | 0.58 | 2.651 | 0.35 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 5 | **1.000** | 0.000 | 0.908 | 0.48 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 5 | **1.000** | 0.000 | 0.808 | -0.02 | 3.595 | 0.28 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 8 | **1.000** | 0.000 | 0.875 | 0.22 | 2.248 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 8 | **1.000** | 0.000 | 1.000 | 0.60 | 2.428 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 11 | **1.000** | 0.000 | 1.000 | 0.39 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 11 | **1.000** | 0.000 | 0.708 | -0.04 | 3.416 | 0.34 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 13 | **1.000** | 0.000 | 0.917 | 0.47 | 2.112 | 0.34 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 13 | **1.000** | 0.000 | 0.917 | 0.56 | 2.337 | 0.39 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 16 | **1.000** | 0.000 | 1.000 | 0.48 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 16 | **1.000** | 0.000 | 0.983 | 0.24 | 2.997 | 0.39 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 17 | **1.000** | 0.000 | 0.892 | 0.37 | 1.851 | 0.38 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 17 | **1.000** | 0.000 | 0.850 | 0.68 | 2.152 | 0.41 | 8.4M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 21 | **1.000** | 0.000 | 0.975 | 0.77 | 1.210 | 0.54 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 21 | **1.000** | 0.000 | 0.933 | 0.79 | 1.135 | 0.59 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 22 | **1.000** | 0.000 | 1.000 | 0.41 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 22 | **1.000** | 0.000 | 0.967 | -0.34 | 2.572 | 0.42 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 25 | **1.000** | 0.000 | 0.950 | 0.95 | 0.392 | 0.75 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 25 | **1.000** | 0.000 | 0.958 | 0.84 | 0.417 | 0.76 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 27 | **1.000** | 0.000 | 0.883 | 0.56 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 27 | **1.000** | 0.000 | 0.950 | 0.66 | 1.472 | 0.57 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.000 | 1.00 | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | 28 | **1.000** | 0.000 | 0.933 | 0.96 | 0.004 | 1.00 | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | 32 | **1.000** | 0.000 | 0.917 | 0.81 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 32 | **1.000** | 0.000 | 0.958 | 0.86 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | styc | style_w | 36 | **1.000** | 0.000 | 0.992 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | 36 | **1.000** | 0.000 | 1.000 | 1.00 | 0.097 | 0.99 | 33.6M |
| qwen3-0.6b | eagle-mlp | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 2.1M |
| qwen3-1.7b | eagle-mlp | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 8.4M |
| qwen3-4b | eagle-mlp | styc | style_w | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | styc | style_w | base | **1.000** | 0.000 | 1.000 | 1.00 | -- | -- | 33.6M |
| qwen3-0.6b | eagle-tf | styc | style_w | 0 | **1.000** | 0.000 | 0.525 | -0.12 | 3.730 | 0.19 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 0 | **1.000** | 0.000 | 0.575 | 0.31 | 4.943 | 0.15 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 0 | **1.000** | 0.000 | 0.508 | 0.13 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 0 | **1.000** | 0.000 | 0.400 | 0.05 | 4.826 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 4 | **1.000** | 0.000 | 0.825 | 0.18 | 2.459 | 0.31 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 4 | **1.000** | 0.000 | 1.000 | 0.55 | 3.230 | 0.27 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 5 | **1.000** | 0.000 | 0.950 | 0.51 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 5 | **1.000** | 0.000 | 0.900 | -0.03 | 4.184 | 0.26 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 8 | **1.000** | 0.000 | 0.942 | 0.02 | 2.356 | 0.33 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 8 | **1.000** | 0.000 | 0.842 | 0.54 | 2.995 | 0.29 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 11 | **1.000** | 0.000 | 1.000 | 0.36 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 11 | **1.000** | 0.000 | 0.950 | 0.25 | 4.592 | 0.19 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 13 | **1.000** | 0.000 | 0.858 | 0.37 | 2.103 | 0.35 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 13 | **1.000** | 0.000 | 0.858 | 0.57 | 2.492 | 0.36 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 16 | **1.000** | 0.000 | 1.000 | 0.37 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 16 | **1.000** | 0.000 | 0.933 | 0.23 | 4.456 | 0.23 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 17 | **1.000** | 0.000 | 0.983 | 0.59 | 1.753 | 0.40 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 17 | **1.000** | 0.000 | 0.842 | 0.72 | 2.173 | 0.41 | 25.2M |
| qwen3-0.6b | eagle-tf | styc | style_w | 21 | **1.000** | 0.000 | 0.983 | 0.79 | 1.192 | 0.54 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 21 | **1.000** | 0.000 | 0.917 | 0.77 | 1.145 | 0.57 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 22 | **1.000** | 0.000 | 0.992 | 0.58 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 22 | **1.000** | 0.000 | 0.983 | 0.04 | 3.636 | 0.30 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 25 | **1.000** | 0.000 | 0.967 | 0.96 | 0.452 | 0.72 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 25 | **1.000** | 0.000 | 0.975 | 0.71 | 0.414 | 0.76 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 27 | **1.000** | 0.000 | 0.933 | 0.50 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 27 | **1.000** | 0.000 | 1.000 | 0.50 | 1.898 | 0.56 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | 28 | **1.000** | 0.000 | 0.950 | 1.00 | 0.000 | 1.00 | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | 28 | **1.000** | 0.000 | 0.942 | 1.00 | 0.009 | 0.95 | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | 32 | **1.000** | 0.000 | 0.892 | 0.80 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 32 | **1.000** | 0.000 | 0.917 | 0.84 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | styc | style_w | 36 | **1.000** | 0.000 | 0.983 | 0.99 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | 36 | **1.000** | 0.000 | 1.000 | 0.99 | 0.078 | 0.90 | 100.7M |
| qwen3-0.6b | eagle-tf | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 6.3M |
| qwen3-1.7b | eagle-tf | styc | style_w | base | **1.000** | 0.000 | 0.950 | 1.00 | -- | -- | 25.2M |
| qwen3-4b | eagle-tf | styc | style_w | base | **1.000** | 0.000 | 0.983 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | styc | style_w | base | **1.000** | 0.000 | 1.000 | 1.00 | -- | -- | 100.7M |
| qwen3-4b | eagle-2l | uf | quality | 0 | **0.840** | 0.361 | 0.424 | 0.80 | 4.431 | 0.17 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 0 | **0.836** | 0.361 | 0.398 | 0.79 | 4.787 | 0.20 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 5 | **0.844** | 0.364 | 0.442 | 0.81 | 3.394 | 0.28 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 5 | **0.840** | 0.357 | 0.394 | 0.80 | 5.004 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 11 | **0.844** | 0.357 | 0.428 | 0.81 | 3.305 | 0.29 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 11 | **0.833** | 0.357 | 0.401 | 0.79 | 4.341 | 0.23 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 16 | **0.848** | 0.361 | 0.420 | 0.80 | 3.871 | 0.23 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 16 | **0.836** | 0.361 | 0.387 | 0.80 | 4.858 | 0.18 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 22 | **0.836** | 0.349 | 0.420 | 0.82 | 3.028 | 0.32 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 22 | **0.840** | 0.357 | 0.450 | 0.79 | 4.832 | 0.22 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 27 | **0.874** | 0.379 | 0.617 | 0.82 | 1.088 | 0.61 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 27 | **0.866** | 0.383 | 0.587 | 0.82 | 3.118 | 0.39 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 32 | **0.941** | 0.394 | 0.639 | 0.65 | 0.641 | 0.72 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 32 | **0.937** | 0.394 | 0.673 | 0.90 | 0.910 | 0.67 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | 36 | **0.996** | 0.413 | 0.673 | 1.00 | 0.114 | 0.87 | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | 36 | **0.974** | 0.424 | 0.688 | 1.00 | 0.208 | 0.83 | 201.4M |
| qwen3-4b | eagle-2l | uf | quality | base | **1.000** | 0.409 | 0.662 | 1.00 | -- | -- | 78.7M |
| qwen3-8b | eagle-2l | uf | quality | base | **1.000** | 0.420 | 0.688 | 1.00 | -- | -- | 201.4M |
| qwen3-4b | eagle-attn | uf | quality | 0 | **0.844** | 0.357 | 0.401 | 0.80 | 4.838 | 0.17 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 0 | **0.836** | 0.361 | 0.375 | 0.79 | 7.815 | 0.07 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 5 | **0.844** | 0.357 | 0.413 | 0.81 | 4.002 | 0.24 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 5 | **0.836** | 0.361 | 0.401 | 0.80 | 5.756 | 0.18 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 11 | **0.848** | 0.361 | 0.454 | 0.79 | 3.168 | 0.29 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 11 | **0.836** | 0.375 | 0.439 | 0.79 | 5.013 | 0.21 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 16 | **0.855** | 0.368 | 0.442 | 0.80 | 3.053 | 0.30 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 16 | **0.840** | 0.364 | 0.450 | 0.79 | 4.134 | 0.25 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 22 | **0.862** | 0.375 | 0.483 | 0.79 | 3.194 | 0.34 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 22 | **0.848** | 0.379 | 0.509 | 0.76 | 4.145 | 0.30 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 27 | **0.870** | 0.383 | 0.580 | 0.85 | 1.624 | 0.51 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 27 | **0.870** | 0.372 | 0.532 | 0.79 | 2.801 | 0.43 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 32 | **0.929** | 0.390 | 0.625 | 0.87 | 0.900 | 0.66 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 32 | **0.926** | 0.390 | 0.654 | 0.89 | 1.384 | 0.62 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | 36 | **0.985** | 0.401 | 0.651 | 1.00 | 0.043 | 0.92 | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | 36 | **0.978** | 0.428 | 0.695 | 1.00 | 0.064 | 0.91 | 67.1M |
| qwen3-4b | eagle-attn | uf | quality | base | **1.000** | 0.409 | 0.662 | 1.00 | -- | -- | 26.2M |
| qwen3-8b | eagle-attn | uf | quality | base | **1.000** | 0.420 | 0.688 | 1.00 | -- | -- | 67.1M |
| qwen3-4b | eagle-mlp | uf | quality | 0 | **0.848** | 0.361 | 0.420 | 0.80 | 5.003 | 0.15 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 0 | **0.833** | 0.364 | 0.390 | 0.79 | 4.961 | 0.18 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 5 | **0.855** | 0.361 | 0.454 | 0.82 | 2.679 | 0.35 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 5 | **0.844** | 0.368 | 0.442 | 0.81 | 3.595 | 0.28 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 11 | **0.844** | 0.372 | 0.457 | 0.81 | 2.374 | 0.41 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 11 | **0.840** | 0.364 | 0.428 | 0.79 | 3.416 | 0.34 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 16 | **0.859** | 0.372 | 0.483 | 0.80 | 2.435 | 0.40 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 16 | **0.844** | 0.368 | 0.476 | 0.79 | 2.997 | 0.39 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 22 | **0.855** | 0.375 | 0.491 | 0.80 | 2.089 | 0.47 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 22 | **0.836** | 0.353 | 0.491 | 0.79 | 2.572 | 0.42 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 27 | **0.881** | 0.379 | 0.602 | 0.87 | 1.145 | 0.60 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 27 | **0.881** | 0.375 | 0.606 | 0.87 | 1.472 | 0.57 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 32 | **0.941** | 0.379 | 0.632 | 0.93 | 0.593 | 0.73 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 32 | **0.941** | 0.405 | 0.665 | 0.93 | 0.711 | 0.72 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | 36 | **0.985** | 0.394 | 0.658 | 0.99 | 0.002 | 1.00 | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | 36 | **0.985** | 0.413 | 0.691 | 0.99 | 0.097 | 0.99 | 33.6M |
| qwen3-4b | eagle-mlp | uf | quality | base | **1.000** | 0.409 | 0.662 | 1.00 | -- | -- | 13.1M |
| qwen3-8b | eagle-mlp | uf | quality | base | **1.000** | 0.420 | 0.688 | 1.00 | -- | -- | 33.6M |
| qwen3-4b | eagle-tf | uf | quality | 0 | **0.848** | 0.375 | 0.420 | 0.80 | 4.203 | 0.19 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 0 | **0.836** | 0.353 | 0.383 | 0.80 | 4.826 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 5 | **0.840** | 0.361 | 0.435 | 0.82 | 3.279 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 5 | **0.840** | 0.357 | 0.413 | 0.81 | 4.184 | 0.26 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 11 | **0.844** | 0.357 | 0.428 | 0.81 | 3.287 | 0.28 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 11 | **0.836** | 0.361 | 0.401 | 0.80 | 4.592 | 0.19 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 16 | **0.862** | 0.361 | 0.461 | 0.81 | 2.748 | 0.34 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 16 | **0.836** | 0.361 | 0.428 | 0.80 | 4.456 | 0.23 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 22 | **0.855** | 0.375 | 0.517 | 0.79 | 2.148 | 0.45 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 22 | **0.851** | 0.368 | 0.517 | 0.78 | 3.636 | 0.30 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 27 | **0.862** | 0.383 | 0.558 | 0.66 | 1.283 | 0.62 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 27 | **0.885** | 0.394 | 0.617 | 0.46 | 1.898 | 0.56 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 32 | **0.933** | 0.379 | 0.636 | 0.85 | 0.622 | 0.72 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 32 | **0.933** | 0.398 | 0.677 | 0.92 | 0.779 | 0.71 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | 36 | **0.989** | 0.405 | 0.665 | 1.00 | 0.033 | 0.92 | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | 36 | **0.989** | 0.416 | 0.688 | 1.00 | 0.078 | 0.90 | 100.7M |
| qwen3-4b | eagle-tf | uf | quality | base | **1.000** | 0.409 | 0.662 | 1.00 | -- | -- | 39.3M |
| qwen3-8b | eagle-tf | uf | quality | base | **1.000** | 0.420 | 0.688 | 1.00 | -- | -- | 100.7M |

## Reading guide

- **floor rnd** is the memorisation ceiling of a bag-of-token-ids probe on a random split. A dataset whose floor rnd is ~1.0 is solvable from vocabulary alone.

- **floor grp** is the same probe on held-out groups. The gap between the two is how much of the dataset is *memorisable* vocabulary rather than *generalisable* vocabulary. A model probe beating floor grp at layer 0 is reading sub-token regularity, not a lookup table (goodfire/RESULTS.md:14-18).

- **tie** at L0 near 1.00 means the read is degenerate, not that the layer is uninformative: both completions end in the same token, so the last-token difference is exactly zero. Compare the `mean` read for that cell.

- **shuffled** should sit at 0.5. Anything materially above it means the rung's capacity is fitting noise and its accuracies are inflated.
