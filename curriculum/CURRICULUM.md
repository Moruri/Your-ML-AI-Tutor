# Curriculum map

This is the whole road, start to finish. Everything is taught in Python.

**How to read this page**

- Lessons are numbered `001`, `002`, ... and taught in strict order. Each one
  builds on the one before it, so please don't skip ahead.
- We publish two lessons every weekday (Monday to Friday). Nothing on weekends.
  That's the day to rest, or to redo an exercise that didn't click.
- Each lesson has one job, written as a one-line goal. If you can do that thing
  when you finish, you're done. Nothing more is expected.
- Lessons **001** and **002** are where we start (Week 1, Day 1: Friday
  2026-09-11). See [PROGRESS.md](PROGRESS.md) for what's published so far.

A rough sense of pace: two lessons a day, five days a week, means a phase of
~14 lessons takes about a week and a half. The whole curriculum is a few months
of steady, unhurried work. That's on purpose. You can't rush the part where
ideas settle.

---

## Phase 0 - Python foundations for data work

*Goal of the phase: be comfortable enough in Python that it stops being the
thing you're thinking about.*

Everything here uses only the standard library. No installs.

| # | Lesson | Goal |
|---|--------|------|
| **001** | **Why ML/AI, and how we'll learn** (start here) | Understand the `data -> pattern -> prediction` mental model and see "learning" happen in ten lines of plain Python. |
| **002** | **Python warm-up for data people** (start here) | Use variables, lists, dicts, loops and functions to read a small CSV with the stdlib `csv` module. |
| 003 | Numbers, strings and the things that bite | Get comfortable with ints vs floats, rounding, f-strings, and the string methods you'll use daily on messy data. |
| 004 | Lists and tuples, properly | Slice, sort, unpack and comprehend lists; know when a tuple is the better fit. |
| 005 | Dictionaries as tiny databases | Group, count and look things up with dicts, `collections.Counter` and `defaultdict`. |
| 006 | Functions that don't lie | Write small, named, single-purpose functions with default arguments and clear return values. |
| 007 | Files, paths and CSV round-trips | Read and write text/CSV files safely with `pathlib` and `csv.DictReader`/`DictWriter`. |
| 008 | Errors, and what to do about them | Read a traceback calmly, use `try`/`except` on purpose, and validate input before it hurts. |
| 009 | Modules, scripts and `if __name__ == "__main__"` | Split code across files, import your own helpers, and run scripts from the command line with arguments. |
| 010 | A little bit of classes | Bundle related data and behaviour with `dataclasses` so later ML code reads cleanly. |
| 011 | Iterators, generators and lazy data | Process data you can't hold in memory using generators and `itertools`. |
| 012 | Mini-project: a stdlib data report | Load a CSV, clean it, compute summary statistics, and print a tidy report with pure Python. |

## Phase 1 - Data science basics

*Goal of the phase: look at a dataset and have honest, useful things to say
about it.*

Installs: `numpy`, `pandas`, `matplotlib`, `scipy` (see `requirements.txt`).

| # | Lesson | Goal |
|---|--------|------|
| 013 | Setting up a real environment | Create a virtual environment, install the Phase 1 packages, and know how to check what's installed. |
| 014 | NumPy arrays: why not just lists? | Create arrays, understand shape and dtype, and see why vectorised code is faster and clearer. |
| 015 | NumPy indexing, broadcasting and reductions | Slice, mask and aggregate arrays; predict what broadcasting will do before running it. |
| 016 | Randomness you can reproduce | Use `numpy.random.default_rng`, seeds, and simple simulations to build intuition for chance. |
| 017 | pandas: Series and DataFrames | Load a CSV into a DataFrame, inspect it with `head`/`info`/`describe`, and select rows and columns. |
| 018 | Cleaning data: missing values, types and duplicates | Find and fix the boring problems that ruin analyses before they start. |
| 019 | Group, aggregate, pivot | Answer "average X per Y" questions with `groupby`, `agg` and `pivot_table`. |
| 020 | Joining tables and reshaping | Merge datasets on keys, and move between wide and long formats with `melt`/`pivot`. |
| 021 | Dates and time series basics | Parse dates, resample, and compute rolling averages without getting lost in timezones. |
| 022 | Plotting that tells the truth | Make line, bar, scatter and histogram plots with matplotlib and label them so they can stand alone. |
| 023 | Distributions and summary statistics | Read means, medians, spreads and skew, and know which one to trust for a given dataset. |
| 024 | Correlation, causation and the traps in between | Compute and interpret correlation, and learn the classic ways it misleads. |
| 025 | Probability intuition for ML | Understand conditional probability, Bayes' rule and expected value through small simulations. |
| 026 | Sampling, confidence and the bootstrap | Quantify "how sure are we?" with resampling instead of formulas you don't trust yet. |
| 027 | Hypothesis tests without the mystery | Run and interpret a t-test and a permutation test, and know what a p-value does and doesn't say. |
| 028 | Mini-project: exploratory data analysis | Take a fresh dataset from raw CSV to a short written analysis with charts you'd show a colleague. |

## Phase 2 - Classical machine learning

*Goal of the phase: train, evaluate and ship a model with scikit-learn, and
know when it's lying to you.*

Installs: `scikit-learn`.

| # | Lesson | Goal |
|---|--------|------|
| 029 | What a model actually is | Define features, targets, parameters and loss, then fit a line by hand before using a library. |
| 030 | Linear regression with scikit-learn | Use the `fit`/`predict` API, read coefficients, and measure error with MAE and RMSE. |
| 031 | Train/test splits and why we need them | Split data properly, see overfitting happen on purpose, and stop trusting training accuracy. |
| 032 | Cross-validation and honest evaluation | Use k-fold cross-validation to get a stable estimate of how a model will really do. |
| 033 | Logistic regression and classification | Predict categories, read probabilities, and understand the decision boundary. |
| 034 | Classification metrics that matter | Choose between accuracy, precision, recall, F1 and ROC-AUC based on what a mistake costs. |
| 035 | Feature scaling and encoding | Standardise numbers and one-hot encode categories, and see which models care. |
| 036 | Pipelines: doing it right every time | Chain preprocessing and models with `Pipeline` and `ColumnTransformer` so nothing leaks. |
| 037 | k-Nearest Neighbours and the curse of dimensionality | Build intuition for distance-based models and why more features aren't always better. |
| 038 | Decision trees you can read | Train, visualise and prune a decision tree; understand impurity and depth. |
| 039 | Random forests and bagging | Combine many trees to cut variance, and read feature importances with healthy suspicion. |
| 040 | Gradient boosting | Understand boosting as "fix the last model's mistakes" and tune a `HistGradientBoosting` model. |
| 041 | Regularisation: Ridge, Lasso and friends | Trade a little bias for a lot less variance, and use Lasso for feature selection. |
| 042 | Hyperparameter search | Use `GridSearchCV` and `RandomizedSearchCV` without accidentally tuning on the test set. |
| 043 | Imbalanced data and threshold tuning | Handle rare classes with reweighting, resampling and choosing a threshold on purpose. |
| 044 | Unsupervised learning: k-means clustering | Find groups in unlabelled data, pick `k` sensibly, and interpret the clusters. |
| 045 | Dimensionality reduction: PCA | Compress features, plot high-dimensional data in 2D, and read explained variance. |
| 046 | Anomaly detection | Flag unusual points with Isolation Forest and density-based methods. |
| 047 | Model interpretation | Explain predictions with permutation importance and partial dependence, and know their limits. |
| 048 | Mini-project: end-to-end tabular ML | Go from raw CSV to a cross-validated, tuned, explained model with a short write-up. |

## Phase 3 - Deep learning foundations

*Goal of the phase: understand neural networks well enough to build and debug
one from scratch, then do it properly in PyTorch.*

Installs: `torch` (CPU is fine).

| # | Lesson | Goal |
|---|--------|------|
| 049 | A neuron in pure NumPy | Implement a single neuron, a loss, and gradient descent by hand. |
| 050 | Backpropagation, step by step | Derive and code the chain rule for a two-layer network without any framework. |
| 051 | Hello, PyTorch: tensors and autograd | Move the NumPy network to PyTorch and let autograd compute the gradients. |
| 052 | Building networks with `nn.Module` | Define layers, forward passes and a training loop the way real PyTorch code does. |
| 053 | Activation functions, initialisation and why training fails | Diagnose dead ReLUs, vanishing gradients and bad learning rates. |
| 054 | Optimisers and learning-rate schedules | Compare SGD, momentum and Adam, and use schedulers to train faster and more stably. |
| 055 | Datasets, DataLoaders and batching | Feed data to a model efficiently, shuffle it correctly, and split it cleanly. |
| 056 | Regularising neural nets | Use dropout, weight decay, early stopping and data augmentation to fight overfitting. |
| 057 | Convolutional networks | Understand convolutions and pooling, then classify small images with a CNN. |
| 058 | Recurrent networks and sequences | Model ordered data with RNNs/GRUs and see where they struggle. |
| 059 | Embeddings: turning categories into geometry | Learn dense representations for words and IDs, and inspect what they capture. |
| 060 | Mini-project: train and debug a real network | Build, train, monitor and improve a classifier with proper logging and checkpoints. |

## Phase 4 - Modern AI: NLP, vision, generative and LLM basics

*Goal of the phase: understand how today's AI systems work, and use them
responsibly from Python.*

Installs: `transformers`, `datasets`, `pillow` (heavier; introduced as needed).

| # | Lesson | Goal |
|---|--------|------|
| 061 | Text as data: tokenisation and bag-of-words | Turn text into numbers with tokenisers, TF-IDF, and a baseline classifier. |
| 062 | Attention, explained slowly | Implement scaled dot-product attention in a few lines and see what it computes. |
| 063 | The Transformer architecture | Put attention, feed-forward layers and positional encodings together into a working block. |
| 064 | Pretrained models and transfer learning | Load a pretrained model from Hugging Face and fine-tune it on a small task. |
| 065 | How large language models are trained | Understand pretraining, instruction tuning and preference tuning at the intuition level. |
| 066 | Prompting as programming | Write clear, testable prompts; use few-shot examples and structured outputs. |
| 067 | Embeddings and semantic search | Embed documents, measure similarity, and build a small search tool. |
| 068 | Retrieval-augmented generation (RAG) | Ground an LLM's answers in your own documents and evaluate whether it helped. |
| 069 | Tool use and agents | Let a model call Python functions, and learn where agent loops go wrong. |
| 070 | Computer vision with pretrained models | Classify, detect and embed images with off-the-shelf vision models. |
| 071 | Generative models: autoencoders and VAEs | Learn to compress and regenerate data, and sample new points from latent space. |
| 072 | Diffusion models, intuitively | Understand noising and denoising, and run a small diffusion pipeline. |
| 073 | Evaluating AI systems | Build small evaluation sets, measure quality honestly, and avoid fooling yourself. |
| 074 | Safety, bias and responsible use | Spot common failure modes and harms, and build the habits that reduce them. |

## Phase 5 - Projects and production intuition

*Goal of the phase: turn "I can train a model" into "I can ship and maintain
one".*

| # | Lesson | Goal |
|---|--------|------|
| 075 | Framing a problem worth solving | Turn a vague request into a measurable ML problem, or decide it doesn't need ML at all. |
| 076 | Project structure, tests and reproducibility | Lay out a real project, pin dependencies, seed everything, and test the data code. |
| 077 | Experiment tracking | Log parameters, metrics and artifacts so you can answer "which run was that?" |
| 078 | Data versioning and feature pipelines | Keep data and features reproducible as they change under you. |
| 079 | Serving a model as an API | Wrap a model in a small web service and call it from another program. |
| 080 | Monitoring and drift | Detect when the world changes and your model quietly gets worse. |
| 081 | Performance: making it fast enough | Profile, batch, cache and quantise so the model fits the budget. |
| 082 | Capstone, part 1: choose and scope | Pick a capstone project, write a plan, and gather your data. |
| 083 | Capstone, part 2: build and evaluate | Build the full pipeline and evaluate it against a baseline. |
| 084 | Capstone, part 3: ship and reflect | Deploy a minimal version, write it up, and decide what you'd do differently next time. |

---

## After the curriculum

Once you're through 084, you'll know enough to read papers, follow library
documentation, and teach yourself what comes next. We may add optional
"deep dive" tracks (time series, reinforcement learning, MLOps at scale) as
weekday extras. They'll be listed here if and when they appear.
