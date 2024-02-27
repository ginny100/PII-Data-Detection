# PII Data Detection with Albert base v2 Training + Inference

## Overview

- Inspired by the Kaggle Competition: [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

- The goal of this model is to detect personally identifiable information (PII) in student writing. Automating the detection and removal of PII from educational data will lower the cost of releasing educational datasets, which will support learning science research and the development of educational tools.

## Data

- The dataset comprises approximately 22,000 essays written by students enrolled in a massively open online course. All of the essays were written in response to a single assignment prompt, which asked students to apply course material to a real-world problem. The goal is to annotate personally identifiable information (PII) found within the essays.

- In order to protect student privacy, the original PII in the dataset has been replaced by surrogate identifiers of the same type using a partially automated process. A majority of the essays are reserved for the test set (70%).

### PII Types

- The model should be able to assign labels to the following seven types of PII:

    - `NAME_STUDENT` - The full or partial name of a student that is not necessarily the author of the essay. This excludes instructors, authors, and other person names.
    - `EMAIL` - A student’s email address.
    - `USERNAME` - A student's username on any platform.
    - `ID_NUM` - A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number.
    - `PHONE_NUM` - A phone number associated with a student.
    - `URL_PERSONAL` - A URL that might be used to identify a student.
    - `STREET_ADDRESS` - A full or partial street address that is associated with the student, such as their home address.

### File and Field Information

- The data is presented in JSON format, which includes a document identifier, the full text of the essay, a list of tokens, information about whitespace, and token annotations. The documents were tokenized using the SpaCy English tokenizer.

- Token labels are presented in BIO (Beginning, Inner, Outer) format. The PII type is prefixed with `B-` when it is the beginning of an entity. If the token is a continuation of an entity, it is prefixed with `I-`. Tokens that are not PII are labeled `O`.

    - `{test|train}.json` - the test and training data; the test data given on this page is for illustrative purposes only, and will be replaced during Code rerun with a hidden test set.

        - (int): the index of the essay
        - `document` (int): an integer ID of the essay
        - `full_text` (string): a UTF-8 representation of the essay
        - `tokens` (list)
            - (string): a string representation of each token
        - `trailing_whitespace` (list)
            - (bool): a boolean value indicating whether each token is followed by whitespace.
        - `labels` (list) [training data only]
            - (string): a token label in BIO format
    
    - `sample_submission.csv` - An example of the correct submission format. See the Submission File section of the Overview page for details.

```
kaggle competitions download -c pii-detection-removal-from-educational-data
```

## Model

- [ALBERT Base v2](https://huggingface.co/albert/albert-base-v2) is applied to this Named Entity Recognition problem
