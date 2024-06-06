# StreamViz: Interactive Data Classification and Visualization

StreamViz is a Streamlit-based application that allows users to upload datasets, perform data cleaning and encoding, visualize data, and apply classification algorithms. The application is designed to be user-friendly and interactive, making it suitable for both beginners and experienced data scientists.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pages](#pages)
  - [Data Upload](#data-upload)
  - [Data Cleaning and Encoding](#data-cleaning-and-encoding)
  - [Classification](#classification)
  - [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview
StreamViz enables users to:
- Upload datasets (CSV or Excel files)
- Clean and encode data
- Visualize data using various plot types
- Apply and compare different classification algorithms

## Features
- **Data Upload:** Easily upload CSV or Excel files and preview the data.
- **Data Cleaning and Encoding:** Encode categorical variables and select target and input features.
- **Classification:** Choose from various classification algorithms like Logistic Regression, SVM, Decision Tree, KNN, and Random Forest, and evaluate their performance.
- **Visualization:** Create different types of plots such as pie charts, bar plots, heatmaps, distribution plots, violin plots, and box plots.

## Installation
To set up the StreamViz application, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/streamviz.git
    cd streamviz
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the StreamViz application, execute the following command:
```bash
streamlit run streamviz.py
