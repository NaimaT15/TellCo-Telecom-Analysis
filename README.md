## Welcome to TellCo-Telecom-User-Engagement-Analysis 👋

## Description

This repository, titled "TellCo-Telecom-User-Engagement-Analysis," provides a comprehensive toolkit for analyzing customer behavior and user engagement within a telecommunications network. The project focuses on aggregating and analyzing session data (xDR records) to uncover insights related to customer engagement, application usage, and traffic distribution. Key features include:

- **User Engagement Analysis:** Aggregates data by user, examining session frequency, session duration, and total traffic (download and upload).

- **Clustering for Customer Segmentation:** Utilizes K-Means clustering to classify customers into engagement groups, enabling tailored service offerings and targeted marketing strategies.

- **Application Traffic Analysis:** Analyzes total traffic per application, identifying the most engaged users and the most popular applications like YouTube, gaming, and social media.

- **Data Visualization:** Contains visualizations such as bar charts and correlation matrices to communicate key insights effectively.

The project is structured into directories for scripts, data, and notebooks, making it modular and extendable, suitable for telecom data scientists and business analysts aiming to improve customer satisfaction and profitability.

## Getting Started

Follow the instructions below to get started with the project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python
- VS Code or any other IDE
- Jupyter Notebook Extension on VS Code

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NaimaT15/TellCo-Telecom-Analysis.git
```

2. Navigate to the project root:
```bash
cd TellCo-Telecom-Analysis
```

3. Setup Virtual Environment (Optional but recommended):

```bash
python -m venv venv
```

```bash  
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. Install Dependencies:
```bash
pip install -r requirements.txt
```

## Usage

- **Exploratory Data Analysis:** Open the `tellco_telecom_analysis.ipynb` notebook within Jupyter Notebook to explore the dataset and clean the data:
```bash
jupyter notebook notebooks/tellco_telecom_analysis.ipynb
```

- **User Engagement Analysis:** Review the `tellco_telecom_analysis.ipynb` for detailed engagement metrics and customer segmentation:
```bash
jupyter notebook notebooks/tellco_telecom_analysis.ipynb
```

- **Running Scripts:** Execute scripts directly from the command line:
```bash
python scripts/tellco_telecom_analysis.py
```

## Features

- **User Engagement Metrics:** Tracks user behavior metrics such as session frequency, session duration, and total traffic.
  
- **Clustering Analysis:** Groups users into three clusters based on engagement levels using K-Means clustering.

- **Application Usage Analysis:** Identifies top applications by traffic and most engaged users for each application.

- **Data Visualization:** Generates charts to visualize user engagement, application usage, and traffic patterns.

## Contribution

Contributions are welcome! Please create a pull request or issue to suggest improvements or add new features.

## Author

👤 **Naima Tilahun**

* Github: [@NaimaT15](https://github.com/NaimaT15)

## Show your support

Give a ⭐️ if this project helped you!
