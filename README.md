# Using color jitter to distinguish individuals in categorically colored visualizations

## Requirements

This module requires Python and the following Python libraries:

- numpy
- matplotlib
- pandas
- distinctipy
- jupyter
- scipy


## Installation

1. Clone this repository or download the module files to a local directory.
2. Navigate to the directory containing `requirements.txt`.
3. Create a virtual environment (recommended):

4. Install the required packages:
`pip install -r requirements.txt`.

5. Start a Jupyter Notebook to run the visualizations:

## Usage

Import the module in your Jupyter Notebook and use the provided functions to create plots:


```
from color_jitter import *

color_wheel_plot(n_categories=8, custom_palette=nightingale_palette, color_jitter=0.08)

```

![Color wheel plot](plots\nightingale_colorwheel_jitter.png)


Or use it right away with the iris dataset:


```
import seaborn as sns

# Load the Iris dataset
iris_df = sns.load_dataset('iris')

# Convert the 'cut' column to a categorical type with the specified order
iris_df['species'] = pd.Categorical(iris_df['species'])#, categories=cut_order, ordered=True)

# Assuming 'categories_in_data' is a list of your data's categories
categories_in_data = iris_df['species'].unique()

# Extract the first N colors from the Nightingale palette
num_categories = len(categories_in_data)
selected_colors = list(nightingale_palette.values())[:num_categories]

# Build the new palette
category_palette = {category: color for category, color in zip(categories_in_data, selected_colors)}

plot_title = 'A good amount of color jitter'
colorjitter_scatterplot(iris_df, 'sepal_length', 'sepal_width', 'species', jitter_intensity=0.1, point_size=20, opacity=1, 
                        custom_palette=category_palette, plot_title=plot_title,
                        x_label='Sepal Length (cm)',
                        y_label='Sepal Width (cm)')

```
![Iris dataset plotted with color jitter](plots\iris_color_jitter.png)

See the Jupyter notebook for more advanced usage.