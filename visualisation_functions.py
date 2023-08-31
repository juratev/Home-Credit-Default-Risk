import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_double_chart(data: pd.DataFrame, feature: str, size: list,
                      palette: dict, hue_feature: str, title: str,
                      top_value: int, bbox_to_anchor_value: tuple) -> None:
    """
    Generates a double chart visualizing data using two subplots.

    Args:
        data (pd.DataFrame): The input pandas DataFrame containing the data to be visualized.
        feature (str): The name of the feature column in the DataFrame to be plotted on the y-axis.
        size (list): A list specifying the size of the figure (width, height) in inches.
        palette (dict): A dictionary defining the color palette for different categories of the hue feature.
        hue_feature (str): The name of the hue feature column in the DataFrame to differentiate the data in the plots.
        title (str): The title of the figure.
        top_value (int): The top margin value for the subplots.
        bbox_to_anchor_value (tuple): A tuple specifying the position of the legend in the histogram subplot.

    Returns:
        None. Displays the double chart using matplotlib and seaborn.
    """
    data = data.copy()
    data[feature].fillna('Unknown', inplace=True)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(size[0], size[1]))

    count_plot = sns.countplot(
        data=data.sort_values(by=[feature, hue_feature],
                              ascending=[True, False]),
        y=feature,
        ax=axs[0],
        color=sns.color_palette()[0]
    )
    axs[0].set_title('Number of people', fontsize=14, weight='bold')
    [label.set_fontweight('bold') for label in axs[0].get_yticklabels()]

    total = len(data[feature])

    for p in count_plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        axs[0].annotate(percentage, (x, y), weight='bold')

    sns.histplot(
        data=data.sort_values(
            by=[feature, hue_feature], ascending=[True, False]),
        y=feature,
        ax=axs[1],
        hue=hue_feature,
        hue_order=palette.keys(),
        multiple='fill',
        stat='proportion',
        discrete=True,
        shrink=0.5,
        palette=palette,
        legend=True
    )
    axs[1].set_title('Proportion', fontsize=14, weight='bold')
    [label.set_fontweight('bold') for label in axs[1].get_yticklabels()]

    legend = axs[1].get_legend()
    handles = legend.legendHandles
    legend.remove()
    axs[1].legend(handles,
                  list(palette.keys()),
                  title='',
                  ncol=len(data[hue_feature].unique()),
                  bbox_to_anchor=bbox_to_anchor_value,
                  loc='upper left')
    fig.suptitle(title, fontsize=18, fontweight='bold')
    for ax in axs.flat:
        ax.margins(x=0.14)
        ax.set(ylabel=None)
        ax.set(xlabel=None)
    plt.subplots_adjust(wspace=0.4, top=top_value)
    plt.show()


def make_multiple_boxplots(t, features, hue_feature, figsize,
                           n_cols, palette):
    """
    Generates multiple boxplots to visualize numerical features in a dataset.

    Args:
        t (pd.DataFrame): The input pandas DataFrame containing the data.
        features (list): A list of feature names to be plotted.
        hue_feature (str): The name of the hue feature column in the DataFrame to differentiate the boxplots.
        figsize (tuple): A tuple specifying the size of the figure (width, height) in inches.
        n_cols (int): The number of columns to organize the subplots.
        palette (dict): A dictionary defining the color palette for different categories of the hue feature.

    Returns:
        None. Displays the multiple boxplots using matplotlib and seaborn.
    """
    font = {'weight': 'bold'}

    n_features = len(features)
    n_rows = n_features // n_cols + (n_features % n_cols > 0)

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(figsize[0], figsize[1]))

    plt.suptitle('Boxplot of all numerical features',
                 fontweight='bold', fontsize=15, y=1)
    for ax, feature in zip(axs.flatten(), features):
        if hue_feature != '':
            sns.boxplot(x=hue_feature, y=feature, data=t,
                        ax=ax, showfliers=False, palette=palette)
        else:
            sns.boxplot(y=feature, data=t, ax=ax, showfliers=False)

        ft = feature.replace('_', ' ').capitalize()
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), weight='bold')
        ax.set_title(ft, fontdict=font, fontsize=12)

    for i in range(len(features), n_rows * n_cols):
        axs.flatten()[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_distribution_of_cat_variables(df, cat_col, nsize, yaxis_size,
                                       figsize, top_space=0.9,
                                       orientation='v', space=[0.2, 0.5]):
    """
    Plots the distribution of categorical variables in a DataFrame.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing the data.
        cat_col (list): A list of categorical column names to be plotted.
        nsize (tuple): A tuple specifying the number of rows and columns for the subplots.
        yaxis_size (float): The size of the y-axis relative to the maximum count of each category.
        figsize (tuple): A tuple specifying the size of the figure (width, height) in inches.
        top_space (float, optional): The top space allocated for the suptitle. Defaults to 0.9.
        orientation (str, optional): The orientation of the countplot, either 'v' (vertical) or 'h' (horizontal). Defaults to 'v'.
        space (list, optional): A list specifying the horizontal and vertical spacing between subplots. Defaults to [0.2, 0.5].

    Returns:
        None. Displays the distribution plots of categorical variables using matplotlib and seaborn.
    """
    fig, axs = plt.subplots(nrows=nsize[0], ncols=nsize[1],
                            figsize=(figsize[0], figsize[1]),
                            gridspec_kw={'wspace': space[0],
                                         'hspace': space[1]})

    total_plots = nsize[0] * nsize[1]
    if orientation == 'v':
        xytext = (0, 10)
    else:
        xytext = (20, 0)

    for i in range(total_plots):
        ax = axs.flatten()[i]

        if i < len(cat_col):
            if orientation == 'v':
                sns.countplot(x=cat_col[i],
                              data=df.sort_values(by=cat_col[i]),
                              ax=ax, color=sns.color_palette()[0])
                max_count = len(df[cat_col[i]])
                ax.set_ylim(0, max_count * yaxis_size)
                plt.setp(ax.get_xticklabels(), weight='bold')
            else:
                sns.countplot(y=cat_col[i],
                              data=df.sort_values(by=cat_col[i]),
                              ax=ax, color=sns.color_palette()[0])
                max_count = len(df[cat_col[i]])
                ax.set_xlim(0, max_count * yaxis_size)
                plt.setp(ax.get_yticklabels(), weight='bold')

            ax.set_title(cat_col[i].replace('_', ' ').capitalize(),
                         fontsize=14, fontweight='bold')

            total = len(df[cat_col[i]])
            for p in ax.patches:
                percentage = f'{p.get_width() / total:.1%}' if orientation == 'h' else f'{p.get_height() / total:.1%}'
                x = p.get_x() + p.get_width() / 2 if orientation == 'v' else p.get_width()
                y = p.get_y() + p.get_height() / 2 if orientation == 'h' else p.get_height()
                ax.annotate(percentage,
                            (x, y),
                            ha='center',
                            va='center',
                            xytext=xytext,
                            textcoords='offset points',
                            weight='bold')

        else:
            ax.axis('off')

    fig.suptitle('Distribution of Categorical Variables',
                 fontsize=18, fontweight='bold')
    fig.subplots_adjust(top=top_space)

    for ax in axs.flat:
        if orientation == 'v':
            ax.margins(y=0.1)
            ax.set(xlabel=None)
        else:
            ax.margins(x=0.1)
            ax.set(ylabel=None)

    plt.show()
