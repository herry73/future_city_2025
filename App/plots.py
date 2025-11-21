import matplotlib.pyplot as plt
import seaborn as sns

def ts_plot(df, col, title=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    df[col].plot(ax=ax)
    ax.set_title(title or col)
    ax.set_xlabel("Time")
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig

def corr_heatmap(df, cols, title="Correlation"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[cols].corr(), annot=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
