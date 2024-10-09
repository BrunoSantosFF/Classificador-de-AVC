import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric_distribution(df, conts):
    """
    Função para plotar a distribuição de variáveis numéricas.
    
    Parâmetros:
    df: DataFrame pandas contendo os dados
    conts: Lista de variáveis contínuas (colunas) a serem plotadas
    """
    fig = plt.figure(figsize=(12, 12), dpi=150, facecolor='#fafafa')
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.1, hspace=0.4)

    background_color = "#fafafa"

    plot = 0
    for row in range(0, 1):
        for col in range(0, 3):
            locals()["ax"+str(plot)] = fig.add_subplot(gs[row, col])
            locals()["ax"+str(plot)].set_facecolor(background_color)
            locals()["ax"+str(plot)].tick_params(axis='y', left=False)
            locals()["ax"+str(plot)].get_yaxis().set_visible(False)
            for s in ["top", "right", "left"]:
                locals()["ax"+str(plot)].spines[s].set_visible(False)
            plot += 1

    plot = 0
    for variable in conts:
        sns.kdeplot(df[variable], ax=locals()["ax"+str(plot)], color='#0f4c81', shade=True, linewidth=1.5, ec='black', alpha=0.9, zorder=3, legend=False)
        locals()["ax"+str(plot)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1, 5))
        plot += 1

    locals()["ax0"].set_xlabel('Age')
    locals()["ax1"].set_xlabel('Avg. Glucose Levels')
    locals()["ax2"].set_xlabel('BMI')

    locals()["ax0"].text(-20, 0.022, 'Numeric Variable Distribution', fontsize=20, fontweight='bold', fontfamily='serif')
    locals()["ax0"].text(-20, 0.02, 'We see a positive skew in BMI and Glucose Level', fontsize=13, fontweight='light', fontfamily='serif')

    plt.show()
