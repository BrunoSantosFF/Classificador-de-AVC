import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def create_plots(str_only, no_str_only):
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.35, hspace=0.27)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])

    background_color = "#f6f6f6"
    fig.patch.set_facecolor(background_color)  # figure background color

    # Plots

    ## Age
    ax0.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    positive = pd.DataFrame(str_only["age"])
    negative = pd.DataFrame(no_str_only["age"])
    sns.kdeplot(positive["age"], ax=ax0, color="#0f4c81", ec='black', label="positive", fill=True)
    sns.kdeplot(negative["age"], ax=ax0, color="#9bb7d4", ec='black', label="negative", fill=True)
    ax0.yaxis.set_major_locator(mtick.MultipleLocator(2))
    ax0.set_ylabel('')
    ax0.set_xlabel('')
    ax0.text(-20, 0.0465, 'Age', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")

    # Smoking
    # Contar os valores de smoking_status para positivos e negativos
    positive = pd.DataFrame(str_only["smoking_status"].value_counts()).reset_index()
    positive.columns = ['smoking_status', 'count']
    negative = pd.DataFrame(no_str_only["smoking_status"].value_counts()).reset_index()
    negative.columns = ['smoking_status', 'count']

    # Combinar os DataFrames de forma que as categorias sejam consistentes
    combined = pd.merge(positive, negative, on='smoking_status', how='outer', suffixes=('_positive', '_negative')).fillna(0)

    # Criar o gráfico de barras horizontais
    ax1.text(0, 4, 'Smoking Status', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax1.barh(combined['smoking_status'], combined['count_positive'], color="#0f4c81", zorder=3, height=0.7, label='With Smoking')
    ax1.barh(combined['smoking_status'], combined['count_negative'], color="#9bb7d4", zorder=3, edgecolor='black', height=0.3, label='Without Smoking')

    # Configurações adicionais para o gráfico
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.xaxis.set_major_locator(mtick.MultipleLocator(10))
    ax1.legend()


    ## GENDER 
    # Gender
    positive = pd.DataFrame(str_only["gender"].value_counts()).reset_index()
    positive.columns = ['gender', 'count']  # Renomeie as colunas
    negative = pd.DataFrame(no_str_only["gender"].value_counts()).reset_index()
    negative.columns = ['gender', 'count']  # Renomeie as colunas

    # Garantir que ambas as DataFrames tenham as mesmas categorias
    all_genders = pd.DataFrame({'gender': ['Male', 'Female']})
    positive = all_genders.merge(positive, on='gender', how='left').fillna(0)
    negative = all_genders.merge(negative, on='gender', how='left').fillna(0)

    # Certifique-se de que os contadores são inteiros
    positive['count'] = positive['count'].astype(int)
    negative['count'] = negative['count'].astype(int)

    x = np.arange(len(positive))

    ax2.text(-0.4, 68.5, 'Gender', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax2.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))

    # Usar a coluna 'count' renomeada
    ax2.bar(x, height=positive['count'], zorder=3, color="#0f4c81", width=0.4)
    ax2.bar(x + 0.4, height=negative['count'], zorder=3, color="#9bb7d4", width=0.4)

    ax2.set_xticks(x + 0.4 / 2)
    ax2.set_xticklabels(['Male', 'Female'])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.yaxis.set_major_locator(mtick.MultipleLocator(10))



    # Heart Dis
    # Heart Disease
    positive = pd.DataFrame(str_only["heart_disease"].value_counts()).reset_index()
    positive.columns = ['heart_disease', 'count']  # Renomeie as colunas
    negative = pd.DataFrame(no_str_only["heart_disease"].value_counts()).reset_index()
    negative.columns = ['heart_disease', 'count']  # Renomeie as colunas

    # Garantir que a coluna 'heart_disease' seja do tipo string em ambos os DataFrames
    positive['heart_disease'] = positive['heart_disease'].astype(str)
    negative['heart_disease'] = negative['heart_disease'].astype(str)

    # Garantir que ambas as DataFrames tenham as mesmas categorias
    all_disease_history = pd.DataFrame({'heart_disease': ['No History', 'History']})
    positive = all_disease_history.merge(positive, on='heart_disease', how='left').fillna(0)
    negative = all_disease_history.merge(negative, on='heart_disease', how='left').fillna(0)

    # Certifique-se de que os contadores são inteiros
    positive['count'] = positive['count'].astype(int)
    negative['count'] = negative['count'].astype(int)

    x = np.arange(len(positive))

    ax3.text(-0.3, 110, 'Heart Disease', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax3.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))

    # Usar a coluna 'count' renomeada
    ax3.bar(x, height=positive['count'], zorder=3, color="#0f4c81", width=0.4)
    ax3.bar(x + 0.4, height=negative['count'], zorder=3, color="#9bb7d4", width=0.4)

    ax3.set_xticks(x + 0.4 / 2)
    ax3.set_xticklabels(['No History', 'History'])
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.yaxis.set_major_locator(mtick.MultipleLocator(20))



    ## AX4 - TITLE
    ax4.spines["bottom"].set_visible(False)
    ax4.tick_params(left=False, bottom=False)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.text(0.5, 0.6, 'Can we see patterns for\n\n patients in our data?', horizontalalignment='center', verticalalignment='center',
            fontsize=22, fontweight='bold', fontfamily='serif', color="#323232")
    ax4.text(0.15, 0.57, "Stroke", fontweight="bold", fontfamily='serif', fontsize=22, color='#0f4c81')
    ax4.text(0.41, 0.57, "&", fontweight="bold", fontfamily='serif', fontsize=22, color='#323232')
    ax4.text(0.49, 0.57, "No-Stroke", fontweight="bold", fontfamily='serif', fontsize=22, color='#9bb7d4')

    # Glucose
    ax5.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    positive = pd.DataFrame(str_only["avg_glucose_level"])
    negative = pd.DataFrame(no_str_only["avg_glucose_level"])
    sns.kdeplot(positive["avg_glucose_level"], ax=ax5, color="#0f4c81", ec='black', label="positive", fill=True)
    sns.kdeplot(negative["avg_glucose_level"], ax=ax5, color="#9bb7d4", ec='black', label="negative", fill=True)
    ax5.text(-55, 0.01855, 'Avg. Glucose Level', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax5.yaxis.set_major_locator(mtick.MultipleLocator(2))
    ax5.set_ylabel('')
    ax5.set_xlabel('')

    ## BMI
    ax6.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    positive = pd.DataFrame(str_only["bmi"])
    negative = pd.DataFrame(no_str_only["bmi"])
    sns.kdeplot(positive["bmi"], ax=ax6, color="#0f4c81", ec='black', label="positive", fill=True)
    sns.kdeplot(negative["bmi"], ax=ax6, color="#9bb7d4", ec='black', label="negative", fill=True)
    ax6.text(-0.06, 0.09, 'BMI', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax6.yaxis.set_major_locator(mtick.MultipleLocator(2))
    ax6.set_ylabel('')
    ax6.set_xlabel('')

    # Work Type
    positive = pd.DataFrame(str_only["work_type"].value_counts()).reset_index()
    positive.columns = ['work_type', 'count']  # Renomear as colunas
    negative = pd.DataFrame(no_str_only["work_type"].value_counts()).reset_index()
    negative.columns = ['work_type', 'count']  # Renomear as colunas

    # Garantir que a coluna 'work_type' seja do tipo string em ambos os DataFrames
    positive['work_type'] = positive['work_type'].astype(str)
    negative['work_type'] = negative['work_type'].astype(str)

    # Criar um gráfico para Work Type
    ax7.bar(negative['work_type'], height=negative['count'], zorder=3, color="#9bb7d4", width=0.05)
    ax7.scatter(negative['work_type'], negative['count'], zorder=3, s=200, color="#9bb7d4")

    ax7.bar(np.arange(len(positive)) + 0.4, height=positive['count'], zorder=3, color="#0f4c81", width=0.05)
    ax7.scatter(np.arange(len(positive)) + 0.4, positive['count'], zorder=3, s=200, color="#0f4c81")

    ax7.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax7.yaxis.set_major_locator(mtick.MultipleLocator(10))
    ax7.set_xticks(np.arange(len(positive)) + 0.4 / 2)
    ax7.set_xticklabels(list(positive['work_type']), rotation=0)
    ax7.text(-0.5, 66, 'Work Type', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")

    # Hypertension
    positive = pd.DataFrame(str_only["hypertension"].value_counts()).reset_index()
    positive.columns = ['hypertension', 'count']  # Renomear as colunas
    negative = pd.DataFrame(no_str_only["hypertension"].value_counts()).reset_index()
    negative.columns = ['hypertension', 'count']  # Renomear as colunas

    # Garantir que a coluna 'hypertension' seja do tipo string em ambos os DataFrames
    positive['hypertension'] = positive['hypertension'].astype(str)
    negative['hypertension'] = negative['hypertension'].astype(str)

    # Criar um gráfico para Hypertension
    x = np.arange(len(positive))

    ax8.text(-0.45, 100, 'Hypertension', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    ax8.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))

    ax8.bar(x, height=positive['count'], zorder=3, color="#0f4c81", width=0.4)
    ax8.bar(x + 0.4, height=negative['count'], zorder=3, color="#9bb7d4", width=0.4)

    ax8.set_xticks(x + 0.4 / 2)
    ax8.set_xticklabels(['No History', 'History'])
    ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))


    plt.show()