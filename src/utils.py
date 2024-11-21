import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib_venn import venn3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap


colors = ['#705557', '#b27b77', '#ddcac6', '#af9294', '#8c9a5b', '#708090', '#c5a880', '#a0522d']

colors_map = [ '#ddcac6', '#af9294',  '#b27b77', '#a0522d']

colors_map_heat = [ '#fcf5f0' , '#eee1de' , '#ddcac6', '#af9294', '#a0522d']

colors_venn = ['#705557',   '#a0522d', '#af9294', '#b27b77',  '#c5a880', '#ddcac6', '#708090']


def plot_grafico_kessner(total_adequado, total_inadequado, total_intermediario, percentuais_adequado, percentuais_inadequado, percentuais_intermediario, categorias, quantidades_adequado, quantidades_inadequado, quantidades_intermediario):
    # Verificar se todas as listas têm o mesmo tamanho
    if not (len(percentuais_adequado) == len(percentuais_inadequado) == len(percentuais_intermediario) == len(quantidades_adequado) == len(quantidades_inadequado) == len(quantidades_intermediario) == len(categorias)):
        raise ValueError("As listas de percentuais, categorias e quantidades devem ter o mesmo tamanho.")

    x = np.arange(len(categorias))  # localização das labels
    width = 0.20  # largura das barras

    fig, ax = plt.subplots(figsize=(14, 8))

    # Barra única para a categoria 'Kessner'
    total_geral = total_adequado + total_inadequado + total_intermediario
    bar_kessner_adequado = ax.bar(-1, total_adequado / total_geral * 100, width, color='#8c9a5b', label='Adequado')  # Verde suave
    bar_kessner_intermediario = ax.bar(-1, total_intermediario / total_geral * 100, width, bottom=total_adequado / total_geral * 100, color='#c5a880', label='Intermediário')  # Amarelo suave
    bar_kessner_inadequado = ax.bar(-1, total_inadequado / total_geral * 100, width, bottom=(total_adequado + total_intermediario) / total_geral * 100, color='#FF8888', label='Inadequado')  # Vermelho suave

    # Ajustar posição para as demais categorias
    x_pos = np.arange(len(categorias))

    # Barras para adequado, intermediário e inadequado nas outras categorias
    bars1 = ax.bar(x_pos - width, percentuais_adequado, width, color='#66BB66')  # Verde
    bars2 = ax.bar(x_pos, percentuais_intermediario, width, color='#FFCC66')  # Amarelo
    bars3 = ax.bar(x_pos + width, percentuais_inadequado, width, color='#FF6666')  # Vermelho

    # Cálculo dos percentuais para garantir que somem 100% por categoria
    percentuais_bars3_adequado = [
        (quantidades_adequado[i] / (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i])) * 100 if (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i]) > 0 else 0
        for i in range(len(quantidades_adequado))
    ]

    percentuais_bars3_intermediario = [
        (quantidades_intermediario[i] / (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i])) * 100 if (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i]) > 0 else 0
        for i in range(len(quantidades_intermediario))
    ]

    percentuais_bars3_inadequado = [
        (quantidades_inadequado[i] / (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i])) * 100 if (quantidades_adequado[i] + quantidades_inadequado[i] + quantidades_intermediario[i]) > 0 else 0
        for i in range(len(quantidades_inadequado))
    ]

    # Ajustar posição das barras totalizadoras para evitar sobreposição
    bars3_adequado = ax.bar(x_pos + width * 2, percentuais_bars3_adequado, width, color='#8c9a5b', alpha=0.6)  # Movido mais à direita
    bars3_intermediario = ax.bar(x_pos + width * 2, percentuais_bars3_intermediario, width, bottom=percentuais_bars3_adequado, color='#c5a880', alpha=0.6)
    bars3_inadequado = ax.bar(x_pos + width * 2, percentuais_bars3_inadequado, width, bottom=[a + b for a, b in zip(percentuais_bars3_adequado, percentuais_bars3_intermediario)], color='#FF8888', alpha=0.6)

    # Adicionar rótulos e título
    ax.set_xlabel('Categorias')
    ax.set_ylabel('Percentual (%)')
    ax.set_title('Comparação de Desfechos Neonatais entre Adequado, Intermediário e Inadequado')
    ax.set_xticks(np.arange(-1, len(categorias)))
    ax.set_xticklabels(['Kessner'] + categorias)  # Inclui 'Kessner' como a primeira categoria
    ax.set_ylim(0, 100)
    ax.legend()

    # Função para adicionar rótulos nas barras
    def autolabel(bars, valores, percentuais):
        for bar, valor, percentual in zip(bars, valores, percentuais):
            height = bar.get_height()
            ax.annotate(f'{percentual:.1f}%\n({valor})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset vertical
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='black')

    # Função para adicionar rótulos nas barras empilhadas
    def autolabel_empilhado(bars, valores, percentuais, bottoms):
        for bar, valor, percentual, bottom in zip(bars, valores, percentuais, bottoms):
            height = bar.get_height()
            ax.annotate(f'{percentual:.1f}%\n({valor})',
                        xy=(bar.get_x() + bar.get_width() / 2, bottom + height * 0.5),  # Ajuste para centralizar dentro da seção
                        textcoords="data",
                        ha='center', va='center',
                        fontsize=9, color='black')

    # Adicionar rótulos para as categorias
    autolabel(bars1, quantidades_adequado, percentuais_adequado)
    autolabel(bars2, quantidades_intermediario, percentuais_intermediario)
    autolabel(bars3, quantidades_inadequado, percentuais_inadequado)
    autolabel_empilhado(bars3_adequado, quantidades_adequado, percentuais_bars3_adequado, [0] * len(percentuais_bars3_adequado))
    autolabel_empilhado(bars3_intermediario, quantidades_intermediario, percentuais_bars3_intermediario, percentuais_bars3_adequado)
    autolabel_empilhado(bars3_inadequado, quantidades_inadequado, percentuais_bars3_inadequado, [a + b for a, b in zip(percentuais_bars3_adequado, percentuais_bars3_intermediario)])

    # Adicionar rótulos para a barra 'Kessner'
    autolabel_empilhado(bar_kessner_adequado, [total_adequado], [total_adequado / total_geral * 100], [0])
    autolabel_empilhado(bar_kessner_intermediario, [total_intermediario], [total_intermediario / total_geral * 100], [total_adequado / total_geral * 100])
    autolabel_empilhado(bar_kessner_inadequado, [total_inadequado], [total_inadequado / total_geral * 100], [(total_adequado + total_intermediario) / total_geral * 100])

    plt.tight_layout()
    plt.show()
    plt.savefig(f"img/graph_kessner.png")

    plot_grafico_kessner(total_adequado, total_inadequado, total_intermediario, percentuais_adequado, percentuais_inadequado, percentuais_intermediario, categorias, quantidades_adequado, quantidades_inadequado, quantidades_intermediario)

    

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
sns.set_palette(sns.color_palette(colors))  


metadados = {
    "IDADE": {"analise": 0},
    "FAIXA_ETARIA": {"analise": 1},
    "ESCOLARIDADE": {"analise": 0},
    "CATEGORIA_ESCOLARIDADE": {"analise": 1},
    "CONSULTAS_PRE_NATAIS": {"analise": 1},
    "TIPO_SANGUINEO_MAE": {"analise": 1},
    "TIPO_DE_GESTACAO": {"analise": 1},
    "STATUS_DE_TRATAMENTO_DE_MORBIDADES": {"analise": 1},
    "TIPO_PARTO": {"analise": 1},
    "IG_SEMANA_FRACIONADA": {"analise": 1},
    "DESFECHO_NEONATAL_RN": {"analise": 1},
    "APGAR_1_MIN": {"analise": 1},
    "APGAR_5_MIN": {"analise": 1},
    "SEXO_RN": {"analise": 1},
    "GESTA": {"analise": 1},
    "CESAREA": {"analise": 1},
    "PARTO_VAGINAL": {"analise": 1},
    "INICIO_PRE_NATAL": {"analise": 1},
    "NASCIDO_VIVO": {"analise": 1},
    "RELIGIAO": {"analise": 1},
    "SITUACAO_CONJUGAL": {"analise": 1},
    "PROCEDENCIA_GESTANTE": {"analise": 1},
    "CLASSE_PARIDADE": {"analise": 1},
    "CATEGORIA_PREMATURIDADE": {"analise": 1},
    "CATEGORIA_CONSULTAS": {"analise": 1},
    "CATEGORIA_PESO_RN": {"analise": 1},
    "CATEGORIA_KESSNER": {"analise": 1},
    "CONVENIO": {"analise": 1},
    "OBITO?": {"analise": 1},
    "CUIDADOS_PERINATAIS_RN": {"analise": 1}
}

def obter_colunas_relevantes(metadados=metadados):
    return [coluna for coluna, info in metadados.items() if info.get('analise') == 1]

df_pre_termo = pd.read_csv('../assets/data/analitycs/df_prontuarios_prematuros.csv')

# Carregar tabelas one-hot e tabela original
df_gestantes = pd.read_csv('../assets/data/analitycs/df_gestante.csv')
df_rn = pd.read_csv('../assets/data/analitycs/df_rn.csv')

df_gestantes_one_hot = pd.read_csv('../assets/data/analitycs/one_hot_gestante.csv')
df_rn_one_hot = pd.read_csv('../assets/data/analitycs/one_hot_rn.csv')

df_intercorrencias_one_hot = pd.read_csv('../assets/data/analitycs/df_intercorrencias_one_hot.csv')
df_morbidade_feto_one_hot = pd.read_csv('../assets/data/analitycs/df_morbidade_feto_one_hot.csv')
df_habitos_de_vida_one_hot = pd.read_csv('../assets/data/analitycs/df_habito_de_vida_one_hot.csv')
df_morbidade_one_hot = pd.read_csv('../assets/data/analitycs/df_morbidade_one_hot.csv')
df_medicamentos_uso_continuo_one_hot = pd.read_csv('../assets/data/analitycs/df_medicamento_uso_continuo_one_hot.csv')

df_soma_intercorrencias = pd.read_csv('../assets/data/analitycs/df_soma_intercorrencias.csv')
df_soma_habito_de_vida = pd.read_csv('../assets/data/analitycs/df_soma_habito_de_vida.csv')
df_soma_medicamento_uso_continuo = pd.read_csv('../assets/data/analitycs/df_soma_medicamento_uso_continuo.csv')
df_soma_morbidade_feto = pd.read_csv('../assets/data/analitycs/df_soma_morbidade_feto.csv')
df_soma_morbidade = pd.read_csv('../assets/data/analitycs/df_soma_morbidade.csv')

multiparas = pd.read_csv('../assets/data/analitycs/multiparas.csv')
primiparas = pd.read_csv('../assets/data/analitycs/primiparas.csv')


prontuarios_one_hot = pd.read_csv('../assets/data/analitycs/one_hot_prontuario_prematuros.csv')




somente_cesarea = multiparas[(multiparas['CESAREA'] > 0) & (multiparas['PARTO_VAGINAL'].isna()) & (multiparas['ABORTO'].isna())]
somente_parto_vaginal = multiparas[(multiparas['CESAREA'].isna()) & (multiparas['PARTO_VAGINAL'] > 0) & (multiparas['ABORTO'].isna())]
somente_aborto = multiparas[(multiparas['CESAREA'].isna()) & (multiparas['PARTO_VAGINAL'].isna()) & (multiparas['ABORTO'] > 0)]

cesarea_aborto = multiparas[(multiparas['CESAREA'] > 0) & (multiparas['ABORTO'] > 0) & (multiparas['PARTO_VAGINAL'].isna())]
cesarea_parto_vaginal = multiparas[(multiparas['CESAREA'] > 0) & (multiparas['PARTO_VAGINAL'] > 0) & (multiparas['ABORTO'].isna())]
parto_vaginal_aborto = multiparas[(multiparas['PARTO_VAGINAL'] > 0) & (multiparas['ABORTO'] > 0) & (multiparas['CESAREA'].isna())]
    
cesarea_parto_vaginal_aborto = multiparas[(multiparas['CESAREA'] > 0) & (multiparas['PARTO_VAGINAL'] > 0) & (multiparas['ABORTO'] > 0)]

total_cesarea = len(somente_cesarea) + len(cesarea_aborto) + len(cesarea_parto_vaginal) + len(cesarea_parto_vaginal_aborto)
total_parto_vaginal = len(somente_parto_vaginal) + len(parto_vaginal_aborto) + len(cesarea_parto_vaginal) + len(cesarea_parto_vaginal_aborto)
total_aborto = len(somente_aborto) + len(cesarea_aborto) + len(parto_vaginal_aborto) + len(cesarea_parto_vaginal_aborto)

total_multiparas = len(multiparas)
total_primiparas = len(primiparas)



total_populacao = len(df_gestantes)


def venn_primiparas_multiparas(somente_cesarea, somente_parto_vaginal, somente_aborto, cesarea_aborto, cesarea_parto_vaginal, parto_vaginal_aborto, cesarea_parto_vaginal_aborto, total_cesarea, total_parto_vaginal, total_aborto, total_multiparas, total_primiparas, total_populacao):
    plt.figure(figsize=(8, 8))
    venn = venn3(subsets=(
        len(somente_cesarea),
        len(somente_parto_vaginal),
        len(cesarea_parto_vaginal),
        len(somente_aborto),
        len(cesarea_aborto),
        len(parto_vaginal_aborto),
        len(cesarea_parto_vaginal_aborto)
    ), set_labels=(
        f'Cesárea: {total_cesarea}',
        f'Parto Vaginal: {total_parto_vaginal}',
        f'Aborto: {total_aborto}'
    ))

    # Aplicar cores personalizadas aos conjuntos
    for i, patch in enumerate(venn.patches):
        if patch:  # Verifica se o patch não é None
            patch.set_color(colors_venn[i % len(colors)])
            patch.set_alpha(0.7)  # Ajustar a transparência para melhor visualização

    for label in venn.set_labels:
        label.set_fontsize(15)  # Ajusta o tamanho da fonte dos rótulos dos conjuntos

    for label in venn.subset_labels:
        if label:  # Verifica se o rótulo não é None
            label.set_fontsize(15)  # Ajusta o tamanho da fonte dos valores

    plt.title(f'Gestações anteriores \n\nTotal de Gestantes: {total_populacao} \n\n Gestantes Multíparas: {total_multiparas} \n\n Gestantes Primíparas {total_primiparas}', fontdict={'fontsize': 15})
    
    plt.savefig(f"img/venn_primiparar_multiparas.png")

    plt.show()

def contar_multivalores(df, coluna, valor_nulo, entidade):
    # Remover valores nulos

    qtd_total = df.shape[0]
        # Substituir valores nulos e garantir que todos os valores sejam strings
    df[coluna] = df[coluna].fillna(valor_nulo).replace(valor_nulo, '').astype(str)

    # Contar multivalores
    contagem = df[coluna].apply(lambda x: len(x.split(', ')) if x else 0)

    # Obter a contagem de cada quantidade de multivalores
    resultado = contagem.value_counts().sort_index()

    saida = []
    for quantidade, Qtd in resultado.items():
        if quantidade == 0:
            saida.append(f"{Qtd} ({((Qtd / qtd_total )* 100 ):.1f}%) {entidade} não tem nenhuma {coluna}")
        elif quantidade == 1:
            saida.append(f"{Qtd} ({((Qtd / qtd_total )* 100 ):.1f}%) {entidade} tem 1 {coluna}")
        else:
            saida.append(f"{Qtd} ({((Qtd / qtd_total )* 100 ):.1f}%) {entidade} tem {quantidade} {coluna}")

    return '\n'.join(saida)

def texto_primiparas_multiparas(total_populacao, total_multiparas, total_primiparas, somente_cesarea, somente_parto_vaginal, somente_aborto, cesarea_aborto, cesarea_parto_vaginal, parto_vaginal_aborto, cesarea_parto_vaginal_aborto, total_cesarea, total_parto_vaginal, total_aborto):
    resultado_texto = f"""
        ### Análise do Diagrama de Venn: Gestantes Multíparas e Primíparas
        O diagrama de Venn ilustra as sobreposições entre diferentes tipos de experiências de parto entre as gestantes multíparas. Vamos explorar cada seção do diagrama:
        **Total de Gestantes:**
        A análise considera um total de {total_populacao} gestantes, das quais {total_multiparas} são multíparas e {total_primiparas} são primíparas.
        **Cesárea Exclusiva:**
        - Existem {len(somente_cesarea)} gestantes que tiveram apenas cesáreas, sem partos vaginais ou abortos.
        **Parto Vaginal Exclusivo:**
        - {len(somente_parto_vaginal)} gestantes tiveram apenas partos vaginais.
        **Aborto Exclusivo:**
        - {len(somente_aborto)} gestantes tiveram apenas abortos, sem partos cesáreos ou vaginais..
        **Combinando Cesárea e Aborto:**
        - {len(cesarea_aborto)} gestantes passaram por cesáreas e abortos, mas não tiveram partos vaginais. Isso pode refletir em condições médicas que afetaram suas experiências de parto.
        **Combinando Cesárea e Parto Vaginal:**
        - {len(cesarea_parto_vaginal)} gestantes experimentaram tanto cesáreas quanto partos vaginais.
        **Combinando Parto Vaginal e Aborto:**
        - {len(parto_vaginal_aborto)} gestantes tiveram partos vaginais e abortos, mas não cesáreas.
        **Todas as Experiências (Cesárea, Parto Vaginal e Aborto):**
        - {len(cesarea_parto_vaginal_aborto)} gestantes tiveram experiências com cesáreas, partos vaginais e abortos. Este grupo é o mais complexo em termos de experiências reprodutivas.
        ### Resumo:
        - **Total de Cesáreas**: {total_cesarea} ({(total_cesarea / total_multiparas) * 100:.1f}% das multíparas)
        - **Total de Partos Vaginais**: {total_parto_vaginal} ({(total_parto_vaginal / total_multiparas) * 100:.1f}% das multíparas)
        - **Total de Abortos**: {total_aborto} ({(total_aborto / total_multiparas) * 100:.1f}% das multíparas)
        Este diagrama de Venn oferece uma representação visual clara das complexidades das experiências de parto entre gestantes multíparas, destacando tanto as exclusividades quanto as combinações de eventos reprodutivos.
        """
    print(resultado_texto)





def print_df_numerico(df_numerico, entidade='Gestante'):
    sns.set_theme(style="whitegrid")

    # Verificando se as colunas são numéricas
    numeric_cols = df_numerico.select_dtypes(include=[np.number]).columns

    # Criar histogramas para as variáveis numéricas
    fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(15, len(numeric_cols) * 5))
    fig.suptitle(f'Distribuição das Variáveis numéricas - {entidade}', fontsize=20)

    # Iterar sobre cada coluna e o respectivo eixo
    for ax, (coluna, color) in zip(axes, zip(numeric_cols, colors * len(numeric_cols))):
        # Remover valores ausentes antes de plotar
        data = df_numerico[coluna].dropna()

        # Verificar se há dados para plotar
        if data.empty:
            ax.text(0.5, 0.5, f'Sem dados para {coluna}', fontsize=12, ha='center')
            continue

        # Criar o histograma
        sns.histplot(data, bins=15, kde=True, ax=ax, color=color, edgecolor='black')

        # Calcular estatísticas
        media = data.mean()
        mediana = data.median()
        desvio_padrao = data.std()
        minimo = data.min()
        maximo = data.max()

        # Adicionar linhas de referência para média e mediana
        ax.axvline(media, color='red', linestyle='dashed', linewidth=1.5, label=f'Média: {media:.2f}')
        ax.axvline(mediana, color='green', linestyle='dashed', linewidth=1.5, label=f'Mediana: {mediana:.2f}')

        # Configurar rótulos e legendas
        ax.set_title(f'Distribuição de {coluna}', fontsize=16)
        ax.set_xlabel(coluna, fontsize=14)
        ax.set_ylabel('Frequência', fontsize=14)
        ax.legend()
        # Criar o texto explicativo
        texto_explicativo = f""
        texto_explicativo += f"Média: {media:.2f}"
        texto_explicativo += f"; Mediana: {mediana:.2f}"
        texto_explicativo += f"; DP: {desvio_padrao:.2f}"
        texto_explicativo += f"; Min: {minimo:.2f}"
        texto_explicativo += f"; MAX: {maximo:.2f}"

        # Análise básica da distribuição
        if media > mediana:
            texto_explicativo += "- A distribuição é assimétrica à direita (positivamente enviesada).\n"
        elif media < mediana:
            texto_explicativo += "- A distribuição é assimétrica à esquerda (negativamente enviesada).\n"
        else:
            texto_explicativo += "- A distribuição é simétrica.\n"

        # Adicionar o texto explicativo abaixo do gráfico
        ax.text(0.5, -0.3, texto_explicativo, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(f"img/numerico_{entidade}.png")

    plt.show()




def explorar_dados(df, tipo, colunas_relevantes):
    sns.set_theme(style="whitegrid")
    
    # Filtrar apenas as colunas relevantes que estão presentes no DataFrame
    colunas_presentes = [coluna for coluna in colunas_relevantes if coluna in df.columns]
    df = df[colunas_presentes]
    
    print(f"Tamanho do DataFrame de {tipo}: {df.shape}\n")
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(y=column, data=df, order=df[column].value_counts().index, palette=colors)
        total = len(df[column])
        
        # Adicionar rótulos de contagem e percentual
        for p in ax.patches:
            percentage = f'{100 * p.get_width() / total:.1f}%'
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.annotate(f'{int(p.get_width())} ({percentage})', (x + 0.02 * total, y), ha='left', va='center', fontsize=10)
        
        plt.title(f'Distribuição de {column} em {tipo}', fontsize=14)
        plt.xlabel('Contagem', fontsize=12)
        plt.ylabel(column, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        plt.savefig(f"img/categoricos_{column}.png")

        plt.show()
    
    # # Distribuição de variáveis numéricas
    # for column in df.select_dtypes(include=['int64', 'float64']).columns:
    #     plt.figure(figsize=(12, 6))
        
    #     # Histograma com linha de densidade
    #     sns.histplot(df[column], kde=True, color='dodgerblue', bins=30, edgecolor='black', alpha=0.7)
        
    #     # Calcular estatísticas
    #     mean = df[column].mean()
    #     median = df[column].median()
        
    #     # Adicionar linhas verticais para média e mediana
    #     plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Média: {mean:.2f}')
    #     plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Mediana: {median:.2f}')
        
    #     # Anotações de estatísticas
    #     plt.text(mean, plt.ylim()[1] * 0.9, f'Média: {mean:.2f}', color='red', fontsize=12, ha='center', fontweight='bold')
    #     plt.text(median, plt.ylim()[1] * 0.8, f'Mediana: {median:.2f}', color='green', fontsize=12, ha='center', fontweight='bold')
        
    #     plt.title(f'Distribuição de {column} em {tipo}', fontsize=16, fontweight='bold')
    #     plt.xlabel(column, fontsize=14)
    #     plt.ylabel('Frequência', fontsize=14)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.legend(fontsize=12)
    #     plt.show()




def analise_exploratoria_multivalorada(df, coluna_splitada, id_col, df_co, df_original, entidade='entidade', max_label_length=30, min_frequency=1, palavra_chave='AA',  min_coocorrencia=0):
    sns.set_theme(style="whitegrid")

    # Verificar se a coluna existe
    if coluna_splitada not in df.columns:
        print(f"A coluna {coluna_splitada} não existe no DataFrame.")
        return
    
    df_replace = df_original[coluna_splitada].fillna(' ') 
    testev = df_replace == palavra_chave
    qtd_palavra_chave = df_original[testev]

    # Remover duplicatas para garantir contagem correta
    categorias_unicas = df.drop_duplicates(subset=[id_col, coluna_splitada])
    
    # Calcular a frequência de cada categoria
    frequencias = categorias_unicas[coluna_splitada].value_counts().sort_values(ascending=False)
    
    # Filtrar categorias com frequência abaixo do mínimo especificado
    frequencias = frequencias[frequencias >= min_frequency]
    
    # Verificar se há dados para plotar
    if frequencias.empty:
        print("Nenhuma categoria atende ao critério de frequência mínima.")
        return
    
    # Truncar rótulos se max_label_length for especificado
    if max_label_length is not None:
        frequencias.index = [label[:max_label_length] + '...' if len(label) > max_label_length else label for label in frequencias.index]
    
    # Calcular percentual em relação ao total de linhas do DataFrame
    total_linhas = df.shape[0] + qtd_palavra_chave.shape[0]
    percentuais = (frequencias / total_linhas) * 100
    
    # Visualizar a distribuição de categorias
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=frequencias.index, y=frequencias.values, palette=colors[:len(frequencias)])
    
    # Título e subtítulo
    plt.title(f'Distribuição de {coluna_splitada} por {entidade}', fontsize=18, fontweight='bold')
    
    # Rótulos dos eixos
    plt.xlabel('Categoria', fontsize=12)
    plt.ylabel(f'Número de {entidade}', fontsize=12)
    
    # Ajustar rotação e tamanho dos rótulos do eixo x
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=11)
    
    # Anotações para destacar categorias importantes
    for i, (v, p) in enumerate(zip(frequencias.values, percentuais.values)):
        ax.text(i, v + 0.5, f'{v} \n ({p:.2f}%)', color='black', ha='center', fontsize=8)
    
    plt.yticks(fontsize=11)
    plt.tight_layout()  # Ajustar layout para evitar corte de rótulos
    
    plt.savefig(f"img/multi_valor_{coluna_splitada}.png")

    plt.show()

    cmap_custom = LinearSegmentedColormap.from_list('custom_palette', colors_map, N=256)
    
    coocorrencias = df_co.set_index(coluna_splitada)
    
    # Ordenar as linhas e colunas alfabeticamente
    coocorrencias = coocorrencias.sort_index().sort_index(axis=1)
    
    frequencias_totais = df[coluna_splitada].value_counts()

    # Filtrar linhas e colunas com soma total de co-ocorrências abaixo do mínimo especificado
    coocorrencias = coocorrencias.loc[coocorrencias.sum(axis=1) > min_coocorrencia, coocorrencias.sum(axis=0) > min_coocorrencia]
    
    # Verificar se a matriz de co-ocorrências está vazia após o filtro
    if coocorrencias.empty:
        print("Nenhuma co-ocorrência atende ao critério de frequência mínima.")
        return
    
    # Adicionar auto-coocorrências na diagonal
    for index in coocorrencias.index:
        if index in frequencias_totais.index:
            coocorrencias.at[index, f"{coluna_splitada}_{index}"] = frequencias_totais[index]
    
    # Criar um heatmap para visualizar as co-ocorrências
    plt.figure(figsize=(22, 18))
    ax = sns.heatmap(coocorrencias, cmap=cmap_custom, fmt='s', cbar_kws={'label': 'Número de Co-ocorrências'},
                     annot=coocorrencias.applymap(lambda v: f'{v}' if v != 0 else ''), annot_kws={"size": 12},
                     linewidths=0.5)  # Linhas de grade mais proeminentes
    
    # Adicionar os totais nos rótulos do eixo
    x_labels = [f"{col.replace(f'{coluna_splitada}_', '')} ({frequencias_totais[col.replace(f'{coluna_splitada}_', '')]})" for col in coocorrencias.columns]
    y_labels = [f"{row} ({frequencias_totais[row]})" if row in frequencias_totais.index else row for row in coocorrencias.index]
    
    # Truncar rótulos se max_label_length for especificado
    if max_label_length is not None:
        x_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label for label in x_labels]
        y_labels = [label[:max_label_length] + '...' if isinstance(label, str) and len(label) > max_label_length else label for label in y_labels]

    # Título e rótulos dos eixos
    plt.title(f'Mapa de Calor das Co-ocorrências de {coluna_splitada}', fontsize=20, fontweight='bold')
    plt.xlabel(coluna_splitada, fontsize=13)
    plt.ylabel(coluna_splitada, fontsize=13)
    
    # Ajustar rótulos do eixo x e y
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right', fontsize=13)
    ax.set_yticklabels(y_labels, fontsize=13)
    
    plt.tight_layout()  # Ajustar layout para evitar corte de rótulos

    plt.savefig(f"img/co_ocorrencia_{coluna_splitada}.png")

    plt.show()



def agrupar_e_somar(df_one_hot, categoria, lista_de_tuplas, id=None):
    # Remover a coluna original não necessária, se existir, apenas uma vez
    if f"{categoria}_ORIGINAL" in df_one_hot.columns:
        df_one_hot = df_one_hot.drop(columns=[f"{categoria}_ORIGINAL"])
    
    # Criar uma cópia do DataFrame original para não modificar diretamente a entrada
    df_resultado = df_one_hot.copy()

    df_resultado['count'] = 1

    # Iterar sobre cada tupla na lista fornecida
    for nome_nova_coluna, valores_a_substituir in lista_de_tuplas:
        # Substituir os valores na coluna de categoria usando replace dentro do loop
        df_resultado[categoria] = df_resultado[categoria].replace(valores_a_substituir, nome_nova_coluna)

        # Identificar colunas que precisam ser somadas
        colunas_a_somar = [
            col for col in df_resultado.columns
            if any(f"{categoria}_{valor}" in col for valor in valores_a_substituir)
        ]

        # Somar as colunas especificadas na tupla
        if colunas_a_somar:
            df_resultado[nome_nova_coluna] = df_resultado[colunas_a_somar].sum(axis=1)
            # Remover as colunas originais após a soma
            df_resultado = df_resultado.drop(columns=colunas_a_somar)

    # Agrupar por ID e Categoria e somar
    df_agrupado = df_resultado.groupby([id, categoria]).sum().reset_index()

    # Renomear as colunas agrupadas
    for nome_nova_coluna, _ in lista_de_tuplas:
        coluna_agrupada_nome = f"{categoria}_{nome_nova_coluna}_AGRUPADA"
        if nome_nova_coluna in df_agrupado.columns:
            df_agrupado = df_agrupado.rename(columns={nome_nova_coluna: coluna_agrupada_nome})

    return df_agrupado



def agrupar_e_somar_personalizado(df_one_hot, categoria, lista_de_tuplas):
    # Remover a coluna original não necessária, se existir, apenas uma vez
    if f"{categoria}_ORIGINAL" in df_one_hot.columns:
        df_one_hot = df_one_hot.drop(columns=[f"{categoria}_ORIGINAL"])
    
    # Criar uma cópia do DataFrame original para não modificar diretamente a entrada
    df_resultado = df_one_hot.copy()

    df_resultado['count'] = 1

    # Iterar sobre cada tupla na lista fornecida
    for nome_nova_coluna, valores_a_substituir in lista_de_tuplas:
        # Substituir os valores na coluna de categoria usando replace dentro do loop
        df_resultado[categoria] = df_resultado[categoria].replace(valores_a_substituir, nome_nova_coluna)

        # Identificar colunas que precisam ser somadas
        colunas_a_somar = [
            col for col in df_resultado.columns
            if any(f"{categoria}_{valor}" in col for valor in valores_a_substituir)
        ]

        # Somar as colunas especificadas na tupla
        if colunas_a_somar:
            df_resultado[nome_nova_coluna] = df_resultado[colunas_a_somar].sum(axis=1)
            # Remover as colunas originais após a soma
            df_resultado = df_resultado.drop(columns=colunas_a_somar)

    # Agrupar por ID e Categoria e somar
    df_agrupado = df_resultado.groupby([categoria]).sum().reset_index()

    # Renomear as colunas agrupadas
    for nome_nova_coluna, _ in lista_de_tuplas:
        coluna_agrupada_nome = f"{categoria}_{nome_nova_coluna}_AGRUPADA"
        if nome_nova_coluna in df_agrupado.columns:
            df_agrupado = df_agrupado.rename(columns={nome_nova_coluna: coluna_agrupada_nome})

    return df_agrupado





# Configuração do gráfico
def plot_grafico_menininos_meninas(total_meninos, total_meninas, percentuais_meninos, percentuais_meninas, categorias, quantidades_meninos, quantidades_meninas):
    x = np.arange(len(categorias) - 1)  # localização das labels, exclui 'Sexo' para barras separadas
    width = 0.25  # largura das barras

    fig, ax = plt.subplots(figsize=(14, 8))

# Barra única para a categoria 'Sexo'
    ax.bar('Sexo', percentuais_meninos[0], color='lightblue')
    ax.bar('Sexo', percentuais_meninas[0], bottom=percentuais_meninos[0], color='lightpink')

# Ajustar posição para as demais categorias
    x_pos = np.arange(1, len(categorias))


# Barras para meninos, meninas e total nas outras categorias
    bars1 = ax.bar(x_pos - width, percentuais_meninos[1:], width, color='royalblue')
    bars2 = ax.bar(x_pos, percentuais_meninas[1:], width, color='deeppink')


# Cálculo dos percentuais para garantir que somem 100% por categoria
    percentuais_bars3_meninos = [
    (quantidades_meninos[i] / (quantidades_meninos[i] + quantidades_meninas[i])) * 100 if (quantidades_meninos[i] + quantidades_meninas[i]) > 0 else 0
    for i in range(1, len(quantidades_meninos))
]

    percentuais_bars3_meninas = [
    (quantidades_meninas[i] / (quantidades_meninos[i] + quantidades_meninas[i])) * 100 if (quantidades_meninos[i] + quantidades_meninas[i]) > 0 else 0
    for i in range(1, len(quantidades_meninas))
]


    bars3_meninos = ax.bar(x_pos + width, percentuais_bars3_meninos, width , color='lightblue')
    bars3_meninas = ax.bar(x_pos + width, percentuais_bars3_meninas, width, bottom=percentuais_bars3_meninos, color='lightpink')


# Adicionar rótulos e título
    ax.set_xlabel('Categorias')
    ax.set_ylabel('Percentual (%)')
    ax.set_title('Comparação de Desfechos Neonatais entre Meninos e Meninas')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categorias[1:])  # Ajuste para corresponder ao tamanho dos dados
    ax.set_ylim(0, 100)
    ax.legend()

# Adicionar rótulos de texto nas barras com percentuais e valores


# Adicionar rótulos para a barra 'Sexo'
    ax.annotate(f'Meninos:\n {percentuais_meninos[0]:.2f}%\n({total_meninos})',
            xy=(0, percentuais_meninos[0] / 2),
            xytext=(0, 0),  # no offset
            textcoords="offset points",
            ha='center', va='center', color='black')


    ax.annotate(f'Meninas:\n {percentuais_meninas[0]:.2f}%\n({total_meninas})',
            xy=(0, percentuais_meninos[0] + percentuais_meninas[0] / 2),
            xytext=(0, 0),  # no offset
            textcoords="offset points",
            ha='center', va='center', color='black')


    def autolabel(bars, valores, percentuais):
        for bar, valor, percentual in zip(bars, valores, percentuais):
            height = bar.get_height()
        # Calcular o offset dinamicamente com base na altura da barra
            offset = height * 0.05  # Ajuste percentual da altura para o deslocamento
            ax.annotate(f'{percentual:.2f}%\n({valor})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # Offset vertical dinâmico
                    textcoords="offset points",
                    ha='center', va='bottom')
    
        
 # Função para adicionar rótulos nas barras empilhadas
    def autolabel_empilhado(bars, valores, percentuais, bottoms):
        for bar, valor, percentual, bottom in zip(bars, valores, percentuais, bottoms):
            height = bar.get_height() + bottom - 10
            ax.annotate(f'{percentual:.2f}%\n({valor})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset vertical
                    textcoords="offset points",
                    ha='center', va='bottom')       

# Adicionar rótulos para as outras categorias com ajuste dinâmico
    autolabel(bars1, quantidades_meninos[1:], percentuais_meninos[1:])
    autolabel(bars2, quantidades_meninas[1:], percentuais_meninas[1:])
    autolabel_empilhado(bars3_meninos, quantidades_meninos[1:], percentuais_bars3_meninos, [0] * len(percentuais_bars3_meninos))
    autolabel_empilhado(bars3_meninas, quantidades_meninas[1:], percentuais_bars3_meninas, percentuais_bars3_meninos)



    plt.tight_layout()
    plt.savefig(f"img/analise_meninos_meninas.png")

    plt.show()


def func_print_meninos_meninas(total_rn, total_meninos, total_meninas, apgar_meninos, apgar_meninas, ucin_meninos, ucin_meninas, uti_meninos, uti_meninas, obito_meninos, obito_meninas, percent_apgar_meninos, percent_apgar_meninas, percent_ucin_meninos, percent_ucin_meninas, percent_uti_meninos, percent_uti_meninas, percent_obito_meninos, percent_obito_meninas):
    # Texto narrativo detalhado
    resultado_texto = f"""
### Análise Detalhada dos Desfechos Neonatais por Sexo
**População Geral:**
A análise abrange um total de {total_rn} recém-nascidos, dos quais {total_meninos} ({(total_meninos / total_rn) * 100:.1f}%) são meninos e {total_meninas} ({(total_meninas / total_rn) * 100:.1f}%) são meninas.
**APGAR < 7:**
- Meninos {total_meninos} ({(total_meninos / total_rn) * 100:.1f}%):
  - {apgar_meninos} ({percent_apgar_meninos:.2f}%) tiveram APGAR < 7. Isso representa {(apgar_meninos / (apgar_meninos + apgar_meninas) * 100):.1f}% de todos os casos de APGAR < 7.
- Meninas {total_meninas} ({(total_meninas / total_rn) * 100:.1f}%):
  - {apgar_meninas} ({percent_apgar_meninas:.2f}%) tiveram APGAR < 7. Isso representa {(apgar_meninas / (apgar_meninos + apgar_meninas) * 100):.1f}% de todos os casos de APGAR < 7.
**Cuidados Perinatais (UCIN):**
- Meninos {total_meninos} ({(total_meninos / total_rn) * 100:.1f}%):
  - {ucin_meninos} ({percent_ucin_meninos:.2f}%) necessitaram de cuidados na UCIN. Isso representa {(ucin_meninos / (ucin_meninos + ucin_meninas) * 100):.1f}% do total de casos que requereram UCIN.
- Meninas {total_meninas} ({(total_meninas / total_rn) * 100:.1f}%):
  - {ucin_meninas} ({percent_ucin_meninas:.2f}%) necessitaram de cuidados na UCIN. Isso representa {(ucin_meninas / (ucin_meninos + ucin_meninas) * 100):.1f}% do total de casos que requereram UCIN.
**Cuidados Perinatais (UTI):**
- Meninos {total_meninos} ({(total_meninos / total_rn) * 100:.1f}%):
  - {uti_meninos} ({percent_uti_meninos:.2f}%) precisaram de cuidados na UTI. Isso representa {(uti_meninos / (uti_meninos + uti_meninas) * 100):.1f}% de todos os casos que necessitaram de UTI.
- Meninas {total_meninas} ({(total_meninas / total_rn) * 100:.1f}%):
  - {uti_meninas} ({percent_uti_meninas:.2f}%) precisaram de cuidados na UTI. Isso representa {(uti_meninas / (uti_meninos + uti_meninas) * 100):.1f}% de todos os casos que necessitaram de UTI.
**Óbitos:**
- Meninos {total_meninos} ({(total_meninos / total_rn) * 100:.1f}%):
  - {obito_meninos} ({percent_obito_meninos:.2f}%) foram óbitos. Isso representa {(obito_meninos / (obito_meninos + obito_meninas) * 100):.1f}% de todos os casos de óbito.
- Meninas {total_meninas} ({(total_meninas / total_rn) * 100:.1f}%):
  - {obito_meninas} ({percent_obito_meninas:.2f}%) foram óbitos. Isso representa {(obito_meninas / (obito_meninos + obito_meninas) * 100):.1f}% de todos os casos de óbito.
"""

# Imprimir o texto
    print(resultado_texto)


def calcular_percentuais_intercorrencias(df, rupreme_df, tpp_df, pe_df, outras_intercorrencias_df, sem_intercorrencias_df):
    # Filtrar dados para cada intercorrência
    total_rn = df.shape[0]

    # Função para calcular quantidades e percentuais
    def calcular_quantidades_percentuais(sub_df, total_rn):
        apgar = sub_df[sub_df['APGAR_1_MIN'] < 7].shape[0]
        ucin = sub_df[sub_df['CUIDADOS_PERINATAIS_RN'] == 'UCIN'].shape[0]
        uti = sub_df[sub_df['CUIDADOS_PERINATAIS_RN'] == 'UTIN'].shape[0]
        obito = sub_df[sub_df['OBITO?'] == 'SIM'].shape[0]
        total = sub_df.shape[0]
        total_populacao = total_rn.shape[0]

        percent_total = (total / total_populacao * 100) if total > 0 else 0

        percent_apgar = (apgar / total * 100) if total > 0 else 0
        percent_ucin = (ucin / total * 100) if total > 0 else 0
        percent_uti = (uti / total * 100) if total > 0 else 0
        percent_obito = (obito / total * 100) if total > 0 else 0

        return [percent_total, percent_apgar, percent_ucin, percent_uti, percent_obito]

    # Calcular percentuais para cada categoria

    percentuais_rupreme = calcular_quantidades_percentuais(rupreme_df, df_rn)
    percentuais_tpp = calcular_quantidades_percentuais(tpp_df, df_rn)
    percentuais_pe = calcular_quantidades_percentuais(pe_df, df_rn)
    percentuais_outras = calcular_quantidades_percentuais(outras_intercorrencias_df, df_rn)
    percentuais_sem = calcular_quantidades_percentuais(sem_intercorrencias_df, df_rn)

    return {
        'RUPREME': percentuais_rupreme,
        'TPP': percentuais_tpp,
        'PE': percentuais_pe,
        'Outras': percentuais_outras,
        'Sem Intercorrências': percentuais_sem
    }



def print_intercorrencias_ucin_uti_obitos_apgar(total_rn, total_rupreme, total_tpp, total_pe, total_outras, total_sem, percentuais_intercorrencias):
    # Exibir análise detalhada das intercorrências
    print("Análise Detalhada das Intercorrências\n")

# População Geral
    print(f"População Geral:")
    print(f"A análise abrange um total de {total_rn} recém-nascidos.\n")



# Função para calcular e imprimir detalhes de cada categoria
    def imprimir_detalhes(categoria, total, percentuais, total_rn):
        percent_total_pop = (total / total_rn) * 100
        apgar = int(percentuais[1] / 100 * total)
        percent_apgar_total_pop = (apgar / total_rn) * 100
        ucin = int(percentuais[2] / 100 * total)
        percent_ucin_total_pop = (ucin / total_rn) * 100
        uti = int(percentuais[3] / 100 * total)
        percent_uti_total_pop = (uti / total_rn) * 100
        obito = int(percentuais[4] / 100 * total)
        percent_obito_total_pop = (obito / total_rn) * 100

        print(f"{categoria} {total} ({percent_total_pop:.1f}%):")
        print(f"- APGAR < 7: {apgar} ({percentuais[1]:.2f}%) tiveram APGAR < 7. E isso representa {percent_apgar_total_pop:.1f}% de todos que tiveram APGAR < 7")
        print(f"- Necessitaram de UCIN: {ucin} ({percentuais[2]:.2f}%) necessitaram de UCIN. E isso representa {percent_ucin_total_pop:.1f}% de todos que necessitaram de UCIN")
        print(f"- Necessitaram de UTI: {uti} ({percentuais[3]:.2f}%) necessitaram de UTI. E isso representa {percent_uti_total_pop:.1f}% de todos que necessitaram de UTI")
        print(f"- Óbitos: {obito} ({percentuais[4]:.2f}%) foram óbitos. E isso representa {percent_obito_total_pop:.1f}% de todos os óbitos\n")

# Detalhes para cada intercorrência
    imprimir_detalhes("RUPREME", total_rupreme, percentuais_intercorrencias['RUPREME'], total_rn)
    imprimir_detalhes("TPP", total_tpp, percentuais_intercorrencias['TPP'], total_rn)
    imprimir_detalhes("PE", total_pe, percentuais_intercorrencias['PE'], total_rn)
    imprimir_detalhes("Outras Intercorrências", total_outras, percentuais_intercorrencias['Outras'], total_rn)
    imprimir_detalhes("Sem Intercorrências", total_sem, percentuais_intercorrencias['Sem Intercorrências'], total_rn)

# Cálculo dos totais gerais
    total_apgar = sum([
    int(percentuais_intercorrencias['RUPREME'][1] / 100 * total_rupreme),
    int(percentuais_intercorrencias['TPP'][1] / 100 * total_tpp),
    int(percentuais_intercorrencias['PE'][1] / 100 * total_pe),
    int(percentuais_intercorrencias['Outras'][1] / 100 * total_outras),
    int(percentuais_intercorrencias['Sem Intercorrências'][1] / 100 * total_sem)
])

    total_ucin = sum([
    int(percentuais_intercorrencias['RUPREME'][2] / 100 * total_rupreme),
    int(percentuais_intercorrencias['TPP'][2] / 100 * total_tpp),
    int(percentuais_intercorrencias['PE'][2] / 100 * total_pe),
    int(percentuais_intercorrencias['Outras'][2] / 100 * total_outras),
    int(percentuais_intercorrencias['Sem Intercorrências'][2] / 100 * total_sem)
])

    total_uti = sum([
    int(percentuais_intercorrencias['RUPREME'][3] / 100 * total_rupreme),
    int(percentuais_intercorrencias['TPP'][3] / 100 * total_tpp),
    int(percentuais_intercorrencias['PE'][3] / 100 * total_pe),
    int(percentuais_intercorrencias['Outras'][3] / 100 * total_outras),
    int(percentuais_intercorrencias['Sem Intercorrências'][3] / 100 * total_sem)
])

    total_obitos = sum([
    int(percentuais_intercorrencias['RUPREME'][4] / 100 * total_rupreme),
    int(percentuais_intercorrencias['TPP'][4] / 100 * total_tpp),
    int(percentuais_intercorrencias['PE'][4] / 100 * total_pe),
    int(percentuais_intercorrencias['Outras'][4] / 100 * total_outras),
    int(percentuais_intercorrencias['Sem Intercorrências'][4] / 100 * total_sem)
])

# Exibir totais gerais
    print("Totais Gerais na População:")
    print(f"- APGAR < 7: {total_apgar}")
    print(f"- Necessitaram de UCIN: {total_ucin}")
    print(f"- Necessitaram de UTI: {total_uti}")
    print(f"- Óbitos: {total_obitos}\n")





def plotar_heatmap_categorico(detalhes_gestacoes_df, colunas_multivaloradas, titulo_base='Comparação da Característica'):
    # Selecionar colunas categóricas
    colunas_categoricas = detalhes_gestacoes_df.select_dtypes(exclude=['number'])

    # Criar um gradiente de cores do azul forte ao azul claro
    cmap = cm.get_cmap('Blues', len(detalhes_gestacoes_df))
    colors = cmap(np.linspace(0, 1, len(detalhes_gestacoes_df)))

    for coluna in colunas_categoricas.columns:
        # Criar figura com layout controlado
        fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
        
        # Verificar se a coluna é multivalorada
        if coluna in colunas_multivaloradas:
            # Expandir valores multivalorados
            detalhes_gestacoes_df_expanded = detalhes_gestacoes_df[coluna].str.get_dummies(sep=', ')
            # Transpor e somar para obter a soma total por categoria
            dados_categoricos = detalhes_gestacoes_df_expanded.groupby(detalhes_gestacoes_df.index).sum().T
        else:
            # Para colunas não multivaloradas, usar crosstab normal
            dados_categoricos = pd.crosstab(detalhes_gestacoes_df[coluna], detalhes_gestacoes_df.index)

        # Verificar se a tabela de frequência não está vazia
        if not dados_categoricos.empty:
            # Criar máscara para valores zero
            mask = dados_categoricos == 0

            # Plotar heatmap com ajuste de tamanho dos quadrados
            ax = sns.heatmap(dados_categoricos, annot=True, cmap='coolwarm', cbar=False, fmt='d', mask=mask, linewidths=.5, square=True)
            ax.set_title(f'{titulo_base}: {coluna} entre as 10 Gestações Mais Prematuras', fontsize=16)
            ax.set_xlabel('Gestação')
            ax.set_ylabel('Categoria')
            
            # Ajustar rótulos do eixo x para começar em 1
            ax.set_xticklabels([f'{i+1}' for i in range(len(dados_categoricos.columns))])

            # Ajustar rótulos do eixo y para ficarem na vertical com os nomes corretos
            ax.set_yticklabels(dados_categoricos.index, rotation=0)

            # Colorir os rótulos do eixo x com fundo para contraste
            for xtick, color in zip(ax.get_xticklabels(), colors):
                xtick.set_color(color)
                xtick.set_backgroundcolor('lightgrey')  # Adicionar fundo claro para contraste

            # Criar legenda associando cores a gestações e idades gestacionais
            handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(dados_categoricos.columns))]
            labels = [f'Gestante {i+1}: {detalhes_gestacoes_df["IG"].iloc[i]}' for i in range(len(dados_categoricos.columns))]
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title='Idade Gestacional', fontsize='small')

            # Adicionar gráfico de barras horizontal para total de quantidades por categoria
            soma_categorias = dados_categoricos.sum(axis=1)
            max_value = soma_categorias.max()
            for i, (value, ytick) in enumerate(zip(soma_categorias, ax.get_yticks())):
                # Alinhar as barras com as categorias do eixo y
                ax.barh(ytick, value, color='darkblue', edgecolor='none', height=0.6, left=len(dados_categoricos.columns) + 0.5)
                ax.text(len(dados_categoricos.columns) + value + 0.5 + 0.5, ytick, f'{value}', va='center', ha='left', color='black')

            # Ajustar o limite do eixo x para não estender desnecessariamente
            ax.set_xlim(0, len(dados_categoricos.columns) + max_value + 1)

            # Usar constrained_layout para melhor dimensionamento
            plt.savefig(f"img/heat_map_categorico.png")

            plt.show()

        else:
            print(f"A tabela de frequência para a coluna '{coluna}' está vazia e não pode ser plotada.")





def plotar_comparacao_caracteristicas_numericas(df_pre_termo, df_bom_desfecho, detalhes_gestacoes_df, colunas_numericas):
    # Calcular métricas para o conjunto completo
    media_completa = df_pre_termo[colunas_numericas.columns].mean()
    media_df_bom_desfecho = df_bom_desfecho[colunas_numericas.columns].mean()

    # Calcular métricas para gestações sem bom desfecho
    df_sem_bom_desfecho = df_pre_termo[df_pre_termo['OBITO?'] == 'SIM']
    media_sem_bom_desfecho = df_sem_bom_desfecho[colunas_numericas.columns].mean()

    # Criar um gradiente de cores do azul forte ao azul claro
    cmap = cm.get_cmap('Blues', len(detalhes_gestacoes_df))

    # Plotar características numéricas separadamente com gradiente de cores
    for coluna in colunas_numericas.columns:
        plt.figure(figsize=(12, 6))
        colors = cmap(np.linspace(0, 1, len(detalhes_gestacoes_df)))
        ax = detalhes_gestacoes_df[coluna].plot(kind='bar', color=colors)
        
        # Adicionar rótulos de valor em cima de cada barra
        for i, value in enumerate(detalhes_gestacoes_df[coluna]):
            if np.isfinite(value):  # Verificar se o valor é finito
                ax.text(i, value + (value * 0.01), f'{value:.2f}', ha='center', va='bottom')

        # Adicionar linhas de média e mediana do conjunto completo
        plt.axhline(y=media_completa[coluna], color='green', linestyle='--', label='Média Completa')
        plt.axhline(y=media_sem_bom_desfecho[coluna], color='red', linestyle='--', label='Média Sem Bom Desfecho')
        plt.axhline(y=media_df_bom_desfecho[coluna], color='grey', linestyle='--', label='Média Bom Desfecho')

        # Configurar título e rótulos
        plt.title(f'Comparação de {coluna} entre as 10 Gestantes com Menor Idade Gestacional que tiveram um bom desfecho')
        plt.xlabel('Gestante')
        plt.ylabel('Valor')
        plt.xticks(ticks=np.arange(len(detalhes_gestacoes_df)), labels=[f'{i+1}' for i in range(len(detalhes_gestacoes_df))], rotation=0)

        # Criar legenda associando cores a gestações e idades gestacionais
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(detalhes_gestacoes_df))]
        labels = [f'Gestante {i+1}: {detalhes_gestacoes_df["IG"].iloc[i]} semanas' for i in range(len(detalhes_gestacoes_df))]
        plt.legend(handles + [plt.Line2D([0], [0], color='green', linestyle='--'), 
                              plt.Line2D([0], [0], color='grey', linestyle='--'),                       
                              plt.Line2D([0], [0], color='red', linestyle='--')
                             ],
                   labels + [f'Média Completa {media_completa[coluna]:.2f}', f'Média Bom Desfecho  {media_df_bom_desfecho[coluna]:.2f}', f'Média Sem Bom Desfecho  {media_sem_bom_desfecho[coluna]:.2f}' ],
                   bbox_to_anchor=(1.05, 1), loc='upper left', title='Idade Gestacional')

        plt.tight_layout()
        
        plt.savefig(f"img/comparacao_caracteristicas_numericas.png")

        plt.show()



def plotar_matriz_co_ocorrencia(analise_estatistica, prefixo_intercorrencias='INTERCORRENCIAS_', prefixo_morbidades='MORBIDADE_', max_label_length=20):
    # Selecionar colunas de intercorrências e morbidades
    intercorrencias_cols = [col for col in analise_estatistica.columns if col.startswith(prefixo_intercorrencias)]
    morbidades_cols = [col for col in analise_estatistica.columns if col.startswith(prefixo_morbidades) and not col.startswith('MORBIDADE_FETO')]

    # Criar DataFrame para co-ocorrências
    co_ocorrencia = pd.DataFrame(index=intercorrencias_cols, columns=morbidades_cols)

    for inter in intercorrencias_cols:
        for morb in morbidades_cols:
            # Calculando a co-ocorrência
            co_ocorrencia.loc[inter, morb] = (analise_estatistica[inter] & analise_estatistica[morb]).sum()

    # Convertendo para valores numéricos
    co_ocorrencia = co_ocorrencia.astype(int)

    # Função para truncar rótulos
    def truncate_label(label, max_length=max_label_length):
        label = label.replace(prefixo_intercorrencias, '').replace(prefixo_morbidades, '')
        return label[:max_length] + '...' if len(label) > max_length else label

    # Aplicar truncamento de rótulos
    co_ocorrencia.index = [truncate_label(label) for label in co_ocorrencia.index]
    co_ocorrencia.columns = [truncate_label(label) for label in co_ocorrencia.columns]

    # Ordenar índices e colunas
    co_ocorrencia = co_ocorrencia.loc[sorted(co_ocorrencia.index), sorted(co_ocorrencia.columns)]

    # Mascarar valores zero
    mask = co_ocorrencia == 0

    # Visualizar a matriz de co-ocorrência com um mapa de calor
    plt.figure(figsize=(30, 16))
    sns.heatmap(co_ocorrencia, annot=True, fmt="d", cmap="YlGnBu", mask=mask, cbar_kws={'label': 'Frequência de Co-ocorrência'},
                linewidths=1, square=True, annot_kws={"size": 18})
    plt.title('Matriz de Co-ocorrência entre Intercorrências e Morbidades')
    plt.xlabel('Morbidades')
    plt.ylabel('Intercorrências')
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.tight_layout()
    
    plt.savefig(f"img/matriz_co_ocorrencia_.png")

    plt.show()



def plotar_heatmap_sociodemografico(analise_estatistica, metadados_colunas, max_chars=30):
    # Selecionar colunas sociodemográficas com base nos metadados
    colunas_sociodemograficas = [col for col, meta in metadados_colunas.items() if meta['relevancia'] == 'Sociodemografico']
    colunas_selecionadas = [
        col for col in analise_estatistica.columns
        if any(col == soc_col or col.startswith(soc_col + '_') for soc_col in colunas_sociodemograficas)
    ]
    df_sociodemografico = analise_estatistica[colunas_selecionadas]

    # Calcular a matriz de correlação
    correlation_matrix = df_sociodemografico.corr()

    # Função para formatar rótulos dos eixos
    def formatar_rotulos(labels, max_chars=max_chars):
        return [label.replace("CATEGORIAS_", "").replace("CATEGORIA_", "")[:max_chars] for label in labels]

    # Formatar rótulos dos eixos
    x_labels = formatar_rotulos(correlation_matrix.columns)
    y_labels = formatar_rotulos(correlation_matrix.index)

    # Criar uma máscara para a parte superior da matriz de correlação
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Definir a paleta de cores personalizada
    cmap_custom = LinearSegmentedColormap.from_list('custom_palette', colors_map_heat, N=256)

    # Criar o heatmap com a máscara
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap_custom, fmt=".2f",
                vmin=-1, vmax=1, center=0,  # Centralizar o mapa de cores em zero
                xticklabels=x_labels, yticklabels=y_labels, square=True,
                cbar_kws={'shrink': .8}, linewidths=0.5)

    plt.title("Matriz de Correlação das Variáveis Sociodemográficas", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(f"img/heatmap_sociodemografico.png")

    plt.show()




def calcular_etnia(df, coluna_etnia):

    if coluna_etnia not in df.columns:
        print(f"A coluna {coluna_etnia} não está disponível no DataFrame fornecido.")
        return {}

    # Inicializando um dicionário para armazenar os resultados
    etnias = ['BRANCA', 'PARDA', 'NEGRA']
    resultados = {}

    # Calculando o número e a porcentagem de registros para cada etnia
    total_registros = len(df)
    for etnia in etnias:
        numero_registros = df[df[coluna_etnia].str.upper() == etnia].shape[0]
        porcentagem_registros = numero_registros / total_registros * 100
        
        # Armazenando os resultados no dicionário
        resultados[etnia] = {
            'numero': numero_registros,
            'porcentagem': porcentagem_registros
        }

    # Exibindo os resultados
    for etnia, valores in resultados.items():
        print(f"Etnia: {etnia}")
        print(f"Número de registros: {valores['numero']} ({valores['porcentagem']:.2f}%)\n")

    return resultados



def calcular_destinos_rn(df, coluna_destino):

    if coluna_destino not in df.columns:
        print(f"A coluna {coluna_destino} não está disponível no DataFrame fornecido.")
        return {}

    # Inicializando um dicionário para armazenar os resultados
    destinos = ['ALOJAMENTO CONJUNTO', 'UCIN', 'UTIN', 'NAO SE APLICA']
    resultados = {}

    # Calculando o número e a porcentagem de RN em cada destino
    total_rn = len(df)
    for destino in destinos:
        numero_rn = df[df[coluna_destino].str.upper() == destino].shape[0]
        porcentagem_rn = numero_rn / total_rn * 100
        
        # Armazenando os resultados no dicionário
        resultados[destino] = {
            'numero': numero_rn,
            'porcentagem': porcentagem_rn
        }

    # Exibindo os resultados
    for destino, valores in resultados.items():
        print(f"Destino: {destino}")
        print(f"Número de RN: {valores['numero']} ({valores['porcentagem']:.2f}%)\n")

    return resultados



def calcular_ausencia_caracteristicas(df, colunas_interesse, entidade):
    resultados = {}

    # Iterando sobre as colunas de interesse
    for coluna in colunas_interesse:
        if coluna in df.columns:
            # Considerando que a ausência de características é indicada por NaN ou por valores específicos
            sem_caracteristica = (
                df[coluna].isna() |
                (df[coluna] == '') |
                (df[coluna].str.upper() == 'NAO CONSTA') |
                (df[coluna].str.upper() == 'NAO SE APLICA') |
                (df[coluna].str.upper() == 'NEGA') |
                (df[coluna].str.upper() == 'SEM INTERCORRENCIAS')
            )
            
            numero_sem_caracteristica = df[sem_caracteristica].shape[0]
            porcentagem_sem_caracteristica = numero_sem_caracteristica / len(df) * 100
            
            # Armazenando os resultados no dicionário
            resultados[coluna] = {
                'numero': numero_sem_caracteristica,
                'porcentagem': porcentagem_sem_caracteristica
            }
        else:
            # Se a coluna não existir no DataFrame, registrar como não disponível
            resultados[coluna] = {
                'numero': 'N/A',
                'porcentagem': 'N/A'
            }

    # Exibindo os resultados
    for coluna, valores in resultados.items():
        if valores['numero'] != 'N/A':
            print(f"Coluna: {coluna}")
            print(f"Número de {entidade} sem {coluna.lower()}: {valores['numero']} ({valores['porcentagem']:.2f}%)\n")
        else:
            ""

    return resultados



def plot_kessner_distribution(df_gestantes):
    contingencia = pd.crosstab(df_gestantes['CONVENIO'], df_gestantes['CATEGORIA_KESSNER'], normalize='index')
    contingencia_counts = pd.crosstab(df_gestantes['CONVENIO'], df_gestantes['CATEGORIA_KESSNER'])

# Configurações do gráfico
    plt.figure(figsize=(12, 8))

# Definir uma paleta de cores personalizada
    cores = {
    'ADEQUADO': '#8c9a5b',  # Verde
    'INTERMEDIÁRIO': '#c5a880',  # Laranja
    'INADEQUADO': '#FF6666'  # Vermelho
}

    ax = contingencia.plot(kind='bar', stacked=True, color=[cores.get(col, '#333333') for col in contingencia.columns], ax=plt.gca())

# Título e rótulos dos eixos
    plt.title('Distribuição da Categoria Kessner por Tipo de Convênio', fontsize=16, fontweight='bold')
    plt.xlabel('Tipo de Convênio', fontsize=12)
    plt.ylabel('Proporção de Pacientes', fontsize=12)

# Rotação dos rótulos do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')

# Adicionar rótulos de dados nas barras com percentuais e quantidades
    for container, counts in zip(ax.containers, contingencia_counts.values.T):
        labels = [f'{count} ({v.get_height() * 100:.1f}%)' if v.get_height() > 0 else '' for v, count in zip(container, counts)]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=10)

# Melhorar a disposição da legenda
    plt.legend(title='Categoria Kessner', fontsize=10, loc='upper right')

# Ajustar o layout para evitar sobreposição
    plt.tight_layout()

# Exibir o gráfico
    plt.savefig(f"img/kessner_distribuition.png")

    plt.show()