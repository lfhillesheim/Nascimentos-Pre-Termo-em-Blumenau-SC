# Análise de Partos Prematuros

Bem-vindo ao repositório do projeto de análise de partos prematuros realizado no Hospital Santo Antônio (HSAN), Blumenau-SC. Este repositório contém todo o material necessário para replicar o estudo, incluindo dados anonimizados, scripts de análise e documentação detalhada.

## Visão Geral do Projeto

O estudo foi delineado como descritivo, transversal e retrospectivo, com o objetivo de analisar fatores sociodemográficos, clínicos e neonatais associados a partos prematuros. Utilizamos dados de prontuários médicos para identificar padrões e associações significativas.

## Estrutura do Repositório

- **`data/`**: Contém arquivos CSV com dados anonimizados utilizados no estudo.
- **`notebooks/`**: Inclui notebooks Jupyter com análises passo a passo.
- **`scripts/`**: Scripts Python para processamento e análise de dados.
- **`results/`**: Resultados das análises, incluindo gráficos e tabelas.
- **`README.md`**: Este documento, com informações detalhadas sobre o projeto e instruções de uso.

## Requisitos do Sistema

Para executar as análises, você precisará de:

- Python 3.8 ou superior
- Jupyter Notebook
- Bibliotecas Python: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels

## Instalação

1. **Clone o repositório:**

        git clone https://github.com/lfhillesheim/Nascimentos-Pre-Termo-em-Blumenau-SC Analise-Parto-Prematuro
2. **Crie um ambiente virtual (opcional, mas recomendado):**
3. 
        python -m venv venv
        # Para Linux/macOS
        venv/bin/activate 
        # Para Windows
        venv\Scripts\activate 

4. **Instale as dependências:**

        pip install -r requirements.txt

## Uso

### Executando Análises

1. **Abra o Jupyter Notebook:**

        jupyter notebook
2. **Navegue até o diretório `notebooks/` e abra o arquivo desejado.** Siga as instruções no notebook para executar as análises passo a passo.

### Estrutura dos Notebooks

Os notebooks estão organizados para guiar o usuário através de diferentes etapas do estudo:

- **Importação e Limpeza de Dados:** Carregamento e preparação dos dados para análise.
- **Análise Descritiva:** Exploração inicial dos dados com estatísticas descritivas.
- **Análise de Correlações:** Identificação de padrões e associações entre variáveis.
- **Visualização de Dados:** Criação de gráficos e mapas de calor.
- **Interpretação dos Resultados:** Discussão sobre as descobertas e suas implicações.

## Contribuição

Contribuições são bem-vindas! Se você deseja propor melhorias ou correções, siga estas etapas:

1. Faça um fork do projeto.
2. Crie uma nova branch para suas alterações (`git checkout -b feature/nova-funcionalidade`).
3. Commit suas alterações (`git commit -m 'Adiciona nova funcionalidade'`).
4. Faça o push para a branch (`git push origin feature/nova-funcionalidade`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

Para dúvidas ou mais informações sobre o projeto, entre em contato com lucas.hillesheim@gmail.com ou abra uma issue no GitHub.
