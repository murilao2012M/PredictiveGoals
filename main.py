import math
import random
import numpy as np
import pandas as pd
import os
import shap  # para interpretabilidade do modelo
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom, skellam
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from modules import gerar_ranking
from modules.gerar_relatorio import gerar_relatorio
from modules.simular_cenario import simular_cenario  # Modelo adicional

# =============================== FUNÇÕES BASE ===============================

def save_to_excel(data, filename="registros_jogos.xlsx"):
    try:
        existing_data = pd.read_excel(filename)
        new_data = pd.concat([existing_data, data], ignore_index=True).drop_duplicates()
    except FileNotFoundError:
        new_data = data
    new_data.to_excel(filename, index=False)
    print(f"✅ Dados atualizados em {filename}")

def mostrar_escudo(time):
    """
    Exibe o escudo do time, se disponível na pasta 'escudos'.
    O arquivo deve ter o formato 'NomeDoTime.png'.
    """
    caminho_escudo = f'escudos/{time}.png'
    if os.path.exists(caminho_escudo):
        img = Image.open(caminho_escudo)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Escudo do {time}", fontsize=14, weight='bold')
        plt.show()
    else:
        print(f"⚠️ Escudo do {time} não encontrado.")

# =============================== CLASSE TeamStats ===============================
class TeamStats:
    def __init__(self, name, golsMarcados, golsSofridos):
        # Validação básica dos dados: se as listas não forem vazias e contiverem números
        if not isinstance(golsMarcados, list) or not isinstance(golsSofridos, list):
            raise ValueError("golsMarcados e golsSofridos devem ser listas.")
        self.name = name
        self.golsMarcados = golsMarcados
        self.golsSofridos = golsSofridos

    def average_goals_scored(self):
        if not self.golsMarcados:
            print(f"⚠️ Não há dados de gols marcados para {self.name}.")
            return 0
        return sum(self.golsMarcados) / len(self.golsMarcados)

    def average_goals_conceded(self):
        if not self.golsSofridos:
            print(f"⚠️ Não há dados de gols sofridos para {self.name}.")
            return 0
        return sum(self.golsSofridos) / len(self.golsSofridos)

    def last_n_game_performance(self, n=5):
        n = min(n, len(self.golsMarcados), len(self.golsSofridos))
        if n == 0:
            print(f"⚠️ Dados insuficientes para calcular desempenho dos últimos jogos de {self.name}.")
            return 0, 0
        last_n_gols = self.golsMarcados[-n:]
        last_n_gols_sofridos = self.golsSofridos[-n:]
        avg_gols_marcados = sum(last_n_gols) / n
        avg_gols_sofridos = sum(last_n_gols_sofridos) / n
        return avg_gols_marcados, avg_gols_sofridos

    def recent_wins(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm > gs)

    def recent_draws(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm == gs)

    def recent_losses(self):
        return sum(1 for gm, gs in zip(self.golsMarcados[-10:], self.golsSofridos[-10:]) if gm < gs)

    def eficiencia_ofensiva(self):
        return self.average_goals_scored()

    def eficiencia_defensiva(self):
        return self.average_goals_conceded()

    def prever_gols(self, time_adversario):
        if not self.golsMarcados or not time_adversario.golsSofridos:
            print("⚠️ Dados insuficientes para previsão de gols.")
            return 0
        mean_gols = (self.average_goals_scored() + time_adversario.average_goals_conceded()) / 2
        return np.random.poisson(mean_gols)

    def simular_partida_monte_carlo(self, time_adversario, n_simulacoes=5000):
        try:
            resultados = [simulate_match(self, time_adversario) for _ in range(n_simulacoes)]
        except Exception as e:
            print(f"Erro na simulação de partida: {e}")
            return 0, 0, 0
        vitorias = sum(1 for g_a, g_b in resultados if g_a > g_b)
        empates = sum(1 for g_a, g_b in resultados if g_a == g_b)
        derrotas = sum(1 for g_a, g_b in resultados if g_a < g_b)
        prob_vitoria = (vitorias / n_simulacoes) * 100
        prob_empate = (empates / n_simulacoes) * 100
        prob_derrota = (derrotas / n_simulacoes) * 100
        print(f"Probabilidades ({n_simulacoes} simulações):")
        print(f"🔹 {self.name} Vitória: {prob_vitoria:.2f}%")
        print(f"🔸 Empate: {prob_empate:.2f}%")
        print(f"🔻 {time_adversario.name} Vitória: {prob_derrota:.2f}%")
        plt.figure(figsize=(6, 4))
        plt.bar(['Vitória', 'Empate', 'Derrota'], [prob_vitoria, prob_empate, prob_derrota])
        plt.ylabel('Probabilidade (%)')
        plt.title(f"Simulação Monte Carlo: {self.name} vs {time_adversario.name}")
        plt.show()
        return prob_vitoria, prob_empate, prob_derrota

    def adicionar_resultado(self, gols_marcados, gols_sofridos):
        if not isinstance(gols_marcados, (int, float)) or not isinstance(gols_sofridos, (int, float)):
            raise ValueError("Os valores de gols devem ser numéricos.")
        self.golsMarcados.append(gols_marcados)
        self.golsSofridos.append(gols_sofridos)
        print(f"✅ Resultado adicionado para {self.name}: {gols_marcados} marcados, {gols_sofridos} sofridos.")

    def average_goals_scored_weighted(self, weight_factor=0.9):
        if not self.golsMarcados:
            print(f"⚠️ Não há dados para calcular a média ponderada de gols marcados para {self.name}.")
            return 0
        pesos = [weight_factor ** i for i in range(len(self.golsMarcados))]
        pesos.reverse()  # O jogo mais recente terá o maior peso
        total_pesos = sum(pesos)
        media_ponderada = sum(g * p for g, p in zip(self.golsMarcados, pesos)) / total_pesos
        return media_ponderada

    def average_goals_conceded_weighted(self, weight_factor=0.9):
        if not self.golsSofridos:
            print(f"⚠️ Não há dados para calcular a média ponderada de gols sofridos para {self.name}.")
            return 0
        pesos = [weight_factor ** i for i in range(len(self.golsSofridos))]
        pesos.reverse()
        total_pesos = sum(pesos)
        media_ponderada = sum(g * p for g, p in zip(self.golsSofridos, pesos)) / total_pesos
        return media_ponderada

# ========================== FUNÇÕES AUXILIARES PARA PROBABILIDADES ==========================
def compute_basic_expected_values(team_a, team_b, home_advantage=0.1, recent_performance_weight=1.5, confidence_level=0.95):
    avg_goals_a = team_a.average_goals_scored() * recent_performance_weight + home_advantage
    avg_goals_b = team_b.average_goals_scored() * recent_performance_weight
    defense_factor_a = max(0.5, 1 - team_a.average_goals_conceded())
    defense_factor_b = max(0.5, 1 - team_b.average_goals_conceded())
    expected_a = max(0.5, min(avg_goals_a * defense_factor_b, 5)) * confidence_level
    expected_b = max(0.5, min(avg_goals_b * defense_factor_a, 5)) * confidence_level
    return expected_a, expected_b

def compute_score_probability(expected_a, expected_b, gols_a, gols_b):
    p_a = poisson.pmf(gols_a, expected_a)
    p_b = poisson.pmf(gols_b, expected_b)
    return p_a * p_b

# ========================== FUNÇÕES DE TREINAMENTO DOS MODELOS ==========================
def preparar_dados_para_treinamento(df_completo):
    linhas_treinamento = []
    for _, row in df_completo.iterrows():
        gols_marc_mandante = row['FTHG']
        gols_sofr_mandante = row['FTAG']
        ftr = row['FTR']
        if ftr == 'H':
            res_mandante = 2
            vit_mandante, emp_mandante, der_mandante = 1, 0, 0
        elif ftr == 'D':
            res_mandante = 1
            vit_mandante, emp_mandante, der_mandante = 0, 1, 0
        else:
            res_mandante = 0
            vit_mandante, emp_mandante, der_mandante = 0, 0, 1
        linhas_treinamento.append({
            'Gols Marcados': gols_marc_mandante,
            'Gols Sofridos': gols_sofr_mandante,
            'Vitórias': vit_mandante,
            'Empates': emp_mandante,
            'Derrotas': der_mandante,
            'Resultado': res_mandante
        })
        gols_marc_visitante = row['FTAG']
        gols_sofr_visitante = row['FTHG']
        if ftr == 'A':
            res_visitante = 2
            vit_visit, emp_visit, der_visit = 1, 0, 0
        elif ftr == 'D':
            res_visitante = 1
            vit_visit, emp_visit, der_visit = 0, 1, 0
        else:
            res_visitante = 0
            vit_visit, emp_visit, der_visit = 0, 0, 1
        linhas_treinamento.append({
            'Gols Marcados': gols_marc_visitante,
            'Gols Sofridos': gols_sofr_visitante,
            'Vitórias': vit_visit,
            'Empates': emp_visit,
            'Derrotas': der_visit,
            'Resultado': res_visitante
        })
    return pd.DataFrame(linhas_treinamento)

def train_model_random_forest(df):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    y = df['Resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== RandomForest ===")
    print("Acurácia (hold-out):", accuracy_score(y_test, y_pred))
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print("Acurácia Cross-Val. (5 folds):", scores.mean().round(3), "+/-", scores.std().round(3))
    return model

def train_model_xgboost(df):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    y = df['Resultado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print("\n=== XGBoost ===")
    print("Acurácia (hold-out):", accuracy_score(y_test, y_pred))
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='accuracy')
    print("Acurácia Cross-Val. (5 folds):", scores.mean().round(3), "+/-", scores.std().round(3))
    return xgb_model

def explicar_modelo_shap(model, df, model_name=""):
    X = df[['Gols Marcados', 'Gols Sofridos', 'Vitórias', 'Empates', 'Derrotas']]
    print(f"\n-- Gerando explicações SHAP para o modelo {model_name} --")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.title(f"SHAP Summary Bar - {model_name}")
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.show()
    plt.title(f"SHAP Summary Plot - {model_name}")
    shap.summary_plot(shap_values, X, show=False)
    plt.show()

# =========================== HISTÓRICO DE SIMULAÇÕES ===========================
historico_simulacoes = []

def salvar_simulacao(team_a, team_b, resultado):
    historico_simulacoes.append({
        "Time A": team_a.name,
        "Time B": team_b.name,
        "Resultado": resultado
    })
    print("✅ Simulação salva no histórico!")

def visualizar_historico():
    if not historico_simulacoes:
        print("\n📜 Nenhum histórico de simulação disponível.")
        return
    print("\n📜 Histórico de Simulações:")
    for i, sim in enumerate(historico_simulacoes, 1):
        print(f"{i}. {sim['Time A']} vs {sim['Time B']} - {sim['Resultado']}")

# ============================ CARREGAR DADOS EXCEL ============================
def carregar_dados_excel(arquivos):
    dfs = []
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            df = pd.read_excel(arquivo)
            dfs.append(df)
    if dfs:
        df_completo = pd.concat(dfs, ignore_index=True)
        media_gols_mandante = df_completo.groupby('HomeTeam')['FTHG'].mean()
        media_gols_visitante = df_completo.groupby('AwayTeam')['FTAG'].mean()
        return {
            'media_gols_mandante': media_gols_mandante,
            'media_gols_visitante': media_gols_visitante,
            'dados_partidas': df_completo
        }
    else:
        raise FileNotFoundError("Nenhum arquivo Excel encontrado.")

# ============================ GET TEAM DATA ============================
def get_team_data(name, dados_partidas=None):
    try:
        golsMarcados = []
        golsSofridos = []
        print(f"\n[Estilo Flashscore] Insira os dados dos últimos jogos do time '{name}':")
        print(" Obs.: Jogo 1 = MAIS ANTIGO; jogo n = MAIS RECENTE.")
        jogos_manuais = int(input("Quantos jogos deseja inserir manualmente? (ex.: 5 ou 10): ").strip())
        for i in range(1, jogos_manuais + 1):
            gm = int(input(f"Gols marcados pelo {name} no Jogo {i} (ANTIGO->RECENTE): "))
            gs = int(input(f"Gols sofridos pelo {name} no Jogo {i} (ANTIGO->RECENTE): "))
            golsMarcados.append(gm)
            golsSofridos.append(gs)
        peso = 1.0
        media_gols = sum(golsMarcados) / len(golsMarcados) if golsMarcados else 0
        if media_gols > 2:
            peso = 1.2
        elif media_gols < 1.5:
            peso = 0.8
        golsMarcados = [g * peso for g in golsMarcados]
        golsSofridos = [g * peso for g in golsSofridos]
        golsMarcados.reverse()
        golsSofridos.reverse()
        print(f"\n➡ Ordem final: índice 0 = jogo MAIS RECENTE; índice -1 = MAIS ANTIGO.")
        return TeamStats(name, golsMarcados, golsSofridos)
    except Exception as e:
        raise ValueError(f"Erro ao processar os dados do time '{name}': {e}")

# ============================ H2H STATISTICS ============================
def calculate_h2h_statistics(team_a, team_b, dados_partidas=None, ultimos=5, peso_recente=1.5):
    try:
        jogos_h2h = []
        if input("\nDeseja adicionar confrontos manuais entre os dois times? (s/n): ").strip().lower() == "s":
            num_jogos = int(input("Quantos confrontos diretos deseja adicionar? "))
            for i in range(num_jogos):
                ga = int(input(f"Gols marcados por {team_a.name} no jogo {i+1}: "))
                gb = int(input(f"Gols marcados por {team_b.name} no jogo {i+1}: "))
                jogos_h2h.append({
                    "HomeTeam": team_a,
                    "AwayTeam": team_b,
                    "FTHG": ga,
                    "FTAG": gb
                })
        if dados_partidas is not None and not jogos_h2h:
            df_h2h = dados_partidas[
                ((dados_partidas['HomeTeam'] == team_a.name) & (dados_partidas['AwayTeam'] == team_b.name)) |
                ((dados_partidas['HomeTeam'] == team_b.name) & (dados_partidas['AwayTeam'] == team_a.name))
            ]
            jogos_h2h.extend(df_h2h.to_dict(orient="records"))
        jogos_h2h = jogos_h2h[-ultimos:]
        if not jogos_h2h:
            return None
        def calc_vitorias(jogos, team, is_mandante):
            if is_mandante:
                return sum(1 for j in jogos if (j["HomeTeam"] == team.name and j["FTHG"] > j["FTAG"]))
            else:
                return sum(1 for j in jogos if (j["AwayTeam"] == team.name and j["FTAG"] > j["FTHG"]))
        vit_a_mandante = calc_vitorias(jogos_h2h, team_a, True)
        vit_a_visitante = calc_vitorias(jogos_h2h, team_a, False)
        vit_b_mandante = calc_vitorias(jogos_h2h, team_b, True)
        vit_b_visitante = calc_vitorias(jogos_h2h, team_b, False)
        empates = sum(1 for j in jogos_h2h if j["FTHG"] == j["FTAG"])
        def get_gols_time(jogo, team):
            if isinstance(jogo["HomeTeam"], TeamStats):
                if jogo["HomeTeam"].name == team.name:
                    return jogo["FTHG"]
                else:
                    return jogo["FTAG"]
            else:
                if jogo["HomeTeam"] == team.name:
                    return jogo["FTHG"]
                else:
                    return jogo["FTAG"]
        gols_a = [get_gols_time(j, team_a) for j in jogos_h2h]
        gols_b = [get_gols_time(j, team_b) for j in jogos_h2h]
        media_gols_a = np.mean(gols_a) if len(gols_a) else 0
        media_gols_b = np.mean(gols_b) if len(gols_b) else 0
        if len(jogos_h2h) > 3:
            recent_games = jogos_h2h[-3:]
            vit_a_recent = sum(1 for rg in recent_games if get_gols_time(rg, team_a) > get_gols_time(rg, team_b))
            vit_b_recent = sum(1 for rg in recent_games if get_gols_time(rg, team_b) > get_gols_time(rg, team_a))
            vit_a_mandante += vit_a_recent * peso_recente
            vit_b_mandante += vit_b_recent * peso_recente
        total_jogos = len(jogos_h2h)
        prob_vit_a = (vit_a_mandante + vit_a_visitante) / total_jogos * 100 if total_jogos else 0
        prob_vit_b = (vit_b_mandante + vit_b_visitante) / total_jogos * 100 if total_jogos else 0
        prob_empate = (empates / total_jogos) * 100 if total_jogos else 0
        return {
            'total_jogos': total_jogos,
            'vitorias_a_mandante': vit_a_mandante,
            'vitorias_a_visitante': vit_a_visitante,
            'vitorias_b_mandante': vit_b_mandante,
            'vitorias_b_visitante': vit_b_visitante,
            'empates': empates,
            'media_gols_a': round(media_gols_a, 2),
            'media_gols_b': round(media_gols_b, 2),
            'prob_vitoria_a': round(prob_vit_a, 2),
            'prob_empate': round(prob_empate, 2),
            'prob_vitoria_b': round(prob_vit_b, 2),
        }
    except Exception as e:
        raise ValueError(f"Erro ao calcular estatísticas H2H: {e}")

# ===================== SIMULAÇÃO PARTIDA / GOLS / ETC =====================
def simulate_match(team_a, team_b, home_advantage=0.1, recent_performance_weight=1.5,
                   confidence_level=0.95, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    avg_goals_a = team_a.average_goals_scored() * recent_performance_weight + home_advantage
    avg_goals_b = team_b.average_goals_scored() * recent_performance_weight
    defense_factor_a = max(0.5, 1 - team_a.average_goals_conceded())
    defense_factor_b = max(0.5, 1 - team_b.average_goals_conceded())
    adjusted_goals_a = max(0.5, min(avg_goals_a * defense_factor_b, 5))
    adjusted_goals_b = max(0.5, min(avg_goals_b * defense_factor_a, 5))
    gols_a = int(np.round(adjusted_goals_a * confidence_level))
    gols_b = int(np.round(adjusted_goals_b * confidence_level))
    return gols_a, gols_b

def calcular_recorrencia_gols(gols, golsMarcados):
    ocorrencias = golsMarcados.count(gols)
    total_jogos = len(golsMarcados)
    return ocorrencias / total_jogos if total_jogos > 0 else 0

def calculate_recent_performance(dados_partidas, team, home_games=True, n_games=5):
    jogos = dados_partidas[dados_partidas['HomeTeam'] == team.name].tail(n_games) if home_games else \
            dados_partidas[dados_partidas['AwayTeam'] == team.name].tail(n_games)
    if home_games:
        gols_marcados = jogos['FTHG'].sum()
        gols_sofridos = jogos['FTAG'].sum()
    else:
        gols_marcados = jogos['FTAG'].sum()
        gols_sofridos = jogos['FTHG'].sum()
    return gols_marcados, gols_sofridos

def calcular_fatores(team_a, team_b, dados_partidas, peso_recente=1.0):
    recent_a_goals, recent_a_conceded = calculate_recent_performance(dados_partidas, team_a, home_games=True)
    recent_b_goals, recent_b_conceded = calculate_recent_performance(dados_partidas, team_b, home_games=False)
    performance_a = team_a.average_goals_scored() - team_b.average_goals_conceded() + (recent_a_goals * peso_recente)
    performance_b = team_b.average_goals_scored() - team_a.average_goals_conceded() + (recent_b_goals * peso_recente)
    return performance_a, performance_b

def plotar_probabilidades_mercado(probabilities, title="Probabilidades de Mercado", bar_width=0.4):
    mercados = list(probabilities.keys())
    overs = [probabilities[m]['over'] * 100 for m in mercados]
    unders = [probabilities[m]['under'] * 100 for m in mercados]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(mercados))
    ax.bar(x, overs, width=bar_width, label='Over (%)', alpha=0.7)
    ax.bar([p + bar_width for p in x], unders, width=bar_width, label='Under (%)', alpha=0.7)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels([f"Over/Under {m}" for m in mercados], fontsize=10)
    ax.set_ylabel("Probabilidade (%)", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("\nProbabilidades detalhadas:")
    for market, values in probabilities.items():
        print(f"Over/Under {market}: Over = {values['over']*100:.2f}%, Under = {values['under']*100:.2f}%")

def plotar_tendencias(team_a, team_b, n_ultimos_jogos=10, suavizar=False):
    qt_a = min(n_ultimos_jogos, len(team_a.golsMarcados), len(team_a.golsSofridos))
    qt_b = min(n_ultimos_jogos, len(team_b.golsMarcados), len(team_b.golsSofridos))
    if qt_a == 0 and qt_b == 0:
        print("\n⚠️ Não há dados suficientes para plotar tendências.")
        return
    gm_a = team_a.golsMarcados[-qt_a:]
    gs_a = team_a.golsSofridos[-qt_a:]
    gm_b = team_b.golsMarcados[-qt_b:]
    gs_b = team_b.golsSofridos[-qt_b:]
    x_a = range(1, qt_a + 1)
    x_b = range(1, qt_b + 1)
    if suavizar:
        window = 3
        if len(gm_a) >= window:
            gm_a = np.convolve(gm_a, np.ones(window)/window, mode='valid')
            gs_a = np.convolve(gs_a, np.ones(window)/window, mode='valid')
            x_a = range(1, len(gm_a) + 1)
        if len(gm_b) >= window:
            gm_b = np.convolve(gm_b, np.ones(window)/window, mode='valid')
            gs_b = np.convolve(gs_b, np.ones(window)/window, mode='valid')
            x_b = range(1, len(gm_b) + 1)
    plt.figure(figsize=(12, 6))
    if len(gm_a) > 0:
        plt.plot(x_a, gm_a, marker='o', label=f"{team_a.name} - Gols Marcados")
        plt.plot(x_a, gs_a, marker='o', linestyle='--', label=f"{team_a.name} - Gols Sofridos")
    if len(gm_b) > 0:
        plt.plot(x_b, gm_b, marker='o', label=f"{team_b.name} - Gols Marcados")
        plt.plot(x_b, gs_b, marker='o', linestyle='--', label=f"{team_b.name} - Gols Sofridos")
    plt.title("Tendências de Desempenho", fontsize=16)
    plt.xlabel("Últimos Jogos", fontsize=12)
    plt.ylabel("Gols", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_goal_market_probabilities(team_a, team_b):
    avg_goals_a = team_a.average_goals_scored()
    avg_goals_b = team_b.average_goals_scored()
    avg_total_goals = (avg_goals_a + avg_goals_b) / 2
    if avg_goals_a > 2.5:
        avg_total_goals *= 1.25
    if avg_goals_b > 2.5:
        avg_total_goals *= 1.25
    over_under_markets = {m: {"over": 0, "under": 0} for m in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]}
    for market in over_under_markets.keys():
        prob_under = poisson.cdf(market, avg_total_goals)
        prob_over = 1 - prob_under
        over_under_markets[market] = {"over": prob_over, "under": prob_under}
    return over_under_markets

def get_best_market(probabilities, avg_goals_a, avg_goals_b):
    if avg_goals_a > 2.5 and avg_goals_b > 2.5:
        markets_to_check = [3.5, 4.5]
    else:
        markets_to_check = [1.5, 2.5]
    best_market = None
    best_weight = 0
    for m in markets_to_check:
        over_prob = probabilities[m]["over"]
        under_prob = probabilities[m]["under"]
        maior = max(over_prob, under_prob)
        if maior > best_weight:
            best_weight = maior
            best_market = m
    return best_market, best_weight, probabilities[best_market]["over"] * 100

def calculate_extended_match_result_probability(team_a, team_b, max_goals=10):
    avg_goals_a = team_a.average_goals_scored()
    avg_goals_b = team_b.average_goals_scored()
    prob_result = {
        "Vitória A por 1 gol": 0,
        "Vitória A por 2 gols": 0,
        "Vitória A por 3 gols": 0,
        "Vitória A por 4 gols": 0,
        "Vitória A": 0,
        "Empate": 0,
        "Vitória B por 1 gol": 0,
        "Vitória B por 2 gols": 0,
        "Vitória B por 3 gols": 0,
        "Vitória B por 4 gols": 0,
        "Vitória B": 0
    }
    for ga in range(0, max_goals):
        for gb in range(0, max_goals):
            pa = poisson.pmf(ga, avg_goals_a)
            pb = poisson.pmf(gb, avg_goals_b)
            total_prob = pa * pb
            if ga > gb:
                diff = ga - gb
                if diff == 1:
                    prob_result["Vitória A por 1 gol"] += total_prob
                elif diff == 2:
                    prob_result["Vitória A por 2 gols"] += total_prob
                elif diff == 3:
                    prob_result["Vitória A por 3 gols"] += total_prob
                elif diff == 4:
                    prob_result["Vitória A por 4 gols"] += total_prob
                else:
                    prob_result["Vitória A"] += total_prob
            elif ga < gb:
                diff = gb - ga
                if diff == 1:
                    prob_result["Vitória B por 1 gol"] += total_prob
                elif diff == 2:
                    prob_result["Vitória B por 2 gols"] += total_prob
                elif diff == 3:
                    prob_result["Vitória B por 3 gols"] += total_prob
                elif diff == 4:
                    prob_result["Vitória B por 4 gols"] += total_prob
                else:
                    prob_result["Vitória B"] += total_prob
            else:
                prob_result["Empate"] += total_prob
    soma = sum(prob_result.values())
    for r in prob_result:
        prob_result[r] = (prob_result[r] / soma) * 100
    return prob_result

def explicar_aposta(market, prob_over, prob_under):
    if prob_over > prob_under:
        return f"🔍 Recomendamos o mercado *Over {market}* com prob. de {prob_over*100:.2f}% (tendência ofensiva)."
    else:
        return f"🔍 Recomendamos o mercado *Under {market}* com prob. de {prob_under*100:.2f}% (tendência defensiva)."

def comparar_times(team_a, team_b):
    print(f"\n🔍 Comparação entre {team_a.name} e {team_b.name}:\n")
    print("⚽ Desempenho Ofensivo:")
    print(f"{team_a.name}: {team_a.average_goals_scored():.2f} gols/jogo")
    print(f"{team_b.name}: {team_b.average_goals_scored():.2f} gols/jogo")
    print("\n🛡️ Desempenho Defensivo:")
    print(f"{team_a.name}: {team_a.average_goals_conceded():.2f} gols sofridos/jogo")
    print(f"{team_b.name}: {team_b.average_goals_conceded():.2f} gols sofridos/jogo")
    recent_a = team_a.golsMarcados[-5:]
    recent_b = team_b.golsMarcados[-5:]
    recent_sa = team_a.golsSofridos[-5:]
    recent_sb = team_b.golsSofridos[-5:]
    ma = sum(recent_a)/len(recent_a) if recent_a else 0
    mb = sum(recent_b)/len(recent_b) if recent_b else 0
    msa = sum(recent_sa)/len(recent_sa) if recent_sa else 0
    msb = sum(recent_sb)/len(recent_sb) if recent_sb else 0
    print("\n📊 Forma Recente (Últimos 5 Jogos):")
    print(f"{team_a.name}: {ma:.2f} gols marcados, {msa:.2f} gols sofridos")
    print(f"{team_b.name}: {mb:.2f} gols marcados, {msb:.2f} gols sofridos")
    tendencia_a = "Melhora" if ma > msa else "Declínio"
    tendencia_b = "Melhora" if mb > msb else "Declínio"
    print("\n📈 Tendência:")
    print(f"{team_a.name}: {tendencia_a}")
    print(f"{team_b.name}: {tendencia_b}")
    wins_a = sum(1 for i in range(len(recent_a)) if recent_a[i] > recent_sa[i])
    draws_a = sum(1 for i in range(len(recent_a)) if recent_a[i] == recent_sa[i])
    losses_a = sum(1 for i in range(len(recent_a)) if recent_a[i] < recent_sa[i])
    wins_b = sum(1 for i in range(len(recent_b)) if recent_b[i] > recent_sb[i])
    draws_b = sum(1 for i in range(len(recent_b)) if recent_b[i] == recent_sb[i])
    losses_b = sum(1 for i in range(len(recent_b)) if recent_b[i] < recent_sb[i])
    print("\n🏆 Performance Recente (Últimos 5 Jogos):")
    print(f"{team_a.name}: {wins_a} vitórias, {draws_a} empates, {losses_a} derrotas")
    print(f"{team_b.name}: {wins_b} vitórias, {draws_b} empates, {losses_b} derrotas")

def calcular_xg(team, fator_ataque=1.2, fator_defesa=1.1, ajuste_recente=True):
    if ajuste_recente:
        fator_ataque = ajustar_fatores(team, "ataque")
        fator_defesa = ajustar_fatores(team, "defesa")
    xg = team.average_goals_scored() * fator_ataque
    xga = team.average_goals_conceded() * fator_defesa
    return {"xG": xg, "xGA": xga}

def exibir_xg(team_a, team_b):
    xg_a = calcular_xg(team_a)
    xg_b = calcular_xg(team_b)
    print(f"\n🔢 Estatísticas Avançadas (xG e xGA):")
    print(f"{team_a.name} - xG: {xg_a['xG']:.2f}, xGA: {xg_a['xGA']:.2f}")
    print(f"{team_b.name} - xG: {xg_b['xG']:.2f}, xGA: {xg_b['xGA']:.2f}")

def ajustar_fatores(team, tipo="ataque", intensidade=True):
    if tipo == "ataque":
        vitorias = team.recent_wins()
        empates = team.recent_draws()
        fator_ataque = 1.0 + vitorias*0.05 - empates*0.02
        if intensidade:
            fator_ataque += sum(team.golsMarcados[5:]) * 0.01
        return fator_ataque
    elif tipo == "defesa":
        derrotas = team.recent_losses()
        fator_defesa = 1.0 - derrotas*0.05
        if intensidade:
            fator_defesa -= sum(team.golsSofridos[5:]) * 0.01
        return fator_defesa

def adjust_offensive_defensive_factors(team):
    """
    Retorna os fatores de ataque e defesa com base na performance recente.
    """
    attack = ajustar_fatores(team, tipo="ataque", intensidade=True)
    defense = ajustar_fatores(team, tipo="defesa", intensidade=True)
    return attack, defense

# ===================== NOVA FUNÇÃO: Simulação Avançada com Variação (mais realista!) =====================
def simulate_match_with_variation(team_a, team_b, base_confidence=1.0, seed=None):
    """
    Simula uma partida com variação nos fatores de ataque e defesa.
    Aumentamos a variabilidade para resultados mais realistas.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    sigma_attack = 0.3   # Maior variabilidade
    sigma_defense = 0.3
    base_attack_a, base_defense_a = adjust_offensive_defensive_factors(team_a)
    base_attack_b, base_defense_b = adjust_offensive_defensive_factors(team_b)
    attack_a = base_attack_a + np.random.normal(0, sigma_attack)
    defense_a = base_defense_a + np.random.normal(0, sigma_defense)
    attack_b = base_attack_b + np.random.normal(0, sigma_attack)
    defense_b = base_defense_b + np.random.normal(0, sigma_defense)
    avg_goals_a = team_a.average_goals_scored() * attack_a
    avg_goals_b = team_b.average_goals_scored() * attack_b
    avg_goals_a *= max(0.5, 1 - team_b.average_goals_conceded() * defense_a)
    avg_goals_b *= max(0.5, 1 - team_a.average_goals_conceded() * defense_b)
    avg_goals_a = max(0.5, min(avg_goals_a, 5))
    avg_goals_b = max(0.5, min(avg_goals_b, 5))
    confidence = base_confidence + np.random.normal(0, 0.05)
    confidence = max(0.8, min(confidence, 1.2))
    gols_a = np.random.poisson(avg_goals_a * confidence)
    gols_b = np.random.poisson(avg_goals_b * confidence)
    return gols_a, gols_b

def validar_simulacoes(team, num_simulacoes=1000):
    simulated_gols = []
    for _ in range(num_simulacoes):
        simulated_gols.append(np.random.poisson(team.average_goals_scored()))
    media_simulada = np.mean(simulated_gols)
    desvio_simulado = np.std(simulated_gols)
    media_historica = team.average_goals_scored()
    desvio_historico = np.std(team.golsMarcados) if team.golsMarcados else 0
    print(f"\nEquipe: {team.name}")
    print(f"Média Histórica: {media_historica:.2f}, Média Simulada: {media_simulada:.2f}")
    print(f"Desvio Padrão Histórico: {desvio_historico:.2f}, Desvio Simulado: {desvio_simulado:.2f}")

# ===================== NOVA FUNÇÃO: Simulação de Temporada =====================
def simulate_season(team, n_matches=38, home_advantage=0.1, performance_weight=1.5):
    """
    Simula uma temporada inteira para um time.
    Retorna a soma de pontos (3 por vitória, 1 por empate) e saldo de gols.
    """
    pontos = 0
    saldo_gols = 0
    resultados = []
    for i in range(n_matches):
        gols = np.random.poisson(team.average_goals_scored() * performance_weight)
        if gols == 0:
            pontos += 1
            resultado = "Empate"
        else:
            pontos += 3
            resultado = "Vitória"
        saldo_gols += gols - np.random.randint(0, 2)  # Simulação simplificada do saldo
        resultados.append((gols, resultado))
    print(f"\n🔮 Simulação de Temporada para {team.name}:")
    print(f"Pontos Totais: {pontos}")
    print(f"Saldo de Gols: {saldo_gols}")
    return pontos, saldo_gols, resultados

# ===================== NOVA FUNÇÃO: Exportar Relatório para CSV =====================
def export_simulation_report(data, filename="relatorio_simulacoes.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Relatório exportado para {filename}")

# ===================== NOVA FUNÇÃO: Simulação de Impacto de Lesões =====================
def simulate_injury_impact(team, injury_factor=0.8):
    """
    Simula o impacto de uma lesão, reduzindo o fator de ataque.
    injury_factor < 1 reduz a capacidade ofensiva.
    """
    adjusted_goals = team.average_goals_scored() * injury_factor
    print(f"\n⚠️ Simulação de lesão para {team.name}:")
    print(f"Média original de gols: {team.average_goals_scored():.2f}")
    print(f"Média ajustada (com lesão): {adjusted_goals:.2f}")
    return adjusted_goals

# ===================== NOVA FUNÇÃO: Análise de Sensibilidade =====================
def sensitivity_analysis(team_a, team_b, param_name="home_advantage", values=[0.05, 0.1, 0.15, 0.2]):
    """
    Executa simulações variando um parâmetro (ex.: home_advantage) e mostra o impacto no placar.
    """
    resultados = {}
    for v in values:
        gols_a, gols_b = simulate_match(team_a, team_b, home_advantage=v)
        resultados[v] = (gols_a, gols_b)
    print("\n🔍 Análise de Sensibilidade:")
    for k, v in resultados.items():
        print(f"{param_name} = {k}: Placar Simulado: {team_a.name} {v[0]} x {team_b.name} {v[1]}")
    return resultados

# ===================== NOVAS FUNÇÕES ADICIONADAS PARA REGISTRO DE SIMULAÇÕES =====================

# Lista global para armazenar os registros das simulações
simulation_records = []

def registrar_resultado_simulacao():
    """
    Permite registrar manualmente o resultado de uma simulação.
    O usuário deverá informar:
      - Nome do Time A e Time B
      - Placar (ex.: "2-1")
      - Mercado escolhido
      - Se a simulação foi ganhadora (Sim/Não)
    """
    time_a = input("Informe o nome do Time A: ").strip()
    time_b = input("Informe o nome do Time B: ").strip()
    placar = input("Informe o placar (ex.: 2-1): ").strip()
    mercado = input("Informe o mercado escolhido: ").strip()
    ganhadora = input("Essa simulação foi ganhadora? (s/n): ").strip().lower()
    registro = {
        "Time A": time_a,
        "Time B": time_b,
        "Placar": placar,
        "Mercado": mercado,
        "Ganhadora": "Sim" if ganhadora in ["s", "sim"] else "Não"
    }
    simulation_records.append(registro)
    print("✅ Resultado registrado com sucesso!")

def exibir_registros_simulacoes():
    """
    Exibe todos os registros de simulação armazenados.
    """
    if not simulation_records:
        print("⚠️ Nenhum registro de simulação encontrado.")
        return
    print("\n📜 Registros de Simulações:")
    for i, reg in enumerate(simulation_records, 1):
        print(f"{i}. Time A: {reg['Time A']} | Time B: {reg['Time B']} | Placar: {reg['Placar']} | Mercado: {reg['Mercado']} | Ganhadora: {reg['Ganhadora']}")

# ===================== NOVAS FUNÇÕES PARA PREVISÕES NO MUNDO DO FUTEBOL =====================
# 1. Distribuição de Poisson para prever o número de gols
def poisson_distribution_probabilities(mean, max_goals=5):
    """
    Calcula a probabilidade de marcar 0, 1, ..., max_goals utilizando a Distribuição de Poisson.
    """
    probabilities = {}
    for k in range(0, max_goals + 1):
        probabilities[k] = poisson.pmf(k, mean)
    return probabilities

# 2. Distribuição Skellam para modelar a diferença de gols
def skellam_distribution_probability(lambda_a, lambda_b, k):
    """
    Calcula a probabilidade de a diferença de gols (time A - time B) ser igual a k,
    usando a Distribuição Skellam.
    k pode ser negativo, zero ou positivo.
    """
    return skellam.pmf(k, mu1=lambda_a, mu2=lambda_b)

# 3. Modelo de Regressão (Placeholder)
def predict_outcome_regression(features):
    """
    Placeholder para um modelo de regressão (ex.: regressão logística) que preveria
    a probabilidade de vitória, empate e derrota com base em variáveis explicativas.
    Retorna um dicionário de probabilidades.
    """
    # Exemplo dummy:
    return {"win": 0.4, "draw": 0.3, "loss": 0.3}

# 4. Atualização de Rating Elo
def update_elo_rating(current_rating, opponent_rating, result, k_factor=20):
    """
    Atualiza o rating Elo com base no resultado do jogo.
    result: 1 para vitória, 0 para derrota, 0.5 para empate.
    """
    expected = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
    new_rating = current_rating + k_factor * (result - expected)
    return new_rating

# 5. (Opcional) Outras métricas, como Expected Goals (xG), já estão integradas nas funções calcular_xg e exibir_xg.

# =============================================================================
# Estas funções podem ser utilizadas para complementar as análises e previsões do futebol,
# seja para encontrar apostas de valor ou para monitorar a performance de equipes ao longo do tempo.
# =============================================================================

# ===================== MAIN =====================
def main():
    arquivos = [
        "../A.P.G 2/dados1.xlsx",
        "../A.P.G 2/dados2.xlsx",
        "../A.P.G 2/dados3.xlsx",
        "../A.P.G 2/dados4.xlsx",
        "../A.P.G 2/dados5.xlsx"
    ]
    try:
        dados = carregar_dados_excel(arquivos)
    except Exception as e:
        print(f"\n❌ Erro ao carregar os dados do Excel: {e}")
        return
    print("\n🔁 Preparando dados para treinamento...")
    df_preparado = preparar_dados_para_treinamento(dados['dados_partidas'])
    print(f"Dataset de treinamento gerado com {len(df_preparado)} linhas.")
    print("\n🔁 Treinando modelo RandomForest...")
    modelo_rf = train_model_random_forest(df_preparado)
    print("\n🔁 Treinando modelo XGBoost (opcional, mas recomendado)...")
    modelo_xgb = train_model_xgboost(df_preparado)
    print("\n✅ Modelos treinados e armazenados: [modelo_rf, modelo_xgboost].")

    def mostrar_escudo_do_time(time):
        print(f"\n🔍 Buscando escudo do {time}...")
        mostrar_escudo(time)

    def escolher_times():
        while True:
            try:
                mandante = input("\nInsira o nome do Time Mandante (Time A): ")
                visitante = input("Insira o nome do Time Visitante (Time B): ")
                team_a = get_team_data(mandante, dados['dados_partidas'])
                team_b = get_team_data(visitante, dados['dados_partidas'])
                return team_a, team_b, mandante, visitante
            except ValueError as e:
                print(f"\n❌ Erro ao buscar dados dos times: {e}. Tente novamente.")

    def exibir_menu_principal():
        print("\n================= MENU =================")
        print("1  Simular partida (método interno)")
        print("2  Comparar times")
        print("3  Visualizar tendências")
        print("4  Configurar cenário personalizado")
        print("5  Gerar relatório")
        print("6  Ver probabilidades de mercado (Over/Under)")
        print("7  Voltar e escolher novos times")
        print("8  Ranking de Força (focado em Gols)")
        print("9  Exibir XG (Expected Goals)")
        print("10 Calcular Recorrência de Gols")
        print("11 Simulação Avançada (Monte Carlo)")
        print("12 Explicar Modelos (SHAP) [Opcional]")
        print("13 Simulação Avançada com Variação")
        print("14 Validar Simulações (Dados Históricos)")
        print("15 Simular Temporada")
        print("16 Exportar Relatório")
        print("17 Simular Impacto de Lesões")
        print("18 Análise de Sensibilidade")
        print("19 Registrar Resultado de Simulação")
        print("20 Exibir Registros de Simulações")
        print("21 Previsão com Distribuição de Poisson")
        print("22 Previsão com Distribuição Skellam")
        print("23 Atualizar Rating Elo")
        print("24 Previsão via Regressão (Placeholder)")
        print("0  Sair")
        print("========================================")
        return input("\nDigite sua escolha: ").strip()

    team_a, team_b, mandante, visitante = escolher_times()
    mostrar_escudo_do_time(mandante)
    mostrar_escudo_do_time(visitante)

    while True:
        escolha = exibir_menu_principal()
        if escolha == "1":
            gols_a, gols_b = simulate_match(team_a, team_b)
            expected_a, expected_b = compute_basic_expected_values(team_a, team_b)
            prob_score = compute_score_probability(expected_a, expected_b, gols_a, gols_b)
            print(f"\n⚽ Placar Simulado: {team_a.name} {gols_a} x {gols_b} {team_b.name}")
            print(f"📊 Probabilidade: {prob_score*100:.2f}%")
        elif escolha == "2":
            comparar_times(team_a, team_b)
        elif escolha == "3":
            plotar_tendencias(team_a, team_b)
        elif escolha == "4":
            simular_cenario(team_a, team_b)
        elif escolha == "5":
            gols_a, gols_b = simulate_match(team_a, team_b)
            expected_a, expected_b = compute_basic_expected_values(team_a, team_b)
            prob_score = compute_score_probability(expected_a, expected_b, gols_a, gols_b)
            print(f"\n⚽ Placar Simulado: {team_a.name} {gols_a} x {gols_b} {team_b.name}")
            print(f"📊 Probabilidade: {prob_score*100:.2f}%")
            prob_market = calculate_goal_market_probabilities(team_a, team_b)
            h2h_stats = calculate_h2h_statistics(team_a, team_b, dados['dados_partidas'])
            gerar_relatorio(team_a, team_b, h2h_stats, prob_market, {'gols_a': gols_a, 'gols_b': gols_b},
                            team_a.golsMarcados[-5:], team_b.golsMarcados[-5:],
                            {"home": team_a.average_goals_scored(), "away": team_a.average_goals_conceded()},
                            {"home": team_b.average_goals_scored(), "away": team_b.average_goals_conceded()})
        elif escolha == "6":
            prob_market = calculate_goal_market_probabilities(team_a, team_b)
            plotar_probabilidades_mercado(prob_market)
        elif escolha == "7":
            print("\n🔄 Voltando para escolher novos times...")
            team_a, team_b, mandante, visitante = escolher_times()
            mostrar_escudo_do_time(mandante)
            mostrar_escudo_do_time(visitante)
        elif escolha == "8":
            gerar_ranking(dados['dados_partidas'])
        elif escolha == "9":
            exibir_xg(team_a, team_b)
        elif escolha == "10":
            gols = int(input("\nInforme o número de gols para calcular a recorrência: "))
            qual_time = input("Escolha o time para cálculo (A/B): ").strip().upper()
            recorrencia = calcular_recorrencia_gols(gols, team_a.golsMarcados if qual_time == "A" else team_b.golsMarcados)
            print(f"\n🔢 Probabilidade de {gols} gols: {recorrencia * 100:.2f}%")
        elif escolha == "11":
            team_a.simular_partida_monte_carlo(team_b, n_simulacoes=50)
        elif escolha == "12":
            explicar_modelo_shap(modelo_rf, df_preparado, model_name="RandomForest")
            explicar_modelo_shap(modelo_xgb, df_preparado, model_name="XGBoost")
        elif escolha == "13":
            gols_a, gols_b = simulate_match_with_variation(team_a, team_b, base_confidence=1.0)
            print(f"\n⚽ Simulação Avançada com Variação: {team_a.name} {gols_a} x {team_b.name} {gols_b}")
            expected_a, expected_b = compute_basic_expected_values(team_a, team_b)
            prob_score = compute_score_probability(expected_a, expected_b, gols_a, gols_b)
            print(f"📊 Probabilidade: {prob_score*100:.2f}%")
        elif escolha == "14":
            validar_simulacoes(team_a, num_simulacoes=1000)
        elif escolha == "15":
            n_matches = int(input("\nQuantos jogos na temporada? (ex.: 38): "))
            pontos, saldo, resultados = simulate_season(team_a, n_matches=n_matches)
            print(f"\n🏆 Temporada simulada para {team_a.name}: {pontos} pontos, Saldo de gols: {saldo}")
            export_choice = input("Deseja exportar o relatório da temporada? (s/n): ").strip().lower()
            if export_choice == "s":
                rel_data = [{"Jogo": i+1, "Placar": f"{r[0]} (resultado: {r[1]})"} for i, r in enumerate(resultados)]
                export_simulation_report(rel_data, filename=f"temporada_{team_a.name}.csv")
        elif escolha == "16":
            print("\n📤 Exportando relatório...")
            rel_choice = input("Deseja exportar histórico de simulações? (s/n): ").strip().lower()
            if rel_choice == "s":
                export_simulation_report(historico_simulacoes, filename="historico_simulacoes.csv")
            else:
                print("❌ Nenhum relatório exportado.")
        elif escolha == "17":
            injury_factor = float(input("\nInforme o fator de redução (ex.: 0.8 para 20% de redução no ataque): "))
            adjusted = simulate_injury_impact(team_a, injury_factor=injury_factor)
            print(f"Nova média ofensiva de {team_a.name} com lesão: {adjusted:.2f}")
        elif escolha == "18":
            param = input("\nParâmetro a variar (ex.: home_advantage): ").strip()
            values = list(map(float, input("Informe os valores (separados por vírgula, ex.: 0.05,0.1,0.15): ").split(",")))
            sensitivity_analysis(team_a, team_b, param_name=param, values=values)
        elif escolha == "19":
            registrar_resultado_simulacao()
        elif escolha == "20":
            exibir_registros_simulacoes()
        elif escolha == "21":
            # Previsão de resultado utilizando a Distribuição de Poisson
            media_total = float(input("\nInforme a média de gols esperada (λ): "))
            max_gols = int(input("Informe o número máximo de gols a considerar: "))
            probs = poisson_distribution_probabilities(media_total, max_gols)
            print("\nProbabilidades calculadas pela Distribuição de Poisson:")
            for k, prob in probs.items():
                print(f"Golos = {k}: {prob*100:.2f}%")
        elif escolha == "22":
            # Previsão utilizando a Distribuição Skellam para diferença de gols
            lambda_a = float(input("\nInforme o λ do time A (média de gols): "))
            lambda_b = float(input("Informe o λ do time B (média de gols): "))
            diff = int(input("Informe a diferença de gols desejada (pode ser negativa): "))
            prob_diff = skellam_distribution_probability(lambda_a, lambda_b, diff)
            print(f"\nProbabilidade de uma diferença de {diff} gols: {prob_diff*100:.2f}%")
        elif escolha == "23":
            # Atualização de Rating Elo
            current_rating = float(input("\nInforme o rating atual do time: "))
            opponent_rating = float(input("Informe o rating do adversário: "))
            result = float(input("Resultado (1 para vitória, 0.5 para empate, 0 para derrota): "))
            new_rating = update_elo_rating(current_rating, opponent_rating, result)
            print(f"\nNovo rating atualizado: {new_rating:.2f}")
        elif escolha == "24":
            # Previsão via modelo de regressão (placeholder)
            features = input("\nInforme os features separados por vírgula (ex.: 1.2,0.8,1.0): ")
            features = list(map(float, features.split(",")))
            outcome_probs = predict_outcome_regression(features)
            print("\nProbabilidades previstas pelo modelo de regressão:")
            for outcome, prob in outcome_probs.items():
                print(f"{outcome}: {prob*100:.2f}%")
        elif escolha == "0":
            print("\n👋 Saindo do programa. Até logo!")
            break
        else:
            print("\n❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
    print("\n👋 Saindo do programa. Até logo!")
    exit()
    