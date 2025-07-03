import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from utils import get_classes_cdf, plot_hist, plot_cdf_ks

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss



class ThresholdOptimizer:
    """
    Otimizador baseado em gradiente descendente para encontrar o melhor threshold de binarização
    de uma coluna de score (ex: PD estimada), de forma a minimizar a diferença entre a perda esperada
    e a perda realizada.

    A função de custo é baseada em uma versão binarizada do score e utiliza as colunas de LGD, EAD
    e perda realizada para calcular o erro.

    Parâmetros:
    -----------
    df : pd.DataFrame
        Conjunto de dados contendo as colunas necessárias para cálculo de perdas.
    col_score : str
        Nome da coluna com a probabilidade/score a ser binarizado.
    col_lgd : str
        Nome da coluna com os valores de LGD.
    col_ead : str
        Nome da coluna com os valores de EAD.
    col_loss_real : str
        Nome da coluna com a perda observada (realizada).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        col_score: str = "pd_estimada",
        col_lgd: str = "lgd",
        col_ead: str = "ead",
        col_loss_real: str = "loss_realizada"
    ):
        self.df = df.copy()
        self.col_score = col_score
        self.col_lgd = col_lgd
        self.col_ead = col_ead
        self.col_loss_real = col_loss_real

    def _loss_diff(self, threshold: float) -> float:
        """
        Calcula a diferença absoluta entre a soma da perda estimada binarizada
        e a perda realizada.

        Parâmetros:
        -----------
        threshold : float
            Valor de corte para binarização.

        Retorno:
        --------
        float
            Erro absoluto entre perda esperada e perda realizada.
        """
        pd_bin = (self.df[self.col_score] >= threshold).astype(float)
        loss_esp = pd_bin * self.df[self.col_lgd] * self.df[self.col_ead]
        loss_real = self.df[self.col_loss_real]
        return abs(loss_esp.sum() - loss_real.sum())

    def fit(
        self,
        lr: float = 0.001,
        n_iter: int = 1000,
        initial: float = 0.01,
        tol: float = 0.01,
        verbose: bool = True
    ) -> float:
        """
        Executa o processo de otimização via gradiente descendente.

        Parâmetros:
        -----------
        lr : float
            Taxa de aprendizado.
        n_iter : int
            Número máximo de iterações.
        initial : float
            Valor inicial do threshold.
        tol : float
            Tolerância percentual para parada antecipada.
        verbose : bool
            Se True, imprime progresso.

        Retorno:
        --------
        float
            Threshold ótimo encontrado.
        """
        threshold = max(initial, 1e-3)
        delta = 1e-4
        loss_real_total = self.df[self.col_loss_real].sum()

        for i in range(n_iter):
            t_plus = min(threshold + delta, 1)
            t_minus = max(threshold - delta, 0)

            grad = (self._loss_diff(t_plus) - self._loss_diff(t_minus)) / (t_plus - t_minus)

            # Atualização
            threshold -= lr * grad
            threshold = np.clip(threshold, 0, 1)

            # Cálculo do erro percentual
            pd_bin = (self.df[self.col_score] >= threshold).astype(float)
            loss_esp = pd_bin * self.df[self.col_lgd] * self.df[self.col_ead]
            erro_percentual = abs(loss_esp.sum() - loss_real_total) / loss_real_total * 100

            if verbose and (i % 10 == 0 or i == n_iter - 1):
                print(f"[Iter {i:04d}] Threshold: {threshold:.4f} | Erro %: {erro_percentual:.2f}%")

            if erro_percentual <= tol * 100:
                if verbose:
                    print(f"✅ [Iter {i:04d}]: Early stopping com erro percentual {erro_percentual:.2f}% <= {tol*100:.2f}%")
                break

        self.best_threshold_ = threshold
        return threshold
    
# 🎛️ Parâmetros de controle
SAMPLE_SIZE = 100_000
DEFAULT_RATE_TARGET = 0.05
NOISE_SCALE = 1
np.random.seed(42)

# 🔢 Simulação das variáveis
age = np.random.randint(18, 70, size=SAMPLE_SIZE)
income = np.random.normal(5000, 2000, size=SAMPLE_SIZE).clip(1000, 20000)
loan_amount = np.random.normal(20000, 10000, size=SAMPLE_SIZE).clip(5000, 100000)
loan_term = np.random.choice([12, 24, 36, 48, 60], size=SAMPLE_SIZE)
employment_status = np.random.choice(['CLT', 'autônomo', 'desempregado'], size=SAMPLE_SIZE, p=[0.6, 0.3, 0.1])
credit_score = np.random.normal(600, 100, size=SAMPLE_SIZE).clip(300, 850)
region = np.random.choice(['Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste'], size=SAMPLE_SIZE)
past_due_days = np.random.poisson(5, size=SAMPLE_SIZE)

# 🔁 Codificação para função de risco
employment_map = {'CLT': 0, 'autônomo': 1, 'desempregado': 2}
employment_encoded = pd.Series(employment_status).map(employment_map).values

logit_base = (
    -6.0
    - 0.01 * credit_score
    + 0.0002 * loan_amount
    - 0.0003 * income
    + 0.5 * employment_encoded
    + NOISE_SCALE * np.random.normal(0, 1, SAMPLE_SIZE)
)

def adjust_intercept(intercept):
    logits = logit_base + intercept
    probs = 1 / (1 + np.exp(-logits))
    return abs(probs.mean() - DEFAULT_RATE_TARGET)

res = minimize_scalar(adjust_intercept, bounds=(-10, 10), method='bounded')
optimal_intercept = res.x

logits = logit_base + optimal_intercept
default_prob = 1 / (1 + np.exp(-logits))
default = np.random.binomial(1, default_prob)

lgd = np.where(default == 1, np.random.beta(2, 5, size=SAMPLE_SIZE), 0)
ead = loan_amount * np.random.uniform(0.8, 1.2, size=SAMPLE_SIZE)
realized_loss = default * lgd * ead

# 📋 Dataset completo
df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'employment_status': employment_status,
    'credit_score': credit_score,
    'region': region,
    'past_due_days': past_due_days,
    'default': default,
    'lgd': lgd,
    'ead': ead,
    'loss_realizada': realized_loss
})

print(f"Taxa de inadimplência simulada: {df['default'].mean():.2%}")

# 🔧 Modelagem
expected_dummies = [
    'employment_status_autônomo', 'employment_status_desempregado',
    'region_Nordeste', 'region_Sudeste', 'region_Sul', 'region_Centro-Oeste'
]
df = pd.get_dummies(df, columns=['employment_status', 'region'], drop_first=True)
for col in expected_dummies:
    if col not in df.columns:
        df[col] = 0

features = [
    'age', 'income', 'loan_amount', 'loan_term',
    'credit_score', 'past_due_days'
] + expected_dummies

# 🧪 Separação com índices
train_idx, test_idx = train_test_split(
    df.index, test_size=0.3, stratify=df['default'], random_state=42
)

# 🔍 Treinamento
model = LogisticRegression()
model.fit(df.loc[train_idx, features], df.loc[train_idx, 'default'])

# 🔮 Previsão
df['pd_estimada'] = model.predict_proba(df[features])[:, 1]

# 🔬 Resultado no conjunto de teste


print(f"Test set size: {len(test_idx)}")

otimizador = ThresholdOptimizer(
    df=df.loc[train_idx],
    col_score='pd_estimada',
    col_lgd='lgd',
    col_ead='ead',
    col_loss_real='loss_realizada'
)

melhor_threshold = otimizador.fit(lr=0.0001, n_iter=5000, tol=0.01, initial=0.01)

print(f"🔍 Melhor threshold encontrado: {melhor_threshold:.4f}")