

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LeadScoringEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # Mapeamento legível das ações para exibição e logs
    NOMES_ACOES = {
        0: "Descartar (0)",
        1: "Nutrir (1)",
        2: "Enviar para Vendas (2)",
    }

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)

     
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.state: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
      
        super().reset(seed=seed)

        self.state = self.np_random.uniform(low=0.0, high=1.0, size=(4,)).astype(np.float32)

        info: Dict[str, Any] = {
            "descricao_estado": {
                "orcamento": float(self.state[0]),
                "engajamento": float(self.state[1]),
                "tempo_na_base": float(self.state[2]),
                "fit_produto": float(self.state[3]),
            }
        }

        return self.state.copy(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
       
        if self.state is None:
            raise RuntimeError("O método 'reset()' deve ser chamado antes de 'step()'.")

        if not self.action_space.contains(action):
            raise ValueError(f"Ação inválida: {action}. Esperado: 0, 1 ou 2.")

        orcamento, engajamento, tempo_na_base, fit_produto = self.state

        if action == 2: 
            if (engajamento + fit_produto) > 1.2:
                reward = 100.0
                motivo = "Venda fechada! Lead tinha alto engajamento + fit."
            else:
                reward = -50.0
                motivo = "Vendedor perdeu tempo. Lead imaturo para abordagem comercial."

        elif action == 0:  
            if fit_produto > 0.8:
                reward = -30.0
                motivo = "Perdeu um cliente com alto potencial (fit > 0.8)."
            else:
                reward = 10.0
                motivo = "Descarte correto. Economia de recursos com lead sem perfil."

        elif action == 1: 
            fator_sorte = float(self.np_random.uniform(0.0, 10.0))
            reward = round(fator_sorte * float(engajamento), 2)
            motivo = "Lead em fluxo de nutrição (recompensa proporcional ao engajamento)."

       
        terminated = True
        truncated = False

        info: Dict[str, Any] = {
            "acao_nome": self.NOMES_ACOES[int(action)],
            "motivo_recompensa": motivo,
            "soma_engajamento_fit": float(engajamento + fit_produto),
        }

        return self.state.copy(), float(reward), terminated, truncated, info


if __name__ == "__main__":
   
    env = LeadScoringEnv()

    NUM_EPISODIOS = 5
    recompensa_total_acumulada = 0.0

    print("=" * 75)
    print(" SIMULAÇÃO DO AMBIENTE: LEAD SCORING RL AGENT ")
    print("=" * 75)

    for episodio in range(1, NUM_EPISODIOS + 1):
        observacao, info_reset = env.reset()
        orcamento, engajamento, tempo_na_base, fit_produto = observacao

        acao = env.action_space.sample()

        nova_obs, recompensa, terminated, truncated, info_step = env.step(acao)
        recompensa_total_acumulada += recompensa

        print(f"\n EPISÓDIO #{episodio}")
        print("-" * 75)
        print("  Estado do Lead (Observação):")
        print(f"      • Orçamento      : {orcamento:.2f}")
        print(f"      • Engajamento    : {engajamento:.2f}")
        print(f"      • Tempo na Base  : {tempo_na_base:.2f}")
        print(f"      • Fit do Produto : {fit_produto:.2f}")
        print(f"      • (Engaj. + Fit) : {info_step['soma_engajamento_fit']:.2f}")
        print(f"\n    Ação Escolhida   : {info_step['acao_nome']}")
        print(f"      Recompensa Obtida: {recompensa:+.2f} pontos")
        print(f"      Diagnóstico      : {info_step['motivo_recompensa']}")
        print(f"      Episódio Fim?    : terminated={terminated} | truncated={truncated}")

    print("\n" + "=" * 75)
    print(f"Simulação concluída! Recompensa acumulada nos {NUM_EPISODIOS} episódios: {recompensa_total_acumulada:+.2f}")
    print("=" * 75)

    env.close()

