# Lead Scoring RL Agent

Implementação de um ambiente customizado de Aprendizado por Reforço (Reinforcement Learning) utilizando a biblioteca `gymnasium` para o problema de qualificação e roteamento de leads comerciais.

## Modelagem do Ambiente

O problema é formulado na classe `LeadScoringEnv` com a seguinte especificação formal:

- **Espaço de Observação (`Box(4,)`, intervalo `[0.0, 1.0]`, `float32`)**:
  1. `orcamento`: Capacidade financeira estimada do lead.
  2. `engajamento`: Nível de interação com a marca.
  3. `tempo_na_base`: Tempo de permanência no funil.
  4. `fit_produto`: Aderência do perfil do lead à solução.

- **Espaço de Ações (`Discrete(3)`)**:
  - `0`: Descartar lead.
  - `1`: Nutrir lead (fluxo de marketing).
  - `2`: Encaminhar para Vendas.

- **Função de Recompensa**:
  - Ação `2` (Vendas) com `(engajamento + fit_produto) > 1.2`: `+100`
  - Ação `2` (Vendas) com `(engajamento + fit_produto) <= 1.2`: `-50`
  - Ação `0` (Descartar) com `fit_produto > 0.8`: `-30`
  - Ação `0` (Descartar) com `fit_produto <= 0.8`: `+10`
  - Ação `1` (Nutrir): valor contínuo entre `0` e `10` proporcional ao `engajamento`.

## Requisitos

- Python 3.10+
- `gymnasium`
- `numpy`

## Como Executar a Simulação

Instale as dependências no ambiente virtual (caso ainda não estejam instaladas):

```bash
./.venv/bin/python -m pip install gymnasium numpy
```

Execute a simulação com o comando:

```bash
./.venv/bin/python main.py
```

Alternativamente, ativando o ambiente virtual no terminal:

```bash
source .venv/bin/activate
python main.py
```
