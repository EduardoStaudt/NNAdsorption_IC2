# Configuração atual do melhor modelo

Desde o estudo Optuna de 19 de novembro de 2026, o padrão das etapas é o trial 26 do estudo. Desde a
Fase 5 (21 de novembro), o melhor modelo da etapa (c) é `runs/phase5c_long_seed1`: o mesmo padrão
treinado por 200 mil passos (`configs/phase5_stage_c_long.yaml`), só física, com y R² 0,962 nos 20
casos de validação da etapa (c) (a outra seed deu 0,953). Antes era `runs/optuna_b/confirm_c/t0026_seed0`,
com 50 mil passos e 0,941. A Tabela 1 mostra o que mudou em relação ao modelo antigo. Fora isso, a
configuração é a do `phase3c_long`, que este documento descreve a partir da seção "O modelo
phase3c_long". As únicas outras diferenças são o modo determinístico ligado e checkpoints e
avaliações mais frequentes (a cada 25 mil passos no treino longo).

Há também uma alternativa barata, registrada na Fase 5 mas que não é o padrão: a mesma configuração
com rede de 6 × 128 (`configs/phase5_cheap_{b,c}.yaml`). Com 50 mil passos ela ficou 0,011 abaixo do
padrão na etapa (b) e 0,012 abaixo na (c), com metade do tempo por passo (10 ms contra 20 ms). O
fine-tune da híbrida (`configs/phase3_finetune_c.yaml`) parte de `runs/phase5c_long_seed1`.

Na Fase 6 (22 de novembro), a colocação concentrada nos casos quentes (`theta_colloc.hot_frac` 0,5,
`configs/phase6_hot_200k.yaml`) chegou a 0,963 com 200 mil passos nas duas seeds. Ela é candidata a
novo padrão da etapa (c), mas o padrão continua com `hot_frac` 0 até a aprovação. Treinar 400 mil
passos não melhorou nem o uniforme nem o quente (`docs/phase6_results.md`).

## Mudanças do estudo Optuna

Tabela 1. O que mudou no padrão das etapas com o estudo Optuna.

| Parâmetro | Antes (phase3c_long) | Agora (trial 26) | Efeito medido |
|---|---|---|---|
| Ativação (`model.activation`) | tanh | SiLU | A SiLU é x vezes a sigmoide de x. É suave como a tanh, mas não satura para x grande. Sozinha, levou o y R² da etapa (b) de 0,842 para 0,921 aos 15 mil passos, o maior ganho isolado. |
| Escala das Fourier features (`sigma_z`, `sigma_t`) | 2 em z, 1 em t | 0,59 em z, 0,30 em t | Frequências mais baixas, então a rede começa com funções mais suaves. Sozinha, levou o y R² de 0,842 para 0,879. |
| Peso da energia (`loss.pde_terms`) | 10 | 2,2 | Sozinho, levou o y R² de 0,842 para 0,866. O peso 10 veio da Fase 0, quando, com peso 1, a rede caía na solução trivial T̃ ≈ 0. Com 2,2 isso não aconteceu nem com tanh (T_out RMSE de 22,2 K, igual ao do modelo antigo). |
| Camadas do tronco (`model.depth`) | 6 | 8 | Não medido sozinho (ver abaixo). |
| Neurônios por camada (`model.width`) | 128 | 224 | Não medido sozinho. |
| Fourier features (`model.fourier.n_features`) | 64 frequências, 128 entradas | 128 frequências, 256 entradas | Não medido sozinho. |
| Taxa de aprendizado máxima (`train.lr`) | 10⁻³ | 6,2 × 10⁻⁴ | Não medido sozinho. O cosseno continua terminando em 1% do máximo. |
| Perda de Cauchy (`loss.robust_c`) | c = 10, só na etapa (c) | c = 24,3, em todas as etapas | Com c maior, a perda continua quase quadrática até resíduos maiores, e só os pontos absurdos são amortecidos. |
| θ sorteados (`theta_colloc.n_theta`) | 2.048 | 1.024 | Não medido sozinho. O pool da etapa (c) cai de 1.074.176 para 549.888 pontos. |
| Troca dos θ (`theta_colloc.resample_every_epochs`) | a cada 10 passadas pelo pool | a cada 19,5 passadas (etapa c) | Com o pool menor, o número de passadas quase dobra, e a troca continua a cada cerca de 2.600 passos. |
| Pesos treináveis | 136.710 | 475.750 | |
| Tempo por passo, sozinho na GPU | 9,4 ms | 20,9 ms | 50 mil passos custam 0,30 GPU-h em vez de 0,14. |

Fonte: `runs/optuna_b/confirm_c/t0026_seed0/config.yaml`, `configs/phase2_base.yaml` e
`docs/optuna_stage_b_results.md`.

Os três primeiros itens foram medidos um de cada vez, partindo do modelo antigo. Juntos, ainda com a
rede de 6 × 128, eles levaram o y R² da etapa (b) a 0,939 aos 15 mil passos. As outras mudanças do
trial 26 (rede maior, mais Fourier features, taxa de aprendizado, Cauchy e número de θ) somaram
mais 0,013, chegando a 0,952, e ajudaram principalmente o pior caso, que foi de 0,752 para 0,839.
Na rede nova, a matriz B das Fourier features é 2 × 128, o tronco recebe 289 entradas (256 das
Fourier features, as 2 coordenadas e os 31 parâmetros) e as cabeças também usam SiLU.

## O modelo phase3c_long

Até o estudo, o melhor modelo era o treino longo da etapa (c) da Fase 3, em `runs/phase3c_long`. É
uma PINN só com física, treinada com os 31 parâmetros variando, e chega a y R² 0,932 nos 20 casos
de validação da etapa (c). As continuações da Fase 4 (0,931 a 0,933) não passaram desse valor. Ele
continua sendo o treino mais longo feito até agora.

A configuração completa está em `runs/phase3c_long/config.yaml`. Ela junta
`configs/phase3_stage_c_long.yaml` com os arquivos que ele herda, nesta ordem:
`phase2_stage_c_before_optuna.yaml` (que guarda os valores de antes do estudo), `phase2_stage_c.yaml`,
`phase2_base.yaml`, `pinn_parametric.yaml` e `pinn_base.yaml`. Este run foi treinado antes do modo
determinístico. Uma rodada nova com a mesma configuração sai determinística, que é o padrão desde a
Fase 4, mas não idêntica bit a bit a este run.

## Como a rede calcula a saída

Para um ponto (z̃, t̃) de um caso com parâmetros θ, a conta tem quatro passos.

1. As coordenadas x = (z̃, t̃/2), as duas entre 0 e 1, passam pelas Fourier features:
   γ(x) = [sen(2π x B), cos(2π x B)]. B é uma matriz fixa 2 × 64 sorteada de uma normal com média 0,
   desvio padrão 2 na linha de z e 1 na linha de t. Isso dá 128 números.
2. Esses 128 números, as 2 coordenadas e os 31 parâmetros normalizados (161 entradas) entram no
   tronco, que tem 6 camadas densas de 128 neurônios. Cada camada calcula h ← tanh(W h + b).
3. A saída do tronco vai para 4 cabeças. Cada cabeça tem 1 camada oculta de 64 neurônios (tanh) e
   uma camada linear, e devolve N_c (2 valores), N_q (2), N_T (1) e N_v (1).
4. As restrições exatas montam a saída física. Com s = 1 − exp(−t̃/0,01):
   c̃ = (1 − s) c̃₀ + s softplus(N_c), q̃ = (1 − s) q̃₀ + s softplus(N_q), T̃ = s N_T e
   ṽ = 1 + z̃ s N_v. Em t̃ = 0 a saída é exatamente o estado inicial. Como softplus(x) = ln(1 + eˣ)
   é sempre positiva, c̃ e q̃ nunca ficam negativos. Na entrada (z̃ = 0) a velocidade é exatamente
   v_s.

São 136.710 pesos treináveis: 103.296 no tronco e 33.414 nas cabeças. A matriz B não é treinada.

## Tabela de parâmetros

Tabela 2. Configuração do run `phase3c_long`.

| Parâmetro | Valor | O que faz |
|---|---|---|
| Camadas do tronco (`model.depth`) | 6 | Número de camadas densas em sequência. Mais camadas representam funções mais complexas, com custo maior por passo. |
| Neurônios por camada (`model.width`) | 128 | Largura de cada camada. Cada matriz W do tronco é 128 × 128. |
| Ativação (`model.activation`) | tanh | Função não linear aplicada em cada neurônio. É suave, então as derivadas segundas que aparecem nos resíduos existem e são contínuas. |
| Cabeças (`model.head_hidden`, `model.head_width`) | 1 camada de 64 neurônios por cabeça | Uma rede pequena para cada grupo de saída (c, q, T e v), em cima do tronco comum. |
| Fourier features (`model.fourier.n_features`) | 64 frequências, 128 entradas | Senos e cossenos de z̃ e t̃ em 64 frequências sorteadas. Ajudam a rede a montar frentes íngremes, que uma rede só com tanh aprende devagar. |
| Escala das Fourier features (`sigma_z`, `sigma_t`) | 2 em z, 1 em t | Desvio padrão das frequências sorteadas. Escala maior gera frequências mais altas, com detalhes mais finos e treino mais ruidoso. |
| Fourier treinável (`model.fourier.trainable`) | não | A matriz B fica fixa durante o treino. |
| Estado inicial exato (`model.hard_ic`, `model.ic_delta`) | sim, δ = 0,01 | O estado inicial sai da própria fórmula da saída. δ é o tempo adimensional em que a saída passa do estado inicial para o valor da rede. |
| Otimizador | Adam | Gradiente com momento. Guarda médias móveis do gradiente e do seu quadrado e ajusta o passo de cada peso. |
| Taxa de aprendizado máxima (`train.lr`) | 10⁻³ | Tamanho do passo do Adam no começo do treino. |
| Scheduler (`train.lr_final_frac`) | cosseno, de 10⁻³ até 10⁻⁵ | A taxa cai como meio período de cosseno ao longo do treino. Passos grandes no começo exploram, passos pequenos no fim refinam. |
| Passos (`train.steps`) | 200.000 | Número de atualizações dos pesos. Levou 29,7 min (0,50 GPU-h), com 8,7 ms por passo. |
| Compilação XLA (`train.jit_compile`) | sim | Compila o passo de treino inteiro para a GPU, cerca de 3,6 vezes mais rápido que sem compilar. |
| Precisão (`runtime.dtype`, `runtime.tf32`) | float32, TF32 desligado | Números de 32 bits. O TF32 (um atalho de precisão menor das GPUs novas) fica desligado porque o erro dele, cerca de 3 × 10⁻⁴ por produto de matrizes, atrapalha as derivadas segundas. |
| Pontos no interior por passo (`sampling.n_interior`) | 4.096 | Pontos (z̃, t̃, caso) sorteados do pool a cada passo. É neles que os resíduos das equações são calculados. |
| Pontos de contorno por passo (`sampling.n_boundary`) | 512 na entrada e 512 na saída | Pontos novos a cada passo em z̃ = 0 (Danckwerts) e em z̃ = 1 (gradiente nulo). |
| Pontos de estado inicial e de dados | não usados | O estado inicial já é exato e esta versão não usa dados do solver. |
| RAR (`sampling.rar`) | a cada 1.000 passos, 1.024 pontos entre 32.768 candidatos | Sorteia 32.768 pontos, calcula o resíduo em cada um e coloca no pool os 1.024 piores. Concentra pontos onde a rede ainda erra. |
| θ de colocação sorteados (`theta_colloc.n_theta`) | 2.048 | Casos sorteados por Sobol no domínio dos 31 parâmetros, sem solução do solver. Os θ que o pré-filtro do IC1 rejeita são trocados por outros (5,9% dos sorteados). |
| θ fixos | 50 | Os θ dos 50 casos MOL de treino. Entram só os parâmetros, sem as soluções. |
| Pontos por θ (`theta_colloc.points_per_theta`) | 512 | Pontos (z̃, t̃) iniciais por caso. O pool começa com 2.098 × 512 = 1.074.176 pontos. |
| Troca dos θ (`theta_colloc.resample_every_epochs`) | a cada 10 passadas pelo pool (2.622 passos) | Os 2.048 θ sorteados são trocados por novos. Ao todo a rede viu 157.696 θ diferentes. |
| Normalização de θ (`theta_colloc.scaler_n_theta`, `theta.log_params`) | média e desvio de 8.192 θ de Sobol e dos 50 casos de treino, log em B_ref, kL, dp e Dm | Cada parâmetro vira (x − média)/desvio padrão. Os quatro que variam por ordens de grandeza passam por log antes. |
| Pesos dos termos da EDP (`loss.pde_terms`) | massa 1, LDF 1, energia 10, global 1 | Multiplicadores fixos de cada resíduo dentro do grupo da EDP. A energia tem 10 porque, com peso 1, a rede caía na solução trivial T̃ ≈ 0 (Fase 0). |
| Peso do grupo EDP, λ_pde (`loss.weights.pde`) | 1 | Grupo de referência para os pesos adaptativos. |
| Peso do contorno, λ_bc (`loss.weights.bc`, `loss.adaptive`) | começa em 1, adaptativo, limite 100 | Recalculado a cada 100 passos pela razão entre os gradientes da EDP e do contorno (detalhe abaixo). Neste run chegou ao limite de 100 no passo 300 e ficou lá até o fim. |
| Peso dos dados, λ_data (`loss.weights.data`) | 0 | Versão física pura: nenhum dado do solver entra na perda. |
| Perda de Cauchy (`loss.robust_c`) | c = 10 | Troca a soma ponderada dos resíduos ao quadrado s de cada ponto por ρ(s) = c ln(1 + s/c). Para s pequeno, ρ ≈ s. Para s grande, ρ cresce só como ln s. |
| Pesos causais (`loss.causal`) | 32 fatias de tempo, ε = 0,1 | A fatia i do tempo recebe peso exp(−ε × soma das perdas das fatias anteriores). ε podia crescer até 10, mas ficou em 0,1 o treino todo. |
| Piso de temperatura | T ≥ T_in/2 | Nas relações constitutivas (isoterma e C_T = P/RT), T é limitada por baixo para evitar divisão por zero e NaN com θ extremos. |
| Escala de temperatura (`scales.dT_ratio`) | 0,42 | ΔT_ref = max(1 K, 0,42 × ΔT_feed). O 0,42 é a mediana, nos 50 casos de treino, da razão entre o aquecimento medido e a estimativa ΔT_feed. |
| Horizonte (`scales.t_final_factor`) | 2 | O tempo simulado vai até 2 t_ref. |
| Seed (`runtime.seed`) | 0 | Sorteia a inicialização da rede e os pontos de contorno. Os θ e o pool usam outra seed (`theta_colloc.seed` = 1). |

Fonte: `runs/phase3c_long/config.yaml` e `runs/phase3c_long/summary.json`.

## Detalhes das partes menos óbvias

**Taxa de aprendizado com cosseno.** No passo k de K = 200.000, a taxa é
η(k) = η_max [α + (1 − α)(1 + cos(π k/K))/2], com η_max = 10⁻³ e α = 0,01. Ela começa em 10⁻³, vale
5,05 × 10⁻⁴ no meio do treino e termina em 10⁻⁵.

**Pesos adaptativos.** Se o gradiente da perda de contorno for muito menor que o da EDP, o contorno é
praticamente ignorado. Por isso, a cada 100 passos o código calcula
λ̂_bc = max|∇_w L_pde| / média|∇_w L_bc|, onde ∇_w é o gradiente em relação aos pesos da rede, e
atualiza λ_bc ← 0,9 λ_bc + 0,1 λ̂_bc (uma média móvel). O valor fica limitado entre 0,001 e 100.

**Pesos causais.** O tempo t̃ é dividido em 32 fatias, e L_i é o resíduo médio da fatia i. A fatia i
recebe peso w_i = exp(−ε (L_1 + ... + L_(i−1))). Enquanto o começo do tempo ainda tem resíduo alto,
o fim quase não pesa. Isso imita a ordem de uma simulação, em que o fim depende do começo.

**Perda de Cauchy.** Em estatística, minimizar a soma dos quadrados é o mesmo que supor erros com
distribuição normal. A perda ρ(s) = c ln(1 + s/c), com s = r², corresponde a supor erros com
distribuição de Cauchy, que tem caudas pesadas, então pontos absurdos puxam bem menos. No gradiente,
cada ponto entra com peso 1/(1 + s/c): com c = 10, um ponto com s = 1 pesa 0,91, um com s = 10 pesa
0,5 e um com s = 10⁹ pesa cerca de 10⁻⁸. Isso foi necessário porque cerca de 1% dos θ de Sobol gera
resíduos dessa ordem e, sem a Cauchy, eles dominavam a perda e o treino divergia.

**O que fica desligado.** L-BFGS no fim do treino (`train.lbfgs.enabled`), dados do solver
(λ_data = 0) e pontos de estado inicial (o estado inicial já é exato).
