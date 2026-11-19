# PINN para adsorção em leito fixo (IC2): resumo das Fases 0 a 4

Eduardo A. Staudt, 18 de novembro de 2026 (branch `pinn-v23b`)

## 1. Objetivo

O objetivo do IC2 é treinar uma PINN (rede neural treinada com as equações do modelo, e não só com
dados) para o modelo de leito fixo do solver v23b, com velocidade variável. A rede recebe os 31
parâmetros de entrada do IC1 e precisa valer para qualquer caso desse domínio. No fim ela vai ser
comparada com uma MLP (rede densa treinada só com dados do solver) em erro, casos de solver
necessários e GPU-h. Comecei com 2 gases e 1 camada de adsorvente, com o código já parametrizado
até 5 componentes. Todos os números abaixo usam como referência o solver MOL (método das linhas)
deste repositório, porque o dataset v23b ainda não chegou, e vão ser refeitos com ele.

## 2. Modelo físico

O modelo segue as Eqs. 1 a 6 do v23b. Cada gás tem balanço de massa com convecção e dispersão
axial, a carga segue a cinética LDF e o equilíbrio vem da isoterma LRC multicomponente do IC1,
dependente da temperatura. O balanço de energia tem condução axial, calor de adsorção e perda pela
parede, e a velocidade v(z, t) sai do balanço molar global com gás ideal a pressão constante. As
condições de contorno são Danckwerts na entrada e gradiente nulo na saída. Tudo é adimensional, com
o tempo escalado pelo maior tempo estequiométrico t_ref e horizonte de 2 t_ref.

## 3. Arquitetura e treino

A entrada da rede é z̃, t̃, 64 Fourier features (senos e cossenos de z̃ e t̃ que ajudam a representar
frentes abruptas) e os 31 parâmetros normalizados. O corpo é uma MLP de 6 camadas de 128 neurônios
(tanh) com uma cabeça para cada grupo de saída (c, q, T e v), num total de cerca de 137 mil
parâmetros. O estado inicial, c e q não negativos e v = v_s na entrada são garantidos pela forma da
saída.

A perda soma os resíduos das Eqs. 1 a 6 em pontos de colocação (valores de z, t e parâmetros θ onde
as equações são cobradas), com pesos adaptativos, pesos causais (o fim do tempo só pesa quando o
começo já está certo), RAR (pontos extras onde o resíduo está alto) e perda de Cauchy (cresce
devagar para resíduos enormes). Os θ de colocação são os dos 50 casos de treino mais 2048 sorteados
por Sobol (sequência quase aleatória que cobre o domínio de forma uniforme), trocados a cada cerca
de 2600 passos. A versão física pura não usa nenhuma solução do solver, e a híbrida acrescenta os
dados de saída (y_i e T). Dividi o problema em três etapas treinadas do zero: na (a) variam 4
parâmetros, na (b) 10 e na (c) os 31. Numa RTX 5070 Ti, um passo leva cerca de 9 ms (0,13 GPU-h a
cada 50 mil passos), e a rede treinada dá a curva de saída de um caso novo em 1 a 2 ms.

## 4. Resultados

A métrica principal é o R² da fração molar na saída (y R²) em casos MOL fora do treino: 10 na
Fase 1 e 20 por etapa nas Fases 2 a 4. A meta é 0,95. A Tabela 1 resume cada fase.

Tabela 1. Resultado principal de cada fase.

| Fase | O que foi feito | y R² |
|---|---|---|
| 0 | Um caso fixo (H2/CO2), comparado com o MOL | 0,994, com 0,26 K de erro na temperatura de saída |
| 1 | Rede paramétrica com 50 casos MOL de treino | 0,654 (física) e 0,460 (híbrida), contra 0,975 e 0,999 nos casos de treino |
| 2 | Colocação em θ por Sobol, etapas (a), (b) e (c) | 0,967, 0,927 e 0,901 (física pura) |
| 3 | Treino longo da etapa (c), ablação e variações | 0,932 com 200 mil passos |
| 4 | Determinismo e continuação do treino longo | 0,931 a 0,933 |

Fonte: Autoria própria.

### 4.1. O que funcionou

O que mais fez diferença foi a colocação em θ sem dados de solver. Na Fase 1 a colocação usava só
os θ dos 50 casos de treino, e a rede acertava esses casos (0,975) mas errava os de validação
(0,654). Com os θ de Sobol, os mesmos 10 casos de validação foram para 0,818 sem nenhum caso novo de
solver. A etapa (c) só ficou treinável com um piso de temperatura nas relações constitutivas (alguns
θ levavam a T ≤ 0 K e a NaN) e com a perda de Cauchy (cerca de 1% dos θ gera resíduos da ordem de
10⁹). O resultado mais curioso veio da ablação na etapa (b), com 2 seeds: Sobol e Cauchy sozinhos
deram cerca de +0,007 cada, mas juntos deram +0,030.

Treinar 200 mil passos em vez de 50 mil levou a etapa (c) de 0,901 para 0,932. Na Fase 0, com um
caso só, os dados de saída junto da física cortaram pela metade o erro dos campos internos. Na
Fase 4, o determinismo do TensorFlow ficou 16 vezes mais lento por causa de uma única operação, e
depois que troquei essa operação o custo caiu para 1,5% por passo. Agora a mesma configuração com a
mesma seed dá resultado idêntico bit a bit.

### 4.2. O que não funcionou

Os dados de saída de só 50 casos (versão híbrida) ajudaram apenas na etapa (a), com 0,979 contra
0,967. Na etapa (c) eles derrubaram o resultado para 0,707, porque a rede acabou casando as saídas
com uma camada não física perto da saída do leito, onde o resíduo ficou até 42 vezes maior que o da
física pura. A Tabela 2 mostra as outras mudanças sem ganho.

Tabela 2. Variações sem ganho (etapa c, a menos que indicado).

| Variação | y R² | Referência |
|---|---|---|
| Rede mais larga | 0,875 | 0,901 |
| 4 vezes mais θ, com 4 vezes menos pontos por θ | 0,832 | 0,901 |
| Metade dos pontos concentrada nas frentes | 0,930 | 0,932 |
| Marcha no tempo em 4 janelas (etapa b) | 0,821 | 0,900 a 0,927 |
| Segundo ciclo da taxa de aprendizado, partindo do treino longo (2 seeds) | 0,931 e 0,931 | 0,932 |
| Dobro de pontos (z, t) por θ, partindo do treino longo (2 seeds) | 0,933 e 0,931 | 0,932 |

Fonte: Autoria própria.

As duas últimas linhas, da Fase 4, mostram que o treino longo já tinha convergido em 0,93. A meta
de 0,95 só foi atingida na etapa (a). Os erros que sobram têm o mesmo formato desde a Fase 2: as
frentes saem espalhadas (começam cerca de 18% antes e terminam cerca de 23% depois do MOL), e a
temperatura e a velocidade na saída ficam fracas (R² perto de 0,6), principalmente nos casos com
aquecimento grande.

## 5. Limitações

As referências ainda são provisórias (MOL deste repositório, não o v23b), e só testei 2 gases e 1
camada. O caso 21 tem reversão de fluxo: no MOL a velocidade fica negativa a partir de 84,5 s em
z/L de 0,2 a 1, e o solver para logo depois, porque a discretização supõe fluxo sempre no mesmo
sentido. A PINN não reproduz a reversão (v/v_s fica entre 0,77 e 1,03) nem o resfriamento (287,6 K
no ponto mais frio, contra 182,6 K no MOL). O que mais me incomoda é que o resíduo dela fica na
faixa dos outros casos, então nada avisa que a resposta está errada. Nenhum caso dos datasets MOL
tem refluxo, mas o domínio não exclui isso (detalhes em `docs/known_limitations.md`).

O determinismo só garante que a mesma seed repete o resultado. Duas seeds treinadas do zero
chegaram a diferir 0,015 no y R² (Fase 3), então toda comparação continua usando 2 seeds.

## 6. O que falta

Com o dataset v23b, primeiro vou reavaliar a PINN física da etapa (c) nos casos dele. Depois vem o
fine-tune da híbrida, que já está pronto: ele parte da rede física pré-treinada e acrescenta os
dados de saída com λ_data = 0,1 fixo (`configs/phase3_finetune_c.yaml`). Em seguida vem a
comparação com a MLP do outro repositório, no mesmo dataset e holdout. Também quero ver se o solver
v23b resolve o caso 21. Mais pra frente ficam mais de 2 gases (o código já aceita, falta testar) e
as 2 camadas com XPINN (uma rede por camada), que ainda não implementei. Os relatórios de cada fase
estão em `docs/phase0_results.md` a `docs/phase4_results.md`.
