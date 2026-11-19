# PINN para adsorção em leito fixo (IC2): resumo das Fases 0 a 4 e do estudo de hiperparâmetros

Eduardo A. Staudt, 19 de novembro de 2026 (branch `pinn-v23b`)

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

Essa é a configuração das Fases 0 a 4. Depois do estudo da seção 4.4, o padrão das etapas passou a
ser 8 camadas de 224 neurônios com SiLU (x vezes a sigmoide de x, suave como a tanh, mas sem
saturar para x grande), 128 Fourier features com frequências mais baixas, peso 2,2 para a energia,
Cauchy com c = 24,3 e 1024 θ de Sobol. São cerca de 476 mil parâmetros, e um passo leva 20,9 ms
(0,30 GPU-h a cada 50 mil passos).

## 4. Resultados

A métrica principal é o R² da fração molar na saída (y R²) em casos MOL fora do treino: 10 na
Fase 1 e 20 por etapa nas Fases 2 a 4 e no estudo Optuna. A meta é 0,95. A Tabela 1 resume cada fase.

Tabela 1. Resultado principal de cada fase.

| Fase | O que foi feito | y R² |
|---|---|---|
| 0 | Um caso fixo (H2/CO2), comparado com o MOL | 0,994, com 0,26 K de erro na temperatura de saída |
| 1 | Rede paramétrica com 50 casos MOL de treino | 0,654 (física) e 0,460 (híbrida), contra 0,975 e 0,999 nos casos de treino |
| 2 | Colocação em θ por Sobol, etapas (a), (b) e (c) | 0,967, 0,927 e 0,901 (física pura) |
| 3 | Treino longo da etapa (c), ablação e variações | 0,932 com 200 mil passos |
| 4 | Determinismo e continuação do treino longo | 0,931 a 0,933 (platô) |
| Optuna | 60 combinações de hiperparâmetros na etapa (b) e confirmação com 2 seeds | 0,973 na (b) e 0,940 na (c) com 50 mil passos, contra 0,928 e 0,900 da configuração anterior |

Fonte: Autoria própria.

### 4.1. O que funcionou

O que mais fez diferença foi a colocação em θ sem dados de solver. Na Fase 1 a colocação usava só
os θ dos 50 casos de treino, e a rede acertava esses casos (0,975) mas errava os de validação
(0,654). Com os θ de Sobol, os mesmos 10 casos de validação foram para 0,818 sem nenhum caso novo de
solver. Antes disso, na Fase 0, com um caso só, os dados de saída junto da física já tinham cortado
pela metade o erro dos campos internos.

A etapa (c) só ficou treinável com um piso de temperatura nas relações constitutivas (alguns θ
levavam a T ≤ 0 K e a NaN) e com a perda de Cauchy (cerca de 1% dos θ gera resíduos da ordem de
10⁹). O resultado mais curioso veio da ablação na etapa (b), com 2 seeds: Sobol e Cauchy sozinhos
deram cerca de +0,007 cada, mas juntos deram +0,030. Treinar 200 mil passos em vez de 50 mil levou a
etapa (c) de 0,901 para 0,932.

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

Fonte: Autoria própria.

A marcha no tempo foi a pior, mais baixa em todas as janelas de tempo. A meta de 0,95 só foi
atingida na etapa (a). Os erros que sobram têm o mesmo formato desde a Fase 2: as frentes saem
espalhadas (começam cerca de 18% antes e terminam cerca de 23% depois do MOL), e a temperatura e a
velocidade na saída ficam fracas (R² perto de 0,6), principalmente nos casos com aquecimento
grande.

### 4.3. Fase 4: determinismo e platô da etapa (c)

Na Fase 3, duas rodadas com a mesma configuração e a mesma seed deram y R² de 0,927 e 0,903, só por
diferenças de arredondamento na GPU. Na Fase 4 testei o modo determinístico do TensorFlow. A primeira
tentativa ficou 16 vezes mais lenta (146 ms por passo, contra 9 ms), porque uma única operação, a
média do resíduo em cada fatia de tempo dos pesos causais, virava um laço serial. Troquei essa média
por um produto de matrizes que dá o mesmo resultado, e o custo caiu para 1,5% por passo. Agora duas
rodadas da mesma configuração saem idênticas bit a bit, e o modo determinístico ficou ligado por
padrão.

Depois testei se a etapa (c) ainda melhorava a partir do treino longo (0,932), com mais 50 mil
passos e 2 seeds em cada variante. A primeira reinicia a taxa de aprendizado em 3 × 10⁻⁴ (um
segundo ciclo do cosseno) e a segunda usa o dobro de pontos (z, t) por θ. A Tabela 3 mostra que
nenhuma das duas mexeu no resultado.

Tabela 3. Etapa (c) continuada a partir do treino longo (y R², ponto de partida 0,932).

| Variante | Seed 0 | Seed 1 | Média |
|---|---|---|---|
| Segundo ciclo da taxa de aprendizado (máximo 3 × 10⁻⁴) | 0,931 | 0,931 | 0,931 |
| Dobro de pontos (z, t) por θ | 0,933 | 0,931 | 0,932 |

Fonte: Autoria própria.

Com a taxa de aprendizado alta no começo, o y R² caiu para 0,921 a 0,930 aos 10 mil passos e depois
voltou para o mesmo 0,93. Os ganhos foram pequenos e ficaram fora do y R²: a temperatura de saída
melhorou um pouco (RMSE de 15,3 para 15,1 K) e o erro do início das frentes caiu de 20,9% para 19,3
a 19,8%. Com esta rede e esta perda, a etapa (c) chegou num platô de cerca de 0,93. Para passar
disso, o ganho vai ter que vir de outro lugar: dos dados do v23b com o fine-tune da híbrida, ou de
mudanças na rede e nos hiperparâmetros.

### 4.4. Estudo de hiperparâmetros com Optuna (etapa b)

Depois do platô, testei se outra combinação de hiperparâmetros fazia melhor. Usei o Optuna
(biblioteca que propõe cada combinação nova a partir das anteriores e abandona cedo as que vão mal)
na etapa (b), com 60 combinações de 15 mil passos. A busca variou o número de camadas e de
neurônios, a ativação (tanh ou SiLU), o número e a escala das Fourier features, a taxa de
aprendizado, o peso da energia, a escala da Cauchy e o número de θ. O objetivo foi o y R² na
validação da etapa (b).

Para testar mais combinações no mesmo tempo, rodei 4 treinos juntos na GPU, cada um com 3 GB de
memória fixos. Antes medi a vazão com 1 a 4 treinos juntos (Tabela 4). Com 4 a GPU fez 28,6
combinações por hora, contra 21,0 com um só. O ganho é bem menor que 4 vezes porque os treinos
dividem a mesma GPU, e ele vem quase todo de sobrepor as partes que quase não usam a GPU, como a
inicialização, a compilação e as avaliações. Com o determinismo ligado, o mesmo treino deu
exatamente o mesmo y R² sozinho e ao lado de outros três. No meio do caminho, o banco SQLite do
Optuna entregou a mesma combinação para os 4 processos, e troquei para o arquivo de journal do
próprio Optuna, que não tem esse problema.

Tabela 4. Vazão com N treinos juntos na RTX 5070 Ti (modelo anterior, 15 mil passos).

| Treinos juntos | Tempo | Combinações por hora | Ganho | Memória da GPU (pico) |
|---|---|---|---|---|
| 1 | 171 s | 21,0 | 1,00 | 4,6 GB |
| 2 | 289 s | 24,9 | 1,19 | 8,0 GB |
| 3 | 395 s | 27,4 | 1,30 | 11,3 GB |
| 4 | 503 s | 28,6 | 1,36 | 14,6 GB |

Fonte: Autoria própria.

O estudo levou 2,3 h. O modelo anterior ficou em 25º lugar entre as 29 combinações que chegaram ao
fim, com 0,842 em 15 mil passos. A melhor chegou a 0,952, com 8 camadas de 224 neurônios, SiLU,
128 Fourier features com escala 0,30 (frequências mais baixas que as anteriores) e peso da energia
2,2. As 10 melhores têm o mesmo perfil: todas usam SiLU e peso da energia entre 1 e 5,2 (o anterior
era 10), e 9 delas usam 128 Fourier features com escala baixa. A temperatura de saída também
melhorou sem estar no objetivo, com o RMSE caindo de 22,3 para 12,0 K. Pela análise de
importância do Optuna, o parâmetro mais importante foi o peso da energia, com folga, mas a
ablação mais abaixo mostrou outra coisa.

Para confirmar, treinei as 3 melhores e o modelo anterior por 50 mil passos, com 2 seeds cada
(Tabela 5).

Tabela 5. Confirmação na etapa (b) com 50 mil passos (y R² na validação).

| Configuração | Seed 0 | Seed 1 | Média | Pior caso (média) | RMSE de T na saída (média) |
|---|---|---|---|---|---|
| Anterior (6 × 128, tanh) | 0,926 | 0,930 | 0,928 | 0,773 | 13,1 K |
| 1ª do estudo (8 × 224, SiLU) | 0,974 | 0,973 | 0,973 | 0,842 | 9,2 K |
| 2ª do estudo (7 × 192, SiLU) | 0,972 | 0,973 | 0,973 | 0,843 | 9,5 K |
| 3ª do estudo (7 × 192, SiLU) | 0,971 | 0,973 | 0,972 | 0,850 | 9,5 K |

Fonte: Autoria própria.

A melhor ficou 0,045 acima do modelo anterior, três vezes a diferença de 0,015 que duas seeds da
mesma configuração chegam a ter. As 3 do estudo ficaram a menos de 0,001 uma da outra, então
qualquer uma serviria. O pior caso continua sendo o caso 11, a frente mais abrupta da validação (a
curva de saída sobe do início da ruptura até 90% em cerca de 2 s), mas ele subiu de 0,773 para
0,842.

Na etapa (c), com os 31 parâmetros variando, testei a melhor contra a anterior nas mesmas condições
(50 mil passos, 2 seeds, outros 20 casos de validação). A melhor deu 0,941 e 0,940, contra 0,901 e
0,899 da anterior. Com só 50 mil passos ela passou o 0,932 do treino longo de 200 mil passos, que era
o platô da seção 4.3, e o pior caso da etapa (c) foi de 0,306 para 0,617. Com isso a etapa (b)
passou da meta de 0,95 e a (c) ficou perto dela.

A melhor configuração muda os 9 hiperparâmetros de uma vez. Para entender de onde vinha o ganho,
rodei 4 treinos curtos extras (15 mil passos) partindo do modelo anterior e mudando uma coisa por
vez. Só a SiLU levou o y R² de 0,842 para 0,921, só a escala menor das Fourier features para 0,879
e só o peso da energia 2,2 para 0,866. As três mudanças juntas, ainda com a rede de 6 × 128,
chegaram a 0,939, perto do 0,952 da melhor, com menos da metade do custo por passo. O caso mais
curioso foi o peso da energia: a análise de importância do Optuna colocou ele em primeiro, mas
sozinho ele foi o que menos mudou. A análise exagerou o peso da energia porque todas as
combinações com tanh também tinham peso alto (7,5 ou mais), então parte do efeito da tanh foi
parar nele. Mas o peso alto também parece atrapalhar: as duas combinações completas com SiLU e
peso acima de 10 ficaram em 0,905 e 0,701, abaixo de todas as outras com SiLU (0,910 a 0,952).

O ganho de 0,045 na (b) e de 0,040 na (c) é cerca de três vezes a diferença entre seeds, então a
melhor configuração virou o padrão das etapas nos configs (`configs/phase2_base.yaml`). Os valores
antigos ficaram em arquivos separados, que os configs das Fases 3 e 4 herdam, e conferi que cada
um deles continua dando exatamente a mesma configuração de antes. O preço é a rede maior: 476 mil
pesos em vez de 137 mil e 20,9 ms por passo em vez de 9,4 ms, ou seja, 0,30 GPU-h a cada 50 mil
passos em vez de 0,14. Rodar 4 treinos juntos economizou 1,2 h das 5,9 h que o estudo e a
confirmação levariam um por vez. O ganho foi de 1,36 vezes no estudo, de treinos curtos, e de só
1,15 a 1,18 vezes nos treinos longos da confirmação. O relatório completo está em
`docs/optuna_stage_b_results.md`.

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

O estudo Optuna escolheu as combinações pelos mesmos 20 casos de validação da etapa (b) que usei na
confirmação, então o número da (b) pode estar um pouco otimista. O teste na etapa (c), com outros
casos, é o que mostra que o ganho não vem só dessa seleção. O estudo também favorece configurações
que aprendem rápido, porque comparou treinos de só 15 mil passos.

## 6. O que falta

Antes do dataset, dois testes curtos fazem sentido. O primeiro é confirmar com 2 seeds e 50 mil
passos a versão barata da ablação (rede de 6 × 128 com SiLU, escala 0,30 e energia 2,2), cerca de
30 a 40 min. Se ela ficar a menos de 0,015 da melhor, ela vira o padrão no lugar da rede grande,
com menos da metade do custo. O segundo é um treino longo da etapa (c) com o padrão novo, para ver
quanto ele ainda ganha depois dos 50 mil passos (a configuração anterior foi de 0,900 para 0,932
com 200 mil). Com o dataset v23b, primeiro vou reavaliar a PINN física da etapa (c) nos casos
dele. Depois vem o fine-tune da híbrida, que já está pronto: ele parte da rede física pré-treinada
e acrescenta os dados de saída com λ_data = 0,1 fixo (`configs/phase3_finetune_c.yaml`). Hoje ele
parte do treino longo antigo, e o melhor é trocar pelo treino longo da configuração nova. Em
seguida vem a comparação com a MLP do outro repositório, no mesmo dataset e holdout. Também quero ver se o solver
v23b resolve o caso 21. Mais pra frente ficam mais de 2 gases (o código já aceita, falta testar) e
as 2 camadas com XPINN (uma rede por camada), que ainda não implementei. Os relatórios de cada fase
estão em `docs/phase0_results.md` a `docs/phase4_results.md`.
