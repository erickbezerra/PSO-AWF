% 1000 Iteracoes

clear all; close all; clc;
sn = 1;
snmax = 50;

while sn ~= snmax + 1

% Exemplo para FTDNN com treinamento PSOAF
% 1 horas com 7 delays em s√©rie
% Medindo o tempo gatos
tic; 
% Limpar
clearvars -except sn snmax
close all
clc
% Carregar serie temporal
load 'A.dat';

% Normalizacao
gasx = A(1:400,1)';
gasy = A(1:401,1)'; % Defino variavel y
maxgasx = max(gasx);
mingasx = min(gasx);
dgasx = (gasx-0.9*mingasx)/(1.1*maxgasx-0.9*mingasx);% Normalizar saidas
maxgasy = max(gasy);
mingasy = min(gasy);
dgasy = (gasy-0.9*mingasy)/(1.1*maxgasy-0.9*mingasy);% Normalizar saidas

% Definir numero de neuronios da camada intermediaria
PREVISAO = 9;
NNCI = 1; % Numero de Neuronios
Ninb = 5; % Numero de entradas da rede neural
Comeca = 9; % Como sao 7 delays em um intervalo de 12 pontos, comeca a previsao no 13
Limite = Comeca; % Variavel que limita o numero de amostras que sera pego e tambem utilizado na diferenca de k

% Gera populacao incial aleatoria
popinic = 120;
pop = popinic; % Usa 24 individuos (pop)
for ii = 1:pop
    w{ii} = (10-(-10))*rand(NNCI,Ninb) + (-10);% (Note [ W01 W11 W21...; W02 W12 W22... ])
    m{ii} = (10-(-10))*rand(1,NNCI+1) + (-10);% (Note [ m0 m1 m2... ]) (Pesos da camada oculta)
end

geracao = 1;
geracaomax = 1000;

% Elementos do AF
AFmax = 150; % Valor max do AF
WF = 5; % Valor do WF
ParticBorn = 120; % Numero de particulas geradas a cada geracao
% Dados abaixo nao definidos por usuario
AF = zeros(1,pop); % Matriz inicial com valores de AF
AFcont = 0; % Iniciando contador para gerar novas particulas

% Elementos PSO
n1 = 1;% Constante Cognitiva
n2 = 1;% Constante Social
% Velocidade inicial nula
for iii = 1:pop
    vw{iii} = zeros(NNCI,Ninb);
    vm{iii} = zeros(1,NNCI+1);
end
% Peso de inercia max e min
winercmin = 0.5; winercmax = 0.9;
% Reset do PSO
reset = 0;

[Meto,Neto] = size(dgasx);
% Definicao da Pbest incial e Gbest inicial
    for ii = 1:pop
        for i=1:(Neto-Comeca) % Pq como sempre preve o prox, no ultimo n√£o teria o desejado para calcular o erro
            % Calcular a aptidao
            % Fase Feedforward
            % Calcular as saidas da camada intermediaria
            % Definindo tamanho da amostra e quantos serao usados
            % Calcular entradas para funcao de ativacao
            Inpz = [ 1 dgasx(i) dgasx(i+4) dgasx(i+6) dgasx(i+7) ];
            uz = Inpz*w{ii}';
            % Calculada saida da funcao de ativacao Sigmoide Logistica
            for uzd = 1:NNCI
                z(uzd) = 1/(1+exp(-uz(1,uzd)));
            end
            % Definir matriz de entrada da camada de saida
            % z0 = 0.1 Bias camada saida
            z = [ 1 z ];
            % Calcular entrada da funcao de ativacao da camada de saida
            uy = z*m{ii}';
            % Calcular saida da camada de saida
            k = i+Comeca-1; % Sao 7 memorias em um intervalo de 7 pontos entao e sempre previsto o 7+1
            yc(k) = 1/(1+exp(-uy));
                        
            % Calcular o erro
            e(k) = dgasy(k) - yc(k);
            
            % Removo o bias de z para o calculo do prox z
            z(:,1) = [];
            
            % Calcular erro quadratico
            eQ(ii,i) = (e(k))^2;
        end
        % Calculo do erro quadratico
        MSE(1,ii) = sum(eQ(ii,:))/(size(eQ,2)-Limite);
        MSEevol(geracao,ii) = MSE(1,ii);
    end
    
    % Defino estes como Pbest
    wPbest = w;
    mPbest = m;
    MSEPbest = MSE;
    MSEGbest = min(MSE);
    % Encontrar Gbest 
    AptdMin = min(MSE); % Encontrar o valor min de aptidao
    [rAptdMin,cAptdMin] = find( MSE == AptdMin); % Encontrar o ponto X de AptdMin
    wGbest = w{cAptdMin};
    mGbest = m{cAptdMin};
    
    % Inicio da busca
[Meto,Neto] = size(dgasx);
while geracao ~= geracaomax
    for ii = 1:pop
        for i=1:(Neto-Comeca) % Pq como sempre preve o prox, no ultimo n√£o teria o desejado para calcular o erro
            % Calcular a aptidao
            % Fase Feedforward
            % Calcular as saidas da camada intermediaria
            % Definindo tamanho da amostra e quantos serao usados
            % Calcular entradas para funcao de ativacao
            Inpz = [ 1 dgasx(i) dgasx(i+4) dgasx(i+6) dgasx(i+7) ];
            uz = Inpz*w{ii}';
            % Calculada saida da funcao de ativacao Sigmoide Logistica
            for uzd = 1:NNCI
                z(uzd) = 1/(1+exp(-uz(1,uzd)));
            end
            % Definir matriz de entrada da camada de saida
            % z0 = 0.1 Bias camada saida
            z = [ 1 z ];
            % Calcular entrada da funcao de ativacao da camada de saida
            uy = z*m{ii}';
            % Calcular saida da camada de saida
            k = i+Comeca-1; % Sao 7 memorias em um intervalo de 7 pontos entao e sempre previsto o 7+1
            yc(k) = 1/(1+exp(-uy));
                        
            % Calcular o erro
            e(k) = dgasy(k) - yc(k);
            
            % Removo o bias de z para o calculo do prox z
            z(:,1) = [];
            
            % Calcular erro quadratico
            eQ(ii,i) = (e(k))^2;
        end
        % Calculo do erro quadratico
        MSE(1,ii) = sum(eQ(ii,:))/(size(eQ,2)-Limite);
        MSEevol(geracao,ii) = MSE(1,ii);
    end
    
    % Verifico se o fitness encontrado e melhor que o Pbest e o Gbest atual
    for i = 1:pop
        if MSE(1,i) < MSEPbest(1,i)
            wPbest{i} = w{i};
            mPbest{i} = m{i};
            MSEPbest(1,i) = MSE(1,i);
            AF(1,i) = AF(1,i) + 1; % Fator de envelhecimento incrementado
        else
            AF(1,i) = AF(1,i) + 1 + WF; % Fator de envelhecimento incrementado normalmente mais o fator de enfraquecimento
        end
        if MSE(1,i) < MSEGbest
            wGbest = w{i};
            mGbest = m{i};
            MSEGbest = MSE(1,i);
            AF(1,i) = 0; % Fator de envelhecimento reiniciado por se tratar da "melhor" particula
        end
    end
    
    % Encontrar AF MAX
    [rAF,cAF]=size(AF); % Econtro o numero de particulas
    for repeatAF = 1:cAF % Repeticao necessaria caso exista + de uma particula com AFmax
        [rAFmax,cAFmax] = find( AF >= AFmax); % Encontrar o ponto X de AFmax
        % Eliminando a particula com AF maximo
        w(:,cAFmax)=[];
        m(:,cAFmax)=[];
        vw(:,cAFmax)=[];
        vm(:,cAFmax)=[];
        wPbest(:,cAFmax)=[];
        mPbest(:,cAFmax)=[];
        MSE(:,cAFmax)=[];
        MSEPbest(:,cAFmax)=[];
        AF(:,cAFmax)=[];
    end
    % Atualizar o tamanho da populaÁ„o
    [rAF,cAF]=size(AF);
    pop = cAF;
    
    % Criando contagem para gerar novas particulas
    AFcont = AFcont + 1;
    if AFcont == AFmax
        for ii = (pop+1):(pop+ParticBorn)
            w{ii} = (10-(-10))*rand(NNCI,Ninb) + (-10);% (Note [ W01 W11 W21...; W02 W12 W22... ])
            m{ii} = (10-(-10))*rand(1,NNCI+1) + (-10);% (Note [ m0 m1 m2... ]) (Pesos da camada oculta)
        end
        % Velocidade inicial nula
        for iii = (pop+1):(pop+ParticBorn)
            vw{iii} = zeros(NNCI,Ninb);
            vm{iii} = zeros(1,NNCI+1);
        end
        % Adicionando novos AF
        AF = [AF zeros(1,ParticBorn)];
        
       % Preparando nova matriz de wPbest e mPbest
       % Defino estes como Pbest
       wPbest = w;
       mPbest = m;
       
       % Definicao da Pbest incial e Gbest inicial para novas particulas
        for ii = (pop+1):(pop+ParticBorn)
            for i=1:(Neto-Comeca) % Pq como sempre preve o prox, no ultimo n√£o teria o desejado para calcular o erro
                % Calcular a aptidao
                % Fase Feedforward
                % Calcular as saidas da camada intermediaria
                % Definindo tamanho da amostra e quantos serao usados
                % Calcular entradas para funcao de ativacao
                Inpz = [ 1 dgasx(i) dgasx(i+4) dgasx(i+6) dgasx(i+7) ];
                uz = Inpz*w{ii}';
                % Calculada saida da funcao de ativacao Sigmoide Logistica
                for uzd = 1:NNCI
                    z(uzd) = 1/(1+exp(-uz(1,uzd)));
                end
                % Definir matriz de entrada da camada de saida
                % z0 = 0.1 Bias camada saida
                z = [ 1 z ];
                % Calcular entrada da funcao de ativacao da camada de saida
                uy = z*m{ii}';
                % Calcular saida da camada de saida
                k = i+Comeca-1; % Sao 7 memorias em um intervalo de 7 pontos entao e sempre previsto o 7+1
                yc(k) = 1/(1+exp(-uy));
                
                % Calcular o erro
                e(k) = dgasy(k) - yc(k);
                
                % Removo o bias de z para o calculo do prox z
                z(:,1) = [];
                
                % Calcular erro quadratico
                eQ(ii,i) = (e(k))^2;
            end
            % Calculo do erro quadratico
            MSE(1,ii) = sum(eQ(ii,:))/(size(eQ,2)-Limite);
            MSEevol(geracao,ii) = MSE(1,ii);
        end
        
        % Verifico se o fitness encontrado e melhor que o Pbest e o Gbest atual
        MSEPbest = MSE;
        for z = (pop+1):(pop+ParticBorn)
             if MSE(1,z) < MSEGbest
                wGbest = w{z};
                mGbest = m{z};
                MSEGbest = MSE(1,z);               
            end
        end
    
        AFcont = 0; % Reinicio contador para nova geracao
        pop = pop+ParticBorn; % Atualizar o tamanho da populaÁ„o
       
    end
    
    % Reducao linear da ponderacao de inercia
    winerc = winercmax - geracao*(winercmax-winercmin)/geracaomax;
    
    % Atualizar velocidades e posicoes
    for i = 1:pop
        vw{i} = winerc*vw{i} + n1*rand()*(wPbest{i} - w{i}) + n2*rand()*(wGbest -w{i});
        w{i} = w{i} + vw{i};
        vm{i} = winerc*vm{i} + n1*rand()*(mPbest{i} - m{i}) + n2*rand()*(mGbest -m{i});
        m{i} = m{i} + vm{i};
    end
    
%     % Reset do PSO caso caia em um min local
%     if geracao == 80 & min(MSE) > 0.001 & reset <= 3
%         geracao = 1;
%         for ii = 1:pop
%             w{ii} = (10-(-10))*rand(NNCI,Ninb) + (-10);% (Note [ W01 W11 W21...; W02 W12 W22... ])
%             m{ii} = (10-(-10))*rand(1,NNCI+1) + (-10);% (Note [ m0 m1 m2... ]) (Pesos da camada oculta)
%         end
%         reset = reset + 1 ;
%     end
    
    geracao = geracao +1;
end

% Ultimo MSE
MSEtr = MSEGbest;

% Desvpad MSEte
STDtr = std(MSE);

% Agora define o melhor w e o melhor m
w = []; m = [];
w = wGbest; m = mGbest;

% Etapa de execucao da rede
% Normalizacao
gasxV = A(551:951,1)';
gasyV = A(551:952,1)'; % Defino variavel ymaxgasxV = max(gasxV);
maxgasxV = max(gasxV);
mingasxV = min(gasxV);
dgasxV = (gasxV-0.9*mingasxV)/(1.1*maxgasxV-0.9*mingasxV);% Normalizar saidas
maxgasyV = max(gasyV);
mingasyV = min(gasyV);
dgasyV = (gasyV-0.9*mingasyV)/(1.1*maxgasyV-0.9*mingasyV);% Normalizar saidas

[mdyv,ndyv]=size(dgasyV);
ycv = zeros(1,ndyv);

% Limpar variaveis
Inpz = [];
z = [];
k = [];

% Fase Feedforward
% Calcular as saidas da camada intermediaria
[Meto2,Neto2] = size(dgasxV);
for i=1:(Neto2-Comeca) %Pq como sempre preve o prox, no ultimo n√£o teria o desejado para calcular o erro
    % Calcular entradas para funcao de ativacao
    Inpz = [ 1 dgasxV(i) dgasxV(i+4) dgasxV(i+6) dgasxV(i+7) ];% 1 Devido ao Bias e outro a faixa de previsao de 1h
        uz = Inpz*w';
        % Calculada saida da funcao de ativacao Sigmoide Logistica
        for uzd = 1:NNCI
        z(uzd) = 1/(1+exp(-uz(1,uzd)));
        end
        % Definir matriz de entrada da camada de saida
        % z0 = 0.1 Bias camada saida
        z = [ 1 z ];
        % Calcular entrada da funcao de ativacao da camada de saida
        uy = z*m';
        % Calcular saida da camada de saida
        k = i+Comeca-1;
        ycv(k) = 1/(1+exp(-uy));
        % Removo o bias de z para o calculo do prox z
        z(:,1) = [];
end

% Remove normalizacao
%x = dx(maxx-minx)/(x-minx);% Normalizar entradas
%dy = (y-miny)/(maxy-miny);% Normalizar saidas

% Calculo do erro bruto acumulado
for i = Comeca:size(dgasyV,2)
    ErroBruto(i) = abs(dgasyV(1,i)-ycv(1,i));
end
ErroBrutoTotal = sum(ErroBruto);

% Calculo do mean absolute error (MAE) (Forecast-Real)/n
MAEV = ErroBrutoTotal/(size(ErroBruto,2)-Limite);

[m,n] = size(ErroBruto);
% Calculo do MSE
for i = Comeca:n
    eQV(i) = (ErroBruto(i))^2;
end
% Calculo do mean square error (MSE)
jv=0;
for i = Comeca:n
    MSEV(i) = sum(eQV(Comeca:i))/(size(eQV(Comeca:i),2)-Limite);    
end

% Ultimo MSE
MSEte = MSEV(1,size(MSEV,2));

% Desvpad MSEte
STDte = std(MSEV(10:n));

% Calculo do mean absolute percentage error (MAPE)
for i = Comeca:n
    mapetemp(i) = ErroBruto(1,i)/dgasyV(1,i);
end
MAPEV = sum(mapetemp)/(size(mapetemp,2)-Limite);

% Plotar grafico de resultados
plot(dgasyV,'-*'); hold; plot(ycv,'-rd');

%Medindo tempo gasto
tempog = toc;

% Tabela para analise dos dados
TAD = [ tempog MSEtr STDtr MSEte STDte ];

% Gravando arquivo .mat
outfile = ['psoawf1N1000IartigoLGD',int2str(sn-1)];
save(outfile);
sn = sn+1;
end
