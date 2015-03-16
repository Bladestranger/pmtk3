%% Test of continuous HMM
% Dmytro Bobkov

setSeed(100);
nHidStates = 2; 
% T =                  [1/6  1/6   1/6   1/6   1/6   1/6  ;  
%                      1/10  1/10  1/10  1/10  1/10  5/10 
%                      2/6   1/6   1/6   1/6    1/12  1/12
%                      7/12   1/12  1/12  1/12  1/12 1/12];  
% trueModel.emission = tabularCpdCreate(T);                  
    
% trueModel.A = [0.6 0.15 0.20 0.05;
%               0.10 0.70 0.15 0.05
%               0.10 0.30 0.10 0.50
%               0.30 0.10 0.30 0.30];

len = 10000;
       
trueModel.A = [ 0.93 0.07;
                0.05 0.86];

% trueModel.emission.mu = repmat(0.2, 1, len);% 0.8];
% trueModel.emission.Sigma = reshape(repmat(0.1, 1, len), 1, 1, len);%[0.1 0; 0 0.1];
% trueModel.emission.d = 1;

% trueModel.emission.mu = repmat([0.2; 0.8], 1, len);% 0.8];
% trueModel.emission.mu = [0.2 0.8];% 0.8];

trueModel.emission.mu = [0.2 0.8;
                         0.3 0.7];
% trueModel.emission.Sigma = reshape([0.15; 0.15], 2, 1, 1);
% trueModel.emission.Sigma(:,:,1) = [0.15 0.15];
% Sigma = [[0.15 0; 0 0.15]];
% trueModel.emission.Sigma = [0.1 0.15];
trueModel.emission.Sigma = [[0.1 0.15;
                            0.15 0.1];
                            [0.1 0.15;
                            0.15 0.1];];

% Sigma = [   0.15 0;
%             0   0.15];
% reshaped = repmat(Sigma, 1, len);
% trueModel.emission.Sigma = reshape(reshaped, size(Sigma,1), size(Sigma,2), len);%[0.1 0; 0 0.1];
trueModel.emission.d = 2;

%%
% d     = model.emission.d; % model.d;
% E     = model.emission;
% mu    = E.mu; 
% Sigma = E.Sigma; 
          
% trueModel.pi = [0.8 0.1 0.1 0];
trueModel.pi = [0.2 0.8;
                0.3 0.7];
trueModel.type = 'gauss';
%% Sample

[observed, labels] = hmmSample(trueModel, len);

% labels = hidden;


% observed = sequence.observation;
% siz = 10;
% load('/home/ga56den/Work/workspace/pedometer/HMM/statisticsStaticVsDynamic.mat', 'sequence', 'labelNames');
% len = 28;
% % sequence = HMM_mergeSequence(sequence);
% sequence = sequence{1};
% observed = sequence.observation;
% observed = observed / max(observed);
% observed = exp(-10*observed);

% observed = exp(observed);
% labels = sequence.labels;


%
%% Learn the model using EM with random restarts
nrestarts = 2;
modelEM = hmmFit(observed, nHidStates, 'gauss', ...
    'convTol', 1e-5, 'nRandomRestarts', nrestarts, 'verbose', true);

%%
data{1,1} = [0.3 0.3 0.3 0.1 0.3 0.3 0.3 0.3 0.3 0.3; 
             0.3 0.8 0.8 0.8 0.9 0.9 0.86 0.91 0.99 0.88];
data{2,1} = [0.23 0.33 0.13 0.15 0.33 0.32 0.34 0.35 0.36 0.31; 
             0.3 0.8 0.8 0.8 0.9 0.9 0.86 0.91 0.99 0.88];
         
data{3,1} = randn(2,100);

d = 2;

% test with a bogus prior
prior.mu = ones(1, d);
prior.Sigma = 0.1*eye(d);
prior.k = d;
prior.dof = prior.k + 1;


model = hmmFitEm(data, 2, 'gauss', 'verbose', true, 'piPrior', [3 2], ...
    'emissionPrior', prior, 'nRandomRestarts', 2, 'maxIter', 10);




% modelEM = hmmFit(data, nHidStates, 'gauss', ...
%     'convTol', 1e-5, 'nRandomRestarts', nrestarts, 'verbose', true);



% modelEM = hmmFit(data, nHidStates, 'gauss', ...
%     'convTol', 1e-5, 'nRandomRestarts', nrestarts, 'verbose', true);



%%
fprintf('estimated mu %f %f\n', modelEM.emission.mu(1), modelEM.emission.mu(2));
fprintf('estimated Sigma %f %f\n', modelEM.emission.Sigma(1), modelEM.emission.Sigma(2));

N = len;
% N = 100;
decodedFromViterbi = hmmMap(modelEM, observed(1:N));
norm(decodedFromViterbi - labels(1:N), 1) / numel(labels(1:N))

return;


%% How different are the respective log probabilities?
fprintf('trueModel LL: %g\n', hmmLogprob(trueModel, observed));
fprintf('emModel LL: %g\n', hmmLogprob(modelEM, observed)); 

%% Decode using true model
decodedFromTrueViterbi = hmmMap(trueModel, observed);
decodedFromTrueViterbi = bestPermutation(decodedFromTrueViterbi, hidden);
trueModelViterbiError = mean(decodedFromTrueViterbi ~= hidden)

decodedFromTrueMaxMarg = maxidx(hmmInferNodes(trueModel, observed), [], 1);
decodedFromTrueMaxMarg = bestPermutation(decodedFromTrueMaxMarg, hidden);
trueModelMaxMargError = mean(decodedFromTrueMaxMarg ~= hidden)

%% Decode using the EM model
decodedFromEMviterbi = hmmMap(modelEM, observed);
decodedFromEMviterbi = bestPermutation(decodedFromEMviterbi, hidden);

emModelViterbiError = mean(decodedFromEMviterbi ~= hidden)

decodedFromEMmaxMarg = maxidx(hmmInferNodes(modelEM, observed), [], 1);
decodedFromEMmaxMarg = bestPermutation(decodedFromEMmaxMarg, hidden);

emModelMaxMargError = mean(decodedFromEMmaxMarg ~= hidden)



