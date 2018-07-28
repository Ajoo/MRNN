T = 30;
BATCH_SIZE = 1000;
HIDDEN_SIZE = 100;

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
p0 = mdl.params;

opt = ADAMOptimizer(mdl, 2e-3);
% opt = SGDOptimizer(mdl, 1e-3, 'momentum', 0.6);

%%
[x, y] = sampleaddition(BATCH_SIZE, T);
[xval, yval] = sampleaddition(BATCH_SIZE, T);

[~, loss] = call(mdl, x, y);
[~, valloss] = evaluate(mdl, xval, yval);
i = 1;
%%
N = 500;
DECIMATION = 50;
loss = [loss; zeros(N,1)];

for i=i:i+N
    step(opt);
    [~, loss(i)] = recall(mdl);
    if mod(i, DECIMATION) == 0
        [~, valloss(end+1)] = evaluate(mdl, xval, yval);
        fprintf('Iter: %i, Train Error: %1.4f, Val Error: %1.4f\n', ...
            i, 2*loss(i)/BATCH_SIZE, 2*valloss(end)/BATCH_SIZE);
    end
end
% disp(['loss: ', num2str(2*l/BATCH_SIZE)])
% disp(['rho: ', num2str(rho)])
% disp(['iter: ', num2str(iter)])
% disp(['damping: ', num2str(opt.state.damping)])
% disp(['step size: ', num2str(norm(opt.state.previousstep))])


%%
plot(2*loss/BATCH_SIZE);
hold on;
plot(1:DECIMATION:numel(loss), 2*valloss/BATCH_SIZE, '.-')
