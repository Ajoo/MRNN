classdef IndRNN < RNN
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    properties
        whlimit;
    end
    
    methods
        function rnn = IndRNN(inputsize, hiddensize, whlimit, activation)
            if nargin < 4
                activation = 'relu';
            end
            if nargin < 3
                whlimit = Inf;
            end
            rnn@RNN(inputsize, hiddensize, activation);
            rnn.whlimit = whlimit;
            
            over_limit = abs(rnn.Wh) > rnn.whlimit;
            rnn.Wh(over_limit) = sign(rnn.Wh(over_limit))*rnn.whlimit;
%             over_gamma = abs(wh) > rnn.gamma;
%             wh(over_gamma) = sign(wh(over_gamma))*gamma;
        end
        function initialize(rnn, inputsize, hiddensize)
            stddev = 1/sqrt(hiddensize);
            wh = 2*rand(hiddensize,1)-1;
            
            rnn.Wh = spdiags(wh, 0, hiddensize, hiddensize);
            rnn.Wx = (2*rand(hiddensize, inputsize)-1)*stddev;
            rnn.b = (2*randn(hiddensize,1)-1)*stddev;
        end
        
        function [Vh, Vx, vb] = getweights(rnn, v)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            V = reshape(v, nh, []);
            Vh = spdiags(V(:,1), 0, nh, nh);
            Vx = V(:,2:1+nx);
            vb = V(:,1+nx+1);
        end
        function p = set_params_hook(rnn, p)
            % introduce modifications to p prior to assignment
            nh = rnn.hiddensize;
            over_limit = find(p(1:nh) > rnn.whlimit);
            p(over_limit) = sign(p(over_limit))*rnn.whlimit;
        end

        function v = vecweights(rnn, V)
            nh = rnn.hiddensize;
            if nargin == 1
                v = [spdiags(rnn.Wh, 0); rnn.Wx(:); rnn.b];
            else
                v = [spdiags(V, 0); reshape(V(:,nh+1:end), [], 1)];
            end
        end
        function np = getparamsize(rnn)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            np = nh*(nx+2);            
        end
    end
end



