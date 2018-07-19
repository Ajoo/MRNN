classdef RNNLinearRegressor < handle
    %RNNLINEARREGRESSOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rnn
        Wo
        b
        loss
        dloss
        d2loss
    end
    properties (Dependent)
        params
        paramsize
    end
    properties %(Access=private)
        h_end_
        dl_
        d2l_
        y_
    end
    
    methods
        function mdl = RNNLinearRegressor(rnn, loss)
            %RNNLINEARREGRESSOR Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 2
                loss = 'MSE';
            end
            mdl.rnn = rnn;
            nh = rnn.hiddensize;
            stddev = 1/sqrt(nh);
            mdl.Wo = (2*rand(1,nh)-1)*stddev;
            mdl.b = 0;
            switch upper(loss)
                case 'MSE'
                    mdl.loss = @(y, yh) sum((y-yh).^2)/2;
                    mdl.dloss = @(y, yh) yh-y;
                    mdl.d2loss = @(y, yh) []; % empty interpreted as Id
            end
        end
        
        function [yh, l] = evaluate(mdl, x, y)
            h_end = evaluate(mdl.rnn, x);
            yh = mdl.Wo*h_end(:,:) + mdl.b;
            l = mdl.loss(y, yh);
        end
        
        function [yh, l] = call(mdl, x, y)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            h_end = call(mdl.rnn, x);
            yh = mdl.Wo*h_end(:,:) + mdl.b;
            l = mdl.loss(y, yh);
            
            % setup buffers
            mdl.h_end_ = h_end;
            mdl.dl_ = mdl.dloss(y, yh);
            mdl.d2l_ = mdl.d2loss(y,yh);
            mdl.y_ = y;
        end
        
        function [yh, l] = recall(mdl)
            h_end = recall(mdl.rnn);
            yh = mdl.Wo*h_end(:,:) + mdl.b;
            l = mdl.loss(mdl.y_, yh);
            
            mdl.h_end_ = h_end;
            mdl.dl_ = mdl.dloss(mdl.y_, yh);
            mdl.d2l_ = mdl.d2loss(mdl.y_,yh);
        end
        
        function [yh_v, l_v] = fdiff(mdl, v)
            nrnnp = mdl.rnn.paramsize;
            h_end_vrnn = fdiff(mdl.rnn, v(1:nrnnp));
            
            yh_vrnn = mdl.Wo*h_end_vrnn;
            v_o = v(mdl.rnn.paramsize+1:end-1)';
            yh_v = (yh_vrnn + v_o*mdl.h_end_ + v(end));
            l_v = mdl.dl_*yh_v';
        end
        
        function u_yh = bdiff(mdl, u, ul)
            if nargin < 2 || isempty(u)
                u = zeros(size(mdl.dl_));
            end
            if nargin >= 3
            	u = u + ul*mdl.dl_;
            end
            u_yh = [bdiff(mdl.rnn, mdl.Wo'*u);
                    mdl.h_end_*u';
                    sum(u)];
        end
        
        % TODO: versions of bdiff and fdiff that don't include loss
        function [gv, n2jv] = gvp(mdl, v)
            yh_v = fdiff(mdl, v);
            if ~isempty(mdl.d2l_) % empty interpreted as id
                yh_v_H = yh_v*mdl.d2l_;
            else
                yh_v_H = yh_v;
            end
            n2jv = yh_v_H*yh_v';
            gv = bdiff(mdl, yh_v_H);
        end
        
        function f = fimdiag(mdl, sample)
            % Computes the diagonal of the FIM
            % optionally, approximates it based on a random subsample
            % of the batch
            nb = numel(mdl.dl_);
            if nargin < 2
                I = 1:nb;
                w = 1;
            else
                I = randperm(nb, sample);
                w = nb/sample;
            end
            np = mdl.paramsize;
            f = zeros(np,1);
            for i=I
                u = zeros(1, nb); u(i) = 1;
                f = f + w*bdiff(mdl, u).^2;
            end
        end
        
        function np = get.paramsize(mdl)
            np = mdl.rnn.paramsize + numel(mdl.Wo) + 1;
        end
        function set.params(mdl, p)
            np = mdl.rnn.paramsize;
            no = numel(mdl.Wo);
            mdl.rnn.params = p(1:np);
            mdl.Wo(:) = p(np+1:np+no);
            mdl.b = p(np+no+1);
        end
        function p = get.params(mdl)
            p = [mdl.rnn.params; mdl.Wo(:); mdl.b];
        end
    end
end

