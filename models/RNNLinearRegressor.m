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
        fullseq
    end
    properties (Dependent)
        params
        paramsize
    end
    properties %(Access=private)
        h_
        dl_
        d2l_
        y_
    end
    
    methods
        function mdl = RNNLinearRegressor(rnn, loss, fullseq)
            %RNNLINEARREGRESSOR Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 3 || isempty(fullseq)
                fullseq = false;
            end
            if nargin < 2 || isempty(loss)
                loss = 'MSE';
            end
            mdl.rnn = rnn;
            nh = rnn.hiddensize;
            stddev = 1/sqrt(nh);
            mdl.Wo = (2*rand(1,nh)-1)*stddev;
            mdl.b = 0;
            switch upper(loss)
                case 'MSE'
                    mdl.loss = @(y, yh) sum(vec((y-yh).^2))/2;
                    mdl.dloss = @(y, yh) yh-y;
                    mdl.d2loss = @(y, yh) []; % empty interpreted as Id
            end
            mdl.fullseq = fullseq;
        end
        
        function [yh, l] = evaluate(mdl, x, y)
            if mdl.fullseq
                [~, h] = evaluate(mdl.rnn, x);
            else
                h = evaluate(mdl.rnn, x);
            end
            sh = size(h);
            yh = reshape(mdl.Wo*h(:,:) + mdl.b, [1, sh(2:end)]);
            l = mdl.loss(y, yh);
        end
        
        function [yh, l] = call(mdl, x, y)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            if mdl.fullseq
                [~, h] = call(mdl.rnn, x);
            else
                h = call(mdl.rnn, x);
            end
            sh = size(h);
            yh = reshape(mdl.Wo*h(:,:) + mdl.b, [1, sh(2:end)]);
            l = mdl.loss(y, yh);
            
            % setup buffers
            mdl.h_ = h;
            mdl.dl_ = mdl.dloss(y, yh);
            mdl.d2l_ = mdl.d2loss(y,yh);
            mdl.y_ = y;
        end
        
        function [yh, l] = recall(mdl)
            if mdl.fullseq
                [~, h] = recall(mdl.rnn);
            else
                h = recall(mdl.rnn);
            end
            sh = size(h);
            yh = reshape(mdl.Wo*h(:,:) + mdl.b, [1, sh(2:end)]);
            l = mdl.loss(mdl.y_, yh);
            
            mdl.h_ = h;
            mdl.dl_ = mdl.dloss(mdl.y_, yh);
            mdl.d2l_ = mdl.d2loss(mdl.y_,yh);
        end
        
        function [yh_v, l_v] = fdiff(mdl, v)
            nrnnp = mdl.rnn.paramsize;
            if mdl.fullseq
                [~, h_vrnn] = fdiff(mdl.rnn, v(1:nrnnp));
            else
                h_vrnn = fdiff(mdl.rnn, v(1:nrnnp));
            end
            sh = size(h_vrnn);
            yh_vrnn = mdl.Wo*h_vrnn(:,:);
            v_o = v(mdl.rnn.paramsize+1:end-1)';
            yh_v = (yh_vrnn + v_o*mdl.h_(:,:) + v(end));
            l_v = yh_v*mdl.dl_(:);
            yh_v = reshape(yh_v, [1, sh(2:end)]);
        end
        
        function u_yh = bdiff(mdl, u, ul)
            if nargin < 2 || isempty(u)
                u = zeros(size(mdl.dl_));
            end
            if nargin >= 3
            	u = u + ul*mdl.dl_;
            end
            if mdl.fullseq
                u_h = bdiff(mdl.rnn, 0, mdl.Wo.'.*u);
            else
                u_h = bdiff(mdl.rnn, mdl.Wo'*u);
            end
            u_yh = [u_h;
                    mdl.h_(:,:)*u(:);
                    sum(u(:))];
        end
        
        % TODO: versions of bdiff and fdiff that don't include loss
        function [gv, n2jv] = gvp(mdl, v)
            yh_v = fdiff(mdl, v);
            if ~isempty(mdl.d2l_) % empty interpreted as id
                yh_v_H = yh_v*mdl.d2l_;
            else
                yh_v_H = yh_v;
            end
            n2jv = yh_v_H(:)'*yh_v(:);
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

function m = vec(M)
    m = M(:);
end
