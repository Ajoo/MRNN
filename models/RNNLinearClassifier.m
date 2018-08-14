classdef RNNLinearClassifier < handle
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
        classsize
    end
    properties %(Access=private)
        h_
        p_
        dl_
        %d2l_
        %y_
    end
    
    methods
        function mdl = RNNLinearClassifier(rnn, K, loss, fullseq)
            %RNNLINEARREGRESSOR Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 3 || isempty(fullseq)
                fullseq = false;
            end
            if nargin < 2 || isempty(loss)
                loss = 'CROSSENTROPY';
            end
            mdl.rnn = rnn;
            nh = rnn.hiddensize;
            stddev = 1/sqrt(nh);
            mdl.Wo = (2*rand(K, nh)-1)*stddev;
            mdl.b = zeros(K, 1);
            switch upper(loss)
                case 'CROSSENTROPY'
                    mdl.loss = @score_crossentropy;
                    mdl.dloss = @(y, p) p-y;
                    mdl.d2loss = @d2crossentropy_vprod; % empty interpreted as Id
            end
            mdl.fullseq = fullseq;
        end
        
        function [a, l] = evaluate(mdl, x, y)
            if mdl.fullseq
                [~, h] = evaluate(mdl.rnn, x);
            else
                h = evaluate(mdl.rnn, x);
            end
            sh = size(h);
            nk = mdl.classsize;
            a = reshape(mdl.Wo*h(:,:) + mdl.b, [nk, sh(2:end)]); %scores
            [l, ~] = mdl.loss(y, a);
        end
        
        function [a, l] = call(mdl, x, y)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            if mdl.fullseq
                [~, h] = call(mdl.rnn, x);
            else
                h = call(mdl.rnn, x);
            end
            sh = size(h);
            nk = mdl.classsize;
            a = reshape(mdl.Wo*h(:,:) + mdl.b, [nk, sh(2:end)]); %scores
            [l, p] = mdl.loss(y, a);
            
            % setup buffers
            mdl.h_ = h;
            mdl.p_ = p;
            mdl.dl_ = mdl.dloss(y, p);
%             mdl.d2l_ = mdl.d2loss(y, p);
%             mdl.y_ = y;
        end
        
        function [a, l] = recall(mdl)
            if mdl.fullseq
                [~, h] = recall(mdl.rnn);
            else
                h = recall(mdl.rnn);
            end
            sh = size(h);
            nk = mdl.classsize;
            a = reshape(mdl.Wo*h(:,:) + mdl.b, [nk, sh(2:end)]); %scores
            [l, p] = mdl.loss(y, a);
            
            % setup buffers
            mdl.h_ = h;
            mdl.p_ = p;
            mdl.dl_ = mdl.dloss(y, p);
%             mdl.d2l_ = mdl.d2loss(y, p);
%             mdl.y_ = y;
        end
        
        function [a_v, l_v] = fdiff(mdl, v)
            nrnnp = mdl.rnn.paramsize;
            nk = mdl.classsize;
            if mdl.fullseq
                [~, h_vrnn] = fdiff(mdl.rnn, v(1:nrnnp));
            else
                h_vrnn = fdiff(mdl.rnn, v(1:nrnnp));
            end
            sh = size(h_vrnn);
            a_vrnn = mdl.Wo*h_vrnn(:,:);
            v_o = reshape(v(nrnnp+1:nrnnp+nk*sh(1)), nk, sh(1));
            v_b = v(end-nk+1:end);
            a_v = (a_vrnn + v_o*mdl.h_(:,:) + v_b(:));
            
            l_v = a_v(:)'*mdl.dl_(:);
            a_v = reshape(a_v, [nk, sh(2:end)]);
        end
        
        function u_yh = bdiff(mdl, u, ul)
            if nargin < 2 || isempty(u)
                u = zeros(size(mdl.dl_));
            end
            if nargin >= 3
            	u = u + ul*mdl.dl_;
            end
            if mdl.fullseq
                su = size(u);
                nh = mdl.rnn.hiddensize;
                u_rnn = reshape(mdl.Wo'*u(:,:), [nh, su(2:end)]);
                u_h = bdiff(mdl.rnn, 0, u_rnn);
            else
                u_h = bdiff(mdl.rnn, mdl.Wo'*u);
            end
            u_yh = [u_h;
                    reshape(u(:,:)*mdl.h_(:,:)', [], 1);
                    sum(u(:,:), 2)];
        end
        
        % TODO: versions of bdiff and fdiff that don't include loss
        function [gv, n2jv] = gvp(mdl, v)
            a_v = fdiff(mdl, v);
            if ~isempty(mdl.d2loss) % empty interpreted as id
                a_v_H = mdl.d2loss(mdl.p_, a_v);
            else
                a_v_H = a_v;
            end
            n2jv = a_v_H(:)'*a_v(:);
            gv = bdiff(mdl, a_v_H);
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
        
        function nk = get.classsize(mdl)
            nk = size(mdl.Wo, 1);
        end
        
        function np = get.paramsize(mdl)
            np = mdl.rnn.paramsize + numel(mdl.Wo) + numel(mdl.b);
        end
        function set.params(mdl, p)
            np = mdl.rnn.paramsize;
            no = numel(mdl.Wo);
            mdl.rnn.params = p(1:np);
            mdl.Wo(:) = p(np+1:np+no);
            mdl.b = p(np+no+1:end);
        end
        function p = get.params(mdl)
            p = [mdl.rnn.params; mdl.Wo(:); mdl.b];
        end
    end
end

function [Ha] = d2crossentropy_vprod(p, da)
    Ha = p.*da;
    Ha = Ha - p.*sum(Ha, 1);
end

function [l, p] = score_crossentropy(y, a)
    s = logsumexp(a, 1);
    p = exp(a - s);
    s = s - sum(y.*a, 1);
    l = sum(s(:));
end

function s = logsumexp(a, dim)
    if nargin < 2
        dim = 1;
    end
    max_a = max(a, [], dim);
    s = max_a + log(sum(exp(a - max_a), dim));
end

function m = vec(M)
    m = M(:);
end
