classdef DeterministicLROptimizer < DeterministicOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        lr = 1e-3;
        
        % log
        log;
    end
    
    methods
        function opt = DeterministicLROptimizer(model, lr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt@DeterministicOptimizer(model);
            if nargin >= 2 && ~isempty(lr)
                opt.lr = lr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
            
            opt.update_params = struct('lr', [1/4 2]);
            opt.log = Log('lr', 'reductionratio');
        end
        
        function plot(opt)
%             figure(1);
%             plot(getfield(opt.log, 'loss')), hold on

%             figure(2);
            subplot(2,1,1); plot(max(getfield(opt.log, 'reductionratio'),0)), hold on
            subplot(2,1,2); semilogy(getfield(opt.log, 'lr')), hold on
        end
        
        function newloss = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            lr_ = opt.lr;
            
            [step, predchange] = compute_step(opt);
            make_step(opt, lr_*step);
            predchange = lr_*predchange;
            
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            if ~update(opt, reductionratio, true)
                make_step(opt, -lr_*step); % unmake step
                newloss = loss;
            end
            append(opt.log, lr_, reductionratio)
        end
        
        function accept = update(opt, reductionratio, limited)
            if reductionratio < 0.25
                opt.lr = opt.lr*opt.lr_decrease;
            elseif reductionratio > 0.75 && limited
                opt.lr = opt.lr*opt.lr_increase;
                opt.lr = min(opt.lr, opt.lr_max);
            end
            accept = (reductionratio > opt.rejection_threshold);
        end
    end
    
end