classdef DeterministicOptimizer < ModelOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        lr = 1e-3;
        
        % acceptence test
        lr_max = Inf;
        lr_increase = 2;
        lr_decrease = 1/4;
        rejection_threshold = 0;
        
        log
    end
    
    methods
        function opt = DeterministicOptimizer(model, lr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt@ModelOptimizer(model);
            if nargin >= 2 && ~isempty(lr)
                opt.lr = lr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
            
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
            
            [step, predchange] = computestep(opt);
            make_step(opt, lr_*step);
            
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            if ~update(opt, reductionratio)
                opt.model.params = opt.model.params - step;
                newloss = loss;
            end
            append(opt.log, lr_, reductionratio)
        end
        
        function accept = update(opt, reductionratio)
            if reductionratio < 0.25
                opt.lr = opt.lr*opt.lr_decrease;
            elseif reductionratio > 0.75
                opt.lr = opt.lr*opt.lr_increase;
                opt.lr = min(opt.lr, opt.lr_max);
            end
            accept = (reductionratio > opt.rejection_threshold);
        end
    end
    
end