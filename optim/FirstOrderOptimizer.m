classdef FirstOrderOptimizer < handle
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        model
        lr = 1e-3;
        % regularization
        l2 = 0;
        weightdecay = 0;
        
        % acceptence test
        accept = false;
        lr_max = Inf;
        lr_increase = 2;
        lr_decrease = 1/4;
        rejection_threshold = 0;
        
        log = Log('lr', 'reductionratio');
    end
    
    methods
        function opt = FirstOrderOptimizer(model, lr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            if nargin >= 2 && ~isempty(lr)
                opt.lr = lr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
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
            l2_ = opt.l2;
            weightdecay_ = opt.weightdecay;
            
            g = bdiff(opt.model, [], 1);
            if l2_ > 0
                g = g + l2_*opt.model.params;
            end
            step = lr_*computestep(opt, g);
            
            if weightdecay_ > 0
                opt.model.params = (1-lr_*weightdecay_)*opt.model.params;
            end
            opt.model.params = opt.model.params + step;
            
            newloss = loss;
            if opt.accept
                [~, newloss] = recall(opt.model);
                predchange = g'*step;
                reductionratio = (newloss-loss)/predchange;
                if ~update(opt, reductionratio)
                    opt.model.params = opt.model.params - step;
                    newloss = loss;
                end
            else
                reductionratio = NaN;
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
    
    methods (Abstract)
        step = computestep(opt, g)
    end
end