classdef StochasticOptimizer < ModelOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        lr = 1e-3;
        log
    end
    
    methods
        function opt = StochasticOptimizer(model, lr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt@ModelOptimizer(model);
            if nargin >= 2 && ~isempty(lr)
                opt.lr = lr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
            
            opt.log = Log('lr');
        end
        
        function plot(opt)
%             figure(1);
%             plot(getfield(opt.log, 'loss')), hold on

%             figure(2);
            subplot(2,1,1); plot(max(getfield(opt.log, 'reductionratio'),0)), hold on
            subplot(2,1,2); semilogy(getfield(opt.log, 'lr')), hold on
        end
        
        function step(opt)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            lr_ = opt.lr;
            
            make_step(opt, lr_*compute_step(opt));
            append(opt.log, lr_)
        end
        
    end
end