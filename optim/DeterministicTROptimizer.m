classdef DeterministicTROptimizer < DeterministicOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        tr = 1e-3;
                
        log;
    end
    
    methods
        function opt = DeterministicTROptimizer(model, tr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt@DeterministicOptimizer(model);
            if nargin >= 2 && ~isempty(tr)
                opt.tr = tr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
            
            opt.update_params = struct('tr', [1/4 2]);
            opt.log = Log('tr', 'reductionratio');
        end
        
        function plot(opt)
%             figure(2);
            subplot(2,1,1); plot(max(getfield(opt.log, 'reductionratio'),0)), hold on
            subplot(2,1,2); semilogy(getfield(opt.log, 'tr')), hold on
        end
        
        function newloss = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            tr_ = opt.tr;
            
            [step, predchange, limited] = compute_step(opt, tr_);
            
            make_step(opt, step);
            [~, newloss] = recall(opt.model);
            reductionratio = (newloss-loss)/predchange;
            
            if ~update(opt, reductionratio, limited)
                make_step(opt, -step); % unmake step
                newloss = loss;
            end
            
            append(opt.log, tr_, reductionratio)
        end

    end
    
end