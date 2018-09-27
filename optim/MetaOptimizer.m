classdef MetaOptimizer < handle
    %METAOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        opt
        gamma = 0.1;
        nsubsteps = 10;
        
        log = Log('l0', 'lf');
    end
    
    methods
        function mopt = MetaOptimizer(opt)
            mopt.opt = opt;
        end
        
        function plot(mopt)
            semilogy([getfield(mopt.log, 'l0') getfield(mopt.log, 'lf')])
        end
        
        function [l0, lf] = step(mopt, x, y)
            reset(mopt.opt); % reset optmizer state
            [~, l0] = call(mopt.opt.model, x, y);
            
            lf = l0;
            p0 = mopt.opt.params;
            for i=1:mopt.nsubsteps
                lf = step(mopt.opt, lf);
            end
            
            gamma = mopt.gamma;
            if gamma <= 0
                B = size(x,2);
                gamma = B/(B-0.9*gamma);
                mopt.gamma = mopt.gamma-B;
            end
            
            mopt.opt.params = mopt.opt.params*gamma + p0*(1-gamma);
            append(mopt.log, l0, lf);
        end        
    end
    
end

