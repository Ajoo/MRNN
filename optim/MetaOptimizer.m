classdef MetaOptimizer < handle
    %METAOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        opt
        gamma = 0.1;
        nsubsteps = 10;
    end
    
    methods
        function mopt = MetaOptimizer(opt)
            mopt.opt = opt;
        end
        
        function l0 = step(mopt, l0)
            reset(mopt.opt); % reset optmizer state
            p0 = mopt.opt.params;
            for i=1:mopt.nsubsteps
                l0 = step(mopt.opt, l0);
            end
            mopt.opt.params = mopt.opt.params*mopt.gamma + p0*(1-mopt.gamma);
        end        
    end
    
end

