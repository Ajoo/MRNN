classdef MetaValOptimizer < handle
    %METAOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        opt
        gamma = 0.1;
        maxsubsteps = 100;
        persistence = 2;
        
        l0_ = Inf;
        log = Log('l0', 'lf', 'nsubsteps');
    end
    
    methods
        function mopt = MetaValOptimizer(opt)
            mopt.opt = opt;
        end
        
        function plot(mopt)
            subplot(2,1,1)
            semilogy([getfield(mopt.log, 'l0') getfield(mopt.log, 'lf')])
            subplot(2,1,2)
            plot(getfield(mopt.log, 'nsubsteps'))
        end
        
        function [l0, lf, i] = step(mopt, x, y)
            reset(mopt.opt); % reset optmizer state
            if isinf(mopt.l0_)
                [~, mopt.l0_] = call(mopt.opt.model, x, y);
                l0 = NaN; lf = NaN; i = 0;
                return
            end
            
            l0 = mopt.l0_;
            [~, lv0] = evaluate(mopt.opt.model, x, y);
            
            p = 0;
            for i=1:mopt.maxsubsteps
                lf = step(mopt.opt, l0);
                [~, lvf] = evaluate(mopt.opt.model, x, y);
                
                if (lvf-lv0) > 0
                    p = p + 1;
                    if p >= mopt.persistence
                        break
                    end
                else
                    p = 0;
                end
%                 l0 = lf;
                lv0 = lvf;
            end
            
            [~, mopt.l0_] = call(mopt.opt.model, x, y);
            
            append(mopt.log, l0, lf, i);
        end        
    end
    
end

