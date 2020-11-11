
classdef t_gp < handle
    
    properties 
       X
       Y       
       kfcn
       theta 
       alpha
       L             
    end        
    
    methods
        function obj = t_gp(X,Y,kfcn, theta , o_learn_theta)            
            obj.X = X;
            obj.Y = Y;          
            obj.kfcn = kfcn;            
            if o_learn_theta == true
                obj.theta = obj.fit(theta);
            else
                obj.theta = theta;
            end
            K = kfcn(X, X, theta);            
            obj.L = chol(K,'lower');
            obj.alpha = obj.L'\(obj.L\Y);
            
        end
        
        function [m,s] = predict(obj, X)            
            Ks = obj.kfcn(obj.X, X, obj.theta);
            Kss = obj.kfcn(X, X, obj.theta);
            m = Ks'* obj.alpha;
            beta = obj.L\Ks;
            %s = diag(Kss - Ks'*(obj.L'\(obj.L\Ks)));
            s = diag(Kss - beta'*beta);
            s = max(s,0);
        end
        
        function theta_ = fit(obj, theta0)
            
            options = optimoptions('fminunc','Display','iter','Algorithm',...
                        'quasi-newton','SpecifyObjectiveGradient',false, ...
                        'MaxFunctionEvaluations', 100, 'MaxIterations', 100, ...
                        'StepTolerance',1e-10, 'Display','iter');
        
            f = @(x_)(obj.margloglik(x_));        
            [theta_, fval_] = fminunc(f,theta0, options);                                               
            disp(theta_)
            obj.theta = theta_;                      
            K = obj.kfcn(obj.X, obj.X, theta_);            
            obj.L = chol(K,'lower');
            obj.alpha = obj.L'\(obj.L\obj.Y);
        end
        
        function loglik = margloglik(obj, theta)
            K = obj.kfcn(obj.X, obj.X, theta);            
            L_ = chol(K,'lower');
            [ny, np] = size(obj.Y);
            alpha_ = L_'\(L_\obj.Y);
            loglik = -0.5 * trace(obj.Y'* alpha_)/ny -sum(log(diag(L_))) -(np/2)*log(2*pi);
            loglik = -loglik ;
        end
        
    end
    
end