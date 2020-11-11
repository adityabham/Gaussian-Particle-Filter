
classdef gp_pf < handle
    % Implements a GP Particle Filter
    % 
    % 
    %
    % usage:
    %   kf = t_kf(x0, F, H, P0, Q, R);
    %   kf.update(y)
    %
    % Author: Tales Imbiriba
    % Date: March, 3, 2020.
    %
    properties 
       particles
       weights
       f            % state transition function
       h            % observation function       
       Q            % state noise covariance matrix
       R            % observation noise covariance matrix
%        gp           % gaussian process obj
       Np           % number of particles
       nx           % state dimension
       kfcn         % kernel function
       theta        % kernel parameters
       var_prob     % probability of sampling from m+s
       o_learn_theta
       gp_verbose
    end
        
    
    methods        
        function obj = gp_pf(particles, f, h, Q, R, kfcn, theta, var_prob, o_learn_theta, gp_verbose)            
            % function obj = t_kf(x0, F, H, P0, Q, R) 
            % Obj constructor            
            obj.particles = particles;
            obj.f = f;
            obj.h = h;                        
            obj.Q = Q;
            obj.R = R;
            obj.Np = size(particles, 1);
            obj.nx = size(particles, 2);
            obj.kfcn = kfcn;
            obj.theta = theta;
            obj.var_prob = var_prob;
            obj.o_learn_theta = o_learn_theta;
            obj.gp_verbose = gp_verbose;
        end        
        
        function [particle_aux, w_aux] = prop_particles(obj, y)
%         function [tgp, support_lims] = prop_particles(obj, y)
            
            particle_aux = zeros(obj.Np, obj.nx);            
            w_aux = zeros(obj.Np, 1);            

            for i=1:obj.Np
                % draw particles
            %     particle_aux(:,i) = state_func(actual_particle(k-1),k) + sqrt(Q)*randn;
                particle_aux(i,:) = obj.f(obj.particles(i,:)') + sqrt(obj.Q)*randn(obj.nx,1);
                % compute weights
            %     w_aux(:,i) = exp(-0.5*(meas-meas_func(particle_aux(:,i)))*inv(R)*(meas-meas_func(particle_aux(:,i))).');
                w_aux(i) = exp(-0.5*(y-obj.h(particle_aux(i,:)'))'* (obj.R\(y-obj.h(particle_aux(i,:)'))));                
%                 w_aux(i) = (y-obj.h(particle_aux(i,:)'))'* (obj.R\(y-obj.h(particle_aux(i,:)')));                
            end

            % normalization
            t=sum(w_aux);          
            for i=1:obj.Np
                w_aux(i)=w_aux(i)/t;                
            end
                        
            
        end
        
        
        function [MMSE_est, MAP_est, Neff] = update(obj, y)
            
            [particle_aux, w_aux] = obj.prop_particles(y);
            % gp fit            
            gp = t_gp(particle_aux, w_aux, obj.kfcn, obj.theta, obj.o_learn_theta);            
            support_lims = [min(particle_aux); max(particle_aux)];
            
            n = 500;
            test_points = zeros(n, obj.nx);
            for i=1:obj.nx                
                test_points(:,i) = linspace(support_lims(1,i), support_lims(2,i), n);
            end
            
            [m,s] = gp.predict(test_points);
            
            [~, idx] = max(m);
%             MAP_est = test_points(idx,:);
            p0 = test_points(idx,:);
            [~,MAP_est] = obj.findmax(gp, p0, support_lims(1,:), support_lims(2,:), false);
            
            MMSE_est = zeros(obj.nx,1);
            for i=1:obj.nx
                MMSE_est(i) =  sum(particle_aux(:,i).*gp.alpha)/ sum(gp.alpha); 
            end
            
            Neff = 1/sum(w_aux.^2);
            
            obj.particles = obj.gp_resampling(particle_aux, gp);
            
            
            if obj.gp_verbose 
                figure(1)
                clf;
                stem3(particle_aux(:,1), particle_aux(:,3), gp.predict(particle_aux))
                stem3(particle_aux(:,1), particle_aux(:,3), w_aux)
                pause(0.01)
            end
%             plot(w_aux - gp.predict(particle_aux))
%             plot(test_points(:,1) , m)
%             hold on
%             stem(particle_aux(:,1),w_aux)
%             stem(obj.particles(:,1),0*ones(size(obj.particles(:,1))))
%             plot(test_points(:,1) , m + 3*sqrt(s), '--k')
%             plot(test_points(:,1) , m - 3*sqrt(s), '--k')
%             legend('$\mu(x)$', 'old particles', 'new particles', '$\mu(x) \pm 3 s(x)$', 'interpreter', 'latex', 'fontsize',16)
            
            
        end

        function [new_particles] = gp_resampling(obj, particles, gp)
            
            if obj.var_prob > 0 
                var_particles = zeros(obj.Np, obj.nx);
                for i=1:obj.Np-1
                    delta = 0.5*(particles(i+1,:) - particles(i,:));
                    var_particles(i,:) = delta + particles(i,:);
                end
                var_particles(end,:) = particles(end,:) + delta;
                var_particles = [var_particles; particles];
                [v_m, v_var] = gp.predict(var_particles);
                var_gp = t_gp(var_particles, v_m + 3 *sqrt(v_var), obj.kfcn, gp.theta , false);                
                np_var = floor(obj.Np * obj.var_prob);
                
                % sampling from mu + 3 sigma:
                p = abs(var_gp.alpha);
                p(p<=max(abs(var_gp.alpha))/2) = 1e-10;
    %             p(p<=0) = 1e-5;
                p = p./sum(p);          
                gm = gmdistribution(var_gp.X, exp(var_gp.theta(1))*eye(obj.nx), p);
                var_new_particles = random(gm, np_var);                 
            else
                np_var = 0;
                var_new_particles = [];
            end
            
            np = obj.Np - np_var;
            
            p = abs(gp.alpha);
            p(p<=max(abs(gp.alpha))/2) = 1e-10;
%             p(p<=0) = 1e-5;
            p = p./sum(p);          
            gm = gmdistribution(gp.X, exp(gp.theta(1))*eye(obj.nx), p);
            new_particles = random(gm, np);
            
            new_particles = [new_particles; var_new_particles];
            
%             % implementing entropy
%             % 1 - determine support
%             % for now I'll determine the support as [min(particle), max(particle)]
%             ll = max(particles) - min(particles);
%             support = [min(particles); max(particles)];    
%             sup_len = (support(2,:) - support(1,:));
% 
%             % 2 - grid for acessing the largest density value
%             n = 500;
%             grid = zeros(n, obj.nx);
%             for i=1:obj.nx                
%                 grid(:,i) = linspace(support(1,i), support(2,i), n);
%             end
%  
%             % # of particles
%             np = length(particles);
%             new_particles = zeros(size(particles));
%             
%             [~, idx] = max(gp.predict(particles));
% %             p0 = (support(2,:)-support(1,:))/2 + support(1,:);
%             p0 = particles(idx,:);
%             
%             max_val = obj.findmax(gp, p0, support(1,:), support(2,:), false);                
%             Mm = 1.1*max_val;
%             
%             max_val = obj.findmax(gp, p0, support(1,:), support(2,:), true);
%             Mv = 1.1*max_val;
%             
% 
%             for j=1:np
%                 pvar = rand;
%                 if pvar < obj.var_prob
%                     M = Mv;
%                 else
%                     M = Mm;
%                 end
%                 while true
%                     u1 = rand(1,obj.nx).* sup_len + support(1,:);                   
%                     
%                     [mu,su] = gp.predict(u1);  
%                     if mu < M/5
%                         continue;
%                     end
%                     
%                     u2 = rand * M;
%                     if pvar < obj.var_prob
%                         if u2 < mu + 3*sqrt(su)
%                             new_particles(j,:) = u1;
%                             break
%                         end
%                     else
%                         if u2 < mu
%                             new_particles(j,:) = u1;
%                             break
%                         end
%                     end
%                 end
% 
%             end
        end
        
    end
    
    methods(Static)
        function [max_val, p] = findmax(gp, p0, lb, ub, o_var)
             f = @(x_)(gp_pf.cost(x_, gp, o_var));
             evalc('p = fmincon(f,p0,[],[],[],[],lb,ub)');
             max_val = gp.predict(p);
        end
        function c = cost(x_, gp, o_var)
            [m,s] = gp.predict(x_);
            if o_var
                c = m + 3*sqrt(s);
            else
                c = m;
            end
            c=-c;
        end
    end
end