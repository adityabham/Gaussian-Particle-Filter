
classdef t_pf < handle
    % Implements a GP Particle Filter
    % 
    % 
    %
    % usage:
    %   pf = t_pf(x0, F, H, P0, Q, R);
    %   pf.update(y)
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
       Np           % number of particles
       nx           % state dimension       
    end
        
    
    methods        
        function obj = t_pf(particles, f, h, Q, R)            
            % function obj = t_kf(x0, F, H, P0, Q, R) 
            % Obj constructor            
            obj.particles = particles;
            obj.f = f;
            obj.h = h;                        
            obj.Q = Q;
            obj.R = R;
            obj.Np = size(particles, 1);
            obj.nx = size(particles, 2);
            
        end        
        
        function [particle_aux, w_aux] = prop_particles(obj, y)
            
            particle_aux = zeros(obj.Np, obj.nx);            
            w_aux = zeros(obj.Np, 1);            

            for i=1:obj.Np
                % draw particles
                particle_aux(i,:) = obj.f(obj.particles(i,:)') + sqrt(obj.Q)*randn(obj.nx,1);
                % compute weights         
%                 w_aux(i) = exp(-0.5*(y-obj.h(particle_aux(i,:)'))'* (obj.R\(y-obj.h(particle_aux(i,:)'))));                
%                 w_aux(i) = (y-obj.h(particle_aux(i,:)'))'* (obj.R\(y-obj.h(particle_aux(i,:)')));          
                w_aux(i) = mvnpdf(y, obj.h(particle_aux(i,:)'), obj.R);         
            end

            % normalization
            t=sum(w_aux);          
            for i=1:obj.Np
                w_aux(i)=w_aux(i)/t;                
            end
                        
            
        end
        
        
        function [MMSE_est, MAP_est, Neff] = update(obj, y)
            
            [particle_aux, w_aux] = obj.prop_particles(y);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%% Compute desired estimates based on particles at time k %%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            MAP_est=MAP_estimation(w_aux,particle_aux);     %% MAP estimate            
            MMSE_est=MMSE_estimation(w_aux,particle_aux);   %% MMSE estimate
            
            Neff = 1/sum(w_aux.^2);
            
            obj.particles = obj.resampling(particle_aux, w_aux);
                       
            
        end

        function [particles] = resampling(obj, particles, w)
            % RESAMPLING algorithm to discard particles with small weights and to
            % concentrante on those with large weights
            %
            % actual_particle       actual particles at time k
            % actual_w              actual weights at time k
            %
            % particle              resampled particles at time k
            % w                     updated weights at time k
            % parent                index of the new particle parent (useful for ASIR)
            %
            % Pau Closas closas@gps.tsc.upc.edu

            Ns = size(particles,1);

            %for k=1:NUM_STATES
                c(1) = w(1);                         % Cumulative Density Function
                for i = 2:Ns
                    c(i) = c(i-1)+w(i);
                end
                i=1;
                u(1) = rand/Ns;
                for j = 1:Ns
                    u(j) = u(1) + (j-1)/Ns;	
                    while u(j) > c(i)
                        i = i + 1;
                        if i > Ns, i = Ns; break; end			    
                    end
                    particles(j,:) = particles(i,:);
                    w(j) = 1/Ns;                    
                end
        end
                        
    end
    
end


function MAP_est=MAP_estimation(w, particle)
    % MAP_ESTIMATION finds the MAP estimation using the posterior pdf defined
    % by the pairs particle-weights
    %
    % w             weights vector
    % particle      particles vector
    %
    % MAP_est      MAP estimation
    %
    % Pau Closas    closas@gps.tsc.upc.edu


    [V,I_map] = max(w.');
    MAP_est = particle(I_map,:);

end


function MMSE_est=MMSE_estimation(w, particle)
    % MMSE_ESTIMATION finds the MMSE estimation using the posterior pdf defined
    % by the pairs particle-weights
    %
    % w             weights vector
    % particle      particles vector
    %
    % MMSE_est      MMSE estimation
    %
    % Pau Closas    closas@gps.tsc.upc.edu


    % find the mean of the posterior PDF

    % en realitat NO cal ordenar, però per fer gràfiques va millor.
%     vec_aux = [particle.'  w.'];
%     [a,b] = sort(vec_aux(:,1));
%     vec_aux = [a  w(1,b).'];

    %mu_posterior=sum(vec_aux(:,1).*vec_aux(:,2));
    mu_posterior = particle'*w;

    MMSE_est = mu_posterior;

end

