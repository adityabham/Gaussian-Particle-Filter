


classdef t_kf < handle
    % Implements a Kalman filter
    % x_t = F x_{t-1} + q
    % y_t = H x_t + r
    %
    % usage:
    %   kf = t_kf(x0, F, H, P0, Q, R);
    %   kf.update(y)
    %
    % Author: Tales Imbiriba
    % Date: March, 3, 2020.
    %
    properties 
       x    % state vector
       F    % state transition matrix
       H    % observation matrix
       P    % state covariance matrix
       Q    % state noise covariance matrix
       R    % observation noise covariance matrix
    end
        
    
    methods        
        function obj = t_kf(x0, F, H, P0, Q, R)            
            % function obj = t_kf(x0, F, H, P0, Q, R) 
            % Obj constructor            
            obj.x = x0;
            obj.F = F;
            obj.H = H;            
            obj.P = P0;
            obj.Q = Q;
            obj.R = R;
            
        end        
        
        function [x_pred, P_pred] = predict(obj)            
           x_pred = obj.F * obj.x;
           P_pred = obj.F * obj.P * obj.F' + obj.Q;
        end
        
        function update(obj, y)
            [x_pred, P_pred] = obj.predict();
            e = y - obj.H*x_pred;
            S = obj.H * P_pred * obj.H' + obj.R;
            K = P_pred * obj.H' / S;
            obj.x = x_pred + K*e;
            obj.P = (eye(length(obj.x)) - K* obj.H) * P_pred;
        end
        
    end
    
end