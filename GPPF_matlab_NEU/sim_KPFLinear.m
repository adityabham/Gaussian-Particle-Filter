
clear all;
close all;

% Example taken from 
% 
% Chang, C. and Ansari, R., 2005. Kernel particle filter for visual 
% tracking. IEEE signal processing letters, 12(3), pp.242-245.

Nt = 100;
Np = 100;

var_prob = 0.1;

% kernel function and gp config
o_learn_theta = false;       % if true theta acts as a theta0 and the gp will learn theta
reg = 1e-6;                 % (K + reg*I)

% theta = [log(1), log(1e-5)];    % kernel parameters
% kfcn = @(XN,XM,theta) (exp(theta(2))^2)*exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2)) + reg*(pdist2(XN,XM) == 0);

theta = [log(0.01)];    % kernel parameters
kfcn = @(XN,XM,theta) exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2)) + reg*(pdist2(XN,XM) == 0);


% x_t = F x_t-1 + Gamma * w_t 
F = [1 1 0 0;
     0 1 0 0;
     0 0 1 1;
     0 0 0 1];

Gamma = [0.5 0;
        1   0;
        0  0.5;
        0  1];
     
H = [1 0 0 0;
     0 0 1 0 ];

% w_t ~ N (0, Q) 
q = 2e-3;
Q = q^2 * eye(2);
%Q4 = diag([sqrt(0.5)*q^2, q^2, sqrt(0.5)*q^2, q^2]);
Q4 = Gamma*Q*Gamma';
r = 5e-2;
% r = 1;
R = r^2 * eye(2);

x0 = [5.3, 0.43, 4.5, -0.52]';
P0 = diag([1, 0.1, 1, 0.1]);
 

x = x0;
P = P0;
save_x = zeros(Nt,2);
save_y = zeros(Nt,2);

save_kf_x = zeros(Nt,2);
save_kf_y = zeros(Nt,2);

save_gppf_x_map = zeros(Nt,2);
save_gppf_x_mmse = zeros(Nt,2);

save_pf_x_map = zeros(Nt,2);
save_pf_x_mmse = zeros(Nt,2);


kf = t_kf(x0, F, H, P0, Q4, R);

particles = randn(Np,length(x0)) * sqrt(Q4) * 5 + repmat(x0', Np,1);

pf = t_pf(particles, @(z) F*z ,@(z) H*z, Q4, R);
gppf = gp_pf(particles, @(z) F*z ,@(z) H*z, Q4, R, kfcn, theta, var_prob, o_learn_theta);

for n=1:Nt
    x = F *x + Gamma * mvnrnd([0, 0], Q)';
    y = H * x + mvnrnd([0, 0], R)';
    
    kf.update(y);
    [MSE_x, MAP_x] = gppf.update(y);
    
    [MSE_x_pf, MAP_x_pf] = pf.update(y);
    
    save_x(n,:) = [x(1),  x(3)]';
    save_y(n,:) = y';
    
    save_kf_x(n,:) = [kf.x(1),  kf.x(3)]';
    save_kf_y(n,:) = y';
    
    save_gppf_x_map(n,:) = [MAP_x(1),  MAP_x(3)]';
    save_gppf_x_mmse(n,:) = [MSE_x(1),  MSE_x(3)]';
    
    save_pf_x_map(n,:) = [MAP_x_pf(1),  MAP_x_pf(3)]';
    save_pf_x_mmse(n,:) = [MSE_x_pf(1),  MSE_x_pf(3)]';
    
end
 
 
 
figure;
plot(save_x(:,1),save_x(:,2))
hold on
plot(save_kf_x(:,1),save_kf_x(:,2))
plot(save_gppf_x_map(:,1),save_gppf_x_map(:,2), '--')
plot(save_gppf_x_mmse(:,1),save_gppf_x_mmse(:,2), '--')

plot(save_pf_x_map(:,1),save_pf_x_map(:,2),'-.')
plot(save_pf_x_mmse(:,1),save_pf_x_mmse(:,2), '-.')


RMSE_KF = sqrt((norm(save_kf_x - save_x).^2)/length(save_x))
RMSE_GP_MAP = sqrt((norm(save_gppf_x_map - save_x).^2)/length(save_x))
RMSE_GP_MMSE = sqrt((norm(save_gppf_x_mmse - save_x).^2)/length(save_x))
RMSE_PF_MAP = sqrt((norm(save_pf_x_map - save_x).^2)/length(save_x))
RMSE_PF_MMSE = sqrt((norm(save_pf_x_mmse - save_x).^2)/length(save_x))
legend('True','KF','GPPF (MAP)', 'GPPF (MMSE)', 'PF (MAP)', 'PF (MMSE)')


figure;
plot(sum(sqrt((save_gppf_x_map - save_x).^2), 2))
hold on
plot(sum(sqrt((save_gppf_x_mmse - save_x).^2), 2))
plot(sum(sqrt((save_pf_x_map - save_x).^2), 2),'-.')
plot(sum(sqrt((save_pf_x_mmse - save_x).^2), 2),'-.')
xlabel('samples')
ylabel('RMSE')
legend('GPPF (MAP)', 'GPPF (MMSE)', 'PF (MAP)', 'PF (MMSE)')

% plot(save_y(:,1),save_y(:,2))
 
% plot(save_kf_x(:,1),save_kf_x(:,2))

% plot(save_kf_y(:,1),save_kf_y(:,2))
 